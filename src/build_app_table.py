"""Precompute the risk tables the app reads, so the app never trains anything.

Final models are trained on ALL data (2023 - Sep 2026); the test scores in the README come
from model.py, which holds out 2026. Output: app/risk_station.parquet, app/risk_line.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from model import CAT, NUM, build_grid, load_weather, service_hour

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
APP.mkdir(exist_ok=True)

SCENARIOS = ["Typical for the month", "Rain", "Heavy snow", "Deep cold (below -10 C)"]


def weather_scenarios(w: pd.DataFrame) -> pd.DataFrame:
    """Typical weather per month x hour, plus three 'what if' versions of it."""
    w = w.assign(month=w.hour_start.dt.month, hour=w.hour_start.dt.hour)
    typ = w.groupby(["month", "hour"])[["temp", "precip", "snow", "wind"]].median().reset_index()
    out = []
    for s in SCENARIOS:
        t = typ.copy()
        t["scenario"] = s
        if s == "Rain":
            t["precip"], t["snow"], t["temp"] = 2.0, 0.0, t.temp.clip(lower=2)
        elif s == "Heavy snow":
            t["precip"], t["snow"], t["temp"] = 1.5, 1.5, t.temp.clip(upper=-3)
        elif s.startswith("Deep cold"):
            t["temp"], t["precip"], t["snow"] = -15.0, 0.0, 0.0
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t["cold"] = (t.temp < -10).astype(int)
    t["snowing"] = (t.snow > 0).astype(int)
    return t


def steady_weather(q: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Apply each weather scenario as ONE average adjustment instead of cell-by-cell model output.

    Heavy snow and deep cold are rare in the data (41 and 309 hours), so in rare combinations
    (e.g. Sunday 8am + snow) the model extrapolates wildly. We keep the model's prediction for
    typical weather, measure each scenario's average effect on the odds across all cells, and
    apply that same odds ratio everywhere.
    """
    odds = lambda p: p / (1 - p)
    typ = q[q.scenario == SCENARIOS[0]].set_index(keys).risk
    out = []
    for s in SCENARIOS:
        part = q[q.scenario == s].copy()
        if s != SCENARIOS[0]:
            ratio = float(np.exp((np.log(odds(part.set_index(keys).risk)) - np.log(odds(typ))).mean()))
            o = odds(part.set_index(keys).index.map(typ).values) * ratio
            part["risk"] = o / (1 + o)
            print(f"  {s}: odds x{ratio:.2f}")
        out.append(part)
    q = pd.concat(out)
    q["risk"] = q.risk.astype("float32")
    return q


def service_rows(index_cols: dict) -> pd.DataFrame:
    """Every weekday x passenger-service hour."""
    rows = [(wd, h) for wd in range(7) for h in range(24)
            if h <= 1 or (h >= 8 if wd == 6 else h >= 6)]
    return pd.DataFrame(rows, columns=["weekday", "hour"])


def fit_gb(X, y, cats):
    return HistGradientBoostingClassifier(
        categorical_features=cats, learning_rate=0.08, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=0).fit(X, y)


def main():
    d = pd.read_parquet(ROOT / "data" / "delays_clean.parquet")
    w = load_weather()
    sc = weather_scenarios(w)
    base = service_rows({})

    # ---- Station model --------------------------------------------------------
    grid = build_grid(d, w)
    X = grid[CAT + NUM].copy()
    cats = {c: pd.Categorical(X[c]).categories for c in ["station", "line"]}
    for c, cat in cats.items():
        X[c] = pd.Categorical(X[c], categories=cat)
    gb = fit_gb(X, grid.y, ["station", "line"])

    stations = grid[["station", "line"]].drop_duplicates()
    q = (stations.merge(base, how="cross").merge(pd.DataFrame({"month": range(1, 13)}), how="cross")
         .merge(sc, on=["month", "hour"]))
    q["first_service_hour"] = (((q.weekday < 6) & (q.hour == 6)) | ((q.weekday == 6) & (q.hour == 8))).astype(int)
    Xq = q[CAT + NUM].copy()
    for c, cat in cats.items():
        Xq[c] = pd.Categorical(Xq[c], categories=cat)
    q["risk"] = gb.predict_proba(Xq)[:, 1]
    q = steady_weather(q, ["station", "weekday", "hour", "month"])
    st_out = q[["station", "line", "weekday", "hour", "month", "scenario", "risk"]]
    st_out.to_parquet(APP / "risk_station.parquet", index=False)

    # Observed history per station, for context in the app
    hist = grid.groupby("station").agg(hours=("y", "size"), delay_hours=("y", "sum")).reset_index()
    hist["observed_rate"] = hist.delay_hours / hist.hours
    hist.to_parquet(APP / "station_history.parquet", index=False)

    # ---- Line model (any 5+ min delay anywhere on the line that hour) ----------
    hrs = pd.date_range(w.hour_start.min(), w.hour_start.max(), freq="h")
    hrs = hrs[service_hour(pd.Series(hrs)).values]
    lg = pd.MultiIndex.from_product([["YU", "BD", "SHP"], hrs], names=["line", "hour_start"]).to_frame(index=False)
    h = d[(d.min_delay >= 5) & ~d.is_closure & d.line.isin(["YU", "BD", "SHP"])].copy()
    h["hour_start"] = h.timestamp.dt.floor("h")
    h = h.groupby(["line", "hour_start"]).size().rename("n").reset_index()
    lg = lg.merge(h, how="left").merge(w, how="left")
    lg["y"] = lg.n.notna().astype(int)
    ts = lg.hour_start
    lg["hour"], lg["weekday"], lg["month"] = ts.dt.hour, ts.dt.dayofweek, ts.dt.month
    LX = ["line", "hour", "weekday", "month", "temp", "precip", "snow", "wind"]
    lcat = pd.Categorical(lg.line).categories
    lgX = lg[LX].copy(); lgX["line"] = pd.Categorical(lgX.line, categories=lcat)
    lm = HistGradientBoostingClassifier(categorical_features=["line"], max_iter=300, learning_rate=0.05,
                                        min_samples_leaf=100, random_state=0).fit(lgX, lg.y)
    lq = (pd.DataFrame({"line": ["YU", "BD", "SHP"]}).merge(base, how="cross")
          .merge(pd.DataFrame({"month": range(1, 13)}), how="cross").merge(sc, on=["month", "hour"]))
    lqX = lq[LX].copy(); lqX["line"] = pd.Categorical(lqX.line, categories=lcat)
    lq["risk"] = lm.predict_proba(lqX)[:, 1]
    lq = steady_weather(lq, ["line", "weekday", "hour", "month"])
    lq[["line", "weekday", "hour", "month", "scenario", "risk"]].to_parquet(APP / "risk_line.parquet", index=False)

    print(f"station table {len(st_out):,} rows, line table {len(lq):,} rows")
    print("network average 5+ min delay rate per station-hour:", round(grid.y.mean(), 4))
    for f in APP.glob("*.parquet"):
        print(f.name, round(f.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
