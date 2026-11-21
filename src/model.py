"""Predict the chance of a 5+ minute subway delay at a station in a given hour.

One row = one station x one passenger-service hour. Target = 1 if that station logged
a delay of 5+ minutes in that hour (planned closures excluded).

Honest test: train on the past, test on a later year the model never saw.
Usage: python src/model.py [train_end_year] [test_year]   (default 2024 2025)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TRAIN_END = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
TEST_YEAR = int(sys.argv[2]) if len(sys.argv) > 2 else 2025

SRT = {"LAWRENCE EAST", "ELLESMERE", "MIDLAND", "SCARBOROUGH CENTRE", "MCCOWAN"}  # Line 3 closed July 2023
CAT = ["station", "line", "weekday", "hour", "month"]
NUM = ["first_service_hour", "temp", "precip", "snow", "wind", "cold", "snowing"]


def service_hour(ts: pd.Series) -> pd.Series:
    """True for hours with passenger trains: 6am-1:59am Mon-Sat, 8am-1:59am Sunday."""
    h, wd = ts.dt.hour, ts.dt.dayofweek
    late = h <= 1
    day = np.where(wd == 6, h >= 8, h >= 6)
    return late | day


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(ROOT / "data_raw" / "weather_hourly.csv", skiprows=3)
    w.columns = ["hour_start", "temp", "precip", "snow", "wind"]
    w["hour_start"] = pd.to_datetime(w["hour_start"])
    return w


def build_grid(d: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    # Stations and the line each mostly sits on
    st = d[(d.location_kind == "station") & ~d.station.isin(SRT)]
    station_line = (st[st.line.isin(["YU", "BD", "SHP"])].groupby("station").line
                    .agg(lambda s: s.value_counts().index[0]))
    # Interchanges get their own label: delays there can come from either line
    for s in ["BLOOR-YONGE", "ST GEORGE", "SPADINA", "SHEPPARD-YONGE"]:
        if s in station_line.index:
            station_line[s] = "INTERCHANGE"

    hours = pd.date_range(w.hour_start.min(), w.hour_start.max(), freq="h")
    hours = hours[service_hour(pd.Series(hours)).values]
    grid = pd.MultiIndex.from_product([station_line.index, hours], names=["station", "hour_start"]).to_frame(index=False)

    # Target: any 5+ minute delay at that station in that hour
    hit = st[(st.min_delay >= 5) & ~st.is_closure].copy()
    hit["hour_start"] = hit.timestamp.dt.floor("h")
    hit = hit.groupby(["station", "hour_start"]).size().rename("n_delays").reset_index()
    grid = grid.merge(hit, on=["station", "hour_start"], how="left")
    grid["y"] = grid.n_delays.notna().astype(int)

    # Features
    grid["line"] = grid.station.map(station_line)
    ts = grid.hour_start
    grid["hour"] = ts.dt.hour
    grid["weekday"] = ts.dt.dayofweek
    grid["month"] = ts.dt.month
    grid["year"] = ts.dt.year
    grid["first_service_hour"] = (((grid.weekday < 6) & (grid.hour == 6)) | ((grid.weekday == 6) & (grid.hour == 8))).astype(int)
    grid = grid.merge(w, on="hour_start", how="left")
    grid["cold"] = (grid.temp < -10).astype(int)
    grid["snowing"] = (grid.snow > 0).astype(int)
    return grid.drop(columns="n_delays")


def lift_table(y, p, bins=10):
    """Share of real delays caught in the riskiest 10%, 20%... of hours."""
    order = np.argsort(-p)
    y_sorted = np.asarray(y)[order]
    cum = np.cumsum(y_sorted) / y_sorted.sum()
    out = {}
    for q in (0.05, 0.10, 0.20, 0.30):
        out[f"top_{int(q*100)}pct_hours_catch"] = float(cum[int(len(cum) * q) - 1])
    return out


def evaluate(name, y, p):
    r = {"model": name, "auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
         "brier": brier_score_loss(y, p)}
    r.update(lift_table(y, p))
    return r


def main():
    d = pd.read_parquet(ROOT / "data" / "delays_clean.parquet")
    w = load_weather()
    grid = build_grid(d, w)
    train = grid[grid.year <= TRAIN_END]
    test = grid[grid.year == TEST_YEAR]
    print(f"grid rows {len(grid):,} | train {len(train):,} ({train.year.min()}-{TRAIN_END}) | test {len(test):,} ({TEST_YEAR})")
    print(f"delay rate train {train.y.mean():.3%} | test {test.y.mean():.3%}")

    results = []

    # Baseline 0: always predict the training delay rate (AUC 0.5 by definition)
    results.append(evaluate("Always the average", test.y, np.full(len(test), train.y.mean())))

    # Baseline 1: historical rate for this station x hour x day type (no ML, just a lookup)
    key = lambda df: df.station + "|" + df.hour.astype(str) + "|" + (df.weekday >= 5).astype(str)
    rates = train.groupby(key(train)).y.mean()
    p_hist = key(test).map(rates).fillna(train.y.mean()).values
    results.append(evaluate("Historical rate lookup", test.y, p_hist))

    # Model 1: logistic regression (one-hot categories + scaled weather)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
    ])
    lr = make_pipeline(pre, LogisticRegression(max_iter=300, C=1.0))
    lr.fit(train[CAT + NUM], train.y)
    p_lr = lr.predict_proba(test[CAT + NUM])[:, 1]
    results.append(evaluate("Logistic regression", test.y, p_lr))

    # Model 2: gradient boosting (learns interactions like hour x weekday x station)
    X_tr, X_te = train[CAT + NUM].copy(), test[CAT + NUM].copy()
    for c in ["station", "line"]:
        cats = pd.Categorical(X_tr[c]).categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_te[c] = pd.Categorical(X_te[c], categories=cats)
    gb = HistGradientBoostingClassifier(
        categorical_features=["station", "line"], learning_rate=0.08, max_iter=400,
        max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0)
    gb.fit(X_tr, train.y)
    p_gb = gb.predict_proba(X_te)[:, 1]
    results.append(evaluate("Gradient boosting", test.y, p_gb))

    res = pd.DataFrame(results).set_index("model")
    print(res.round(4).to_string())

    # What does the boosted model rely on? (permutation importance on a test sample)
    idx = test.sample(200_000, random_state=0).index
    pi = permutation_importance(gb, X_te.loc[idx], test.y.loc[idx], scoring="roc_auc",
                                n_repeats=3, random_state=0)
    imp = pd.Series(pi.importances_mean, index=CAT + NUM).sort_values(ascending=False)
    print("\nAUC lost when each feature is shuffled:\n", imp.round(4).to_string())

    # Calibration check: predicted vs actual by risk decile
    t = test[["station", "hour_start", "y"]].copy()
    t["p"] = p_gb
    t["decile"] = pd.qcut(t.p.rank(method="first"), 10, labels=range(1, 11))
    calib = t.groupby("decile", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"))
    print("\nCalibration by risk decile:\n", calib.round(4).to_string())

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    res.to_csv(out / f"model_results_test{TEST_YEAR}.csv")
    imp.to_csv(out / f"feature_importance_test{TEST_YEAR}.csv", header=["auc_drop"])
    calib.to_csv(out / f"calibration_test{TEST_YEAR}.csv")
    t.to_parquet(ROOT / "data" / f"predictions_test{TEST_YEAR}.parquet", index=False)
    json.dump({"train": f"2023-{TRAIN_END}", "test": TEST_YEAR, "test_rows": int(len(test)),
               "test_delay_rate": float(test.y.mean())}, open(out / f"run_test{TEST_YEAR}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
