"""First-look charts for the TTC subway delay data. Writes PNGs to figures/."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"  # blue + orange: the pair most colour-blind people can tell apart
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.axisbelow": True, "axes.titlelocation": "left",
})


def load():
    d = pd.read_parquet(ROOT / "data" / "delays_clean.parquet")
    return d[d.is_delay & ~d.is_closure]


def hourly_profile(dl):
    days = dl.timestamp.dt.normalize()
    wk = dl.weekday < 5
    n_wk, n_we = days[wk].nunique(), days[~wk].nunique()
    a = dl[wk].groupby("hour").size().reindex(range(24), fill_value=0) / n_wk
    b = dl[~wk].groupby("hour").size().reindex(range(24), fill_value=0) / n_we
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(a.index, a.values, color=BLUE, lw=2, label="Weekdays")
    ax.plot(b.index, b.values, color=ORANGE, lw=2, label="Weekends")
    ax.text(17, a[17] + 0.04, "Weekdays", color=INK2, ha="center", va="bottom")
    ax.text(20, b[20] - 0.06, "Weekends", color=INK2, ha="center", va="top")
    ax.set(xlim=(0, 23), ylim=(0, None), xticks=range(0, 24, 3),
           xlabel="Hour of day", ylabel="Delays per day, that hour")
    ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 3)])
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Two peaks: when service starts, and the evening rush")
    fig.tight_layout(); fig.savefig(FIG / "01_hourly_profile.png", dpi=160); plt.close(fig)


def top_stations(dl):
    s = (dl[dl.location_kind == "station"].groupby("station").min_delay.sum()
         .sort_values().tail(12) / 1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(s.index.str.title(), s.values, color=BLUE, height=0.62)
    for y, v in enumerate(s.values):
        ax.text(v + 120, y, f"{v:,.0f}", va="center", color=INK2, fontsize=9)
    ax.set(xlabel="Total delay minutes, Jan 2023 - Sep 2026")
    ax.grid(axis="y", visible=False)
    ax.set_title("Interchange and terminal stations lose the most time")
    fig.tight_layout(); fig.savefig(FIG / "02_top_stations.png", dpi=160); plt.close(fig)


def causes(dl):
    c = dl.groupby("cause_group").min_delay.sum()
    c = c[c.index != "Other"].sort_values()
    share = c / c.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(share.index, share.values, color=BLUE, height=0.6)
    for y, v in enumerate(share.values):
        ax.text(v + 0.6, y, f"{v:.0f}%", va="center", color=INK2, fontsize=9)
    ax.set(xlabel="Share of all delay minutes (%)", xlim=(0, 40))
    ax.grid(axis="y", visible=False)
    ax.set_title("Passenger, security and misc. incidents: two thirds of lost time")
    fig.tight_layout(); fig.savefig(FIG / "03_causes.png", dpi=160); plt.close(fig)


def heatmap(dl):
    days = dl.timestamp.dt.normalize().drop_duplicates()
    n_per_wd = days.dt.dayofweek.value_counts().sort_index()
    m = dl.groupby(["weekday", "hour"]).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
    m = m.div(n_per_wd, axis=0).loc[:, 5:]  # service starts ~6am; 5am shows opening incidents
    fig, ax = plt.subplots(figsize=(9, 3.4))
    im = ax.imshow(m.values, aspect="auto", cmap=SEQ)
    ax.set_yticks(range(7)); ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, m.shape[1], 2)); ax.set_xticklabels([f"{h}" for h in m.columns[::2]])
    ax.set_xlabel("Hour of day"); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.outline.set_visible(False)
    cb.set_label("Delays per day", color=INK2)
    ax.set_title("When delays happen: weekday 4-6pm is the hot zone (Sunday 8am = Sunday opening)")
    fig.tight_layout(); fig.savefig(FIG / "04_heatmap.png", dpi=160); plt.close(fig)


def weather_effect(dl):
    w = pd.read_csv(ROOT / "data_raw" / "weather_hourly.csv", skiprows=3)
    w.columns = ["time", "temp", "precip", "snow", "wind"]
    w["time"] = pd.to_datetime(w["time"])
    x = dl[dl.min_delay >= 5]
    cnt = x.groupby(x.timestamp.dt.floor("h")).size().rename("n")
    p = w.set_index("time").join(cnt).fillna({"n": 0})
    p = p[p.index.hour >= 6]
    groups = {
        "Normal\n(-10 to 25 C, dry)": (p.temp.between(-10, 25)) & (p.precip == 0),
        "Rain or snow\nfalling": p.precip > 0,
        "Heavy snow\n(1 cm+ per hour)": p.snow >= 1,
        "Deep cold\n(below -10 C)": p.temp < -10,
    }
    rates = pd.Series({k: p.loc[m, "n"].mean() for k, m in groups.items()})
    hours = pd.Series({k: int(m.sum()) for k, m in groups.items()})
    rel = rates / rates.iloc[0] * 100 - 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(rates.index, rates.values, color=BLUE, width=0.55)
    for i, (r, h) in enumerate(zip(rates.values, hours.values)):
        lab = f"{r:.2f}" if i == 0 else f"{r:.2f} ({rel.iloc[i]:+.0f}%)"
        ax.text(i, r + 0.02, lab, ha="center", color=INK2, fontsize=9)
        ax.text(i, 0.03, f"{h:,} h", ha="center", color="white", fontsize=8)
    ax.set(ylabel="Delays of 5+ min per hour", ylim=(0, 1.1))
    ax.grid(axis="x", visible=False)
    ax.set_title("Snow and deep cold raise the delay rate by about a third")
    fig.tight_layout(); fig.savefig(FIG / "05_weather.png", dpi=160); plt.close(fig)
    return rates, hours


if __name__ == "__main__":
    dl = load()
    hourly_profile(dl); top_stations(dl); causes(dl); heatmap(dl)
    print(weather_effect(dl))
    print(sorted(p.name for p in FIG.iterdir()))
