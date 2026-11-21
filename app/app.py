"""TTC subway delay risk: pick a station, day, hour and weather; see the chance of a 5+ minute delay.

Run locally:  streamlit run app/app.py
The risk tables are precomputed by src/build_app_table.py.
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"
LINE_NAMES = {"YU": "Line 1 Yonge-University", "BD": "Line 2 Bloor-Danforth", "SHP": "Line 4 Sheppard"}
INTERCHANGES = {"BLOOR-YONGE": ["YU", "BD"], "ST GEORGE": ["YU", "BD"], "SPADINA": ["YU", "BD"],
                "SHEPPARD-YONGE": ["YU", "SHP"]}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
REPO = "https://github.com/HidayaPatel/ttc-delay-predictor"

st.set_page_config(page_title="TTC Subway Delay Risk", page_icon="🚇", layout="centered")


@st.cache_data
def load():
    s = pd.read_parquet(HERE / "risk_station.parquet")
    l = pd.read_parquet(HERE / "risk_line.parquet")
    typical = s[s.scenario == "Typical for the month"].risk
    return s, l, float(typical.mean()), float(typical.quantile(0.5)), float(typical.quantile(0.9))


def hour_label(h: int) -> str:
    return {0: "12 am", 12: "12 pm"}.get(h, f"{h % 12} {'am' if h < 12 else 'pm'}")


def nice(name: str) -> str:
    return name.title().replace("Tmu", "TMU")


def lines_for(station: str, line: str) -> list:
    return INTERCHANGES.get(station, [line] if line in LINE_NAMES else [])


s, l, avg, p50, p90 = load()

st.title("🚇 TTC Subway Delay Risk")
st.caption("How likely is a delay of 5+ minutes at your station, at your time, in this weather? "
           "Based on every delay the TTC logged from January 2023 to September 2026.")

# ---- Inputs -------------------------------------------------------------------
stations = sorted(s.station.unique())
c1, c2 = st.columns(2)
station = c1.selectbox("Station", stations, index=stations.index("BLOOR-YONGE"), format_func=nice)
day = c2.selectbox("Day", DAYS, index=1)
wd = DAYS.index(day)
service_hours = sorted(s[(s.station == station) & (s.weekday == wd)].hour.unique(),
                       key=lambda h: (h < 5, h))  # morning first, 12am/1am last
c3, c4, c5 = st.columns([1, 1, 1.4])
hour = c3.selectbox("Hour", service_hours, index=service_hours.index(17) if 17 in service_hours else 0,
                    format_func=hour_label)
month = c4.selectbox("Month", range(1, 13), index=10, format_func=lambda m: MONTHS[m - 1])
scenario = c5.selectbox("Weather", sorted(s.scenario.unique(), key=lambda x: x != "Typical for the month"))

# ---- Station risk ---------------------------------------------------------------
pick = s[(s.station == station) & (s.weekday == wd) & (s.hour == hour) & (s.month == month) & (s.scenario == scenario)]
risk = float(pick.risk.iloc[0])
line_code = pick.line.iloc[0]
band = "High" if risk >= p90 else ("Elevated" if risk >= p50 else "Low")
band_icon = {"High": "🔴", "Elevated": "🟠", "Low": "🟢"}[band]

st.divider()
m1, m2 = st.columns(2)
m1.metric(f"Chance of a 5+ min delay at {nice(station)}", f"{risk:.1%}",
          help="Probability that at least one delay of 5+ minutes is logged at this station in this hour.")
m2.metric("Compared with an average station-hour", f"{risk / avg:.1f}x",
          help=f"The average across all stations and service hours is {avg:.2%}.")
st.markdown(f"**{band_icon} {band} risk.** About **1 in {max(1, round(1 / risk))}** hours like this "
            f"has a delay of 5+ minutes at this station.")

# ---- Line risk ------------------------------------------------------------------
codes = lines_for(station, line_code)
if codes:
    lr = l[(l.weekday == wd) & (l.hour == hour) & (l.month == month) & (l.scenario == scenario)].set_index("line").risk
    parts = [f"**{LINE_NAMES[c]}: {lr[c]:.0%}**" for c in codes]
    st.markdown("Chance of a 5+ min delay **anywhere on the line** this hour: " + " · ".join(parts))

# ---- Risk through the day -----------------------------------------------------------
st.subheader(f"Through the day at {nice(station)}")
day_df = s[(s.station == station) & (s.weekday == wd) & (s.month == month) & (s.scenario == scenario)].copy()
day_df["order"] = day_df.hour.where(day_df.hour >= 5, day_df.hour + 24)
day_df["Time"] = day_df.hour.map(hour_label)
day_df["Risk"] = day_df.risk
order = day_df.sort_values("order").Time.tolist()
base = alt.Chart(day_df).encode(
    x=alt.X("Time:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
    y=alt.Y("Risk:Q", title="Chance of a 5+ min delay", axis=alt.Axis(format="%"), scale=alt.Scale(zero=True)),
    tooltip=[alt.Tooltip("Time:N"), alt.Tooltip("Risk:Q", format=".1%")])
line = base.mark_line(color=BLUE, strokeWidth=2)
dots = base.mark_point(color=BLUE, filled=True, size=40, opacity=0.001)  # invisible hover targets
here = alt.Chart(day_df[day_df.hour == hour]).mark_point(color=ORANGE, filled=True, size=110).encode(
    x=alt.X("Time:N", sort=order), y="Risk:Q", tooltip=[alt.Tooltip("Time:N"), alt.Tooltip("Risk:Q", format=".1%")])
st.altair_chart((line + dots + here).properties(height=260), width="stretch")
st.caption(f"Orange dot = your chosen hour ({hour_label(hour)}). {day}s in {MONTHS[month - 1]}, weather: {scenario.lower()}.")

# ---- Riskiest stations right now ------------------------------------------------------
st.subheader(f"Riskiest stations, {day} at {hour_label(hour)}")
top = (s[(s.weekday == wd) & (s.hour == hour) & (s.month == month) & (s.scenario == scenario)]
       .nlargest(10, "risk").assign(Station=lambda x: x.station.map(nice)))
top["Selected"] = top.station == station
bars = alt.Chart(top).mark_bar(cornerRadiusEnd=4, height=16).encode(
    x=alt.X("risk:Q", title="Chance of a 5+ min delay", axis=alt.Axis(format="%")),
    y=alt.Y("Station:N", sort="-x", title=None),
    color=alt.condition("datum.Selected", alt.value(ORANGE), alt.value(BLUE)),
    tooltip=["Station", alt.Tooltip("risk:Q", format=".1%", title="Risk")])
st.altair_chart(bars.properties(height=300), width="stretch")

# ---- How it works ----------------------------------------------------------------------
with st.expander("How this works, and its limits"):
    st.markdown(f"""
- **Data:** 97,227 incidents the TTC logged, January 2023 to September 2026 (City of Toronto Open Data), plus hourly weather (Open-Meteo).
- **Model:** gradient boosting on every station x passenger-service hour. Inputs: station, line, hour, weekday, month, first hour of service, temperature, rain, snow, wind.
- **Tested honestly:** trained on 2023-2025 and tested on 2026, which it never saw. **AUC 0.645**; the riskiest 10% of station-hours had **6x the delay rate** of the safest 10%. The app uses a version retrained on all the data.
- **What it can't do:** most delays (ill passengers, disorderly passengers, people on the track) are close to random. The model predicts *risk*, like a weather forecast, not specific incidents. It has no live data.
- **Weather scenarios** apply one average adjustment measured from the data (heavy snow raises the odds of a delay by about a third). Heavy snow is rare in the data, so these are rough. Weather matters far less than station and time of day.

[Code and full write-up on GitHub]({REPO})
""")
st.caption("Built by Hidaya Aiyub · Not affiliated with the TTC")
