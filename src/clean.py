"""Clean and combine the TTC subway delay files (2023 to the latest month).

Output: data/delays_clean.parquet, one row per reported incident.

Problems this fixes, found while exploring the raw files:
- 2023 and 2024 come as Excel, 2025+ as CSV with an extra _id column.
- Station is free text cut at 22 characters: 1,687 spellings for about 75 stations
  ("KENNEDY BD STATION (PL", "FINCH STATION - PLATFO", "VMC STATION" vs "VAUGHAN MC STATION").
- Two stations were renamed during the period: Dundas -> TMU (2024), Eglinton West -> Cedarvale (2025).
- Line has 40+ spellings ("YU/BD", "YU / BD", "BD/YU", "999", "77 SWANSEA").
- 65% of incidents have a 0-minute delay: they are logged events, not delays.
- About 3% of codes are missing from the code list.
"""
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "data_raw")
OUT = Path(__file__).resolve().parent.parent / "data"

# Canonical stations: raw prefix -> (station, line it sits on when the Line field is unclear)
STATIONS = {
    # Line 1 Yonge-University
    "VAUGHAN MC": "VAUGHAN METROPOLITAN CENTRE", "VMC": "VAUGHAN METROPOLITAN CENTRE",
    "HIGHWAY 407": "HIGHWAY 407", "PIONEER VILLAGE": "PIONEER VILLAGE",
    "YORK UNIVERSITY": "YORK UNIVERSITY", "FINCH WEST": "FINCH WEST",
    "DOWNSVIEW PARK": "DOWNSVIEW PARK", "SHEPPARD WEST": "SHEPPARD WEST",
    "WILSON": "WILSON", "YORKDALE": "YORKDALE", "LAWRENCE WEST": "LAWRENCE WEST",
    "GLENCAIRN": "GLENCAIRN", "EGLINTON WEST": "CEDARVALE", "CEDARVALE": "CEDARVALE",
    "ST CLAIR WEST": "ST CLAIR WEST", "DUPONT": "DUPONT",
    "SPADINA": "SPADINA", "ST GEORGE": "ST GEORGE", "MUSEUM": "MUSEUM",
    "QUEEN'S PARK": "QUEEN'S PARK", "QUEENS PARK": "QUEEN'S PARK",
    "ST PATRICK": "ST PATRICK", "OSGOODE": "OSGOODE", "ST ANDREW": "ST ANDREW",
    "UNION": "UNION", "KING": "KING", "QUEEN": "QUEEN",
    "DUNDAS": "TMU", "TMU": "TMU", "COLLEGE": "COLLEGE", "WELLESLEY": "WELLESLEY",
    "BLOOR": "BLOOR-YONGE", "YONGE BD": "BLOOR-YONGE", "YONGE": "BLOOR-YONGE",
    "ROSEDALE": "ROSEDALE", "SUMMERHILL": "SUMMERHILL", "ST CLAIR": "ST CLAIR",
    "DAVISVILLE": "DAVISVILLE", "EGLINTON": "EGLINTON", "LAWRENCE": "LAWRENCE",
    "YORK MILLS": "YORK MILLS", "SHEPPARD-YONGE": "SHEPPARD-YONGE",
    "SHEPPARD YONGE": "SHEPPARD-YONGE", "SHEPPARD": "SHEPPARD-YONGE",
    "NORTH YORK CTR": "NORTH YORK CENTRE", "NORTH YORK CENTRE": "NORTH YORK CENTRE",
    "FINCH": "FINCH",
    # Line 2 Bloor-Danforth
    "KIPLING": "KIPLING", "ISLINGTON": "ISLINGTON", "ROYAL YORK": "ROYAL YORK",
    "OLD MILL": "OLD MILL", "JANE": "JANE", "RUNNYMEDE": "RUNNYMEDE",
    "HIGH PARK": "HIGH PARK", "KEELE": "KEELE", "DUNDAS WEST": "DUNDAS WEST",
    "LANSDOWNE": "LANSDOWNE", "DUFFERIN": "DUFFERIN", "OSSINGTON": "OSSINGTON",
    "CHRISTIE": "CHRISTIE", "BATHURST": "BATHURST", "BAY": "BAY",
    "SHERBOURNE": "SHERBOURNE", "CASTLE FRANK": "CASTLE FRANK",
    "BROADVIEW": "BROADVIEW", "CHESTER": "CHESTER", "PAPE": "PAPE",
    "DONLANDS": "DONLANDS", "GREENWOOD": "GREENWOOD", "COXWELL": "COXWELL",
    "WOODBINE": "WOODBINE", "MAIN STREET": "MAIN STREET",
    "VICTORIA PARK": "VICTORIA PARK", "WARDEN": "WARDEN", "KENNEDY": "KENNEDY",
    # Line 4 Sheppard
    "BAYVIEW": "BAYVIEW", "BESSARION": "BESSARION", "LESLIE": "LESLIE",
    "DON MILLS": "DON MILLS",
    # Line 3 Scarborough RT (closed July 2023)
    "LAWRENCE EAST": "LAWRENCE EAST", "ELLESMERE": "ELLESMERE", "MIDLAND": "MIDLAND",
    "SCARBOROUGH CTR": "SCARBOROUGH CENTRE", "SCARBOROUGH CENTRE": "SCARBOROUGH CENTRE",
    "MCCOWAN": "MCCOWAN",
}
# Longest prefix first, so "DUNDAS WEST" wins over "DUNDAS" and "ST CLAIR WEST" over "ST CLAIR".
PREFIXES = sorted(STATIONS, key=len, reverse=True)

CAUSE_GROUPS = {
    "EU": "Train equipment", "ER": "Train equipment",
    "MU": "Passenger & miscellaneous", "MR": "Passenger & miscellaneous",
    "PU": "Track, signals & power", "PR": "Track, signals & power",
    "SU": "Security", "SR": "Security",
    "TU": "Operations & staff", "TR": "Operations & staff",
}


def norm_text(s: str) -> str:
    s = str(s).upper().strip()
    s = s.replace("ST.", "ST").replace("STN", "STATION")
    s = re.sub(r"\s+", " ", s)
    return s


def match_station(raw: str):
    """Return (station, kind). kind is 'station', 'yard', 'line-wide' or 'unknown'."""
    s = norm_text(raw)
    if re.search(r"\b(YARD|CARHOUSE|HOSTLER|SHOP|BUILD UP|TRACK PLANT)\b", s):
        return None, "yard"
    if s.startswith(("LINE ", "BLOOR DANFORTH", "BLOOR-DANFORTH", "YONGE UNIVERSITY", "YONGE-UNIVERSITY",
                     "TORONTO TRANSIT", "YU ", "BD ")):
        return None, "line-wide"
    for p in PREFIXES:
        if s == p or s.startswith(p + " ") or s.startswith(p + "-") or s.startswith(p + "("):
            # "YONGE" alone is ambiguous; only accept it as Bloor-Yonge in "YONGE BD"
            if p == "YONGE" and not s.startswith("YONGE STATION"):
                continue
            return STATIONS[p], "station"
    return None, "unknown"


def norm_line(raw) -> str:
    if pd.isna(raw):
        return "UNKNOWN"
    s = re.sub(r"[\s&/]+", "/", str(raw).upper().strip())
    has = {
        "YU": bool(re.search(r"YU|YONGE|LINE/?1", s)),
        "BD": bool(re.search(r"BD|BLOOR|DANFORTH|LINE/?2", s)),
        "SHP": bool(re.search(r"SHP|SHEP", s)),
        "SRT": "SRT" in s,
    }
    found = [k for k, v in has.items() if v]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        return "MULTI"
    return "UNKNOWN"


def load_raw(raw: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(raw.iterdir()):
        name = f.name.lower()
        if "readme" in name or "code" in name:
            continue
        if f.suffix == ".xlsx" and "delay-data-20" in name:
            frames.append(pd.read_excel(f))
        elif f.suffix == ".csv" and "delay_data" in name.lower():
            frames.append(pd.read_csv(f).drop(columns=["_id"], errors="ignore"))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_codes(raw: Path) -> pd.Series:
    f = next(p for p in raw.iterdir() if p.name.endswith("Code_Descriptions.csv"))
    codes = pd.read_csv(f, encoding="utf-8")
    # The city's file has mojibake dashes ("â") from a bad encoding round trip
    desc = codes["DESCRIPTION"].str.replace("â", "-", regex=False).str.replace(r"\s+", " ", regex=True)
    return pd.Series(desc.str.strip().values, index=codes["CODE"].str.strip())


def clean(raw: Path = RAW) -> pd.DataFrame:
    df = load_raw(raw)
    n_raw = len(df)

    df["timestamp"] = pd.to_datetime(df["date"].astype(str).str[:10] + " " + df["time"].astype(str))
    matched = df["station"].map(match_station)
    df["station_raw"] = df["station"]
    df["station"] = matched.str[0]
    df["location_kind"] = matched.str[1]
    df["line"] = df["line"].map(norm_line)
    df["bound"] = df["bound"].where(df["bound"].isin(["N", "S", "E", "W"]))
    df["code"] = df["code"].str.strip()

    codes = load_codes(raw)
    df["code_desc"] = df["code"].map(codes).fillna("Not in code list")
    df["cause_group"] = df["code"].str[:2].map(CAUSE_GROUPS).fillna("Other")

    df["is_delay"] = df["min_delay"] > 0
    df = df.drop_duplicates(subset=["timestamp", "station_raw", "code", "min_delay", "line"])

    # Delays over 6 hours are almost always planned closures logged as one event
    df["is_closure"] = df["code"].isin(["MUCP", "MUCU", "MUFM"]) | (df["min_delay"] > 360)

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    cols = ["timestamp", "year", "month", "weekday", "hour", "line", "station", "location_kind",
            "bound", "code", "code_desc", "cause_group", "min_delay", "min_gap", "is_delay",
            "is_closure", "vehicle", "station_raw"]
    df = df[cols].sort_values("timestamp").reset_index(drop=True)
    print(f"raw rows {n_raw:,} -> clean rows {len(df):,} (duplicates removed {n_raw - len(df):,})")
    return df


if __name__ == "__main__":
    out = clean()
    OUT.mkdir(exist_ok=True)
    out.to_parquet(OUT / "delays_clean.parquet", index=False)
    print(out["location_kind"].value_counts(normalize=True).round(3).to_dict())
    print(out["line"].value_counts().to_dict())
    print("stations:", out["station"].nunique())
