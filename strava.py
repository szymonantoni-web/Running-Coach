"""Reading Strava data into one tidy frame of runs.

The primary path is Strava's bulk export (Settings → My Account → Download or
Delete Your Account → Request your archive), which arrives as a zip containing
`activities.csv`. That file needs no OAuth app, no API keys and no rate limits,
which is why it is the default here.

Strava's export is messier than it looks: several columns share the name
"Distance", the units differ between them, the date format follows the
account's locale, and column sets have changed across export versions. So
nothing is read by position — every column is matched by name and every unit
is sniffed from the data.

`from_api_activities` takes the JSON shape Strava's `/athlete/activities`
endpoint returns, for when you want to wire up OAuth later.
"""

from __future__ import annotations

import math

import pandas as pd

RUN_TYPES = {"run", "trail run", "treadmill", "virtual run", "track run", "long run"}

COLUMNS = {
    "activity_id": ["activity id", "id"],
    "date": ["activity date", "start date local", "start date", "date"],
    "name": ["activity name", "name"],
    "type": ["activity type", "type", "sport type"],
    "moving_seconds": ["moving time", "moving time seconds"],
    "elapsed_seconds": ["elapsed time", "elapsed time seconds"],
    "distance": ["distance"],
    "elevation_m": ["elevation gain", "total elevation gain"],
    "avg_hr": ["average heart rate", "average heartrate"],
    "max_hr": ["max heart rate", "max heartrate"],
}


def _normalise(label: object) -> str:
    text = str(label).lower().strip()
    kept = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in text]
    return " ".join("".join(kept).split())


def _find_columns(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Every column matching one of these names, in the order given.

    pandas suffixes duplicate headers ('Distance', 'Distance.1'), and the
    suffix is stripped before matching so both are found.
    """
    found = []
    for column in frame.columns:
        base = _normalise(str(column).rsplit(".", 1)[0] if str(column).rsplit(".", 1)[-1].isdigit()
                          else column)
        if base in candidates:
            found.append(column)
    return found


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pick_distance_km(frame: pd.DataFrame, assume_miles: bool = False) -> pd.Series:
    """Distance in kilometres, whichever column and unit the export used.

    Strava writes distance twice: once in the athlete's display unit (km or
    miles) and once in metres. The metres column is preferred because it is
    unit-independent; it is identified by magnitude, since a typical run is
    thousands of metres but single-digit kilometres.
    """
    columns = _find_columns(frame, COLUMNS["distance"])
    if not columns:
        return pd.Series([math.nan] * len(frame), index=frame.index, dtype="float64")

    best, best_median = None, -1.0
    for column in columns:
        values = _numeric(frame[column]).dropna()
        if values.empty:
            continue
        median = float(values.median())
        if median > best_median:
            best, best_median = column, median

    if best is None:
        return pd.Series([math.nan] * len(frame), index=frame.index, dtype="float64")

    values = _numeric(frame[best])
    if best_median > 1000:                 # metres
        return values / 1000.0
    if assume_miles:                       # athlete display unit is miles
        return values * 1.609344
    return values                          # already kilometres


def _parse_dates(series: pd.Series) -> pd.Series:
    """Strava's date strings vary by locale. Try the permissive parser, then
    day-first, and keep whichever resolved more rows."""
    first = pd.to_datetime(series, errors="coerce", format="mixed")
    if first.notna().mean() > 0.9:
        return first
    second = pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
    return second if second.notna().sum() > first.notna().sum() else first


def load_activities(path_or_buffer, assume_miles: bool = False,
                    runs_only: bool = True) -> pd.DataFrame:
    """Read a Strava `activities.csv` into a tidy frame of runs.

    Returns columns: date, name, type, distance_km, moving_seconds,
    pace_sec_per_km, elevation_m, avg_hr — sorted oldest first.
    """
    raw = pd.read_csv(path_or_buffer, low_memory=False)
    if raw.empty:
        raise ValueError("That file has no rows in it.")

    out = pd.DataFrame(index=raw.index)

    date_columns = _find_columns(raw, COLUMNS["date"])
    if not date_columns:
        raise ValueError(
            "No activity date column found. This does not look like a Strava "
            "activities.csv — check you picked the right file out of the export zip."
        )
    out["date"] = _parse_dates(raw[date_columns[0]])

    for key in ("name", "type"):
        columns = _find_columns(raw, COLUMNS[key])
        out[key] = raw[columns[0]].astype(str) if columns else ""

    # Moving time is the honest denominator for pace; elapsed time includes
    # standing at traffic lights.
    for key, target in (("moving_seconds", "moving_seconds"), ("elapsed_seconds", "elapsed_seconds")):
        columns = _find_columns(raw, COLUMNS[key])
        best = None
        for column in columns:
            values = _numeric(raw[column]).dropna()
            if not values.empty and (best is None or values.median() > 0):
                best = column
                break
        out[target] = _numeric(raw[best]) if best else math.nan

    out["distance_km"] = _pick_distance_km(raw, assume_miles=assume_miles)

    for key in ("elevation_m", "avg_hr", "max_hr"):
        columns = _find_columns(raw, COLUMNS[key])
        out[key] = _numeric(raw[columns[0]]) if columns else math.nan

    out["moving_seconds"] = out["moving_seconds"].fillna(out["elapsed_seconds"])
    out = out.drop(columns=["elapsed_seconds"])

    return _finalise(out, runs_only=runs_only)


def from_api_activities(activities: list[dict]) -> pd.DataFrame:
    """Build the same frame from Strava's REST API payloads.

    The API is already tidy: distance in metres, moving_time in seconds,
    start_date_local as ISO 8601. Wire OAuth up to `/athlete/activities` and
    hand the resulting list straight to this function.
    """
    if not activities:
        raise ValueError("No activities were returned.")
    frame = pd.DataFrame(activities)
    out = pd.DataFrame(index=frame.index)
    out["date"] = pd.to_datetime(frame.get("start_date_local", frame.get("start_date")),
                                 errors="coerce", format="mixed")
    out["name"] = frame.get("name", "")
    out["type"] = frame.get("sport_type", frame.get("type", ""))
    out["distance_km"] = pd.to_numeric(frame.get("distance"), errors="coerce") / 1000.0
    out["moving_seconds"] = pd.to_numeric(frame.get("moving_time"), errors="coerce")
    out["elevation_m"] = pd.to_numeric(frame.get("total_elevation_gain"), errors="coerce")
    out["avg_hr"] = pd.to_numeric(frame.get("average_heartrate"), errors="coerce")
    out["max_hr"] = pd.to_numeric(frame.get("max_heartrate"), errors="coerce")
    return _finalise(out)


def _finalise(out: pd.DataFrame, runs_only: bool = True) -> pd.DataFrame:
    """Shared cleanup: filter to runs, drop unusable rows, compute pace."""
    if runs_only:
        kinds = out["type"].astype(str).str.lower().str.strip()
        keep = kinds.isin(RUN_TYPES) | kinds.str.contains("run", na=False)
        if keep.any():
            out = out[keep]

    out = out.dropna(subset=["date"])
    out = out[(out["distance_km"] > 0.3) & (out["moving_seconds"] > 60)]
    if out.empty:
        raise ValueError(
            "No runs found. Check the activity type column, and that distances "
            "and times came through as numbers."
        )

    out["pace_sec_per_km"] = out["moving_seconds"] / out["distance_km"]
    # Anything outside 2:00–15:00 per km is a data error, not a run.
    out = out[(out["pace_sec_per_km"] > 120) & (out["pace_sec_per_km"] < 900)]

    out = out.sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.tz_localize(None) if getattr(out["date"].dt, "tz", None) else out["date"]
    return out


def append_runs(runs: pd.DataFrame, added: list[dict]) -> pd.DataFrame:
    """Add manually confirmed runs (typically read off a screenshot) to a
    history frame.

    Duplicates are dropped on date and distance, so re-adding the same
    screenshot twice does not double-count a run.
    """
    if not added:
        return runs

    extra = pd.DataFrame([{
        "date": pd.Timestamp(row["date"]),
        "name": row.get("name") or "Run",
        "type": "Run",
        "distance_km": float(row["distance_km"]),
        "moving_seconds": float(row["moving_seconds"]),
        "elevation_m": row.get("elevation_m"),
        "avg_hr": row.get("avg_hr"),
        "max_hr": row.get("max_hr"),
    } for row in added])

    combined = pd.concat([runs, extra], ignore_index=True) if not runs.empty else extra
    combined["pace_sec_per_km"] = combined["moving_seconds"] / combined["distance_km"]
    combined = combined.sort_values("date").reset_index(drop=True)

    key = list(zip(combined["date"].dt.strftime("%Y-%m-%d"), combined["distance_km"].round(2)))
    combined = combined[~pd.Series(key, index=combined.index).duplicated(keep="last")]
    return combined.sort_values("date").reset_index(drop=True)


def best_efforts(runs: pd.DataFrame, min_km: float = 3.0) -> pd.DataFrame:
    """Candidate runs for estimating current fitness: the fastest sustained
    efforts of at least `min_km`, most recent first.

    A whole-run average understates a race effort and overstates an interval
    session (whose average includes the jog recovery), so this is a starting
    point the athlete confirms rather than an automatic answer.
    """
    candidates = runs[runs["distance_km"] >= min_km].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values("pace_sec_per_km").head(10).sort_values("date", ascending=False)
