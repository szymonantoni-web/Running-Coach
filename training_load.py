"""Training load: weekly volume, intensity distribution, acute:chronic ratio.

One honest limitation up front. A Strava bulk export gives one average pace per
activity, not the splits. So an interval session — 5 × 1 km fast with jog
recoveries — shows up at its *average* pace, which lands somewhere around
marathon effort and understates how hard it actually was. Everything here is
computed from what the export contains, and the session classifier leans on
activity names as well as pace to compensate. Wire up the Strava API's lap data
if you want this to be exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

QUALITY_WORDS = (
    "interval", "tempo", "threshold", "fartlek", "track", "repeat", "rep",
    "race", "parkrun", "time trial", "progression", "workout", "session",
    "hills", "hill", "vo2", "speed", "strides",
)

LONG_RUN_MIN_KM = 15.0          # below this it is a normal run, not a long run
LONG_RUN_SHARE = 0.28           # or ≥28% of the week's volume

INTENSITY_ORDER = {"easy": 0, "moderate": 1, "hard": 2, "unknown": -1}


@dataclass(frozen=True)
class SessionType:
    """What the runner says a session was, overriding what its average says.

    The export gives one average pace per activity, and that average is the
    root of most of the misclassification here: an interval session's average
    includes the jog recoveries, so a genuinely hard session reads as easy and
    stops counting as quality. Telling the app what the session was is the
    cheapest possible fix for that, and it beats every heuristic.

    `intensity` pins the classification outright — used where the label is more
    reliable than the pace, as with a recovery run taken briskly on a downhill.
    `intensity_floor` only raises it — an interval session is at least moderate,
    but if the average pace says hard, hard stands.
    """
    label: str
    quality: bool | None = None         # None: decide from pace and name
    long_run: bool | None = None        # None: decide from distance
    intensity: str | None = None        # exact override
    intensity_floor: str | None = None  # raise to at least this
    note: str = ""


AUTO_SESSION = SessionType("Auto — work it out from the data", note=(
    "Pace and the activity name decide. Fine for easy running; unreliable for "
    "anything with recoveries in it."))

SESSION_TYPES: dict[str, SessionType] = {
    t.label: t for t in [
        AUTO_SESSION,
        SessionType("Easy run", quality=False, intensity="easy",
                    note="Counts toward the easy 80%. Still flagged as the long run if it is "
                         "the week's longest."),
        SessionType("Recovery run", quality=False, long_run=False, intensity="easy",
                    note="Never the long run and never quality, whatever the pace says."),
        SessionType("Long run", quality=False, long_run=True,
                    note="Resets the long-run clock, so the coach stops asking for one."),
        SessionType("Long run with quality", quality=True, long_run=True,
                    intensity_floor="moderate",
                    note="Race-pace or tempo segments inside a long run — both clocks reset."),
        SessionType("Threshold / tempo", quality=True, long_run=False,
                    intensity_floor="moderate",
                    note="Resets the quality clock; no second hard session for three days."),
        SessionType("Intervals", quality=True, long_run=False, intensity_floor="moderate",
                    note="The one the average pace gets most wrong — the jog recoveries drag "
                         "it into the easy band."),
        SessionType("Fartlek", quality=True, long_run=False, intensity_floor="moderate"),
        SessionType("Hill session", quality=True, long_run=False, intensity_floor="moderate"),
        SessionType("Progression run", quality=True, intensity_floor="moderate"),
        SessionType("Race / time trial", quality=True, intensity_floor="hard",
                    note="Also makes the run a candidate for your fitness benchmark."),
    ]
}


def session_type(label: str | None) -> SessionType:
    """The named type, or the auto rule for anything unrecognised."""
    return SESSION_TYPES.get(str(label or "").strip(), AUTO_SESSION)


def suggest_session_type(name: str, distance_km: float, pace_sec: float,
                         zones: dict[str, float]) -> str:
    """A first guess at what a run was, for pre-selecting the dropdown.

    A suggestion, never a decision — it is wrong in exactly the cases the
    dropdown exists for. The name is trusted over the pace, because a session
    named "5x1k" is a session whatever its average says.
    """
    text = str(name or "").lower()
    for words, label in (
        (("interval", "track", "repeat", "vo2", " x ", "×"), "Intervals"),
        (("tempo", "threshold"), "Threshold / tempo"),
        (("fartlek",), "Fartlek"),
        (("hill",), "Hill session"),
        (("race", "parkrun", "time trial", "tt"), "Race / time trial"),
        (("progression",), "Progression run"),
        (("long",), "Long run"),
        (("recovery", "shakeout"), "Recovery run"),
    ):
        if any(word in text for word in words):
            return label

    if math.isfinite(distance_km) and distance_km >= LONG_RUN_MIN_KM:
        return "Long run"
    if classify_intensity(pace_sec, zones) == "easy":
        return "Easy run"
    return AUTO_SESSION.label


def _apply_intensity_rule(rule: SessionType, measured: str) -> str:
    if rule.intensity is not None:
        return rule.intensity
    if rule.intensity_floor is not None and \
            INTENSITY_ORDER.get(measured, -1) < INTENSITY_ORDER[rule.intensity_floor]:
        return rule.intensity_floor
    return measured


def classify_intensity(pace_sec: float, zones: dict[str, float]) -> str:
    """easy / moderate / hard, from average pace against the athlete's zones.

    Boundaries sit slightly outside the exact zone paces, because nobody runs
    a session at precisely their threshold pace.
    """
    threshold = zones.get("threshold", math.nan)
    marathon = zones.get("marathon", math.nan)
    if not math.isfinite(pace_sec) or not math.isfinite(threshold) or not math.isfinite(marathon):
        return "unknown"
    if pace_sec <= threshold * 1.02:
        return "hard"
    if pace_sec >= marathon * 1.03:
        return "easy"
    return "moderate"


def session_load(distance_km: float, pace_sec: float, easy_pace_sec: float) -> float:
    """Intensity-weighted kilometres.

    load = distance × (easy pace ÷ session pace)²

    An easy run scores its own distance; faster running scores more, on the
    duration × intensity² shape that underlies most training-stress metrics.
    It is a transparent proxy, not a validated model — its job is to make
    "hard week" and "easy week" comparable, and it does that without needing
    heart rate.
    """
    if not all(math.isfinite(v) for v in (distance_km, pace_sec, easy_pace_sec)):
        return math.nan
    if pace_sec <= 0 or easy_pace_sec <= 0:
        return math.nan
    return distance_km * (easy_pace_sec / pace_sec) ** 2


def add_derived(runs: pd.DataFrame, zones: dict[str, float], easy_pace_sec: float) -> pd.DataFrame:
    """Attach intensity, load, quality and long-run flags to each run.

    A run carrying a `session_type` the runner chose is classified by that
    label; everything else falls back to pace and name.
    """
    out = runs.copy()
    if "session_type" not in out.columns:
        out["session_type"] = ""
    rules = [session_type(label) for label in out["session_type"]]

    measured = [classify_intensity(p, zones) for p in out["pace_sec_per_km"]]
    out["intensity"] = [_apply_intensity_rule(rule, value)
                        for rule, value in zip(rules, measured)]
    out["load"] = [session_load(km, pace, easy_pace_sec)
                   for km, pace in zip(out["distance_km"], out["pace_sec_per_km"])]

    names = out["name"].astype(str).str.lower()
    named_quality = names.apply(lambda n: any(word in n for word in QUALITY_WORDS))
    inferred_quality = named_quality | out["intensity"].isin(["hard", "moderate"])
    out["is_quality"] = [rule.quality if rule.quality is not None else bool(value)
                         for rule, value in zip(rules, inferred_quality)]

    week_totals = out.groupby(out["date"].dt.to_period("W"))["distance_km"].transform("sum")
    out["is_long_run"] = (out["distance_km"] >= LONG_RUN_MIN_KM) | \
                         (out["distance_km"] >= week_totals * LONG_RUN_SHARE)
    # A long run is the week's longest, not merely a long-ish one.
    week_max = out.groupby(out["date"].dt.to_period("W"))["distance_km"].transform("max")
    out["is_long_run"] &= out["distance_km"] >= week_max * 0.95
    out["is_long_run"] = [rule.long_run if rule.long_run is not None else bool(value)
                          for rule, value in zip(rules, out["is_long_run"])]
    return out


def weekly_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar week (Monday start), oldest first."""
    if runs.empty:
        return pd.DataFrame()

    frame = runs.copy()
    frame["week"] = frame["date"].dt.to_period("W").dt.start_time

    easy_km = frame.assign(
        easy=frame["distance_km"].where(frame["intensity"] == "easy", 0.0)
    ).groupby("week")["easy"].sum()

    summary = frame.groupby("week").agg(
        km=("distance_km", "sum"),
        sessions=("distance_km", "size"),
        load=("load", "sum"),
        longest_km=("distance_km", "max"),
        avg_pace=("pace_sec_per_km", "mean"),
    )
    summary["easy_km"] = easy_km
    summary["easy_share"] = (summary["easy_km"] / summary["km"]).where(summary["km"] > 0)
    return summary.reset_index()


@dataclass(frozen=True)
class LoadState:
    acute: float             # load over the last 7 days
    chronic: float           # average weekly load over the last 28 days
    ratio: float             # acute ÷ chronic
    acute_km: float
    chronic_km: float
    verdict: str
    status: str              # good | warning | critical | muted


def acwr(runs: pd.DataFrame, as_of: pd.Timestamp | None = None) -> LoadState:
    """Acute:chronic workload ratio.

    Acute is the last 7 days; chronic is the last 28 days divided by four, so
    the two are on the same weekly scale. A ratio near 1.0 means this week
    looks like the recent norm.

    Treat it as a guardrail, not a law. The ratio is widely used and widely
    criticised — the original injury-risk findings have not replicated
    cleanly, and the arithmetic is sensitive to how you define load. It is
    useful for catching the thing it was built to catch: a sharp spike after
    a quiet spell.
    """
    if runs.empty:
        return LoadState(math.nan, math.nan, math.nan, math.nan, math.nan,
                         "No runs to assess.", "muted")

    as_of = as_of or runs["date"].max()
    window = lambda days: runs[runs["date"] > as_of - pd.Timedelta(days=days)]  # noqa: E731

    acute = float(window(7)["load"].sum())
    chronic = float(window(28)["load"].sum()) / 4.0
    acute_km = float(window(7)["distance_km"].sum())
    chronic_km = float(window(28)["distance_km"].sum()) / 4.0
    ratio = acute / chronic if chronic > 0 else math.nan

    span_days = (as_of - runs["date"].min()).days
    if span_days < 21:
        return LoadState(acute, chronic, ratio, acute_km, chronic_km,
                         "Less than three weeks of history — not enough to judge a trend yet.",
                         "muted")

    if not math.isfinite(ratio):
        verdict, status = "Not enough recent running to compute a ratio.", "muted"
    elif ratio > 1.5:
        verdict, status = ("This week is far above your recent norm. Hold volume flat "
                           "or back off before adding anything."), "critical"
    elif ratio > 1.3:
        verdict, status = ("Ramping faster than usual. Fine for a week, but do not "
                           "stack another jump on top of it."), "warning"
    elif ratio < 0.8:
        verdict, status = ("Below your recent norm — either a deliberate down week, or "
                           "fitness you are letting slip."), "warning"
    else:
        verdict, status = "Load is consistent with your recent training.", "good"

    return LoadState(acute, chronic, ratio, acute_km, chronic_km, verdict, status)


@dataclass(frozen=True)
class RecentContext:
    as_of: pd.Timestamp
    days_since_run: float
    days_since_quality: float
    days_since_long: float
    last_7_km: float
    last_7_sessions: int
    last_28_km: float
    easy_share_28: float
    longest_recent_km: float
    weekly_km_trend: float      # last 4 weeks vs the 4 before, as a ratio


def recent_context(runs: pd.DataFrame, as_of: pd.Timestamp | None = None) -> RecentContext:
    """Everything the prescription engine needs to know about the last month."""
    as_of = as_of or runs["date"].max()

    def days_since(subset: pd.DataFrame) -> float:
        past = subset[subset["date"] <= as_of]
        if past.empty:
            return float("inf")
        return (as_of - past["date"].max()).total_seconds() / 86400.0

    last7 = runs[(runs["date"] > as_of - pd.Timedelta(days=7)) & (runs["date"] <= as_of)]
    last28 = runs[(runs["date"] > as_of - pd.Timedelta(days=28)) & (runs["date"] <= as_of)]
    prior28 = runs[(runs["date"] > as_of - pd.Timedelta(days=56)) &
                   (runs["date"] <= as_of - pd.Timedelta(days=28))]

    easy_28 = float(last28.loc[last28["intensity"] == "easy", "distance_km"].sum())
    total_28 = float(last28["distance_km"].sum())
    prior_total = float(prior28["distance_km"].sum())

    return RecentContext(
        as_of=as_of,
        days_since_run=days_since(runs),
        days_since_quality=days_since(runs[runs["is_quality"]]),
        days_since_long=days_since(runs[runs["is_long_run"]]),
        last_7_km=float(last7["distance_km"].sum()),
        last_7_sessions=int(len(last7)),
        last_28_km=total_28,
        easy_share_28=(easy_28 / total_28) if total_28 > 0 else math.nan,
        longest_recent_km=float(last28["distance_km"].max()) if not last28.empty else math.nan,
        weekly_km_trend=(total_28 / prior_total) if prior_total > 0 else math.nan,
    )
