"""What to run next, and whether the goal is realistic.

Two jobs:

* `next_session` — a decision tree over recent training. Which session is
  missing, is there room for it, and what does the training phase call for.
* `assess_goal` — an honest read on the gap between current fitness and the
  target, including whether the volume needed is compatible with the days a
  week available.

The prescription rules encode mainstream endurance practice: mostly easy
running, one or two quality sessions a week, a weekly long run, volume rising
in steps with a lighter fourth week, and a taper into the race. None of this is
individualised medical advice, and the thresholds are conventions rather than
constants — they are all named here so you can change them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from physiology import (
    MARATHON_M,
    format_duration,
    format_pace,
    pace_zones,
    predict_race_seconds,
    vdot_for_target,
)
from training_load import (
    LoadState,
    RecentContext,
    acwr,
    classify_intensity,
    recent_context,
    session_load,
)

# -- tunable conventions ---------------------------------------------------
MAX_WEEKLY_RAMP = 1.08          # volume increase per week in a build phase
DOWN_WEEK_FACTOR = 0.75         # every fourth week
EASY_TARGET_SHARE = 0.80        # polarised 80/20
MIN_DAYS_BETWEEN_QUALITY = 3
LONG_RUN_MIN_GAP_DAYS = 6
ACWR_CAUTION = 1.3
ACWR_STOP = 1.5

# --------------------------------------------------------------------------
# What the race distance changes
#
# Everything below this point used to be marathon-shaped: an 18/8/3-week
# periodisation, 34 km long runs, threshold-and-marathon-pace emphasis, and a
# volume table keyed off marathon finishing times. Applied to a 5 km goal that
# is wrong in every particular — a 20-minute 5 km would have been read against
# the sub-3-marathon volume band and told to run 80 km a week.
#
# The distance changes four things, so each is a field rather than a constant.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RaceProfile:
    label: str
    min_distance_m: float
    # Weeks-to-race boundaries: above base_from is base, then build, then peak,
    # then taper. Shorter races sharpen later and taper for days, not weeks.
    base_from: float
    build_from: float
    peak_from: float
    long_run_cap_km: float          # no point running 34 km for a 5 km race
    long_run_share: float           # of weekly volume
    base_volume: tuple[float, float]  # typical peak weekly km at VDOT 45
    race_pace_work: bool            # are race-pace segments in the long run useful?
    emphasis: str                   # which quality session carries the block
    why: str                        # one line, shown in the session rationale


RACE_PROFILES = [
    RaceProfile(
        "Marathon", 30000, 18, 8, 3, 34.0, 0.30, (65, 90), True, "threshold",
        "the marathon is decided by the pace you can hold before lactate accumulates, "
        "and by how late in a long run you can still hold it",
    ),
    RaceProfile(
        "Half marathon", 15000, 14, 6, 2, 26.0, 0.28, (55, 80), True, "threshold",
        "the half sits almost exactly at threshold, so raising that pace raises the race",
    ),
    RaceProfile(
        "10 km", 8000, 12, 5, 1.5, 20.0, 0.25, (45, 70), False, "mixed",
        "10 km sits between threshold and VO2max, so the block needs both",
    ),
    RaceProfile(
        "5 km", 3000, 10, 4, 1.0, 16.0, 0.25, (38, 58), False, "intervals",
        "5 km is run close to VO2max, so interval work is the session that moves it",
    ),
    RaceProfile(
        "Short", 0, 8, 3, 1.0, 12.0, 0.22, (30, 50), False, "intervals",
        "short races are limited by speed and economy more than by aerobic ceiling",
    ),
]


def race_profile(distance_m: float) -> RaceProfile:
    """The profile for a race of this distance."""
    for profile in RACE_PROFILES:
        if distance_m >= profile.min_distance_m:
            return profile
    return RACE_PROFILES[-1]


# --------------------------------------------------------------------------
# Periodisation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase:
    name: str
    focus: str
    description: str


PHASES = {
    "base": Phase("Base", "Aerobic volume",
                  "Build easy kilometres and consistency. One quality session a week, "
                  "usually threshold — the aim is a bigger engine, not a sharper one."),
    "build": Phase("Build", "Threshold and volume",
                   "Volume keeps climbing while threshold work becomes the weekly staple, "
                   "with intervals to lift ceiling fitness. The long run grows."),
    "peak": Phase("Peak", "Race-specific work",
                  "Sessions take on the race's own pace and volume tops out. "
                  "Sessions now rehearse the race rather than build general fitness."),
    "taper": Phase("Taper", "Absorb the work",
                   "Volume drops sharply, intensity is kept but shortened. You cannot "
                   "gain fitness now; you can arrive fresh."),
    "offseason": Phase("Off-season", "No race on the calendar",
                       "No target date set, so this is general aerobic maintenance."),
}


def phase_for(weeks_out: float | None, distance_m: float = MARATHON_M) -> Phase:
    """Which training phase a number of weeks before the race falls in.

    The boundaries move with the distance. A marathon build is long and tapers
    for three weeks; a 5 km block sharpens later and tapers for about one,
    because there is far less accumulated fatigue to shed.
    """
    if weeks_out is None or not math.isfinite(weeks_out):
        return PHASES["offseason"]
    profile = race_profile(distance_m)
    if weeks_out > profile.base_from:
        return PHASES["base"]
    if weeks_out > profile.build_from:
        return PHASES["build"]
    if weeks_out > profile.peak_from:
        return PHASES["peak"]
    return PHASES["taper"]


# --------------------------------------------------------------------------
# Goal assessment
# --------------------------------------------------------------------------

@dataclass
class Goal:
    distance_m: float = MARATHON_M
    goal_seconds: float = 3 * 3600
    race_date: pd.Timestamp | None = None
    days_per_week: int = 4
    label: str = "Marathon"


@dataclass
class Assessment:
    current_vdot: float
    required_vdot: float
    vdot_gap: float
    predicted_seconds: float
    goal_seconds: float
    time_gap_seconds: float
    weeks_available: float
    plausible_vdot_gain: float
    reachable_vdot: float
    reachable_seconds: float
    typical_peak_km: tuple[float, float]
    current_weekly_km: float
    verdict: str
    status: str                          # good | warning | serious | critical
    notes: list[str] = field(default_factory=list)


def _vdot_gain_rate(vdot: float) -> float:
    """Plausible VDOT points per week of consistent, well-structured training.

    Improvement slows sharply as fitness rises — the first points are cheap,
    the last ones are not. These figures are a planning heuristic drawn from
    typical development curves, not a guarantee in either direction.
    """
    if vdot < 45:
        return 0.25
    if vdot < 50:
        return 0.20
    if vdot < 55:
        return 0.15
    return 0.10


def typical_peak_volume(distance_m: float, vdot: float) -> tuple[float, float]:
    """Typical peak weekly kilometres for this race at this standard.

    Two things drive it and both are needed. The **distance** sets the base —
    marathoners run more than 5 km runners at the same standard. The **VDOT**
    scales it, because faster runners of any distance run more. Keying off
    finishing time alone, as this used to, is meaningless across distances: a
    20-minute 5 km and a 20-minute 5 km split of a marathon are not remotely
    the same training.

    Ranges, not requirements. People have run every one of these times on less
    and on far more; the number is here to show the gap, not to prescribe.
    """
    profile = race_profile(distance_m)
    low, high = profile.base_volume
    if not math.isfinite(vdot):
        return float(low), float(high)
    # Calibrated so a sub-3 marathon (VDOT ≈ 53.5) lands at the 80–100 km/week
    # that is the conventional figure for that time.
    return (max(20.0, low + (vdot - 45) * 1.8),
            max(30.0, high + (vdot - 45) * 1.5))


def assess_goal(vdot: float, goal: Goal, context: RecentContext,
                today: pd.Timestamp | None = None) -> Assessment:
    """How far the goal is from current fitness, and whether the time left is
    enough to cover it."""
    today = today or context.as_of
    weeks = math.nan
    if goal.race_date is not None:
        weeks = max((pd.Timestamp(goal.race_date) - pd.Timestamp(today)).days / 7.0, 0.0)

    required = vdot_for_target(goal.distance_m, goal.goal_seconds)
    predicted = predict_race_seconds(vdot, goal.distance_m)
    gap = required - vdot

    gain = _vdot_gain_rate(vdot) * weeks if math.isfinite(weeks) else math.nan
    reachable = vdot + gain if math.isfinite(gain) else math.nan
    reachable_seconds = predict_race_seconds(reachable, goal.distance_m) \
        if math.isfinite(reachable) else math.nan

    racing = race_profile(goal.distance_m)
    low, high = typical_peak_volume(goal.distance_m, required)
    weekly = context.last_28_km / 4.0 if math.isfinite(context.last_28_km) else math.nan

    notes: list[str] = []

    # Volume needed vs days available.
    if math.isfinite(weekly) and weekly > 0:
        if weekly < low * 0.7:
            notes.append(
                f"You are averaging {weekly:.0f} km a week. Runners at this standard over "
                f"{racing.label.lower()} usually peak around {low:.0f}–{high:.0f} km. "
                "That gap is the single biggest thing standing between the two numbers."
            )
        elif weekly < low:
            notes.append(
                f"At {weekly:.0f} km a week you are approaching, but not yet at, the "
                f"{low:.0f}–{high:.0f} km typical at this standard over "
                f"{racing.label.lower()}."
            )

    if goal.days_per_week and goal.days_per_week <= 4 and low >= 70:
        longest = min(racing.long_run_cap_km, low * racing.long_run_share * 1.15)
        notes.append(
            f"On {goal.days_per_week} running days a week, {low:.0f} km means averaging "
            f"{low / goal.days_per_week:.0f} km per run — long runs approaching "
            f"{longest:.0f} km and midweek runs of {low / goal.days_per_week * 0.8:.0f} km. "
            "Worth deciding up front whether that fits your week, or whether a fifth "
            "(shorter) day is the easier path."
        )

    if math.isfinite(context.easy_share_28) and context.easy_share_28 < 0.7:
        notes.append(
            f"Only {context.easy_share_28 * 100:.0f}% of your recent volume was at genuine "
            "easy pace, against a conventional target of about 80%. Running easy days "
            "too hard is the most common reason volume stops going up."
        )

    # Verdict.
    if not math.isfinite(weeks):
        verdict, status = "Set a race date to see whether the timeline works.", "muted"
    elif gap <= 0:
        verdict, status = ("Current fitness already projects at or inside the goal. "
                           "The work now is holding it and getting race-specific."), "good"
    elif math.isfinite(reachable) and reachable >= required:
        verdict, status = (f"Reachable. The gap is {gap:.1f} VDOT points and {weeks:.0f} weeks "
                           "of consistent training would typically cover it — with the volume "
                           "to back it up."), "good"
    elif math.isfinite(reachable) and reachable >= required - 1.5:
        verdict, status = (f"Ambitious but not out of reach. You need {gap:.1f} VDOT points in "
                           f"{weeks:.0f} weeks; a typical rate gets you most of the way, so it "
                           "depends on consistency and on the volume actually rising."), "warning"
    else:
        shortfall = format_duration(reachable_seconds) if math.isfinite(reachable_seconds) else "—"
        verdict, status = (f"A stretch on this timeline. The gap is {gap:.1f} VDOT points and "
                           f"{weeks:.0f} weeks at a typical improvement rate projects to about "
                           f"{shortfall}. Worth treating the goal as the ceiling and setting an "
                           "interim target you can chase honestly."), "serious"

    return Assessment(
        current_vdot=vdot, required_vdot=required, vdot_gap=gap,
        predicted_seconds=predicted, goal_seconds=goal.goal_seconds,
        time_gap_seconds=predicted - goal.goal_seconds,
        weeks_available=weeks, plausible_vdot_gain=gain,
        reachable_vdot=reachable, reachable_seconds=reachable_seconds,
        typical_peak_km=(low, high), current_weekly_km=weekly,
        verdict=verdict, status=status, notes=notes,
    )


# --------------------------------------------------------------------------
# Weekly volume target
# --------------------------------------------------------------------------

def weekly_volume_target(context: RecentContext, phase: Phase, goal: Goal,
                         week_index: int = 0) -> tuple[float, str]:
    """Kilometres to aim for this week, and why."""
    base = context.last_28_km / 4.0 if context.last_28_km > 0 else context.last_7_km
    if not math.isfinite(base) or base <= 0:
        return math.nan, "Not enough history to set a weekly target yet."

    racing = race_profile(goal.distance_m)
    low, high = typical_peak_volume(goal.distance_m,
                                    vdot_for_target(goal.distance_m, goal.goal_seconds))
    ceiling = high

    if phase.name == "Taper":
        weeks_out = max((pd.Timestamp(goal.race_date) - context.as_of).days / 7.0, 0) \
            if goal.race_date is not None else 0
        # A marathon taper sheds three weeks of accumulated fatigue; a 5 km
        # taper is a few easy days. Cutting volume as hard for the short race
        # would lose fitness rather than freshen you.
        gentle = racing.peak_from <= 1.5
        schedule = {0: 0.70, 1: 0.85} if gentle else {0: 0.45, 1: 0.60, 2: 0.75}
        factor = schedule.get(int(weeks_out), 0.90 if gentle else 0.80)
        return base * factor, (
            f"Taper week for {racing.label.lower()}: about {factor * 100:.0f}% of recent "
            "volume. Intensity stays, distance goes." +
            ("" if gentle else " The long taper is what a marathon needs; a short race "
                               "would want far less."))

    if week_index % 4 == 3:
        return base * DOWN_WEEK_FACTOR, ("Down week — every fourth week drops about 25% so "
                                         "the previous three actually get absorbed.")

    target = min(base * MAX_WEEKLY_RAMP, ceiling)
    if target <= base * 1.01:
        return base, (f"Holding at {base:.0f} km. That is the top of the range typical for "
                      f"{racing.label.lower()} at this standard ({low:.0f}–{high:.0f} km), "
                      "so consistency beats more volume.")
    return target, (f"Up about {(target / base - 1) * 100:.0f}% on your four-week average of "
                    f"{base:.0f} km — the conventional ceiling is 10% a week.")


# --------------------------------------------------------------------------
# The next session
# --------------------------------------------------------------------------

@dataclass
class Prescription:
    kind: str
    distance_km: float
    structure: str
    target_pace: str
    rationale: str
    status: str = "good"
    alternatives: list[str] = field(default_factory=list)
    # Estimated average pace for the whole session, recoveries and warm-up
    # included. Only used to project a week forward; nan where meaningless.
    avg_pace_sec: float = math.nan
    # Kilometres of the session run faster than easy. The rest is warm-up,
    # cool-down and jog recoveries — which is most of a "hard" session.
    hard_km: float = 0.0


def _blend(total_km: float, work_km: float, work_pace: float, easy_pace: float) -> float:
    """Average pace over a whole session, warm-up and recoveries included.

    A threshold session is not run at threshold pace: most of its distance is
    easy. Projecting a week forward needs the honest session average, because
    that is what the load calculation sees.
    """
    work_km = max(0.0, min(work_km, total_km))
    if total_km <= 0 or not math.isfinite(work_pace) or not math.isfinite(easy_pace):
        return math.nan
    return (work_km * work_pace + (total_km - work_km) * easy_pace) / total_km


def _pace_window(pace_sec: float, tolerance: int = 5) -> str:
    if not math.isfinite(pace_sec):
        return "—"
    return f"{format_pace(pace_sec - tolerance)} – {format_pace(pace_sec + tolerance)}"


def _days(value: float) -> int:
    """Whole days elapsed. Floors rather than rounds, so '2.6 days since' reads
    as 2 and never contradicts a 3-day threshold in the same sentence."""
    if not math.isfinite(value):
        return 0
    return int(math.floor(max(value, 0)))


def next_session(context: RecentContext, load: LoadState, zones: dict[str, float],
                 easy_slow_sec: float, phase: Phase, goal: Goal,
                 weekly_target_km: float) -> Prescription:
    """Decide the next run, in priority order: recover, then long, then
    quality, then easy."""
    racing = race_profile(goal.distance_m)
    easy_window = f"{format_pace(zones['easy'])} – {format_pace(easy_slow_sec)}"
    easy_avg = (zones["easy"] + easy_slow_sec) / 2.0
    remaining = weekly_target_km - context.last_7_km if math.isfinite(weekly_target_km) else math.nan

    # 1. Recovery overrides everything.
    if context.days_since_run < 1 and context.days_since_quality < 1:
        return Prescription(
            "Rest or easy shakeout", 5.0,
            "Complete rest, or 5 km very easy if you want to move.",
            easy_window,
            "You ran a quality session within the last 24 hours. Adaptation happens in the "
            "recovery, not the session — a second hard day now costs more than it buys.",
            status="warning",
            alternatives=["Cross-train easy (bike, swim) if you want the aerobic time without the impact."],
        )

    if math.isfinite(load.ratio) and load.ratio > ACWR_STOP:
        return Prescription(
            "Easy run", min(8.0, max(5.0, context.last_7_km * 0.15)),
            "Easy, flat, no strides.",
            easy_window,
            f"Your last 7 days are running at {load.ratio:.2f}× your four-week norm. That is the "
            "spike pattern worth interrupting deliberately — hold easy until it settles under 1.3.",
            status="critical",
            alternatives=["A rest day works equally well here."],
            avg_pace_sec=easy_avg,
        )

    # 2. Coming back from a gap — ease in before anything else.
    if context.days_since_run >= 7:
        gap_days = _days(context.days_since_run)
        return Prescription(
            "Easy return run", 8.0,
            "8 km easy, or less if it feels like work. No strides, no hills.",
            easy_window,
            f"It has been {gap_days} days since your last run. Fitness holds up for a week or two, "
            "but the tissues that take the impact detrain faster than the engine does — which is why "
            "injuries cluster in the first week back. Give it two or three easy runs before the "
            "long run or a session.",
            status="warning",
            alternatives=["If the break was illness rather than choice, add a day and keep it shorter still."],
            avg_pace_sec=easy_avg,
        )

    # 3. The long run is the week's anchor.
    if context.days_since_long >= LONG_RUN_MIN_GAP_DAYS:
        longest = context.longest_recent_km if math.isfinite(context.longest_recent_km) else 14.0
        # The cap is the race's, not the marathon's: 34 km serves a marathon and
        # is pointless — and costly — in a 5 km block.
        cap = racing.long_run_cap_km
        target = min(longest + 2.0, max(longest, 10.0), cap)
        if phase.name == "Taper":
            target = min(target * (0.85 if racing.peak_from <= 1.5 else 0.7), cap)
        elif phase.name == "Peak":
            target = min(longest + 2.0, cap)

        # Race-pace segments make this a quality session, so it has to respect
        # the same spacing as one. Without this check the long run was being
        # prescribed with segments two days after a threshold session — the
        # long-run branch sits above the quality branch in the tree and was
        # never consulting days_since_quality.
        segments_ok = context.days_since_quality >= MIN_DAYS_BETWEEN_QUALITY
        if phase.name in ("Peak", "Build") and racing.race_pace_work and segments_ok:
            # Goal pace is simply the target time over the target distance — no model
            # needed. Fall back to the VDOT marathon zone if no goal time is set.
            race_pace = (goal.goal_seconds / (goal.distance_m / 1000.0)
                         if goal.goal_seconds and math.isfinite(goal.goal_seconds)
                         else zones["marathon"])
            segment = 6.0 if phase.name == "Peak" else 4.0
            segment = min(segment, max(2.0, target * 0.25))
            return Prescription(
                f"Long run with {racing.label.lower()}-pace work", round(target, 1),
                f"{target - 2 * segment - 2:.0f} km easy → 2 × {segment:.0f} km at "
                f"{racing.label.lower()} pace ({format_pace(race_pace)}) with 2 km easy "
                "between → 2 km easy.",
                f"Easy {easy_window}; segments {_pace_window(race_pace)}",
                f"Longest run in {_days(context.days_since_long)} days, and the {phase.name.lower()} phase "
                "is where the long run stops being just time on feet. Race-pace segments late in a "
                "long run rehearse the thing that actually decides the day: holding form and pace when "
                "you are already tired.",
                alternatives=[f"If you are flat, drop the segments and run all {target:.0f} km easy — "
                              "the distance is worth more than the pace."],
                avg_pace_sec=_blend(target, 2 * segment, race_pace, easy_avg),
                hard_km=2 * segment,
            )

        return Prescription(
            "Long run", round(target, 1),
            f"{target:.0f} km steady, ideally on a route you do not have to think about. "
            "Last 20 minutes slightly quicker if it feels controlled.",
            easy_window,
            f"It has been {_days(context.days_since_long)} days since your longest run. The weekly long run "
            "builds the aerobic base and fatigue resistance nothing else does — it matters for every "
            f"distance, though it carries more of the work the longer the race. Capped at "
            f"{racing.long_run_cap_km:.0f} km here, which is what {racing.label.lower()} asks for. "
            "Keep it genuinely easy; the benefit is the duration, not the pace."
            + ("" if segments_ok or not racing.race_pace_work else
               f" No race-pace segments today — it is only "
               f"{_days(context.days_since_quality)} days since your last quality session, and a long "
               "run with segments is a hard day whatever the pace of the first 14 km."),
            alternatives=["Split into two runs the same day only if the full distance is not realistic yet."],
            avg_pace_sec=easy_avg,
        )

    # 4. Quality, if recovered and load allows.
    if (context.days_since_quality >= MIN_DAYS_BETWEEN_QUALITY
            and (not math.isfinite(load.ratio) or load.ratio <= ACWR_CAUTION)):
        threshold, interval = zones["threshold"], zones["interval"]

        speed_race = racing.emphasis == "intervals"

        if phase.name == "Taper":
            if speed_race:
                return Prescription(
                    "Sharpener", 7.0,
                    f"2 km easy → 4 × 400 m at {format_pace(interval)} with 2 min jog → 2 km easy.",
                    f"Reps {_pace_window(interval, 4)}",
                    f"Taper work for {racing.label.lower()} keeps race rhythm without adding fatigue: "
                    f"race intensity, a fraction of the volume. Short reps because {racing.why}.",
                    avg_pace_sec=_blend(7.0, 1.6, interval, easy_avg),
                    hard_km=1.6,
                )
            return Prescription(
                "Sharpener", 8.0,
                f"2 km easy → 3 × 1 km at threshold ({format_pace(threshold)}) with 2 min jog → 2 km easy.",
                f"Reps {_pace_window(threshold)}",
                "Taper work keeps the legs sharp without adding fatigue: same intensity, much less of it.",
                avg_pace_sec=_blend(8.0, 3.0, threshold, easy_avg),
                hard_km=3.0,
            )

        # Which session carries the block depends on the race. A 5 km is run near
        # VO2max, so intervals are the staple and threshold the support; a
        # marathon is the reverse, and its peak weeks belong to race-specific
        # work rather than to the VO2max ceiling. A 10 km needs both, so it alternates.
        if speed_race:
            do_intervals = phase.name != "Base" or context.days_since_quality >= 5
        elif racing.emphasis == "mixed":
            do_intervals = context.days_since_quality >= 5
        else:
            do_intervals = context.days_since_quality >= 5 and phase.name in ("Build", "Base")

        if do_intervals:
            reps, rep_m = (6, 800) if speed_race else (5, 1000)
            total = round(4.0 + reps * rep_m / 1000.0 + reps * 0.4, 1)
            return Prescription(
                "Intervals (VO2max)", total,
                f"2 km easy → {reps} × {rep_m} m at {format_pace(interval)} with 400 m jog "
                "recovery → 2 km easy.",
                f"Reps {_pace_window(interval, 4)}; recoveries genuinely slow",
                f"{_days(context.days_since_quality)} days since your last quality session, and load is in range "
                f"({load.ratio:.2f}×). Intervals lift maximal aerobic power, and for "
                f"{racing.label.lower()} that matters because {racing.why}. Run the reps at the prescribed "
                "pace, not faster: going too hard turns a VO2max session into a race and costs you the "
                "next three days.",
                alternatives=[f"Feeling flat? Make it {reps - 1} × {rep_m} m rather than pushing through "
                              f"{reps} bad ones."],
                avg_pace_sec=_blend(total, reps * rep_m / 1000.0, interval, easy_slow_sec),
                hard_km=reps * rep_m / 1000.0,
            )

        rep_km = 1.0 if speed_race else 2.0
        work_km = 5.0 if speed_race else (8.0 if phase.name == "Peak" else 6.0)
        reps = max(2, round(work_km / rep_km))
        if racing.emphasis == "threshold":
            why_threshold = (f"Threshold raises the pace you can hold before lactate accumulates — for "
                             f"{racing.label.lower()} that is the session that carries the block, because "
                             f"{racing.why}.")
        else:
            why_threshold = (f"Threshold is the support session in a {racing.label.lower()} block: it raises "
                             "the pace you can hold before lactate accumulates, which is what lets you "
                             "absorb the interval work rather than merely survive it.")
        return Prescription(
            "Threshold", round(reps * rep_km + 4.0, 1),
            f"2 km easy → {reps} × {rep_km:g} km at threshold ({format_pace(threshold)}) "
            f"with 90 s jog → 2 km easy.",
            f"Reps {_pace_window(threshold)}",
            f"{_days(context.days_since_quality)} days since quality work, load at "
            f"{load.ratio:.2f}× your norm. " + why_threshold,
            alternatives=["A continuous 20–30 min tempo works too if you prefer it to broken reps."],
            avg_pace_sec=_blend(reps * rep_km + 4.0, reps * rep_km, threshold, easy_avg),
            hard_km=reps * rep_km,
        )

    # 5. Otherwise, easy.
    distance = 10.0
    if math.isfinite(remaining) and remaining > 0:
        distance = min(max(remaining, 6.0), 14.0)
    reason = (f"{_days(context.days_since_quality)} days since quality work — not enough recovery for another "
              "hard session yet." if context.days_since_quality < MIN_DAYS_BETWEEN_QUALITY else
              f"Load is at {load.ratio:.2f}× your four-week norm, which is above the {ACWR_CAUTION} "
              "caution line, so this is a day to add volume rather than intensity.")
    return Prescription(
        "Easy run", round(distance, 1),
        f"{distance:.0f} km easy, conversational throughout. Add 4 × 20 s strides at the end if you feel good.",
        easy_window,
        reason + " Easy running is not filler — it is where most of the aerobic adaptation happens, "
                 "and it only works if it is actually easy.",
        alternatives=["Split it or shorten it if time is tight; consistency matters more than the exact distance."],
        avg_pace_sec=easy_avg,
    )


# --------------------------------------------------------------------------
# The week ahead
#
# `next_session` answers one question: what should today be. Projecting seven
# days means asking it seven times — but each answer changes the training
# history the next one reads, so the days cannot simply be asked in parallel.
#
# The approach here is to simulate: prescribe a day, write that session into a
# copy of the run history as though it had been run exactly as given, then ask
# again from the new history. That reuses the real load and context maths
# rather than a parallel approximation that would drift out of step with it.
#
# The output is a projection, not a plan. It assumes you run every session as
# prescribed, that nothing hurts, and that the weather cooperates. Run
# something different on Tuesday and Wednesday onward changes — which is the
# point of a rules engine rather than a fixed schedule.
# --------------------------------------------------------------------------

@dataclass
class PlannedDay:
    date: pd.Timestamp
    prescription: Prescription | None      # None on a rest day
    rest_reason: str = ""
    phase_name: str = ""
    load_ratio: float = math.nan

    @property
    def weekday(self) -> str:
        return self.date.strftime("%a")

    @property
    def is_rest(self) -> bool:
        return self.prescription is None

    @property
    def distance_km(self) -> float:
        return 0.0 if self.prescription is None else self.prescription.distance_km


@dataclass
class WeekOutlook:
    days: list[PlannedDay]
    total_km: float
    quality_sessions: int
    longest_km: float
    easy_share: float
    target_km: float
    end_ratio: float
    caveat: str
    shortfall: str = ""


QUALITY_KINDS = ("Threshold", "Intervals", "Sharpener")


def is_quality_kind(kind: str) -> bool:
    """Does this prescription count as a hard session?

    Deliberately keyed off what the engine *prescribed* rather than off the
    session's average pace. A threshold session is mostly easy running by
    distance — 6 km of reps inside a 10 km session — so its average pace can
    land in the easy band and the intensity classifier will call it easy. That
    is correct for a run someone actually did (the classifier only ever sees an
    average), but wrong here, where the structure is known. Reading it back off
    the average made the projection prescribe threshold on four consecutive
    days, because `days_since_quality` never reset.
    """
    return kind.startswith(QUALITY_KINDS) or "pace work" in kind


def _synthetic_run(date: pd.Timestamp, prescription: Prescription,
                   zones: dict[str, float], easy_slow_sec: float) -> dict:
    """The row this session would add to the history if it were run as given."""
    pace = prescription.avg_pace_sec
    if not math.isfinite(pace):
        pace = easy_slow_sec
    intensity = classify_intensity(pace, zones)
    return {
        "date": date,
        "name": prescription.kind,
        "type": "Run",
        "distance_km": prescription.distance_km,
        "moving_seconds": prescription.distance_km * pace,
        "pace_sec_per_km": pace,
        "intensity": intensity,
        "load": session_load(prescription.distance_km, pace, easy_slow_sec),
        "is_quality": is_quality_kind(prescription.kind) or intensity in ("hard", "moderate"),
        "is_long_run": prescription.kind.startswith("Long run"),
        "avg_hr": math.nan,
        "session_type": "",
    }


def _volume_note(total: float, target: float, per_week: int) -> str:
    """Whether the week as projected actually reaches its own volume target."""
    if not math.isfinite(target) or target <= 0 or total <= 0:
        return ""
    gap = target - total
    if abs(gap) <= target * 0.08:
        return ""
    if gap > 0:
        return (f"This week lands about {gap:.0f} km short of its {target:.0f} km target. "
                f"Across {per_week} running days that is roughly {gap / per_week:.1f} km more "
                "per session — or one more running day, which is usually the easier change.")
    return (f"This week runs about {-gap:.0f} km over its {target:.0f} km target. "
            "Trim the easy days rather than the sessions if you want it back in line.")


def week_ahead(runs: pd.DataFrame, zones: dict[str, float], easy_slow_sec: float,
               goal: Goal, today: pd.Timestamp, days: int = 7,
               week_index: int = 0) -> WeekOutlook:
    """Project the next `days` days by asking the engine once per day."""
    sim = runs.copy()
    planned: list[PlannedDay] = []
    scheduled = 0
    rests_taken = 0
    per_week = max(1, min(7, int(goal.days_per_week)))
    rest_budget = max(0, days - per_week)

    for offset in range(days):
        day = pd.Timestamp(today) + pd.Timedelta(days=offset)
        weeks_out = ((pd.Timestamp(goal.race_date) - day).days / 7.0
                     if goal.race_date is not None else None)
        phase = phase_for(weeks_out, goal.distance_m)
        context = recent_context(sim, as_of=day)
        load = acwr(sim, as_of=day)
        target_km, _ = weekly_volume_target(context, phase, goal,
                                            week_index=week_index + offset // 7)

        # Place the rest days where a coach would: after the hard ones. An even
        # spread ignores what the previous day actually was, and a day off after
        # a threshold session is worth more than a day off after an easy 8 km.
        runs_left = per_week - scheduled
        rests_left = rest_budget - rests_taken
        after_hard = bool(planned) and planned[-1].prescription is not None and (
            is_quality_kind(planned[-1].prescription.kind)
            or planned[-1].prescription.kind.startswith("Long run"))

        if runs_left <= 0:
            reason = (f"Rest — that is {per_week} running days, which is what you set. "
                      "The days off are what let the sessions do their work.")
        elif rests_left > 0 and after_hard:
            reason = (f"Rest after {planned[-1].prescription.kind.lower()}. Adaptation happens "
                      "in the recovery, not the session.")
        else:
            reason = ""

        if reason:
            rests_taken += 1
            planned.append(PlannedDay(day, None, reason, phase.name, load.ratio))
            continue

        prescription = next_session(context, load, zones, easy_slow_sec,
                                    phase, goal, target_km)

        # The engine can veto a scheduled running day. Honour that, and do not
        # spend one of the week's running days on it.
        if prescription.kind.startswith("Rest"):
            rests_taken += 1
            planned.append(PlannedDay(day, None, prescription.rationale,
                                      phase.name, load.ratio))
            continue

        planned.append(PlannedDay(day, prescription, "", phase.name, load.ratio))
        scheduled += 1
        sim = pd.concat(
            [sim, pd.DataFrame([_synthetic_run(day, prescription, zones, easy_slow_sec)])],
            ignore_index=True)

    ran = [d for d in planned if not d.is_rest]
    total = sum(d.distance_km for d in ran)
    # Easy share from the sessions' structure, not their average pace: a
    # threshold session is 4 km easy and 6 km of reps, and counting the whole
    # thing as either one misrepresents the week against the 80/20 target.
    easy_km = sum(d.distance_km - (d.prescription.hard_km if d.prescription else 0.0)
                  for d in ran)
    final_context = recent_context(sim, as_of=pd.Timestamp(today) + pd.Timedelta(days=days - 1))
    final_phase = phase_for(((pd.Timestamp(goal.race_date) - pd.Timestamp(today)).days / 7.0
                             if goal.race_date is not None else None), goal.distance_m)
    target_km, _ = weekly_volume_target(final_context, final_phase, goal, week_index=week_index)

    return WeekOutlook(
        days=planned,
        total_km=total,
        quality_sessions=sum(1 for d in ran
                             if d.prescription is not None
                             and is_quality_kind(d.prescription.kind)),
        longest_km=max((d.distance_km for d in ran), default=0.0),
        easy_share=(easy_km / total) if total > 0 else math.nan,
        target_km=target_km,
        end_ratio=acwr(sim, as_of=pd.Timestamp(today) + pd.Timedelta(days=days - 1)).ratio,
        shortfall=_volume_note(total, target_km, per_week),
        caveat=("A projection, not a schedule. Every day assumes the one before it was run "
                "exactly as prescribed. Run something different — or nothing — and the rest "
                "of the week is recalculated from what actually happened."),
    )
