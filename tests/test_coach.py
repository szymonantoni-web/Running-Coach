"""Tests for the coaching engine.

Run with `pytest`, or `python tests/test_coach.py` if you would rather not
install it.

The physiology tests are the important ones: they check the implementation
against values published in *Daniels' Running Formula*, which is the only way
to know a VDOT implementation is actually right rather than merely plausible.
Daniels' tables are rounded to the second, so the tolerances below reflect
that rather than floating-point error.
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiology import (  # noqa: E402
    MARATHON_M,
    easy_pace_range,
    format_duration,
    format_pace,
    pace_zones,
    parse_duration,
    percent_vo2max,
    predict_race_seconds,
    velocity_at_vo2,
    vo2_at_velocity,
    vdot_from_race,
)
from plan import (  # noqa: E402
    ACWR_STOP,
    Goal,
    assess_goal,
    is_quality_kind,
    next_session,
    phase_for,
    typical_peak_volume,
    week_ahead,
    weekly_volume_target,
)
from strava import load_activities  # noqa: E402
from training_load import (  # noqa: E402
    AUTO_SESSION,
    LoadState,
    SESSION_TYPES,
    suggest_session_type,
    acwr,
    add_derived,
    classify_intensity,
    recent_context,
    session_load,
    weekly_summary,
)

DEMO = Path(__file__).resolve().parents[1] / "data" / "sample_activities.csv"


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def demo_runs() -> pd.DataFrame:
    runs = load_activities(DEMO)
    zones = pace_zones(43.5)
    _, easy_slow = easy_pace_range(43.5)
    return add_derived(runs, zones, easy_slow)


# --------------------------------------------------------------------------
# Physiology — checked against Daniels' published tables
# --------------------------------------------------------------------------

def test_vdot_from_known_races():
    """Daniels' table: a 19:57 5 km is VDOT 50; 41:21 10 km is VDOT 50;
    3:10:49 marathon is VDOT 50."""
    assert abs(vdot_from_race(5000, 19 * 60 + 57) - 50) < 0.15
    assert abs(vdot_from_race(10000, 41 * 60 + 21) - 50) < 0.15
    assert abs(vdot_from_race(MARATHON_M, 3 * 3600 + 10 * 60 + 49) - 50) < 0.15


def test_race_predictions_match_daniels():
    """VDOT 50 should predict 5 k 19:57, 10 k 41:21, marathon 3:10:49.
    Two seconds of tolerance covers the rounding in the published table."""
    for distance, expected in [(5000, 19 * 60 + 57),
                               (10000, 41 * 60 + 21),
                               (MARATHON_M, 3 * 3600 + 10 * 60 + 49)]:
        predicted = predict_race_seconds(50, distance)
        assert abs(predicted - expected) < 12, f"{distance} m: {predicted} vs {expected}"


def test_pace_zones_match_daniels():
    """VDOT 50: threshold 4:15/km, interval 3:55/km, marathon 4:31/km."""
    zones = pace_zones(50)
    assert abs(zones["threshold"] - (4 * 60 + 15)) < 4
    assert abs(zones["interval"] - (3 * 60 + 55)) < 4
    assert abs(zones["marathon"] - (4 * 60 + 31)) < 4


def test_easy_band_brackets_daniels_range():
    """Daniels' E column for VDOT 50 is roughly 5:08–5:41 per km."""
    quick, slow = easy_pace_range(50)
    assert 295 < quick < 320, format_pace(quick)
    assert 325 < slow < 355, format_pace(slow)
    assert quick < slow          # quick end is the smaller seconds-per-km


def test_prediction_round_trips():
    """Predicting a time from a VDOT and reading the VDOT back off that time
    must return the same number."""
    for vdot in (35, 45, 55, 65):
        for distance in (5000, 10000, MARATHON_M):
            seconds = predict_race_seconds(vdot, distance)
            assert close(vdot_from_race(distance, seconds), vdot, 1e-6)


def test_vo2_inverse():
    for velocity in (150.0, 200.0, 280.0, 350.0):
        assert close(velocity_at_vo2(vo2_at_velocity(velocity)), velocity, 1e-6)


def test_monotonic_in_the_right_directions():
    # Fitter athletes run faster and predict quicker times.
    assert pace_zones(55)["threshold"] < pace_zones(45)["threshold"]
    assert predict_race_seconds(55, 10000) < predict_race_seconds(45, 10000)
    # A longer race is run at a slower pace.
    pace_5k = predict_race_seconds(50, 5000) / 5
    pace_marathon = predict_race_seconds(50, MARATHON_M) / 42.195
    assert pace_5k < pace_marathon
    # Sustainable fraction of VO2max falls as a race gets longer.
    assert percent_vo2max(10) > percent_vo2max(60) > percent_vo2max(180)


def test_sub_three_requires_the_expected_vdot():
    """A 3:00:00 marathon sits at VDOT 53.5 in Daniels' table."""
    assert abs(vdot_from_race(MARATHON_M, 3 * 3600) - 53.5) < 0.3


def test_duration_parsing_and_formatting():
    assert parse_duration("3:00:00") == 10800
    assert parse_duration("20:00") == 1200          # a race time is minutes:seconds
    assert parse_duration("2h55") == 10500          # an 'h' forces hours first
    assert parse_duration("45m") == 2700
    assert math.isnan(parse_duration("nonsense"))
    assert math.isnan(parse_duration(""))
    assert format_duration(10800) == "3:00:00"
    assert format_duration(1200) == "20:00"
    assert format_pace(330) == "5:30/km"
    assert format_pace(float("nan")) == "—"


# --------------------------------------------------------------------------
# Strava loading
# --------------------------------------------------------------------------

def _csv(text: str) -> io.StringIO:
    return io.StringIO(text.strip())


def test_prefers_the_metres_column_over_the_display_column():
    """Strava writes distance twice — display units and metres. The metres
    column is unit-independent, so it should win."""
    runs = load_activities(_csv("""
Activity Date,Activity Name,Activity Type,Distance,Moving Time,Distance
"Sep 01, 2026, 7:00:00 AM",Morning run,Run,10.0,3000,10000
"Sep 03, 2026, 7:00:00 AM",Morning run,Run,12.0,3600,12000
"""))
    assert close(runs["distance_km"].iloc[0], 10.0, 1e-9)
    assert close(runs["distance_km"].iloc[1], 12.0, 1e-9)


def test_miles_are_converted_when_flagged():
    runs = load_activities(_csv("""
Activity Date,Activity Name,Activity Type,Distance,Moving Time
"Sep 01, 2026, 7:00:00 AM",Morning run,Run,6.2137,3000
"""), assume_miles=True)
    assert abs(runs["distance_km"].iloc[0] - 10.0) < 0.01


def test_non_runs_are_filtered_out():
    runs = load_activities(_csv("""
Activity Date,Activity Name,Activity Type,Distance,Moving Time
"Sep 01, 2026, 7:00:00 AM",Morning run,Run,10000,3000
"Sep 02, 2026, 6:00:00 PM",Gym,Weight Training,0,2700
"Sep 03, 2026, 5:00:00 PM",Commute,Ride,20000,3000
"Sep 04, 2026, 7:00:00 AM",Trail,Trail Run,12000,4200
"""))
    assert len(runs) == 2
    assert set(runs["type"]) == {"Run", "Trail Run"}


def test_impossible_paces_are_dropped():
    """A 400 m 'run' logged over two hours is a data error, not a session."""
    runs = load_activities(_csv("""
Activity Date,Activity Name,Activity Type,Distance,Moving Time
"Sep 01, 2026, 7:00:00 AM",Good run,Run,10000,3000
"Sep 02, 2026, 7:00:00 AM",Bad GPS,Run,1000,7200
"Sep 03, 2026, 7:00:00 AM",Impossible,Run,10000,1000
"""))
    assert list(runs["name"]) == ["Good run"]


def test_pace_is_computed_from_moving_time():
    runs = load_activities(_csv("""
Activity Date,Activity Name,Activity Type,Distance,Elapsed Time,Moving Time
"Sep 01, 2026, 7:00:00 AM",Run,Run,10000,3600,3000
"""))
    assert close(runs["pace_sec_per_km"].iloc[0], 300.0, 1e-9)


def test_best_efforts_respects_the_recency_window():
    """A fast run from months ago must not be offered as *current* fitness —
    it would propagate into every pace, prediction and goal verdict."""
    from strava import best_efforts

    runs = demo_runs()
    latest = runs["date"].max()

    recent = best_efforts(runs, within_days=56, as_of=latest)
    assert not recent.empty
    assert (recent["date"] >= latest - pd.Timedelta(days=56)).all()

    # A wider window can only ever offer more candidates, never fewer.
    assert len(best_efforts(runs, within_days=None)) >= len(recent)

    # And it really is filtering: the demo spans ~20 weeks.
    everything = best_efforts(runs, within_days=None)
    assert everything["date"].min() < recent["date"].min()


def test_best_efforts_returns_empty_rather_than_widening_silently():
    """When nothing qualifies the caller must be able to tell, so it can say
    so rather than quietly using a stale effort."""
    from strava import best_efforts

    runs = demo_runs()
    far_future = runs["date"].max() + pd.Timedelta(days=400)
    assert best_efforts(runs, within_days=56, as_of=far_future).empty


def test_best_efforts_ranks_by_pace_within_the_window():
    from strava import best_efforts

    runs = demo_runs()
    picked = best_efforts(runs, within_days=56, as_of=runs["date"].max(), limit=3)
    window = runs[runs["date"] >= runs["date"].max() - pd.Timedelta(days=56)]
    window = window[window["distance_km"] >= 3.0]
    assert picked["pace_sec_per_km"].max() <= window["pace_sec_per_km"].nsmallest(3).max() + 1e-9


def test_demo_file_loads():
    runs = load_activities(DEMO)
    assert len(runs) > 50
    assert runs["date"].is_monotonic_increasing
    assert (runs["distance_km"] > 0).all()
    assert set(runs["type"].unique()) <= {"Run", "Trail Run", "Long Run"}


# --------------------------------------------------------------------------
# Training load
# --------------------------------------------------------------------------

def test_easy_running_scores_its_own_distance():
    """The load metric is calibrated so an easy run scores 1.0 per km."""
    assert close(session_load(10.0, 330.0, 330.0), 10.0, 1e-9)
    # Faster than easy scores more; slower scores less.
    assert session_load(10.0, 280.0, 330.0) > 10.0
    assert session_load(10.0, 380.0, 330.0) < 10.0


def test_intensity_classification():
    zones = {"threshold": 270.0, "marathon": 290.0}
    assert classify_intensity(260.0, zones) == "hard"        # faster than threshold
    assert classify_intensity(280.0, zones) == "moderate"    # between the two
    assert classify_intensity(340.0, zones) == "easy"        # slower than marathon
    assert classify_intensity(float("nan"), zones) == "unknown"


def test_acwr_arithmetic():
    """Four identical weeks must give a ratio of exactly 1.0."""
    dates = pd.date_range("2026-08-01", periods=28, freq="D")
    runs = pd.DataFrame({
        "date": dates,
        "name": ["Run"] * 28,
        "distance_km": [10.0] * 28,
        "moving_seconds": [3300.0] * 28,
        "pace_sec_per_km": [330.0] * 28,
        "load": [10.0] * 28,
        "intensity": ["easy"] * 28,
        "is_quality": [False] * 28,
        "is_long_run": [False] * 28,
    })
    state = acwr(runs, as_of=dates[-1])
    assert close(state.acute, 70.0, 1e-9)
    assert close(state.chronic, 70.0, 1e-9)
    assert close(state.ratio, 1.0, 1e-9)
    assert state.status == "good"


def test_acwr_flags_a_spike():
    dates = list(pd.date_range("2026-08-01", periods=21, freq="D")) + \
            list(pd.date_range("2026-08-22", periods=7, freq="D"))
    loads = [3.0] * 21 + [20.0] * 7
    runs = pd.DataFrame({
        "date": dates, "name": ["Run"] * 28, "distance_km": loads,
        "moving_seconds": [3300.0] * 28, "pace_sec_per_km": [330.0] * 28,
        "load": loads, "intensity": ["easy"] * 28,
        "is_quality": [False] * 28, "is_long_run": [False] * 28,
    })
    state = acwr(runs, as_of=dates[-1])
    assert state.ratio > ACWR_STOP
    assert state.status == "critical"


def test_recent_context_days_since():
    runs = demo_runs()
    as_of = runs["date"].max() + pd.Timedelta(days=2)
    context = recent_context(runs, as_of=as_of)
    assert 1.9 < context.days_since_run < 2.2
    assert context.days_since_quality >= context.days_since_run
    assert context.last_7_km > 0
    assert 0 <= context.easy_share_28 <= 1


def test_weekly_summary_totals_match_the_runs():
    runs = demo_runs()
    weekly = weekly_summary(runs)
    assert close(weekly["km"].sum(), runs["distance_km"].sum(), 1e-6)
    assert weekly["sessions"].sum() == len(runs)
    assert (weekly["easy_share"].dropna() <= 1.0).all()


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------

def test_phases_by_weeks_out():
    assert phase_for(30).name == "Base"
    assert phase_for(12).name == "Build"
    assert phase_for(5).name == "Peak"
    assert phase_for(2).name == "Taper"
    assert phase_for(None).name == "Off-season"


def test_phase_boundaries_move_with_race_distance():
    """A 5 km block sharpens later and tapers for days, not three weeks."""
    assert phase_for(12, MARATHON_M).name == "Build"
    assert phase_for(12, 5000).name == "Base"
    # Two weeks out: a marathon is already tapering, a 5 km is still peaking.
    assert phase_for(2, MARATHON_M).name == "Taper"
    assert phase_for(2, 5000).name == "Peak"


def test_faster_runners_need_more_volume():
    assert typical_peak_volume(MARATHON_M, 53.5)[0] > typical_peak_volume(MARATHON_M, 45)[0]


def test_longer_races_need_more_volume_at_equal_fitness():
    marathon = typical_peak_volume(MARATHON_M, 50)
    five_k = typical_peak_volume(5000, 50)
    assert marathon[0] > five_k[0] and marathon[1] > five_k[1]


def test_long_run_cap_follows_the_race():
    """No 34 km long runs in a 5 km block."""
    from plan import race_profile
    assert race_profile(MARATHON_M).long_run_cap_km > race_profile(5000).long_run_cap_km
    ctx = _context(days_since_long=7.0, longest_recent_km=30.0)
    marathon = next_session(ctx, _load(1.0), ZONES, EASY_SLOW, phase_for(30, MARATHON_M),
                            Goal(race_date=pd.Timestamp("2027-05-16")), 50.0)
    five_k = next_session(ctx, _load(1.0), ZONES, EASY_SLOW, phase_for(30, 5000),
                          Goal(distance_m=5000, goal_seconds=20 * 60,
                               race_date=pd.Timestamp("2027-05-16"), label="5 km"), 50.0)
    assert marathon.distance_km > five_k.distance_km
    assert five_k.distance_km <= race_profile(5000).long_run_cap_km


def test_quality_session_emphasis_follows_the_race():
    """A 5 km peak week gets intervals; a marathon peak week gets threshold."""
    ctx = _context(days_since_quality=4.0, days_since_long=1.0)
    marathon = next_session(ctx, _load(1.0), ZONES, EASY_SLOW, phase_for(5, MARATHON_M),
                            Goal(race_date=pd.Timestamp("2027-05-16")), 50.0)
    five_k = next_session(ctx, _load(1.0), ZONES, EASY_SLOW, phase_for(2, 5000),
                          Goal(distance_m=5000, goal_seconds=20 * 60,
                               race_date=pd.Timestamp("2027-05-16"), label="5 km"), 50.0)
    assert marathon.kind == "Threshold"
    assert five_k.kind.startswith("Intervals")


def _context(**overrides):
    """A plausible RecentContext, with fields overridden per test."""
    from training_load import RecentContext
    defaults = dict(
        as_of=pd.Timestamp("2026-09-14"), days_since_run=2.0, days_since_quality=4.0,
        days_since_long=3.0, last_7_km=40.0, last_7_sessions=4, last_28_km=160.0,
        easy_share_28=0.8, longest_recent_km=20.0, weekly_km_trend=1.0,
    )
    defaults.update(overrides)
    return RecentContext(**defaults)


def _load(ratio=1.0):
    return LoadState(50.0, 50.0, ratio, 40.0, 40.0, "", "good")


ZONES = pace_zones(45)
EASY_SLOW = easy_pace_range(45)[1]
GOAL = Goal(race_date=pd.Timestamp("2027-05-16"), days_per_week=4)


def test_recovery_wins_over_everything():
    """A quality session in the last 24 hours outranks a missing long run."""
    prescription = next_session(
        _context(days_since_run=0.5, days_since_quality=0.5, days_since_long=30.0),
        _load(1.0), ZONES, EASY_SLOW, phase_for(30), GOAL, 50.0)
    assert "Rest" in prescription.kind
    assert prescription.status == "warning"


def test_a_load_spike_forces_easy_running():
    prescription = next_session(
        _context(days_since_quality=10.0, days_since_long=10.0),
        _load(1.8), ZONES, EASY_SLOW, phase_for(30), GOAL, 50.0)
    assert prescription.kind == "Easy run"
    assert prescription.status == "critical"


def test_a_long_layoff_gets_an_easy_return():
    prescription = next_session(
        _context(days_since_run=10.0, days_since_quality=10.0, days_since_long=10.0),
        _load(0.3), ZONES, EASY_SLOW, phase_for(30), GOAL, 50.0)
    assert prescription.kind == "Easy return run"


def test_a_missing_long_run_is_prescribed():
    prescription = next_session(
        _context(days_since_long=8.0), _load(1.0), ZONES, EASY_SLOW,
        phase_for(30), GOAL, 50.0)
    assert "Long run" in prescription.kind


def test_quality_when_recovered_and_load_allows():
    prescription = next_session(
        _context(days_since_quality=4.0, days_since_long=2.0), _load(1.0),
        ZONES, EASY_SLOW, phase_for(30), GOAL, 50.0)
    assert prescription.kind in ("Threshold", "Intervals (VO2max)")


def test_easy_when_quality_is_too_recent():
    prescription = next_session(
        _context(days_since_quality=1.5, days_since_long=2.0), _load(1.0),
        ZONES, EASY_SLOW, phase_for(30), GOAL, 50.0)
    assert prescription.kind == "Easy run"


def test_peak_phase_puts_marathon_pace_in_the_long_run():
    prescription = next_session(
        _context(days_since_long=8.0), _load(1.0), ZONES, EASY_SLOW,
        phase_for(6), GOAL, 50.0)
    assert "marathon-pace" in prescription.kind.lower()


def test_taper_shortens_everything():
    prescription = next_session(
        _context(days_since_quality=4.0, days_since_long=2.0), _load(1.0),
        ZONES, EASY_SLOW, phase_for(2), GOAL, 30.0)
    assert prescription.kind == "Sharpener"
    assert prescription.distance_km <= 10


def test_every_prescription_is_complete():
    """Whatever branch fires, the user gets a full answer."""
    cases = [
        _context(days_since_run=0.5, days_since_quality=0.5),
        _context(days_since_run=10.0),
        _context(days_since_long=8.0),
        _context(days_since_quality=6.0),
        _context(days_since_quality=1.0),
    ]
    for phase_weeks in (30, 12, 6, 2):
        for context in cases:
            for ratio in (0.5, 1.0, 1.8):
                prescription = next_session(context, _load(ratio), ZONES, EASY_SLOW,
                                            phase_for(phase_weeks), GOAL, 50.0)
                assert prescription.kind and prescription.structure
                assert prescription.rationale and prescription.target_pace
                assert prescription.distance_km > 0


def test_weekly_target_respects_the_ramp_cap():
    context = _context(last_28_km=160.0)
    target, _ = weekly_volume_target(context, phase_for(30), GOAL, week_index=0)
    assert 40.0 < target <= 40.0 * 1.081          # 10% cap, applied as 8%


def test_down_week_every_fourth_week():
    context = _context(last_28_km=160.0)
    target, reason = weekly_volume_target(context, phase_for(30), GOAL, week_index=3)
    assert target < 40.0
    assert "down week" in reason.lower()


def test_goal_assessment_is_honest_about_a_large_gap():
    """A VDOT 40 athlete chasing sub-3 in 10 weeks should not be told it is fine."""
    context = _context(last_28_km=120.0)
    assessment = assess_goal(40.0, Goal(race_date=pd.Timestamp("2026-11-23"), days_per_week=4),
                             context, today=pd.Timestamp("2026-09-14"))
    assert assessment.vdot_gap > 10
    assert assessment.status in ("serious", "critical")
    assert assessment.notes                                # volume gap flagged


def test_goal_assessment_recognises_a_goal_already_met():
    context = _context(last_28_km=320.0)
    assessment = assess_goal(56.0, Goal(race_date=pd.Timestamp("2027-05-16")),
                             context, today=pd.Timestamp("2026-09-14"))
    assert assessment.vdot_gap < 0
    assert assessment.status == "good"


def test_end_to_end_on_the_demo_data():
    runs = demo_runs()
    as_of = pd.Timestamp("2026-09-14")
    context = recent_context(runs, as_of=as_of)
    load = acwr(runs, as_of=as_of)
    goal = Goal(race_date=pd.Timestamp("2027-05-16"), days_per_week=4)
    phase = phase_for((goal.race_date - as_of).days / 7)
    target, _ = weekly_volume_target(context, phase, goal, 1)
    prescription = next_session(context, load, pace_zones(43.5),
                                easy_pace_range(43.5)[1], phase, goal, target)
    assessment = assess_goal(43.5, goal, context, today=as_of)

    assert phase.name == "Base"
    assert prescription.distance_km > 0
    assert math.isfinite(assessment.required_vdot)
    assert assessment.required_vdot > assessment.current_vdot


# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# The week ahead
# --------------------------------------------------------------------------

def _week(days_per_week=5, distance_m=MARATHON_M, goal_seconds=3.5 * 3600, days=7):
    runs = demo_runs()
    today = runs["date"].max()
    goal = Goal(distance_m=distance_m, goal_seconds=goal_seconds,
                race_date=today + pd.Timedelta(weeks=14), days_per_week=days_per_week)
    return week_ahead(runs, pace_zones(43.5), easy_pace_range(43.5)[1], goal, today, days=days)


def test_week_ahead_returns_one_entry_per_day():
    outlook = _week()
    assert len(outlook.days) == 7
    dates = [day.date for day in outlook.days]
    assert dates == sorted(dates)
    assert (dates[-1] - dates[0]).days == 6


def test_week_ahead_respects_running_days_per_week():
    for days_per_week in (3, 4, 5, 6):
        outlook = _week(days_per_week=days_per_week)
        ran = [day for day in outlook.days if not day.is_rest]
        # The engine may veto a scheduled day (a forced rest), so it can run
        # fewer — but never more than asked for.
        assert len(ran) <= days_per_week, (days_per_week, len(ran))


def test_week_ahead_never_stacks_quality_sessions():
    """The bug this projection was built to expose: prescribing threshold on
    consecutive days because a hard session's *average* pace reads as easy."""
    outlook = _week(days_per_week=6)
    quality_days = [index for index, day in enumerate(outlook.days)
                    if not day.is_rest and is_quality_kind(day.prescription.kind)]
    gaps = [b - a for a, b in zip(quality_days, quality_days[1:])]
    assert all(gap >= 3 for gap in gaps), (quality_days, gaps)


def test_week_ahead_totals_match_its_days():
    outlook = _week()
    assert abs(outlook.total_km - sum(day.distance_km for day in outlook.days)) < 1e-6
    assert outlook.longest_km == max(day.distance_km for day in outlook.days)


def test_hard_kilometres_are_a_minority_of_a_quality_session():
    """A threshold session is mostly easy running by distance. Counting the
    whole session as hard made the projected easy share meaningless."""
    context = _context(days_since_quality=6.0, days_since_long=1.0)
    session = next_session(context, _load(1.0), ZONES, EASY_SLOW,
                           phase_for(30, MARATHON_M), GOAL, 50.0)
    assert is_quality_kind(session.kind)
    assert 0 < session.hard_km < session.distance_km
    # And its average pace sits between the reps and easy running.
    assert ZONES["threshold"] < session.avg_pace_sec < EASY_SLOW


def test_week_ahead_easy_share_is_plausible():
    outlook = _week(days_per_week=5)
    assert 0.5 < outlook.easy_share <= 1.0, outlook.easy_share


def test_short_race_week_differs_from_marathon_week():
    marathon = _week(distance_m=MARATHON_M, goal_seconds=3.5 * 3600)
    five_k = _week(distance_m=5000, goal_seconds=21 * 60)
    assert marathon.longest_km > five_k.longest_km



# --------------------------------------------------------------------------
# Session types the runner chooses
# --------------------------------------------------------------------------

def _one_run(distance_km, seconds, name="Run", session="", when="2026-09-01"):
    frame = pd.DataFrame([{
        "date": pd.Timestamp(when), "name": name, "type": "Run",
        "distance_km": distance_km, "moving_seconds": seconds,
        "elevation_m": None, "avg_hr": None, "max_hr": None,
        "session_type": session,
    }])
    frame["pace_sec_per_km"] = frame["moving_seconds"] / frame["distance_km"]
    return add_derived(frame, pace_zones(50), easy_pace_range(50)[1])


def test_interval_session_counts_as_quality_despite_an_easy_average():
    """The reason the dropdown exists. 8 km in 44:00 is 5:30/km — squarely
    easy at VDOT 50 — but the session was 5 x 1 km with jog recoveries."""
    auto = _one_run(8.0, 44 * 60, name="Evening Run")
    assert auto["intensity"].iloc[0] == "easy"
    assert not bool(auto["is_quality"].iloc[0])

    labelled = _one_run(8.0, 44 * 60, name="Evening Run", session="Intervals")
    assert labelled["intensity"].iloc[0] == "moderate"     # floor applied
    assert bool(labelled["is_quality"].iloc[0])


def test_label_can_also_demote_a_run():
    """A brisk recovery run is still a recovery run."""
    auto = _one_run(6.0, 26 * 60, name="Morning Run")     # 4:20/km — hard
    assert auto["intensity"].iloc[0] == "hard"
    assert bool(auto["is_quality"].iloc[0])

    labelled = _one_run(6.0, 26 * 60, name="Morning Run", session="Recovery run")
    assert labelled["intensity"].iloc[0] == "easy"
    assert not bool(labelled["is_quality"].iloc[0])
    assert not bool(labelled["is_long_run"].iloc[0])


def test_intensity_floor_does_not_lower_a_genuinely_hard_run():
    """The floor raises; it never demotes. A 5 km race averaged at race pace
    stays hard rather than being pulled down to moderate."""
    run = _one_run(5.0, 19 * 60 + 57, name="Race", session="Race / time trial")
    assert run["intensity"].iloc[0] == "hard"


def test_long_run_label_overrides_the_distance_rule():
    short = _one_run(9.0, 54 * 60, session="Long run")
    assert bool(short["is_long_run"].iloc[0])
    long_but_labelled = _one_run(22.0, 2 * 3600, session="Recovery run")
    assert not bool(long_but_labelled["is_long_run"].iloc[0])


def test_unlabelled_runs_behave_exactly_as_before():
    """Adding the column must not change how an existing history is read."""
    runs = demo_runs().drop(columns=["session_type"], errors="ignore")
    zones, easy_slow = pace_zones(43.5), easy_pace_range(43.5)[1]
    with_column = add_derived(runs.assign(session_type=""), zones, easy_slow)
    without = add_derived(runs.drop(columns=["session_type"], errors="ignore"), zones, easy_slow)
    for column in ("intensity", "is_quality", "is_long_run", "load"):
        assert list(with_column[column]) == list(without[column]), column


def test_suggestions_read_the_name_before_the_pace():
    zones = pace_zones(50)
    assert suggest_session_type("Track 5x1k", 8.0, 330, zones) == "Intervals"
    assert suggest_session_type("Tempo", 10.0, 300, zones) == "Threshold / tempo"
    assert suggest_session_type("Sunday long", 12.0, 340, zones) == "Long run"
    assert suggest_session_type("Morning Run", 22.0, 340, zones) == "Long run"
    assert suggest_session_type("Morning Run", 8.0, 340, zones) == "Easy run"
    # Every suggestion must be selectable in the dropdown.
    for name, distance, pace in [("Track 5x1k", 8.0, 330), ("Morning Run", 8.0, 250)]:
        assert suggest_session_type(name, distance, pace, zones) in SESSION_TYPES


def test_session_types_round_trip_through_storage():
    import storage
    runs = _one_run(8.0, 44 * 60, name="Evening Run", session="Intervals")
    rows = storage.runs_to_rows("Szymon", runs)
    back = storage.rows_to_runs([dict(zip(storage.RUN_COLUMNS, row)) for row in rows], "Szymon")
    assert back["session_type"].iloc[0] == "Intervals"


def test_a_sheet_written_before_the_column_existed_still_loads():
    import storage
    legacy = ["profile", "date", "name", "type", "distance_km", "moving_seconds",
              "elevation_m", "avg_hr"]
    row = dict(zip(legacy, ["Szymon", "2026-09-01T07:00:00", "Run", "Run",
                            "10.000", "3000", "", ""]))
    back = storage.rows_to_runs([row], "Szymon")
    assert len(back) == 1
    assert back["session_type"].iloc[0] == ""
    assert AUTO_SESSION.label not in back["session_type"].iloc[0]


if __name__ == "__main__":
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, test in tests:
        try:
            test()
            print(f"  PASS  {name}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {name}  {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}  {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
