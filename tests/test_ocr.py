"""Tests for screenshot reading.

Two halves. The parser tests run anywhere — they feed text in and check the
structure that comes out, which is where all the ambiguity lives. The
end-to-end tests render Strava-like screenshots, push them through Tesseract,
and check the numbers survive; those skip automatically when the Tesseract
binary is not installed.

Run with `pytest`, or `python tests/test_ocr.py`.
"""

from __future__ import annotations

import io
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr import (  # noqa: E402
    ParsedRun,
    _clock_to_seconds,
    _fix_units,
    _parse_date,
    _parse_name,
    cross_validate,
    parse_activity_text,
)
from strava import append_runs, load_activities  # noqa: E402

TODAY = date(2026, 9, 14)


def close(a, b, tol=0.02):
    """Within `tol` relative — OCR reads what the app rounded, so exactness is
    not the standard."""
    if a is None or b is None:
        return False
    return abs(a - b) <= abs(b) * tol


def has_tesseract() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# Small pieces
# --------------------------------------------------------------------------

def test_clock_parsing():
    assert _clock_to_seconds("1:06:52") == 4012
    assert _clock_to_seconds("52:18") == 3138
    assert _clock_to_seconds("0:44:12") == 2652
    assert _clock_to_seconds("1:75:00") is None       # impossible minutes
    assert _clock_to_seconds("12") is None
    assert _clock_to_seconds("not a time") is None


def test_unit_repair():
    """Tesseract reads the m of km as n or rn more often than anything else."""
    assert "/km" in _fix_units("5:23 /kn")
    assert "/km" in _fix_units("5:23 /krn")
    assert "/mi" in _fix_units("8:27 /rni")
    assert "12.4 km" in _fix_units("12.4 kn")


def test_date_parsing():
    assert _parse_date("Today at 7:32 AM", today=TODAY) == TODAY
    assert _parse_date("Yesterday at 6:10 PM", today=TODAY) == date(2026, 9, 13)
    assert _parse_date("September 13, 2026", today=TODAY) == date(2026, 9, 13)
    assert _parse_date("Sep 13, 2026", today=TODAY) == date(2026, 9, 13)
    assert _parse_date("13.09.2026", today=TODAY) == date(2026, 9, 13)
    assert _parse_date("2026-09-13", today=TODAY) == date(2026, 9, 13)
    assert _parse_date("no date here", today=TODAY) is None


def test_a_bare_date_in_the_future_belongs_to_last_year():
    assert _parse_date("Monday, 20 December", today=TODAY) == date(2025, 12, 20)
    assert _parse_date("Monday, 20 August", today=TODAY) == date(2026, 8, 20)


def test_activity_names_survive_unit_words():
    """'5 x 1 km intervals' is a title, even though it contains 'km'."""
    assert _parse_name("5 x 1 km intervals\nSeptember 10, 2026\n") == "5 x 1 km intervals"
    assert _parse_name("Morning Run\nToday at 7:32 AM\n12.41 1:06:52") == "Morning Run"
    assert _parse_name("Distance\nPace\nMoving Time") is None
    assert _parse_name("September 10, 2026\n11.00 km") is None


# --------------------------------------------------------------------------
# Cross-validation — the real safety net
# --------------------------------------------------------------------------

def test_the_third_value_is_derived_from_the_other_two():
    parsed = cross_validate(ParsedRun(distance_km=10.0, moving_seconds=3000.0))
    assert close(parsed.pace_sec_per_km, 300.0)

    parsed = cross_validate(ParsedRun(distance_km=10.0, pace_sec_per_km=300.0))
    assert close(parsed.moving_seconds, 3000.0)

    parsed = cross_validate(ParsedRun(moving_seconds=3000.0, pace_sec_per_km=300.0))
    assert close(parsed.distance_km, 10.0)


def test_three_values_that_disagree_are_flagged_not_reconciled():
    """A misread digit produces three individually plausible numbers that
    cannot all be true. The tool must say so rather than pick a winner."""
    parsed = cross_validate(ParsedRun(distance_km=12.41, moving_seconds=4012.0,
                                      pace_sec_per_km=263.0))
    assert parsed.warnings
    assert parsed.confidence["distance_km"] == "low"
    assert parsed.overall_confidence == "low"
    # Nothing was silently "fixed".
    assert parsed.distance_km == 12.41 and parsed.pace_sec_per_km == 263.0


def test_consistent_values_pass_without_complaint():
    parsed = cross_validate(ParsedRun(distance_km=10.0, moving_seconds=3000.0,
                                      pace_sec_per_km=300.0))
    assert not parsed.warnings


def test_implausible_values_are_caught():
    parsed = cross_validate(ParsedRun(distance_km=1241.0, moving_seconds=4012.0))
    assert any("Distance" in w for w in parsed.warnings)

    parsed = cross_validate(ParsedRun(distance_km=10.0, moving_seconds=600.0))
    assert any("pace" in w.lower() for w in parsed.warnings)

    parsed = cross_validate(ParsedRun(distance_km=10.0, moving_seconds=3000.0, avg_hr=12.0))
    assert parsed.avg_hr is None          # nonsense heart rate is dropped


# --------------------------------------------------------------------------
# Text -> structure
# --------------------------------------------------------------------------

def test_web_layout_text():
    parsed = parse_activity_text(
        "Long run\nSeptember 13, 2026\nDistance 21.48 km\nMoving Time 1:59:37\n"
        "Pace 5:34 /km\nElevation Gain 124 m\nAvg HR 148 bpm\n", today=TODAY)
    assert close(parsed.distance_km, 21.48)
    assert close(parsed.moving_seconds, 7177)
    assert close(parsed.pace_sec_per_km, 334)
    assert parsed.avg_hr == 148
    assert parsed.elevation_m == 124
    assert parsed.run_date == date(2026, 9, 13)
    assert parsed.name == "Long run"


def test_miles_are_converted():
    parsed = parse_activity_text(
        "Evening Run\nToday\n7.71 mi 1:02:14 8:04 /mi\nDistance Time Pace\n", today=TODAY)
    assert close(parsed.distance_km, 12.41)
    assert close(parsed.pace_sec_per_km, 301)


def test_comma_decimals():
    parsed = parse_activity_text("Abendlauf\n13.09.2026\n10,25 km 0:56:42 5:32 /km\n", today=TODAY)
    assert close(parsed.distance_km, 10.25)
    assert parsed.run_date == date(2026, 9, 13)


def test_cells_take_priority_over_flat_text():
    """The column-cropped read is the more reliable one, so it wins."""
    parsed = parse_activity_text(
        "Morning Run\nToday\n12.41 1:06:53:23 /km\nDistance Moving Time Pace\n",
        today=TODAY,
        cells={"distance": "12.41", "time": "1:06:52", "pace": "5:23 /km"},
    )
    assert close(parsed.distance_km, 12.41)
    assert close(parsed.moving_seconds, 4012)
    assert close(parsed.pace_sec_per_km, 323)
    assert parsed.confidence["distance_km"] == "high"
    assert not parsed.warnings


def test_two_of_three_is_enough():
    parsed = parse_activity_text("Recovery jog\nToday\n8.20 km\nMoving Time 0:47:30\n", today=TODAY)
    assert parsed.is_usable
    assert close(parsed.pace_sec_per_km, 2850 / 8.2)


def test_unreadable_input_fails_loudly():
    parsed = parse_activity_text("~~~ ### ???", today=TODAY)
    assert not parsed.is_usable
    assert parsed.warnings
    assert parsed.overall_confidence == "low"


def test_a_missing_date_defaults_to_today_and_says_so():
    parsed = parse_activity_text("Morning Run\n10.00 km\nMoving Time 0:52:00\n", today=TODAY)
    assert parsed.run_date == TODAY
    assert any("date" in w.lower() for w in parsed.warnings)
    assert parsed.confidence["run_date"] == "low"


def test_empty_input():
    parsed = parse_activity_text("", today=TODAY)
    assert not parsed.is_usable
    assert parsed.warnings


# --------------------------------------------------------------------------
# End to end, through real Tesseract
# --------------------------------------------------------------------------

def _end_to_end(name: str):
    from make_screenshots import EXPECTED, render_bytes
    from ocr import read_screenshot

    parsed = read_screenshot(render_bytes(name), today=TODAY)
    expected = EXPECTED[name]
    for field, want in expected.items():
        got = getattr(parsed, field)
        assert close(got, want), f"{name}.{field}: read {got}, expected {want}"
    return parsed


def test_end_to_end_phone_light():
    if not has_tesseract():
        return
    parsed = _end_to_end("phone_light")
    assert parsed.name == "Morning Run"
    assert parsed.run_date == TODAY
    assert not parsed.warnings


def test_end_to_end_phone_dark():
    """Dark mode is inverted before OCR; Tesseract expects dark on light."""
    if not has_tesseract():
        return
    parsed = _end_to_end("phone_dark")
    assert parsed.run_date == date(2026, 9, 13)      # "Yesterday"


def test_end_to_end_phone_miles():
    if not has_tesseract():
        return
    _end_to_end("phone_miles")


def test_end_to_end_web_layouts():
    if not has_tesseract():
        return
    _end_to_end("web_light")
    parsed = _end_to_end("web_dark")
    assert parsed.name == "5 x 1 km intervals"


def test_a_screenshot_with_merged_columns_is_flagged_not_guessed():
    """The overflow case is genuinely unreadable. The requirement is that the
    app refuses to present a confident answer, not that it reads it."""
    if not has_tesseract():
        return
    from make_screenshots import render_bytes
    from ocr import read_screenshot

    parsed = read_screenshot(render_bytes("phone_overflow"), today=TODAY)
    assert parsed.overall_confidence == "low"
    assert parsed.warnings


# --------------------------------------------------------------------------
# Appending to history
# --------------------------------------------------------------------------

def _demo() -> pd.DataFrame:
    return load_activities(Path(__file__).resolve().parents[1] / "data" / "sample_activities.csv")


def test_appending_a_run():
    runs = _demo()
    before = len(runs)
    merged = append_runs(runs, [{
        "date": pd.Timestamp("2026-09-15 07:30"), "name": "Morning Run",
        "distance_km": 12.41, "moving_seconds": 4012.0, "avg_hr": 148.0, "elevation_m": 86.0,
    }])
    assert len(merged) == before + 1
    assert close(float(merged.iloc[-1]["pace_sec_per_km"]), 4012 / 12.41)
    assert merged["date"].is_monotonic_increasing


def test_the_same_screenshot_twice_does_not_double_count():
    runs = _demo()
    row = {"date": pd.Timestamp("2026-09-15 07:30"), "name": "Morning Run",
           "distance_km": 12.41, "moving_seconds": 4012.0}
    once = append_runs(runs, [row])
    twice = append_runs(once, [row])
    assert len(twice) == len(once)


def test_history_survives_an_export_and_reload():
    """The download/re-upload loop is how a run persists between sessions, so
    it has to be lossless."""
    runs = append_runs(_demo(), [{
        "date": pd.Timestamp("2026-09-15 07:30"), "name": "Morning Run",
        "distance_km": 12.41, "moving_seconds": 4012.0, "avg_hr": 148.0, "elevation_m": 86.0,
    }])
    export = pd.DataFrame({
        "Activity Date": runs["date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Activity Name": runs["name"],
        "Activity Type": runs["type"],
        "Moving Time": runs["moving_seconds"].round(0),
        "Distance": (runs["distance_km"] * 1000).round(1),
        "Elevation Gain": runs["elevation_m"],
        "Average Heart Rate": runs["avg_hr"],
    }).to_csv(index=False).encode("utf-8")

    reloaded = load_activities(io.BytesIO(export))
    assert len(reloaded) == len(runs)
    assert abs(reloaded["distance_km"].sum() - runs["distance_km"].sum()) < 0.01
    assert abs(reloaded["moving_seconds"].sum() - runs["moving_seconds"].sum()) < 1


# --------------------------------------------------------------------------

if __name__ == "__main__":
    if not has_tesseract():
        print("  NOTE  Tesseract not found — end-to-end image tests will no-op.\n")
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
