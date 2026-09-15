"""Tests for profiles and where they are stored.

The Sheets backend is exercised against an in-memory fake of the gspread
surface (tests/fake_sheets.py), so everything except the network call itself is
covered. Both backends are run through the *same* test body wherever possible —
if they ever diverge in behaviour, that is the bug.

Run with `pytest`, or `python tests/test_storage.py`.
"""

from __future__ import annotations

import math
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_sheets import FakeSpreadsheet  # noqa: E402
from storage import (  # noqa: E402
    PROFILE_COLUMNS,
    RUN_COLUMNS,
    LocalStore,
    Profile,
    SheetsStore,
    get_store,
    profile_to_row,
    row_to_profile,
    rows_to_runs,
    runs_to_rows,
    slugify,
)


def sample_runs() -> pd.DataFrame:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-01 07:30", "2026-09-03 18:00", "2026-09-06 08:00"]),
        "name": ["Easy run", "5 x 1 km intervals", "Long run"],
        "type": ["Run", "Run", "Run"],
        "distance_km": [10.0, 11.0, 21.48],
        "moving_seconds": [3300.0, 3376.0, 7177.0],
        "elevation_m": [45.0, 31.0, 124.0],
        "avg_hr": [142.0, 163.0, 148.0],
        "max_hr": [None, None, None],
    })
    frame["pace_sec_per_km"] = frame["moving_seconds"] / frame["distance_km"]
    return frame


def sample_profile(name: str = "Szymon") -> Profile:
    return Profile(name=name, goal_race="Marathon", goal_seconds=10800,
                   race_date=date(2027, 5, 16), days_per_week=4,
                   effort_km=11.0, effort_seconds=3376.0)


def both_stores():
    """Yield (label, store) for each backend, over throwaway storage."""
    directory = tempfile.mkdtemp()
    yield "local", LocalStore(directory)
    yield "sheets", SheetsStore(FakeSpreadsheet())


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def test_profile_round_trip():
    original = sample_profile()
    recovered = row_to_profile(dict(zip(PROFILE_COLUMNS, profile_to_row(original))))
    assert recovered.name == original.name
    assert recovered.goal_seconds == original.goal_seconds
    assert recovered.race_date == original.race_date
    assert recovered.days_per_week == original.days_per_week
    assert recovered.effort_km == original.effort_km
    assert recovered.effort_seconds == original.effort_seconds


def test_runs_round_trip():
    runs = sample_runs()
    rows = [dict(zip(RUN_COLUMNS, row)) for row in runs_to_rows("Szymon", runs)]
    recovered = rows_to_runs(rows, profile_name="Szymon")

    assert len(recovered) == len(runs)
    assert abs(recovered["distance_km"].sum() - runs["distance_km"].sum()) < 1e-6
    assert abs(recovered["moving_seconds"].sum() - runs["moving_seconds"].sum()) < 1e-6
    assert list(recovered["name"]) == list(runs["name"])
    assert recovered["date"].is_monotonic_increasing
    # Pace is recomputed rather than stored, so it cannot drift from the inputs.
    assert abs(recovered["pace_sec_per_km"].iloc[0] - 330.0) < 1e-9


def test_runs_are_filtered_by_profile():
    rows = ([dict(zip(RUN_COLUMNS, row)) for row in runs_to_rows("Szymon", sample_runs())] +
            [dict(zip(RUN_COLUMNS, row)) for row in runs_to_rows("Marta", sample_runs().head(1))])
    assert len(rows_to_runs(rows, profile_name="Szymon")) == 3
    assert len(rows_to_runs(rows, profile_name="Marta")) == 1
    assert len(rows_to_runs(rows, profile_name="Nobody")) == 0


def test_empty_and_malformed_rows_are_skipped_not_fatal():
    rows = [
        {"profile": "A", "date": "", "distance_km": "10", "moving_seconds": "3000"},
        {"profile": "A", "date": "2026-09-01", "distance_km": "", "moving_seconds": "3000"},
        {"profile": "A", "date": "not a date", "distance_km": "10", "moving_seconds": "3000"},
        {"profile": "A", "date": "2026-09-01", "distance_km": "10", "moving_seconds": "3000"},
    ]
    recovered = rows_to_runs(rows, profile_name="A")
    assert len(recovered) == 1


def test_no_runs_gives_an_empty_frame_with_the_right_columns():
    empty = rows_to_runs([], profile_name="A")
    assert empty.empty
    for column in ("date", "distance_km", "moving_seconds", "pace_sec_per_km"):
        assert column in empty.columns


def test_missing_optional_fields_survive():
    profile = Profile(name="Minimal")
    recovered = row_to_profile(dict(zip(PROFILE_COLUMNS, profile_to_row(profile))))
    assert recovered.race_date is None
    assert recovered.effort_km is None
    assert recovered.days_per_week == 4


def test_a_row_without_a_name_is_not_a_profile():
    assert row_to_profile({"name": ""}) is None
    assert row_to_profile({}) is None


def test_slugify_is_readable_and_stable():
    assert slugify("Szymon").startswith("szymon-")
    assert slugify("My Marathon Build").startswith("my-marathon-build-")
    assert slugify("   ").startswith("profile-")
    # Stable across calls — the key must still find the file after a restart.
    assert slugify("Szymon") == slugify("Szymon")


def test_slugify_never_collides():
    """'A/B' and 'A-B' normalise to the same text; without a digest one profile
    would silently overwrite the other."""
    names = ["A/B", "A-B", "A B", "a b", "Szymon", "szymon", "Zoë", "Zoe"]
    slugs = [slugify(name) for name in names]
    assert len(set(slugs)) == len(set(names))


def test_two_similar_names_stay_separate_on_disk():
    with tempfile.TemporaryDirectory() as directory:
        store = LocalStore(directory)
        store.save(sample_profile("A/B"), sample_runs())
        store.save(sample_profile("A-B"), sample_runs().head(1))
        assert sorted(store.list_profiles()) == ["A-B", "A/B"]
        assert len(store.load("A/B")[1]) == 3
        assert len(store.load("A-B")[1]) == 1


# --------------------------------------------------------------------------
# Both backends, same expectations
# --------------------------------------------------------------------------

def test_save_then_load():
    for label, store in both_stores():
        store.save(sample_profile(), sample_runs())
        profile, runs = store.load("Szymon")
        assert profile.name == "Szymon", label
        assert profile.race_date == date(2027, 5, 16), label
        assert profile.effort_seconds == 3376.0, label
        assert len(runs) == 3, label
        assert abs(runs["distance_km"].sum() - 42.48) < 1e-6, label
        assert profile.updated_at, label          # save stamps the time


def test_listing_profiles():
    for label, store in both_stores():
        assert store.list_profiles() == [], label
        store.save(sample_profile("Szymon"), sample_runs())
        store.save(sample_profile("Marta"), sample_runs().head(2))
        assert store.list_profiles() == ["Marta", "Szymon"], label


def test_saving_one_profile_leaves_the_other_alone():
    """The failure that matters most: a rewrite that drops somebody else's data."""
    for label, store in both_stores():
        store.save(sample_profile("Szymon"), sample_runs())
        store.save(sample_profile("Marta"), sample_runs().head(1))

        updated = sample_profile("Szymon")
        updated.goal_seconds = 11400
        store.save(updated, sample_runs().head(2))

        marta_profile, marta_runs = store.load("Marta")
        assert marta_profile.name == "Marta", label
        assert len(marta_runs) == 1, label

        szymon_profile, szymon_runs = store.load("Szymon")
        assert szymon_profile.goal_seconds == 11400, label
        assert len(szymon_runs) == 2, label       # replaced, not appended


def test_resaving_replaces_rather_than_duplicates():
    for label, store in both_stores():
        store.save(sample_profile(), sample_runs())
        store.save(sample_profile(), sample_runs())
        _, runs = store.load("Szymon")
        assert len(runs) == 3, label
        assert store.list_profiles() == ["Szymon"], label


def test_deleting_a_profile_removes_its_runs_too():
    for label, store in both_stores():
        store.save(sample_profile("Szymon"), sample_runs())
        store.save(sample_profile("Marta"), sample_runs().head(1))
        store.delete("Szymon")

        assert store.list_profiles() == ["Marta"], label
        _, orphaned = store.load("Szymon")
        assert orphaned.empty, label
        _, marta = store.load("Marta")
        assert len(marta) == 1, label


def test_loading_an_unknown_profile_is_empty_not_an_error():
    for label, store in both_stores():
        profile, runs = store.load("Nobody")
        assert profile.name == "Nobody", label
        assert runs.empty, label


def test_a_profile_with_no_runs():
    for label, store in both_stores():
        store.save(sample_profile("Fresh"), pd.DataFrame())
        profile, runs = store.load("Fresh")
        assert profile.name == "Fresh", label
        assert runs.empty, label


# --------------------------------------------------------------------------
# Sheets specifics
# --------------------------------------------------------------------------

def test_worksheets_are_created_with_headers():
    spreadsheet = FakeSpreadsheet()
    store = SheetsStore(spreadsheet)
    store.save(sample_profile(), sample_runs())

    assert spreadsheet.titles == ["profiles", "runs"]
    assert spreadsheet.raw("profiles")[0] == PROFILE_COLUMNS
    assert spreadsheet.raw("runs")[0] == RUN_COLUMNS
    assert len(spreadsheet.raw("runs")) == 1 + 3


def test_columns_are_read_by_header_not_position():
    """Someone reorders the columns in the sheet by hand. It should still load."""
    spreadsheet = FakeSpreadsheet()
    sheet = spreadsheet.add_worksheet("profiles")
    sheet.seed([
        ["goal_seconds", "name", "race_date", "days_per_week",
         "goal_race", "effort_km", "effort_seconds", "updated_at"],
        ["10800", "Szymon", "2027-05-16", "4", "Marathon", "11.0", "3376", ""],
    ])
    profile, _ = SheetsStore(spreadsheet).load("Szymon")
    assert profile.name == "Szymon"
    assert profile.goal_seconds == 10800
    assert profile.race_date == date(2027, 5, 16)


def test_blank_rows_in_the_sheet_are_ignored():
    spreadsheet = FakeSpreadsheet()
    sheet = spreadsheet.add_worksheet("profiles")
    sheet.seed([PROFILE_COLUMNS,
                ["Szymon", "Marathon", "10800", "2027-05-16", "4", "11.0", "3376", ""],
                ["", "", "", "", "", "", "", ""],
                ["   ", "", "", "", "", "", "", ""]])
    assert SheetsStore(spreadsheet).list_profiles() == ["Szymon"]


def test_everything_is_written_as_text():
    """Sheets will happily reinterpret 5:23 as a time or 1.5 as a date if given
    the chance, so every cell goes out as a string."""
    spreadsheet = FakeSpreadsheet()
    SheetsStore(spreadsheet).save(sample_profile(), sample_runs())
    for title in ("profiles", "runs"):
        for row in spreadsheet.raw(title):
            assert all(isinstance(cell, str) for cell in row), title


def test_a_save_rewrites_rather_than_appending_forever():
    spreadsheet = FakeSpreadsheet()
    store = SheetsStore(spreadsheet)
    for _ in range(4):
        store.save(sample_profile(), sample_runs())
    assert len(spreadsheet.raw("runs")) == 1 + 3
    assert len(spreadsheet.raw("profiles")) == 1 + 1


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

def test_no_credentials_falls_back_to_local():
    with tempfile.TemporaryDirectory() as directory:
        status = get_store({}, directory)
        assert isinstance(status.store, LocalStore)
        assert status.error is None
        assert "No Google Sheets credentials" in status.detail


def test_broken_credentials_fall_back_instead_of_crashing():
    """An app that cannot save is annoying. An app that will not start is
    useless — so a configuration problem must never be fatal."""
    with tempfile.TemporaryDirectory() as directory:
        status = get_store({"gcp_service_account": {"nonsense": True},
                            "sheets": {"spreadsheet_key": "abc"}}, directory)
        assert isinstance(status.store, LocalStore)
        assert status.error is not None
        assert "could not be opened" in status.detail


def test_local_store_can_be_marked_non_durable():
    """On Streamlit Cloud the disk is wiped on restart, and the app should say
    so rather than implying the data is safe."""
    with tempfile.TemporaryDirectory() as directory:
        status = get_store({}, directory, local_is_durable=False)
        assert status.durable is False


def test_unicode_and_awkward_names():
    with tempfile.TemporaryDirectory() as directory:
        store = LocalStore(directory)
        for name in ["Szymon", "Zoë", "Marta's build", "Läufer/2027"]:
            store.save(sample_profile(name), sample_runs().head(1))
        assert len(store.list_profiles()) == 4
        profile, runs = store.load("Läufer/2027")
        assert profile.name == "Läufer/2027"
        assert len(runs) == 1


def test_nan_values_do_not_reach_the_sheet():
    runs = sample_runs()
    runs.loc[0, "avg_hr"] = math.nan
    runs.loc[1, "elevation_m"] = math.nan
    rows = runs_to_rows("Szymon", runs)
    assert rows[0][7] == ""
    assert rows[1][6] == ""
    recovered = rows_to_runs([dict(zip(RUN_COLUMNS, r)) for r in rows], "Szymon")
    assert recovered["avg_hr"].isna().iloc[0]


# --------------------------------------------------------------------------

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
