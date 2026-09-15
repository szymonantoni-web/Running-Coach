"""Named profiles, and where they are kept.

Streamlit Community Cloud's filesystem is ephemeral — anything written to disk
survives until the container restarts, which happens on every redeploy, reboot
and wake-from-sleep. So a local file is a cache, not storage. Profiles that
have to survive need somewhere outside the app's lifecycle.

Three layers here, deliberately separated:

* `Profile` and the `*_to_rows` / `rows_to_*` functions — the *shape* of the
  data. Pure Python, no network, no framework; this is where the bugs would
  otherwise hide, and it is fully tested.
* `LocalStore` — JSON files on disk. Works with no setup at all, and is the
  right backend when running locally. On a hosted deploy it is a cache.
* `SheetsStore` — a Google Spreadsheet, two worksheets. Durable, and you can
  open it and fix a number by hand, which is worth more than it sounds.

Both satisfy the same interface, so the app never knows which it has.
`get_store` picks one based on whether credentials are configured — meaning the
app runs before you have set anything up, and upgrades when you do.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROFILE_SHEET = "profiles"
RUNS_SHEET = "runs"

PROFILE_COLUMNS = ["name", "goal_race", "goal_seconds", "race_date", "days_per_week",
                   "effort_km", "effort_seconds", "updated_at"]
RUN_COLUMNS = ["profile", "date", "name", "type", "distance_km", "moving_seconds",
               "elevation_m", "avg_hr"]

DEMO_PROFILE = "Demo athlete"


# --------------------------------------------------------------------------
# The profile record
# --------------------------------------------------------------------------

@dataclass
class Profile:
    """Everything about an athlete except their runs."""

    name: str
    goal_race: str = "Marathon"
    goal_seconds: float = 3 * 3600
    race_date: date | None = None
    days_per_week: int = 4
    effort_km: float | None = None          # the reference effort behind the VDOT
    effort_seconds: float | None = None
    updated_at: str = ""

    def touch(self) -> "Profile":
        self.updated_at = datetime.now().isoformat(timespec="seconds")
        return self


def slugify(name: str) -> str:
    """A filesystem-safe, collision-free key for a profile name.

    Normalising alone is not enough: 'A/B' and 'A-B' both reduce to 'a-b',
    which would silently make one profile overwrite the other. So every slug
    carries a short digest of the exact original name. It uses sha1 rather
    than hash() because Python randomises string hashing per process, and a
    key that changes on restart would lose the file it points at.
    """
    base = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-") or "profile"
    digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:8]
    return f"{base[:50]}-{digest}"


# --------------------------------------------------------------------------
# Serialisation — pure, and where the real risk lives
# --------------------------------------------------------------------------

def _clean(value):
    """Empty cells come back from a spreadsheet as '' and from JSON as None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _as_float(value) -> float | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _as_int(value, default: int) -> int:
    number = _as_float(value)
    return default if number is None else int(round(number))


def _as_date(value) -> date | None:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return pd.Timestamp(str(value)).date()
    except (ValueError, TypeError):
        return None


def profile_to_row(profile: Profile) -> list[str]:
    """One profile as a spreadsheet row. Everything is written as a string so
    Sheets cannot reinterpret a number as a date."""
    return [
        str(profile.name),
        str(profile.goal_race),
        f"{profile.goal_seconds:.0f}",
        profile.race_date.isoformat() if profile.race_date else "",
        str(int(profile.days_per_week)),
        f"{profile.effort_km:.3f}" if profile.effort_km else "",
        f"{profile.effort_seconds:.0f}" if profile.effort_seconds else "",
        profile.updated_at or "",
    ]


def row_to_profile(row: dict) -> Profile | None:
    name = _clean(row.get("name"))
    if name is None:
        return None
    return Profile(
        name=str(name),
        goal_race=str(_clean(row.get("goal_race")) or "Marathon"),
        goal_seconds=_as_float(row.get("goal_seconds")) or 3 * 3600,
        race_date=_as_date(row.get("race_date")),
        days_per_week=_as_int(row.get("days_per_week"), 4),
        effort_km=_as_float(row.get("effort_km")),
        effort_seconds=_as_float(row.get("effort_seconds")),
        updated_at=str(_clean(row.get("updated_at")) or ""),
    )


def runs_to_rows(profile_name: str, runs: pd.DataFrame) -> list[list[str]]:
    """A run history as spreadsheet rows, tagged with the profile it belongs to."""
    if runs is None or runs.empty:
        return []
    rows = []
    for _, run in runs.iterrows():
        rows.append([
            str(profile_name),
            pd.Timestamp(run["date"]).isoformat(timespec="seconds"),
            str(run.get("name") or "Run"),
            str(run.get("type") or "Run"),
            f"{float(run['distance_km']):.3f}",
            f"{float(run['moving_seconds']):.0f}",
            "" if _clean(run.get("elevation_m")) is None else f"{float(run['elevation_m']):.0f}",
            "" if _clean(run.get("avg_hr")) is None else f"{float(run['avg_hr']):.0f}",
        ])
    return rows


def rows_to_runs(rows: list[dict], profile_name: str | None = None) -> pd.DataFrame:
    """Spreadsheet rows back into the frame the rest of the app expects."""
    records = []
    for row in rows:
        if profile_name is not None and str(_clean(row.get("profile")) or "") != profile_name:
            continue
        distance = _as_float(row.get("distance_km"))
        seconds = _as_float(row.get("moving_seconds"))
        when = _clean(row.get("date"))
        if not distance or not seconds or when is None:
            continue
        try:
            stamp = pd.Timestamp(str(when))
        except (ValueError, TypeError):
            continue
        records.append({
            "date": stamp,
            "name": str(_clean(row.get("name")) or "Run"),
            "type": str(_clean(row.get("type")) or "Run"),
            "distance_km": distance,
            "moving_seconds": seconds,
            "elevation_m": _as_float(row.get("elevation_m")),
            "avg_hr": _as_float(row.get("avg_hr")),
            "max_hr": None,
        })

    if not records:
        return pd.DataFrame(columns=["date", "name", "type", "distance_km", "moving_seconds",
                                     "elevation_m", "avg_hr", "max_hr", "pace_sec_per_km"])

    frame = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    frame["pace_sec_per_km"] = frame["moving_seconds"] / frame["distance_km"]
    return frame


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class LocalStore:
    """Profiles as JSON files in a directory.

    Durable when you run the app on your own machine. On Streamlit Community
    Cloud the container's disk is wiped on every restart, so `is_durable` is
    False there and the app says so rather than quietly losing data.
    """

    def __init__(self, directory: str | Path, durable: bool = True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._durable = durable

    label = "Local files"

    @property
    def is_durable(self) -> bool:
        return self._durable

    def _path(self, name: str) -> Path:
        return self.directory / f"{slugify(name)}.json"

    def list_profiles(self) -> list[str]:
        names = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                names.append(json.loads(path.read_text())["profile"]["name"])
            except (ValueError, KeyError, OSError):
                continue
        return sorted(names, key=str.lower)

    def load(self, name: str) -> tuple[Profile, pd.DataFrame]:
        """An unknown profile comes back empty rather than raising — the same
        as the Sheets backend, so the app never has to know which it has."""
        path = self._path(name)
        if not path.exists():
            return Profile(name=name), rows_to_runs([])
        try:
            payload = json.loads(path.read_text())
        except (ValueError, OSError):
            return Profile(name=name), rows_to_runs([])
        profile = row_to_profile(payload.get("profile", {})) or Profile(name=name)
        return profile, rows_to_runs(payload.get("runs", []))

    def save(self, profile: Profile, runs: pd.DataFrame) -> None:
        profile.touch()
        payload = {
            "profile": dict(zip(PROFILE_COLUMNS, profile_to_row(profile))),
            "runs": [dict(zip(RUN_COLUMNS, row)) for row in runs_to_rows(profile.name, runs)],
        }
        self._path(profile.name).write_text(json.dumps(payload, indent=2))

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)


class SheetsStore:
    """Profiles in a Google Spreadsheet: one `profiles` worksheet, one `runs`.

    Each save rewrites the whole worksheet rather than patching rows. At this
    size — a few profiles, a few hundred runs — that costs nothing and removes
    a whole category of partial-write bug.

    Takes an already-authorised spreadsheet object, so the transport is
    injectable and the logic is testable without a network.
    """

    label = "Google Sheets"
    is_durable = True

    def __init__(self, spreadsheet):
        self.spreadsheet = spreadsheet

    def _worksheet(self, title: str, columns: list[str]):
        """Fetch a worksheet, creating it with its header row if absent."""
        try:
            worksheet = self.spreadsheet.worksheet(title)
        except Exception:  # noqa: BLE001 — gspread raises WorksheetNotFound
            worksheet = self.spreadsheet.add_worksheet(title=title, rows=1000,
                                                       cols=max(len(columns), 8))
            worksheet.append_rows([columns], value_input_option="RAW")
            return worksheet

        if not worksheet.get_all_values():
            worksheet.append_rows([columns], value_input_option="RAW")
        return worksheet

    @staticmethod
    def _records(worksheet, columns: list[str]) -> list[dict]:
        """Rows as dicts. Read by header name rather than position, so a column
        added by hand in the sheet does not shift everything."""
        values = worksheet.get_all_values()
        if not values:
            return []
        header = [str(cell).strip() for cell in values[0]]
        if not any(header):
            header = columns
            values = [header] + values
        return [dict(zip(header, row)) for row in values[1:] if any(str(c).strip() for c in row)]

    def _rewrite(self, worksheet, columns: list[str], rows: list[list[str]]) -> None:
        worksheet.clear()
        worksheet.append_rows([columns] + rows, value_input_option="RAW")

    def list_profiles(self) -> list[str]:
        records = self._records(self._worksheet(PROFILE_SHEET, PROFILE_COLUMNS), PROFILE_COLUMNS)
        names = [str(_clean(r.get("name"))) for r in records if _clean(r.get("name"))]
        return sorted(dict.fromkeys(names), key=str.lower)

    def load(self, name: str) -> tuple[Profile, pd.DataFrame]:
        records = self._records(self._worksheet(PROFILE_SHEET, PROFILE_COLUMNS), PROFILE_COLUMNS)
        profile = next((row_to_profile(r) for r in records
                        if str(_clean(r.get("name")) or "") == name), None)
        run_records = self._records(self._worksheet(RUNS_SHEET, RUN_COLUMNS), RUN_COLUMNS)
        return profile or Profile(name=name), rows_to_runs(run_records, profile_name=name)

    def save(self, profile: Profile, runs: pd.DataFrame) -> None:
        profile.touch()

        sheet = self._worksheet(PROFILE_SHEET, PROFILE_COLUMNS)
        records = self._records(sheet, PROFILE_COLUMNS)
        kept = [r for r in records if str(_clean(r.get("name")) or "") != profile.name]
        rows = [[str(r.get(column, "")) for column in PROFILE_COLUMNS] for r in kept]
        self._rewrite(sheet, PROFILE_COLUMNS, rows + [profile_to_row(profile)])

        runs_sheet = self._worksheet(RUNS_SHEET, RUN_COLUMNS)
        run_records = self._records(runs_sheet, RUN_COLUMNS)
        other = [r for r in run_records if str(_clean(r.get("profile")) or "") != profile.name]
        other_rows = [[str(r.get(column, "")) for column in RUN_COLUMNS] for r in other]
        self._rewrite(runs_sheet, RUN_COLUMNS, other_rows + runs_to_rows(profile.name, runs))

    def delete(self, name: str) -> None:
        sheet = self._worksheet(PROFILE_SHEET, PROFILE_COLUMNS)
        kept = [r for r in self._records(sheet, PROFILE_COLUMNS)
                if str(_clean(r.get("name")) or "") != name]
        self._rewrite(sheet, PROFILE_COLUMNS,
                      [[str(r.get(c, "")) for c in PROFILE_COLUMNS] for r in kept])

        runs_sheet = self._worksheet(RUNS_SHEET, RUN_COLUMNS)
        kept_runs = [r for r in self._records(runs_sheet, RUN_COLUMNS)
                     if str(_clean(r.get("profile")) or "") != name]
        self._rewrite(runs_sheet, RUN_COLUMNS,
                      [[str(r.get(c, "")) for c in RUN_COLUMNS] for r in kept_runs])


# --------------------------------------------------------------------------
# Choosing a backend
# --------------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive.file"]


def open_spreadsheet(service_account_info: dict, key: str | None = None,
                     url: str | None = None, title: str | None = None):
    """Authorise with a service account and open the target spreadsheet.

    Kept separate from SheetsStore so the store itself needs no network and
    stays testable.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    credentials = Credentials.from_service_account_info(dict(service_account_info), scopes=SCOPES)
    client = gspread.authorize(credentials)

    if key:
        return client.open_by_key(key)
    if url:
        return client.open_by_url(url)
    if title:
        return client.open(title)
    raise ValueError("Set one of spreadsheet_key, spreadsheet_url or spreadsheet_name in secrets.")


@dataclass
class StoreStatus:
    store: object
    durable: bool
    label: str
    detail: str
    error: str | None = None


def get_store(secrets: dict | None, local_dir: str | Path,
              local_is_durable: bool = True) -> StoreStatus:
    """Google Sheets when it is configured and reachable; local files otherwise.

    A configuration problem never stops the app — it falls back and reports
    why, because being unable to save is much better than being unable to run.
    """
    secrets = secrets or {}
    account = secrets.get("gcp_service_account")
    sheet_config = secrets.get("sheets", {}) or {}

    if account:
        try:
            spreadsheet = open_spreadsheet(
                dict(account),
                key=sheet_config.get("spreadsheet_key"),
                url=sheet_config.get("spreadsheet_url"),
                title=sheet_config.get("spreadsheet_name"),
            )
            return StoreStatus(SheetsStore(spreadsheet), True, "Google Sheets",
                               "Profiles are saved to your Google Sheet.")
        except Exception as error:  # noqa: BLE001 — any failure falls back
            return StoreStatus(
                LocalStore(local_dir, durable=local_is_durable), local_is_durable,
                "Local files (Sheets unavailable)",
                "Google Sheets is configured but could not be opened, so profiles are "
                "being kept locally instead.",
                error=str(error),
            )

    return StoreStatus(
        LocalStore(local_dir, durable=local_is_durable), local_is_durable, "Local files",
        "No Google Sheets credentials configured — profiles are kept in local files.",
    )
