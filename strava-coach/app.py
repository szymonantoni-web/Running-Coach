"""Strava Coach — a Streamlit front end.

Run locally with:  streamlit run app.py

Layout only. The physiology is in physiology.py, the load maths in
training_load.py, the decisions in plan.py — so you can change what the coach
says without touching how it looks, and the reverse.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
import ocr
import storage
from physiology import (
    COMMON_RACES,
    MARATHON_M,
    easy_pace_range,
    format_duration,
    format_pace,
    pace_zones,
    parse_duration,
    predict_race_seconds,
    vdot_from_race,
    ZONES,
)
from plan import (
    Goal,
    assess_goal,
    next_session,
    phase_for,
    typical_peak_volume,
    weekly_volume_target,
)
from strava import append_runs, best_efforts, load_activities
from training_load import acwr, add_derived, recent_context, weekly_summary

DEMO_PATH = Path(__file__).parent / "data" / "sample_activities.csv"

st.set_page_config(page_title="Strava Coach", page_icon="🏃", layout="wide",
                   initial_sidebar_state="expanded")

STATUS_ICON = {"good": "✅", "warning": "⚠️", "serious": "🟠", "critical": "🛑", "muted": "•"}


# --------------------------------------------------------------------------
# Streamlit compatibility shims — the width argument was renamed; try the new
# name first so the app works on both sides of that change.
# --------------------------------------------------------------------------

def _rerun_now() -> None:
    try:
        st.rerun()
    except AttributeError:  # older Streamlit
        st.experimental_rerun()


def show_chart(chart) -> None:
    if chart is None:
        return
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def show_table(frame, **kwargs) -> None:
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(frame, use_container_width=True, **kwargs)


@st.cache_data(show_spinner=False)
def _load_demo() -> pd.DataFrame:
    return load_activities(DEMO_PATH)


@st.cache_data(show_spinner=False)
def _load_upload(payload: bytes, assume_miles: bool) -> pd.DataFrame:
    import io
    return load_activities(io.BytesIO(payload), assume_miles=assume_miles)


@st.cache_data(show_spinner=False)
def _read_screenshot(payload: bytes, today_iso: str, assume_miles: bool):
    """Cached so editing a field in the form does not re-run Tesseract."""
    return ocr.read_screenshot(payload, today=date.fromisoformat(today_iso),
                               assume_miles=assume_miles)


# --------------------------------------------------------------------------
# Storage and profiles
# --------------------------------------------------------------------------

NEW_PROFILE = "➕ New profile…"
LOCAL_PROFILE_DIR = Path(__file__).parent / "data" / "profiles"


@st.cache_resource(show_spinner=False)
def _get_store(_generation: int = 0) -> storage.StoreStatus:
    """One store per server process, because it holds an authorised connection.

    `_generation` exists so the sidebar's Reconnect button can force a fresh
    one: a cached connection built before the credentials existed would
    otherwise survive until the server is restarted, and report "no
    credentials" long after that stopped being true.
    """
    secrets: dict = {}
    secrets_error: str | None = None
    try:
        secrets = {key: st.secrets[key] for key in st.secrets}
    except Exception as error:  # noqa: BLE001 — no secrets file at all is normal
        secrets_error = f"{type(error).__name__}: {error}"

    # Streamlit Community Cloud checks the repository out under /mount/src and
    # wipes that disk on every restart, which is what makes local files a cache
    # there rather than storage. Nothing else is a reliable marker —
    # STREAMLIT_RUNTIME_ENV is set when running locally too, so testing for it
    # made every local run claim to be a deploy.
    on_cloud = "/mount/src" in str(Path(__file__).resolve())

    status = storage.get_store(secrets, LOCAL_PROFILE_DIR, local_is_durable=not on_cloud)

    # Never fail silently: if credentials were expected but not found, say what
    # was actually seen and where the file was looked for.
    if status.error is None and not secrets.get("gcp_service_account"):
        status.error = (
            (f"Could not read secrets — {secrets_error}." if secrets_error else
             f"No [gcp_service_account] section found. Sections seen: "
             f"{sorted(secrets) if secrets else 'none'}.")
            + f"\n\nStreamlit looks for secrets.toml relative to the folder you ran it from."
            + f"\nThat folder is currently: {Path.cwd()}"
            + f"\nSo it expects: {Path.cwd() / '.streamlit' / 'secrets.toml'}"
        )
    return status


@st.cache_data(show_spinner=False)
def _load_profile(_store, name: str, version: int):
    """`version` is bumped on every write, which is what invalidates the cache.
    `_store` is underscore-prefixed so Streamlit does not try to hash it."""
    if name == storage.DEMO_PROFILE:
        demo = storage.Profile(name=name, goal_race="Marathon", goal_seconds=3 * 3600,
                               race_date=date(2027, 5, 16), days_per_week=4)
        return demo, _load_demo()
    return _store.load(name)


def _bump() -> None:
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1


st.session_state.setdefault("store_generation", 0)
store_status = _get_store(st.session_state["store_generation"])
store = store_status.store
st.session_state.setdefault("data_version", 0)

st.sidebar.title("🏃 Strava Coach")

known_profiles = store.list_profiles()
options = [storage.DEMO_PROFILE] + known_profiles + [NEW_PROFILE]
st.session_state.setdefault("active_profile", storage.DEMO_PROFILE)
if st.session_state["active_profile"] not in options:
    st.session_state["active_profile"] = storage.DEMO_PROFILE

chosen = st.sidebar.selectbox("Profile", options,
                              index=options.index(st.session_state["active_profile"]))
if chosen != st.session_state["active_profile"]:
    st.session_state["active_profile"] = chosen
    st.session_state.pop("added_runs", None)
    _rerun_now()

# -- storage status --------------------------------------------------------
if store_status.durable and store_status.label == "Google Sheets":
    st.sidebar.caption(f"💾 {store_status.label} — saved automatically.")
elif store_status.durable:
    st.sidebar.caption(f"💾 {store_status.label} — saved on this machine.")
else:
    st.sidebar.caption(f"⚠️ {store_status.label} — **not** durable here.")

if store_status.error:
    with st.sidebar.expander("⚠️ Why storage isn't connected", expanded=True):
        st.caption(store_status.detail)
        st.code(store_status.error, language=None)
        st.caption("If you added credentials since this app started, reconnect rather "
                   "than restarting — the connection is cached per session.")
        if st.button("🔄 Reconnect storage", key="reconnect_storage"):
            _get_store.clear()
            st.session_state["store_generation"] += 1
            _rerun_now()

# -- creating a profile ----------------------------------------------------
if chosen == NEW_PROFILE:
    st.title("New profile")
    st.markdown(
        "A profile holds one athlete's run history, goal race and target time. "
        "You and anyone else using this app can see every profile — they are named, "
        "not private."
    )
    new_name = st.text_input("Profile name", placeholder="e.g. Szymon, or Marta — half marathon")
    clean_name = new_name.strip()
    clash = clean_name in known_profiles or clean_name == storage.DEMO_PROFILE

    if clash:
        st.error(f"There is already a profile called “{clean_name}”.")
    if not store_status.durable:
        st.warning(
            "Storage is not durable on this deploy, so a profile created now will not "
            "survive a restart. Set up Google Sheets first — see the README."
        )

    if st.button("Create profile", type="primary", disabled=not clean_name or clash):
        store.save(storage.Profile(name=clean_name), pd.DataFrame())
        _bump()
        st.session_state["active_profile"] = clean_name
        _rerun_now()
    st.stop()

profile_name = chosen
is_demo = profile_name == storage.DEMO_PROFILE
profile, runs = _load_profile(store, profile_name, st.session_state["data_version"])

# The demo profile is read-only, so additions to it are held in the session.
st.session_state.setdefault("added_runs", [])
if is_demo and st.session_state["added_runs"]:
    runs = append_runs(runs, st.session_state["added_runs"])


def save_runs(updated: pd.DataFrame) -> None:
    """Persist a changed run history for the active profile."""
    if is_demo:
        return                       # handled by session state instead
    store.save(profile, updated)
    _bump()


# -- importing a history ---------------------------------------------------
if not is_demo:
    with st.sidebar.expander("Import a Strava export", expanded=runs.empty):
        st.caption(
            "One-time bootstrap. On strava.com: **Settings → My Account → Download or "
            "Delete Your Account → Request your archive**. Upload the `activities.csv` "
            "from inside the zip."
        )
        uploaded = st.file_uploader("activities.csv", type=["csv"], key="history_upload")
        assume_miles = st.checkbox("That account is set to miles", value=False, key="import_miles")
        replace = st.radio("If this profile already has runs", ["Merge", "Replace"],
                           horizontal=True, key="import_mode")
        if uploaded is not None and st.button("Import into this profile", key="do_import"):
            try:
                imported = _load_upload(uploaded.getvalue(), assume_miles)
                merged = imported if replace == "Replace" else append_runs(runs, [
                    {"date": row["date"], "name": row["name"],
                     "distance_km": row["distance_km"], "moving_seconds": row["moving_seconds"],
                     "avg_hr": row.get("avg_hr"), "elevation_m": row.get("elevation_m")}
                    for _, row in imported.iterrows()
                ])
                store.save(profile, merged)
                _bump()
                st.success(f"Imported {len(imported)} runs.")
                _rerun_now()
            except Exception as error:  # noqa: BLE001
                st.error(str(error))

if runs is None or runs.empty:
    st.title(profile_name)
    st.markdown(
        "This profile has no runs yet.\n\n"
        "**Import a Strava export** from the sidebar to bootstrap the history in one go, "
        "or add runs one at a time once there is something to analyse. A handful of weeks "
        "is enough for the coach to say anything useful."
    )
    st.stop()

st.sidebar.divider()
if is_demo:
    st.sidebar.info("The demo athlete is read-only — changes are not saved. "
                    "Create a profile to keep your own history.")

# -- current fitness -------------------------------------------------------
st.sidebar.subheader("Current fitness")
st.sidebar.caption(
    "Everything keys off one recent hard effort. A race is ideal; a solid tempo or "
    "time trial works. Do not use an interval session — its average pace includes "
    "the jog recoveries and understates you."
)

effort_weeks = st.sidebar.slider(
    "Only consider efforts from the last … weeks", 2, 26, 8,
    help="Fitness estimated from an old effort propagates into every pace and "
         "prediction in this app. Keep this short unless nothing recent qualifies.",
)
candidates = best_efforts(runs, within_days=effort_weeks * 7, as_of=runs["date"].max())
widened = False
if candidates.empty:
    candidates = best_efforts(runs, within_days=None)
    widened = not candidates.empty
    if widened:
        st.sidebar.warning(
            f"No run of 3 km or more in the last {effort_weeks} weeks, so the list below "
            "reaches further back. Treat the resulting fitness estimate as optimistic."
        )

saved_label = None
if profile.effort_km and profile.effort_seconds:
    age = ""
    if profile.effort_date:
        days = (today.date() - profile.effort_date).days
        age = f", {days} d ago" if days < 400 else ""
    saved_label = (f"Saved: {profile.effort_km:.1f} km in "
                   f"{format_duration(profile.effort_seconds)}{age}")

effort_options = ([saved_label] if saved_label else []) + ["Enter it manually"] + [
    f"{row.date:%d %b} · {row['name'][:26]} · {row.distance_km:.1f} km · {format_pace(row.pace_sec_per_km)}"
    for _, row in candidates.iterrows()
]
default_index = 0 if saved_label else (1 if len(effort_options) > 1 else 0)
picked = st.sidebar.selectbox("Best recent effort", effort_options, index=default_index)

effort_date = None
if saved_label and picked == saved_label:
    effort_km, effort_seconds = float(profile.effort_km), float(profile.effort_seconds)
    effort_date = profile.effort_date
    if effort_date and (today.date() - effort_date).days > effort_weeks * 7:
        st.sidebar.warning(
            f"That saved effort is {(today.date() - effort_date).days} days old — beyond your "
            f"{effort_weeks}-week window. Every pace and prediction below is based on it, so "
            "pick a more recent run if you have one."
        )
elif picked == "Enter it manually":
    col_a, col_b = st.sidebar.columns(2)
    effort_km = col_a.number_input("Distance (km)", min_value=1.0, max_value=50.0,
                                   value=float(profile.effort_km or 10.0), step=0.5)
    effort_time = col_b.text_input(
        "Time", value=format_duration(profile.effort_seconds) if profile.effort_seconds else "45:00",
        help="45:00, or 1:35:00")
    effort_seconds = parse_duration(effort_time)
    effort_date = today.date()
else:
    offset = 1 if saved_label else 0
    row = candidates.iloc[effort_options.index(picked) - 1 - offset]
    effort_km, effort_seconds = float(row.distance_km), float(row.moving_seconds)
    effort_date = pd.Timestamp(row.date).date()

if not math.isfinite(effort_seconds) or effort_seconds <= 0:
    st.sidebar.error("That time could not be read — try 45:00 or 1:35:00.")
    st.stop()

vdot = vdot_from_race(effort_km * 1000, effort_seconds)
zones = pace_zones(vdot)
easy_quick, easy_slow = easy_pace_range(vdot)
st.sidebar.metric("Estimated VDOT", f"{vdot:.1f}")

# -- goal ------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("Goal")
race_names = list(COMMON_RACES)
race_label = st.sidebar.selectbox(
    "Race", race_names,
    index=race_names.index(profile.goal_race) if profile.goal_race in race_names
    else race_names.index("Marathon"))
goal_time = st.sidebar.text_input("Target time", value=format_duration(profile.goal_seconds))
race_date = st.sidebar.date_input("Race date",
                                  value=profile.race_date or date(2027, 5, 16))
days_per_week = st.sidebar.slider("Running days per week", 3, 7, int(profile.days_per_week))

goal_seconds = parse_duration(goal_time)
if not math.isfinite(goal_seconds) or goal_seconds <= 0:
    st.sidebar.error("That target time could not be read — try 3:00:00.")
    st.stop()

goal = Goal(distance_m=COMMON_RACES[race_label], goal_seconds=goal_seconds,
            race_date=pd.Timestamp(race_date), days_per_week=days_per_week, label=race_label)

# -- saving settings -------------------------------------------------------
if not is_demo:
    unsaved = (profile.goal_race != race_label
               or abs(profile.goal_seconds - goal_seconds) > 0.5
               or profile.race_date != race_date
               or profile.days_per_week != days_per_week
               or (profile.effort_km or 0) != round(effort_km, 3)
               or (profile.effort_seconds or 0) != round(effort_seconds)
               or profile.effort_date != effort_date)

    if st.sidebar.button("💾 Save settings to profile", type="primary" if unsaved else "secondary",
                         disabled=not unsaved):
        profile.goal_race = race_label
        profile.goal_seconds = goal_seconds
        profile.race_date = race_date
        profile.days_per_week = days_per_week
        profile.effort_km = round(effort_km, 3)
        profile.effort_seconds = round(effort_seconds)
        profile.effort_date = effort_date
        store.save(profile, runs)
        _bump()
        _rerun_now()
    st.sidebar.caption("Runs are saved the moment you add them. Goal and fitness "
                       "settings are saved with this button.")

    with st.sidebar.expander("Delete this profile"):
        st.caption(f"Permanently removes “{profile_name}” and all of its runs.")
        if st.text_input("Type the profile name to confirm", key="delete_confirm") == profile_name:
            if st.button("Delete permanently"):
                store.delete(profile_name)
                _bump()
                st.session_state["active_profile"] = storage.DEMO_PROFILE
                _rerun_now()

today = pd.Timestamp(pd.Timestamp.today().date())
as_of = max(today, runs["date"].max())

# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

runs = add_derived(runs, zones, easy_slow)
weekly = weekly_summary(runs)
context = recent_context(runs, as_of=as_of)
load = acwr(runs, as_of=as_of)
weeks_out = (goal.race_date - today).days / 7.0
phase = phase_for(weeks_out)
week_index = max(0, (today - runs["date"].min()).days // 7)
target_km, target_reason = weekly_volume_target(context, phase, goal, week_index=week_index)
prescription = next_session(context, load, zones, easy_slow, phase, goal, target_km)
assessment = assess_goal(vdot, goal, context, today=today)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title(f"{profile_name} — what to run next")
st.caption(
    f"{len(runs)} runs from {runs['date'].min():%d %b %Y} to {runs['date'].max():%d %b %Y}  ·  "
    f"{race_label} on {goal.race_date:%d %b %Y} ({weeks_out:.0f} weeks out)  ·  "
    f"{phase.name} phase"
)

kpi = st.columns(5)
kpi[0].metric("Last 7 days", f"{context.last_7_km:.0f} km", f"{context.last_7_sessions} runs")
kpi[1].metric("4-week average", f"{context.last_28_km / 4:.0f} km/wk")
kpi[2].metric("Load ratio", f"{load.ratio:.2f}×" if math.isfinite(load.ratio) else "—")
kpi[3].metric("Predicted " + race_label.lower(), format_duration(assessment.predicted_seconds),
              format_duration(abs(assessment.time_gap_seconds)) +
              (" slower than goal" if assessment.time_gap_seconds > 0 else " inside goal"),
              delta_color="inverse")
kpi[4].metric("Easy share (28 d)",
              f"{context.easy_share_28 * 100:.0f}%" if math.isfinite(context.easy_share_28) else "—",
              f"{(context.easy_share_28 - 0.80) * 100:+.0f} pp vs 80% target"
              if math.isfinite(context.easy_share_28) else None)

next_tab, add_tab, fitness_tab, goal_tab, history_tab, method_tab = st.tabs(
    ["Next session", "Add a run", "Fitness & paces", "Goal check",
     "Training history", "How it decides"]
)


# --------------------------------------------------------------------------
# Next session
# --------------------------------------------------------------------------

with next_tab:
    icon = STATUS_ICON.get(prescription.status, "•")
    st.subheader(f"{icon} {prescription.kind} — {prescription.distance_km:.0f} km")

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Session**  \n{prescription.structure}")
        st.markdown(f"**Target pace**  \n{prescription.target_pace}")
        st.markdown(f"**Why this, now**  \n{prescription.rationale}")
        for alternative in prescription.alternatives:
            st.caption(f"↔ {alternative}")

    with right:
        st.markdown("**What triggered it**")
        trigger = pd.DataFrame({
            "Signal": ["Days since last run", "Days since quality", "Days since long run",
                       "Last 7 days", "Load ratio", "Phase"],
            "Value": [f"{context.days_since_run:.1f}",
                      f"{context.days_since_quality:.1f}",
                      f"{context.days_since_long:.1f}",
                      f"{context.last_7_km:.0f} km",
                      f"{load.ratio:.2f}×" if math.isfinite(load.ratio) else "—",
                      phase.name],
        })
        show_table(trigger, hide_index=True)

    st.divider()
    st.markdown(f"**This week: aim for about {target_km:.0f} km**" if math.isfinite(target_km)
                else "**Weekly target unavailable**")
    st.caption(target_reason)
    st.caption(f"**{phase.name} phase** — {phase.description}")

    show_chart(charts.weekly_volume_chart(weekly, runs, target_km=target_km))

    load_icon = STATUS_ICON.get(load.status, "•")
    st.markdown(f"{load_icon} **Training load** — {load.verdict}")
    show_chart(charts.acwr_chart(runs))


# --------------------------------------------------------------------------
# Add a run from a screenshot
# --------------------------------------------------------------------------

def _export_history(frame: pd.DataFrame) -> bytes:
    """Write the merged history in a shape this app can read back.

    Distance goes out in metres so there is no unit ambiguity on re-upload —
    the loader identifies the metres column by magnitude.
    """
    export = pd.DataFrame({
        "Activity Date": frame["date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Activity Name": frame["name"],
        "Activity Type": frame.get("type", "Run"),
        "Moving Time": frame["moving_seconds"].round(0),
        "Distance": (frame["distance_km"] * 1000).round(1),
        "Elevation Gain": frame.get("elevation_m"),
        "Average Heart Rate": frame.get("avg_hr"),
    })
    return export.to_csv(index=False).encode("utf-8")


def _run_form(parsed: "ocr.ParsedRun | None", fingerprint: str | None) -> None:
    """The confirm-and-edit form.

    Shown whether or not OCR ran. With a screenshot it arrives pre-filled;
    without one it is four fields to type. Either way nothing is saved until
    the button is pressed — the screenshot is a convenience, never the
    authority.
    """
    left, right = st.columns(2)
    run_date = left.date_input("Date", value=(parsed.run_date if parsed else None) or today.date(),
                               key="add_date")
    run_name = right.text_input("Name", value=(parsed.name if parsed else None) or "Run",
                                key="add_name")

    left, right = st.columns(2)
    distance_km = left.number_input(
        "Distance (km)", min_value=0.1, max_value=300.0, step=0.01, format="%.2f",
        value=float(parsed.distance_km) if (parsed and parsed.distance_km) else 10.0,
        key="add_distance",
    )
    default_time = (format_duration(parsed.moving_seconds)
                    if (parsed and parsed.moving_seconds) else "50:00")
    time_text = right.text_input("Moving time", value=default_time,
                                 help="1:06:52 or 50:00", key="add_time")

    left, right = st.columns(2)
    avg_hr = left.number_input("Average HR (0 if unknown)", min_value=0, max_value=230,
                               value=int((parsed.avg_hr if parsed else 0) or 0), key="add_hr")
    elevation = right.number_input("Elevation gain, m (0 if unknown)", min_value=0, max_value=9000,
                                   value=int((parsed.elevation_m if parsed else 0) or 0),
                                   key="add_elev")

    seconds = parse_duration(time_text)
    valid = math.isfinite(seconds) and seconds > 0 and distance_km > 0

    if not valid:
        st.error("That time could not be read — try 1:06:52 or 50:00.")
    else:
        pace = seconds / distance_km
        if 120 <= pace <= 900:
            band = ("easy" if pace >= zones["marathon"] * 1.03
                    else "hard" if pace <= zones["threshold"] * 1.02 else "moderate")
            st.markdown(f"**Pace: {format_pace(pace)}** — this counts as a **{band}** run "
                        "at your current fitness")
        else:
            st.markdown(f"**Pace: {format_pace(pace)}**")
            st.error("That distance and time give an implausible pace. One of the two is wrong.")
            valid = False

    if parsed is not None:
        with st.expander("What the OCR actually saw"):
            st.code(parsed.raw_text or "(nothing)", language=None)

    if st.button("✅ Add this run", type="primary", disabled=not valid, key="add_button"):
        entry = {
            "date": pd.Timestamp(run_date),
            "name": run_name or "Run",
            "distance_km": float(distance_km),
            "moving_seconds": float(seconds),
            "avg_hr": float(avg_hr) if avg_hr else None,
            "elevation_m": float(elevation) if elevation else None,
        }
        if is_demo:
            # The demo profile is read-only, so its additions stay in the session.
            st.session_state["added_runs"].append(entry)
        else:
            save_runs(append_runs(runs, [entry]))
        if fingerprint:
            st.session_state["processed_shots"].append(fingerprint)
        _rerun_now()


with add_tab:
    st.subheader("Add a run")
    st.caption(
        "For the daily loop: finish a run, screenshot the Strava summary, drop it here. "
        "Nothing is saved until you have checked what was read — OCR on a phone "
        "screenshot is a first draft, not an answer. You can also just type the run in."
    )

    st.session_state.setdefault("processed_shots", [])
    ocr_available, ocr_status = ocr.tesseract_status()

    if not ocr_available:
        st.info(
            "**Screenshot reading is switched off on this machine** — the Tesseract binary "
            "is not installed. Typing a run in below works exactly the same; it is four "
            "fields and about ten seconds."
        )
        with st.expander("Why, and how to turn it on"):
            st.markdown(
                f"Diagnostic: `{ocr_status}`\n\n"
                "**Deployed on Streamlit Community Cloud** — a file named `packages.txt` "
                "must sit in the **repository root** (not a subfolder) containing the single "
                "line `tesseract-ocr`. If it is already there, open **Manage app → Reboot app**: "
                "system packages are installed at build time, so an app built before the file "
                "existed keeps the old environment until it rebuilds.\n\n"
                "**Running locally** — `brew install tesseract` on macOS, "
                "`sudo apt install tesseract-ocr` on Linux, then restart the app.\n\n"
                "Everything else in this app works without it."
            )

    parsed, fingerprint = None, None

    if ocr_available:
        shot = st.file_uploader("Strava screenshot", type=["png", "jpg", "jpeg", "webp"],
                                key="screenshot_upload")
        shot_miles = st.checkbox("This screenshot is in miles", value=False, key="shot_miles")

        if shot is not None:
            payload = shot.getvalue()
            fingerprint = f"{hash(payload)}-{shot_miles}"

            if fingerprint in st.session_state["processed_shots"]:
                st.success("This screenshot has already been added. Upload another, or clear it "
                           "with the ✕ on the file above.")
                fingerprint = None
            else:
                try:
                    with st.spinner("Reading the screenshot…"):
                        parsed = _read_screenshot(payload, today.date().isoformat(), shot_miles)
                except Exception as error:  # noqa: BLE001 — never lose the manual path
                    st.warning(f"{error}\n\nFill the run in by hand below instead.")

                if parsed is not None:
                    image_column, detail_column = st.columns([1, 2])
                    image_column.image(payload, width=260)
                    with detail_column:
                        badge = {"high": "✅ Read cleanly", "medium": "⚠️ Partly read",
                                 "low": "🛑 Needs checking"}[parsed.overall_confidence]
                        st.markdown(f"**{badge}** — correct anything below before adding it.")
                        for warning in parsed.warnings:
                            st.warning(warning)

    if fingerprint is not None or parsed is not None or not ocr_available:
        _run_form(parsed, fingerprint)
    else:
        with st.expander("…or type the run in by hand"):
            _run_form(None, None)

    st.divider()

    st.markdown("**Most recent runs in this profile**")
    recent = runs.sort_values("date", ascending=False).head(8)
    for position, (_, row) in enumerate(recent.iterrows()):
        columns = st.columns([3, 2, 2, 2, 1])
        columns[0].write(str(row["name"])[:34])
        columns[1].write(pd.Timestamp(row["date"]).strftime("%a %d %b"))
        columns[2].write(f"{row['distance_km']:.2f} km")
        columns[3].write(format_pace(row["pace_sec_per_km"]))
        if not is_demo and columns[4].button("✕", key=f"remove_{position}",
                                             help="Delete this run from the profile"):
            keep = runs.drop(index=row.name)
            save_runs(keep)
            st.session_state["processed_shots"] = []
            _rerun_now()

    if is_demo:
        st.caption("The demo profile is read-only — runs added here disappear when you "
                   "close the tab. Create your own profile to keep them.")
    elif store_status.durable:
        st.success(f"Runs are saved to **{store_status.label}** the moment you add them, "
                   "and the recommendation on the Next session tab is already updated.")
    else:
        st.warning(
            f"**{store_status.label}** — this deploy wipes its disk on restart, so these runs "
            "will not survive. Set up Google Sheets (see the README) or keep a downloaded "
            "copy below."
        )

    st.divider()
    st.download_button(
        "⬇️ Download this profile's history (CSV)",
        _export_history(runs),
        file_name=f"{profile_name.replace(' ', '_')}_{today:%Y-%m-%d}.csv",
        mime="text/csv",
        key="download_history",
        help="A backup you can re-import, or take to another tool.",
    )


# --------------------------------------------------------------------------
# Fitness and paces
# --------------------------------------------------------------------------

with fitness_tab:
    left, right = st.columns([2, 3])

    with left:
        st.subheader("Your training paces")
        when = ""
        if effort_date:
            days = (today.date() - effort_date).days
            when = (" run today" if days <= 0 else
                    " run yesterday" if days == 1 else f" run {days} days ago")
        st.caption(f"Derived from VDOT {vdot:.1f}, estimated from "
                   f"{effort_km:.1f} km in {format_duration(effort_seconds)}{when}. "
                   "Everything on this page moves with that one effort, so an old or "
                   "unrepresentative one skews all of it.")
        rows = []
        for zone in ZONES:
            pace = zones[zone.key]
            display = (f"{format_pace(easy_quick)} – {format_pace(easy_slow)}"
                       if zone.key == "easy" else format_pace(pace))
            rows.append({"Zone": zone.name, "Pace": display, "What it does": zone.purpose})
        show_table(pd.DataFrame(rows), hide_index=True)

        with st.expander("How to run each one"):
            for zone in ZONES:
                st.markdown(f"**{zone.name}** — {zone.guidance}")

    with right:
        st.subheader("What this fitness projects")
        predictions = pd.DataFrame([
            {"Race": name, "Predicted": format_duration(predict_race_seconds(vdot, distance))}
            for name, distance in COMMON_RACES.items()
        ])
        show_table(predictions, hide_index=True)
        st.caption(
            "These are equivalent performances at your current fitness, assuming you are "
            "trained for the distance. The marathon prediction in particular assumes "
            "marathon-specific preparation — long runs, fuelling practice, the volume to "
            "back it up. Without that, the real result is slower than the table says."
        )

    st.divider()
    show_chart(charts.pace_scatter(runs, zones))
    st.caption(
        "Dashed lines are your threshold, marathon and quick-easy paces. Easy runs sitting "
        "above the easy line — that is, faster than it — are the most common training error "
        "there is: too hard to recover from, too slow to be a workout."
    )


# --------------------------------------------------------------------------
# Goal check
# --------------------------------------------------------------------------

with goal_tab:
    icon = STATUS_ICON.get(assessment.status, "•")
    st.subheader(f"{icon} {format_duration(goal.goal_seconds)} {race_label.lower()}, "
                 f"{assessment.weeks_available:.0f} weeks away")
    st.markdown(f"### {assessment.verdict}")

    columns = st.columns(4)
    columns[0].metric("Current fitness", f"VDOT {assessment.current_vdot:.1f}",
                      format_duration(assessment.predicted_seconds))
    columns[1].metric("Goal requires", f"VDOT {assessment.required_vdot:.1f}",
                      f"{assessment.vdot_gap:+.1f} to find", delta_color="off")
    columns[2].metric("Typical projection",
                      f"VDOT {assessment.reachable_vdot:.1f}" if math.isfinite(assessment.reachable_vdot) else "—",
                      format_duration(assessment.reachable_seconds))
    columns[3].metric("Volume now vs typical",
                      f"{assessment.current_weekly_km:.0f} km/wk",
                      f"typical {assessment.typical_peak_km[0]:.0f}–{assessment.typical_peak_km[1]:.0f}",
                      delta_color="off")

    for note in assessment.notes:
        st.warning(note)

    show_chart(charts.projection_chart(
        assessment.current_vdot, assessment.required_vdot,
        assessment.weeks_available,
        (assessment.plausible_vdot_gain / assessment.weeks_available)
        if assessment.weeks_available else 0.0,
    ))

    st.caption(
        "The projection uses a typical improvement rate for your current level — roughly "
        "0.25 VDOT points a week below 45, slowing as fitness rises. It is a planning "
        "heuristic, not a promise: it assumes consistent training and no interruptions, and "
        "individual response varies far more than any single rate captures. Use it to judge "
        "whether a timeline is sane, not to predict a finishing time."
    )


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------

with history_tab:
    st.subheader("By week")
    if not weekly.empty:
        display = weekly.copy()
        display["Week of"] = display["week"].dt.strftime("%d %b %Y")
        display["Easy share"] = (display["easy_share"] * 100).round(0).astype("Int64").astype(str) + "%"
        display["Avg pace"] = display["avg_pace"].map(format_pace)
        show_table(
            display[["Week of", "km", "sessions", "longest_km", "Easy share", "Avg pace", "load"]]
            .rename(columns={"km": "km", "sessions": "runs", "longest_km": "longest",
                             "load": "load (wtd km)"})
            .round(1),
            hide_index=True,
        )

    st.subheader("Every run")
    detail = runs.copy()
    detail["Date"] = detail["date"].dt.strftime("%a %d %b %Y")
    detail["Pace"] = detail["pace_sec_per_km"].map(format_pace)
    detail["Time"] = detail["moving_seconds"].map(format_duration)
    detail["Intensity"] = detail["intensity"].str.capitalize()
    show_table(
        detail[["Date", "name", "distance_km", "Time", "Pace", "Intensity", "avg_hr"]]
        .rename(columns={"name": "Run", "distance_km": "km", "avg_hr": "avg HR"})
        .sort_values("Date", ascending=False, key=lambda s: pd.to_datetime(s, format="%a %d %b %Y"))
        .round(2),
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download the analysed runs (CSV)",
        runs.to_csv(index=False).encode("utf-8"),
        file_name="analysed_runs.csv", mime="text/csv",
    )


# --------------------------------------------------------------------------
# Method
# --------------------------------------------------------------------------

with method_tab:
    st.subheader("How the coach decides")
    st.markdown(
        """
**1. Fitness → VDOT.** One recent hard effort goes through the Daniels–Gilbert
equations to produce a VDOT, an "effective VO2max" inferred from performance.
Every training pace and race prediction comes from that single number.

**2. Load → acute:chronic ratio.** Each run scores intensity-weighted kilometres:
`distance × (easy pace ÷ session pace)²`. The last 7 days are compared against the
last 28 divided by four. Around 1.0 means this week looks like the recent norm.

**3. Phase → what kind of session.** Weeks to race set the phase: base above 18
weeks, build 18–8, peak 8–3, taper inside 3. The phase decides whether quality
work means threshold, intervals or marathon-pace segments.

**4. The decision tree, in priority order.**

| Check | If true |
|---|---|
| Quality session within 24 h | Rest or easy shakeout |
| Load ratio above 1.5 | Easy only, until it settles |
| No run for 7+ days | Easy return run |
| No long run for 6+ days | Long run |
| 3+ days since quality, load ≤ 1.3 | Quality session for the phase |
| Otherwise | Easy run, sized to the weekly target |

Every threshold in that table is a named constant at the top of `plan.py`.
They are conventions, not constants of nature — change them if your coach,
your body or your evidence says otherwise.
        """
    )

    st.divider()
    st.subheader("What this does not know")
    st.markdown(
        """
- **How you feel.** Sleep, stress, soreness and illness matter more than any of
  this, and none of them are in a Strava export. A session that the tool says you
  are ready for is still the wrong session if you are run down.
- **Your splits.** The bulk export has one average pace per activity. An interval
  session shows up at its average, which understates it — the classifier uses
  activity names to compensate, but it is a patch, not a fix.
- **Whether it read your screenshot correctly.** It checks that distance, time and
  pace agree with each other and flags them when they do not, but a consistent
  misread is still a misread. That is why the form asks you to confirm rather
  than saving straight away.
- **Whether the ACWR is real.** The acute:chronic ratio is widely used and widely
  criticised; the original injury-risk findings have not replicated cleanly. It is
  here as a spike detector, which is the part that holds up best.
- **You specifically.** Every number here is a population average. The single most
  useful thing you can do is keep your own records and find out where you differ.

Not medical advice. If something hurts in a way that is sharp, one-sided or getting
worse, no algorithm is the right person to ask.
        """
    )
