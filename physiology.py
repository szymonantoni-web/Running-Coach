"""Running physiology: VDOT, training pace zones, race-time prediction.

The model is Jack Daniels and Jimmy Gilbert's, from *Oxygen Power* (1979) and
used throughout *Daniels' Running Formula*. Two equations do the work:

    %VO2max(T) = 0.8 + 0.1894393·e^(-0.012778·T) + 0.2989558·e^(-0.1932605·T)
    VO2(v)     = -4.60 + 0.182258·v + 0.000104·v²

with T in minutes and v in metres per minute. VDOT is the VO2 a race demands
divided by the fraction of VO2max you can hold for that long — an "effective
VO2max" inferred from performance rather than measured in a lab.

Everything here is pure Python. No pandas, no framework: the numbers can be
checked against Daniels' published tables, and tests/test_coach.py does exactly
that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Daniels-Gilbert coefficients
_A, _B, _C = -4.60, 0.182258, 0.000104
_P0, _P1, _K1, _P2, _K2 = 0.8, 0.1894393, -0.012778, 0.2989558, -0.1932605

MARATHON_M = 42195.0
HALF_M = 21097.5

COMMON_RACES = {
    "1500 m": 1500.0,
    "1 mile": 1609.34,
    "3 km": 3000.0,
    "5 km": 5000.0,
    "10 km": 10000.0,
    "15 km": 15000.0,
    "Half marathon": HALF_M,
    "Marathon": MARATHON_M,
}


def percent_vo2max(minutes: float) -> float:
    """Fraction of VO2max sustainable for a race lasting `minutes`."""
    if minutes <= 0:
        return float("nan")
    return _P0 + _P1 * math.exp(_K1 * minutes) + _P2 * math.exp(_K2 * minutes)


def vo2_at_velocity(metres_per_min: float) -> float:
    """Oxygen cost of running at a given velocity, ml/kg/min."""
    return _A + _B * metres_per_min + _C * metres_per_min ** 2


def velocity_at_vo2(vo2: float) -> float:
    """Inverse of vo2_at_velocity — the positive root of the quadratic."""
    discriminant = _B ** 2 - 4 * _C * (_A - vo2)
    if discriminant < 0:
        return float("nan")
    return (-_B + math.sqrt(discriminant)) / (2 * _C)


def vdot_from_race(distance_m: float, seconds: float) -> float:
    """VDOT implied by a race performance.

    >>> round(vdot_from_race(5000, 20 * 60))      # a 20:00 5 km
    50
    """
    if distance_m <= 0 or seconds <= 0:
        return float("nan")
    minutes = seconds / 60.0
    velocity = distance_m / minutes
    return vo2_at_velocity(velocity) / percent_vo2max(minutes)


def predict_race_seconds(vdot: float, distance_m: float) -> float:
    """Time this VDOT should produce over a given distance.

    Both sides of the identity depend on time, so it is solved numerically by
    bisection rather than in closed form. The bracket spans 1 minute to 10
    hours, which covers every distance anyone would put in here.
    """
    if not math.isfinite(vdot) or vdot <= 0 or distance_m <= 0:
        return float("nan")

    def implied_vdot(minutes: float) -> float:
        return vo2_at_velocity(distance_m / minutes) / percent_vo2max(minutes)

    low, high = 1.0, 600.0
    if implied_vdot(low) < vdot or implied_vdot(high) > vdot:
        return float("nan")          # outside the solvable range

    for _ in range(200):             # ~1e-58 minutes of residual bracket
        mid = (low + high) / 2
        if implied_vdot(mid) > vdot:
            low = mid
        else:
            high = mid
    return (low + high) / 2 * 60.0


def vdot_for_target(distance_m: float, seconds: float) -> float:
    """The VDOT a goal time would require — the same maths as vdot_from_race,
    named for how it is used."""
    return vdot_from_race(distance_m, seconds)


# --------------------------------------------------------------------------
# Training pace zones
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Zone:
    key: str
    name: str
    intensity: float | None     # fraction of VDOT; None = derived from a race prediction
    purpose: str
    guidance: str


ZONES = [
    Zone("easy", "Easy / recovery", 0.70,
         "Builds aerobic base, capillary density and mitochondria; the pace that makes volume survivable.",
         "Conversational. Most of your week lives here — if it feels too slow, it is probably right."),
    Zone("marathon", "Marathon pace", None,
         "Race-specific rehearsal: teaches the fuelling, rhythm and effort you will hold on the day.",
         "Comfortably hard, sustainable for hours. Derived from your predicted marathon time."),
    Zone("threshold", "Threshold / tempo", 0.88,
         "Raises the effort you can sustain before lactate accumulates — the single biggest lever for a marathon.",
         "Comfortably hard, about the pace you could race for an hour. 20–40 min of work per session."),
    Zone("interval", "Interval (VO2max)", 0.98,
         "Develops maximal aerobic power. Expensive to recover from, so it is used sparingly.",
         "Hard. Intervals of 3–5 min with roughly equal jog recovery; 5 × 1 km sits here."),
    Zone("repetition", "Repetition / speed", 1.06,
         "Running economy and neuromuscular speed, not aerobic fitness.",
         "Fast but relaxed, short reps (200–600 m) with full recovery. Never to exhaustion."),
]

ZONE_BY_KEY = {z.key: z for z in ZONES}


def pace_for_intensity(vdot: float, intensity: float) -> float:
    """Seconds per kilometre at a given fraction of VDOT."""
    velocity = velocity_at_vo2(vdot * intensity)
    if not math.isfinite(velocity) or velocity <= 0:
        return float("nan")
    return 60_000.0 / velocity


def pace_zones(vdot: float) -> dict[str, float]:
    """Seconds per kilometre for each training zone.

    Marathon pace comes from the race prediction rather than a fixed
    percentage — that is how Daniels' own table is built, and it keeps M pace
    consistent with the predicted finishing time shown elsewhere in the app.
    """
    paces: dict[str, float] = {}
    for zone in ZONES:
        if zone.intensity is None:
            predicted = predict_race_seconds(vdot, MARATHON_M)
            paces[zone.key] = predicted / (MARATHON_M / 1000.0) if math.isfinite(predicted) else float("nan")
        else:
            paces[zone.key] = pace_for_intensity(vdot, zone.intensity)
    return paces


# The literature puts easy running anywhere in 59–74% of VDOT, which is a very
# wide band. Daniels' own published E-pace column is narrower in practice;
# 62–70% reproduces it to within a few seconds per kilometre.
EASY_BAND = (0.62, 0.70)


def easy_pace_range(vdot: float) -> tuple[float, float]:
    """(quick end, slow end) of easy running, in seconds per km. The range
    matters: the slow end is for recovery days, the quick end for steady
    aerobic runs."""
    slow, quick = EASY_BAND
    return pace_for_intensity(vdot, quick), pace_for_intensity(vdot, slow)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def format_pace(seconds_per_km: float) -> str:
    """330.4 -> '5:30/km'."""
    if seconds_per_km is None or not math.isfinite(seconds_per_km) or seconds_per_km <= 0:
        return "—"
    minutes, seconds = divmod(int(round(seconds_per_km)), 60)
    return f"{minutes}:{seconds:02d}/km"


def format_duration(seconds: float) -> str:
    """9000 -> '2:30:00'; 1500 -> '25:00'."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def parse_duration(text: str) -> float:
    """Seconds from the ways people actually write a time.

    '3:00:00' and '2h55' are both hours-first; '20:00' is minutes:seconds,
    because that is what a race time means. An 'h' anywhere forces the first
    number to be hours, which is the ambiguity '2h55' would otherwise have.
    NaN if it cannot be read.
    """
    if text is None:
        return float("nan")
    raw = str(text).strip().lower().replace(" ", "")
    if not raw:
        return float("nan")

    has_hours = "h" in raw
    for marker in ("h", "m", "s", "'", '"'):
        raw = raw.replace(marker, ":")
    parts = [p for p in raw.split(":") if p != ""]

    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return float("nan")
    if not numbers or len(numbers) > 3:
        return float("nan")

    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        # '2h55' is 2 h 55 min; '20:00' is 20 min 0 s.
        hours, minutes, seconds = (numbers[0], numbers[1], 0.0) if has_hours \
            else (0.0, numbers[0], numbers[1])
    else:
        hours, minutes, seconds = (numbers[0], 0.0, 0.0) if has_hours else (0.0, numbers[0], 0.0)

    return hours * 3600 + minutes * 60 + seconds
