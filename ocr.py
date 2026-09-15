"""Reading a single run off a Strava screenshot.

Two layers, deliberately separated:

* `image_to_text` runs Tesseract. It touches pixels, needs the `tesseract`
  binary, and is thin on purpose.
* `parse_activity_text` turns that text into a structured run. It is pure
  Python, so the fiddly part — deciding which number is distance and which is
  elevation — is testable without an image in sight.

Nothing here is trusted. Every field comes back with a confidence and the
result carries warnings; the app shows all of it in an editable form and waits
for a human to confirm before the run touches the training history. OCR on a
phone screenshot is a first draft, not an answer.

The strongest check is not OCR quality at all — it is arithmetic. Distance,
time and pace are mutually determined, so any two of them predict the third.
`cross_validate` uses that to catch a misread digit that looks perfectly
plausible on its own.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

# Tesseract page segmentation mode 6: "assume a single uniform block of text".
# Strava's summary panels are grid-like rather than prose, and mode 6 handles
# that far better than the default full-page analysis.
TESSERACT_CONFIG = "--psm 6"

MILES_TO_KM = 1.609344


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------

@dataclass
class ParsedRun:
    """What the screenshot appears to say. Every field is optional, because a
    screenshot may be cropped, dark-mode, or simply unreadable."""

    distance_km: float | None = None
    moving_seconds: float | None = None
    pace_sec_per_km: float | None = None
    run_date: date | None = None
    name: str | None = None
    avg_hr: float | None = None
    elevation_m: float | None = None

    confidence: dict[str, str] = field(default_factory=dict)   # field -> high|medium|low
    warnings: list[str] = field(default_factory=list)
    raw_text: str = ""

    @property
    def is_usable(self) -> bool:
        """Enough to compute a pace — which is the minimum the coach needs."""
        known = [v is not None for v in (self.distance_km, self.moving_seconds, self.pace_sec_per_km)]
        return sum(known) >= 2

    @property
    def overall_confidence(self) -> str:
        if not self.confidence:
            return "low"
        levels = list(self.confidence.values())
        if any(level == "low" for level in levels) or self.warnings:
            return "low"
        return "high" if all(level == "high" for level in levels) else "medium"


# --------------------------------------------------------------------------
# Tesseract
# --------------------------------------------------------------------------

def preprocess(image):
    """Make a phone screenshot easier for Tesseract to read.

    Three things matter, in order: size (Tesseract wants text around 30 px
    tall, phone screenshots are smaller), contrast, and polarity — Strava's
    dark mode is light text on dark, and Tesseract expects the opposite.
    """
    from PIL import Image, ImageOps

    image = image.convert("L")                       # greyscale

    # Invert if the image is mostly dark, i.e. dark mode.
    histogram = image.histogram()
    dark_pixels = sum(histogram[:110])
    light_pixels = sum(histogram[145:])
    if dark_pixels > light_pixels:
        image = ImageOps.invert(image)

    # Upscale small screenshots; Tesseract's accuracy falls off a cliff below
    # about 30 px of text height.
    if image.width < 1000:
        scale = max(2, math.ceil(1000 / image.width))
        image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)

    return ImageOps.autocontrast(image)


def tesseract_status() -> tuple[bool, str]:
    """Is screenshot reading available, and if not, why not?

    Worth reporting precisely: "not installed" and "installed but not on PATH"
    have different fixes, and on a hosted deploy the user cannot simply look.
    """
    import shutil

    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract is not installed — add it to requirements.txt."

    binary = shutil.which("tesseract") or getattr(
        getattr(pytesseract, "pytesseract", None), "tesseract_cmd", None)
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version} at {binary or 'the configured path'}"
    except Exception as error:  # noqa: BLE001
        return False, (f"pytesseract is installed but the tesseract binary was not found. "
                       f"({error})")


def _missing_tesseract(error: Exception) -> RuntimeError:
    return RuntimeError(
        "Tesseract is not available. On Streamlit Community Cloud, add a file called "
        "packages.txt to the repository root containing the single line `tesseract-ocr`, "
        "then redeploy. Locally: `brew install tesseract` on macOS, "
        f"`apt install tesseract-ocr` on Linux. ({error})"
    )


def image_to_text(image_bytes: bytes) -> str:
    """OCR a screenshot into flat text."""
    import io

    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment problem
        raise RuntimeError(
            "Screenshot reading needs pytesseract and Pillow. "
            "Add them to requirements.txt and redeploy."
        ) from error

    image = Image.open(io.BytesIO(image_bytes))
    try:
        return pytesseract.image_to_string(preprocess(image), config=TESSERACT_CONFIG)
    except Exception as error:  # pragma: no cover - environment problem
        raise _missing_tesseract(error)


# --------------------------------------------------------------------------
# Column-aware reading
#
# Strava's phone layout is a grid: a big value with a small label underneath.
# Run as flat text, neighbouring cells merge — "1:06:52" and "5:23" come back
# as "1:06:53:23", which is unrecoverable by regex. The fix is to find the
# label row, use the label positions as column boundaries, and re-OCR each
# cell on its own. Cropping removes the ambiguity rather than untangling it.
# --------------------------------------------------------------------------

CELL_LABELS = {
    "distance": "distance",
    "pace": "pace",
    "avgpace": "pace",
    "averagepace": "pace",
    "time": "time",
    "movingtime": "time",
    "elapsedtime": "time",
    "duration": "time",
    "elevation": "elevation",
    "elevgain": "elevation",
    "elevationgain": "elevation",
    "avghr": "hr",
    "averagehr": "hr",
    "heartrate": "hr",
    "avgheartrate": "hr",
    "calories": "calories",
}


def _label_key(text: str) -> str | None:
    return CELL_LABELS.get(re.sub(r"[^a-z]", "", text.lower()))


def _group_lines(words: list[dict]) -> list[list[dict]]:
    """Cluster words into visual rows by vertical centre."""
    if not words:
        return []
    heights = sorted(word["height"] for word in words)
    tolerance = max(heights[len(heights) // 2] * 0.6, 6)

    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: w["top"] + w["height"] / 2):
        centre = word["top"] + word["height"] / 2
        if lines:
            last = lines[-1]
            last_centre = sum(w["top"] + w["height"] / 2 for w in last) / len(last)
            if abs(centre - last_centre) <= tolerance:
                last.append(word)
                continue
        lines.append([word])
    for line in lines:
        line.sort(key=lambda w: w["left"])
    return lines


def _line_cells(line: list[dict], gap_factor: float = 1.2) -> list[dict]:
    """Split a row into cells wherever the horizontal gap is unusually wide."""
    if not line:
        return []
    height = sum(word["height"] for word in line) / len(line)
    threshold = height * gap_factor

    cells, current = [], [line[0]]
    for word in line[1:]:
        previous = current[-1]
        if word["left"] - (previous["left"] + previous["width"]) > threshold:
            cells.append(current)
            current = [word]
        else:
            current.append(word)
    cells.append(current)

    return [{
        "text": " ".join(w["text"] for w in group),
        "left": min(w["left"] for w in group),
        "right": max(w["left"] + w["width"] for w in group),
        "top": min(w["top"] for w in group),
        "bottom": max(w["top"] + w["height"] for w in group),
    } for group in cells]


def extract_cells(image) -> dict[str, str]:
    """Read a grid-shaped screenshot as {canonical label: value text}.

    Returns an empty dict when the image is not grid-shaped — the web layout
    puts label and value on one line, where flat-text regex already works.
    """
    import pytesseract

    try:
        data = pytesseract.image_to_data(image, config=TESSERACT_CONFIG,
                                         output_type=pytesseract.Output.DICT)
    except Exception as error:  # pragma: no cover - environment problem
        raise _missing_tesseract(error)

    words = [
        {"text": data["text"][i].strip(), "left": data["left"][i], "top": data["top"][i],
         "width": data["width"][i], "height": data["height"][i]}
        for i in range(len(data["text"]))
        if data["text"][i].strip() and float(data["conf"][i]) >= 0
    ]

    lines = _group_lines(words)
    cells: dict[str, str] = {}

    for index, line in enumerate(lines):
        if index == 0:
            continue
        line_cells = _line_cells(line)
        labelled = [(cell, _label_key(cell["text"])) for cell in line_cells]
        labelled = [(cell, key) for cell, key in labelled if key]
        if len(labelled) < 2:
            continue                       # not a label row

        value_line = lines[index - 1]
        value_top = min(w["top"] for w in value_line)
        value_bottom = max(w["top"] + w["height"] for w in value_line)
        # A label row directly under its values sits close; further apart means
        # these are separate blocks and the pairing would be wrong.
        if min(c["top"] for c in line_cells) - value_bottom > (value_bottom - value_top) * 1.5:
            continue

        # Labels and values are both left-aligned to the column, but a value
        # is usually wider than its label. So the boundary is the NEXT label's
        # left edge, not the midpoint between labels — a midpoint slices
        # through the value it is meant to contain.
        starts = [cell["left"] for cell, _ in labelled]
        padding = max(8, int((value_bottom - value_top) * 0.25))
        nudge = max(4, int((value_bottom - value_top) * 0.3))

        for position, (cell, key) in enumerate(labelled):
            left = 0 if position == 0 else starts[position] - nudge
            right = (image.width if position == len(labelled) - 1
                     else starts[position + 1] - nudge)
            crop = image.crop((max(0, left), max(0, value_top - padding),
                               min(image.width, right), min(image.height, value_bottom + padding)))
            try:
                text = pytesseract.image_to_string(crop, config="--psm 7").strip()
            except Exception:  # pragma: no cover - a single bad crop is not fatal
                continue
            if text and key not in cells:
                cells[key] = text

    return cells


def _fix_units(text: str) -> str:
    """Repair the unit confusions Tesseract makes most often: the m of 'km'
    read as n or rn, and the i of 'mi' lost."""
    text = re.sub(r"/\s*k[mnr]{1,2}\b", "/km", text, flags=re.I)
    text = re.sub(r"/\s*(?:rni|nni|ni|rn)\b", "/mi", text, flags=re.I)
    text = re.sub(r"\b(\d[\d.,]*)\s*k[nr]\b", r"\1 km", text, flags=re.I)
    return text


# --------------------------------------------------------------------------
# Text -> numbers
# --------------------------------------------------------------------------

def _to_float(text: str) -> float | None:
    """'12.41', '12,41' and '12. 41' all become 12.41."""
    cleaned = text.replace(",", ".").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _clock_to_seconds(text: str) -> float | None:
    """'1:06:52' -> 4012; '52:18' -> 3138. Rejects impossible components."""
    parts = text.strip().split(":")
    if not 2 <= len(parts) <= 3:
        return None
    try:
        numbers = [int(p) for p in parts]
    except ValueError:
        return None
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    else:
        hours, minutes, seconds = 0, numbers[0], numbers[1]
    if minutes > 59 or seconds > 59 or hours > 23:
        return None
    return hours * 3600 + minutes * 60 + seconds


# Distance: a decimal immediately before a unit, or on a line labelled Distance.
DISTANCE_PATTERNS = [
    re.compile(r"(\d{1,3}[.,]\d{1,2})\s*(km|kilometer|kilometre)", re.I),
    re.compile(r"(\d{1,3}[.,]\d{1,2})\s*(mi|mile)s?\b", re.I),
    re.compile(r"distance\D{0,12}(\d{1,3}[.,]\d{1,2})", re.I),
    re.compile(r"(\d{1,3}[.,]\d{1,2})\s*\n?\s*distance", re.I),
]

# Pace: m:ss followed by a per-unit marker.
PACE_PATTERNS = [
    re.compile(r"(\d{1,2}:\d{2})\s*/\s*(km|mi)\b", re.I),
    re.compile(r"(\d{1,2}:\d{2})\s*(?:per|/)\s*(kilometer|kilometre|mile)", re.I),
    re.compile(r"pace\D{0,12}(\d{1,2}:\d{2})", re.I),
]

# Duration: labelled, or a bare h:mm:ss.
TIME_PATTERNS = [
    re.compile(r"(?:moving\s*time|elapsed\s*time|\btime\b|duration)\D{0,12}(\d{1,2}:\d{2}(?::\d{2})?)", re.I),
    re.compile(r"(\d{1,2}:\d{2}:\d{2})"),
]

HR_PATTERN = re.compile(r"(\d{2,3})\s*(?:bpm|/\s*bpm)", re.I)
HR_LABEL_PATTERN = re.compile(r"(?:avg|average)?\s*(?:hr|heart\s*rate)\D{0,12}(\d{2,3})", re.I)
ELEV_PATTERN = re.compile(r"(?:elev(?:ation)?\s*gain|elevation)\D{0,12}(\d{1,4})", re.I)


def _find(patterns, text) -> tuple[str, ...] | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.groups()
    return None


def _parse_date(text: str, today: date | None = None) -> date | None:
    """Strava writes dates several ways depending on where you screenshot it."""
    today = today or date.today()
    lowered = text.lower()

    if re.search(r"\btoday\b", lowered):
        return today
    if re.search(r"\byesterday\b", lowered):
        return today - timedelta(days=1)

    patterns = [
        ("%B %d, %Y", r"([A-Z][a-z]+ \d{1,2}, \d{4})"),
        ("%b %d, %Y", r"([A-Z][a-z]{2} \d{1,2}, \d{4})"),
        ("%d %B %Y", r"(\d{1,2} [A-Z][a-z]+ \d{4})"),
        ("%d.%m.%Y", r"(\d{1,2}\.\d{1,2}\.\d{4})"),
        ("%Y-%m-%d", r"(\d{4}-\d{2}-\d{2})"),
    ]
    for fmt, pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return datetime.strptime(match.group(1), fmt).date()
            except ValueError:
                continue

    # "Monday, 14 September" style, with the year implied.
    match = re.search(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
                      r"September|October|November|December)", text, re.I)
    if match:
        try:
            parsed = datetime.strptime(f"{match.group(1)} {match.group(2)} {today.year}",
                                       "%d %B %Y").date()
            # A date in the future means it belongs to last year.
            return parsed.replace(year=today.year - 1) if parsed > today else parsed
        except ValueError:
            pass
    return None


MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")
NAME_STOPWORDS = {"distance", "pace", "time", "elevation", "calories", "heart",
                  "bpm", "avg", "average", "moving", "elapsed", "today",
                  "yesterday", "gain", "hr"}


def _looks_like_a_title(line: str) -> bool:
    if not 3 <= len(line) <= 60 or re.fullmatch(r"[\W_\d]+", line):
        return False
    lowered = line.lower()
    if any(month in lowered for month in MONTHS):
        return False
    words = set(re.findall(r"[a-z]+", lowered))
    if words & NAME_STOPWORDS:
        return False
    # A title has letters; a metrics row is mostly digits and punctuation.
    return sum(ch.isalpha() for ch in line) >= 3


def _parse_name(text: str) -> str | None:
    """The activity title. Strava puts it first, so the first line that reads
    like a title wins — checked before the generic scan so a run called
    '5 x 1 km intervals' is not rejected for containing 'km'."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and _looks_like_a_title(lines[0]):
        return lines[0]
    for line in lines:
        if _looks_like_a_title(line):
            return line
    return None


def cross_validate(parsed: ParsedRun) -> ParsedRun:
    """Use pace = time ÷ distance to fill gaps and catch misreads.

    Any two of the three determine the third. When all three are present they
    must agree; a disagreement beyond 5% means at least one was misread, and
    the honest response is to say so rather than to pick a winner.
    """
    distance, seconds, pace = parsed.distance_km, parsed.moving_seconds, parsed.pace_sec_per_km

    if distance and seconds and not pace:
        parsed.pace_sec_per_km = seconds / distance
        parsed.confidence["pace_sec_per_km"] = "high"      # derived, not read
    elif distance and pace and not seconds:
        parsed.moving_seconds = pace * distance
        parsed.confidence["moving_seconds"] = "high"
    elif seconds and pace and not distance:
        parsed.distance_km = seconds / pace
        parsed.confidence["distance_km"] = "high"
    elif distance and seconds and pace:
        implied = seconds / distance
        if implied > 0 and abs(implied - pace) / implied > 0.05:
            parsed.warnings.append(
                f"The three numbers do not agree: {distance:.2f} km in "
                f"{seconds / 60:.1f} min is {implied / 60:.2f} min/km, but the pace "
                f"reads {pace / 60:.2f} min/km. One of them was misread — check all "
                "three before saving."
            )
            for key in ("distance_km", "moving_seconds", "pace_sec_per_km"):
                parsed.confidence[key] = "low"

    # Sanity bounds. These catch the classic OCR failure of a dropped decimal
    # point far more reliably than any confidence score.
    if parsed.distance_km is not None and not 0.3 <= parsed.distance_km <= 300:
        parsed.warnings.append(f"Distance of {parsed.distance_km:.2f} km looks wrong — check it.")
        parsed.confidence["distance_km"] = "low"
    if parsed.pace_sec_per_km is not None and not 120 <= parsed.pace_sec_per_km <= 900:
        parsed.warnings.append(
            f"A pace of {parsed.pace_sec_per_km / 60:.2f} min/km is outside the plausible "
            "range for running — check the distance and time."
        )
        parsed.confidence["pace_sec_per_km"] = "low"
    if parsed.moving_seconds is not None and not 120 <= parsed.moving_seconds <= 24 * 3600:
        parsed.warnings.append("The duration looks wrong — check it.")
        parsed.confidence["moving_seconds"] = "low"
    if parsed.avg_hr is not None and not 60 <= parsed.avg_hr <= 230:
        parsed.avg_hr = None

    return parsed


def _unit_system(text: str, cells: dict[str, str] | None, assume_miles: bool) -> str:
    """'km' or 'mi'. A pace always carries its unit, so it is the best witness."""
    haystack = " ".join([text] + list((cells or {}).values())).lower()
    if re.search(r"/\s*mi\b|\bmiles?\b", haystack):
        return "mi"
    if re.search(r"/\s*km\b|\bkm\b|kilomet", haystack):
        return "km"
    return "mi" if assume_miles else "km"


def _from_cells(parsed: ParsedRun, cells: dict[str, str], unit: str) -> None:
    """Fill fields from cropped grid cells. These are the most reliable reads
    available — each was OCR'd alone, with no neighbour to merge into."""
    if "distance" in cells:
        match = re.search(r"(\d{1,3}[.,]\d{1,2}|\d{1,3})", cells["distance"])
        value = _to_float(match.group(1)) if match else None
        if value is not None:
            local = "mi" if re.search(r"\bmi\b", cells["distance"], re.I) else unit
            parsed.distance_km = value * MILES_TO_KM if local == "mi" else value
            parsed.confidence["distance_km"] = "high"

    if "time" in cells:
        match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)", cells["time"])
        seconds = _clock_to_seconds(match.group(1)) if match else None
        if seconds is not None:
            parsed.moving_seconds = seconds
            parsed.confidence["moving_seconds"] = "high"

    if "pace" in cells:
        match = re.search(r"(\d{1,2}:\d{2})", cells["pace"])
        seconds = _clock_to_seconds(match.group(1)) if match else None
        if seconds is not None:
            local = "mi" if re.search(r"/\s*mi|\bmi\b", cells["pace"], re.I) else unit
            parsed.pace_sec_per_km = seconds / MILES_TO_KM if local == "mi" else seconds
            parsed.confidence["pace_sec_per_km"] = "high"

    if "hr" in cells:
        match = re.search(r"(\d{2,3})", cells["hr"])
        if match:
            parsed.avg_hr = _to_float(match.group(1))
            parsed.confidence["avg_hr"] = "high"

    if "elevation" in cells:
        match = re.search(r"(\d{1,4})", cells["elevation"])
        if match:
            parsed.elevation_m = _to_float(match.group(1))


def parse_activity_text(text: str, today: date | None = None,
                        assume_miles: bool = False,
                        cells: dict[str, str] | None = None) -> ParsedRun:
    """Turn OCR output into a structured run.

    `cells` is the optional column-cropped read from `extract_cells`. When
    present it wins, because each cell was recognised in isolation; the
    flat-text regexes then fill whatever it missed.
    """
    parsed = ParsedRun(raw_text=text)
    if (not text or not text.strip()) and not cells:
        parsed.warnings.append("Nothing readable was found in that image.")
        return parsed

    text = _fix_units(text or "")
    cells = {key: _fix_units(value) for key, value in (cells or {}).items()}
    unit = _unit_system(text, cells, assume_miles)

    if cells:
        _from_cells(parsed, cells, unit)

    # Distance, with its unit.
    if parsed.distance_km is None:
        found = _find(DISTANCE_PATTERNS, text)
        if found:
            value = _to_float(found[0])
            if value is not None:
                local = found[1].lower() if len(found) > 1 else unit
                is_miles = local.startswith("mi")
                parsed.distance_km = value * MILES_TO_KM if is_miles else value
                parsed.confidence["distance_km"] = "high" if len(found) > 1 else "medium"

    # Pace, with its unit.
    if parsed.pace_sec_per_km is None:
        found = _find(PACE_PATTERNS, text)
        if found:
            seconds = _clock_to_seconds(found[0])
            if seconds is not None:
                local = found[1].lower() if len(found) > 1 else unit
                is_miles = local.startswith("mi")
                parsed.pace_sec_per_km = seconds / MILES_TO_KM if is_miles else seconds
                parsed.confidence["pace_sec_per_km"] = "high" if len(found) > 1 else "medium"

    # Duration. A labelled match is trustworthy; a bare clock might be anything.
    if parsed.moving_seconds is None:
        found = _find(TIME_PATTERNS, text)
        if found:
            seconds = _clock_to_seconds(found[0])
            # A two-part clock without a label is more likely a pace than a duration.
            if seconds is not None and not (found[0].count(":") == 1 and
                                            seconds == parsed.pace_sec_per_km):
                parsed.moving_seconds = seconds
                parsed.confidence["moving_seconds"] = "medium"

    if parsed.avg_hr is None:
        match = HR_PATTERN.search(text) or HR_LABEL_PATTERN.search(text)
        if match:
            parsed.avg_hr = _to_float(match.group(1))
            parsed.confidence["avg_hr"] = "medium"

    if parsed.elevation_m is None:
        match = ELEV_PATTERN.search(text)
        if match:
            parsed.elevation_m = _to_float(match.group(1))

    parsed.run_date = _parse_date(text, today=today)
    if parsed.run_date is None:
        parsed.warnings.append("No date found — it has been set to today, so change it if that is wrong.")
        parsed.run_date = today or date.today()
        parsed.confidence["run_date"] = "low"
    else:
        parsed.confidence["run_date"] = "high"

    parsed.name = _parse_name(text)

    parsed = cross_validate(parsed)

    if not parsed.is_usable:
        parsed.warnings.append(
            "Could not read enough to work out a pace. Fill the fields in by hand below, "
            "or try a tighter crop of the summary panel."
        )
    return parsed


def read_screenshot(image_bytes: bytes, today: date | None = None,
                    assume_miles: bool = False) -> ParsedRun:
    """The whole pipeline: pixels in, a draft run out.

    Reads the image twice — once flat for the title and date, once
    column-cropped for the numbers — and merges the two.
    """
    import io

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment problem
        raise RuntimeError("Screenshot reading needs Pillow. Add it to requirements.txt.") from error

    try:
        image = preprocess(Image.open(io.BytesIO(image_bytes)))
    except Exception as error:  # noqa: BLE001 — a corrupt or unsupported file
        raise RuntimeError(
            "That file could not be opened as an image. PNG and JPEG both work; "
            f"HEIC from an iPhone often does not — convert it first. ({error})"
        ) from error

    import pytesseract
    try:
        text = pytesseract.image_to_string(image, config=TESSERACT_CONFIG)
    except Exception as error:  # pragma: no cover - environment problem
        raise _missing_tesseract(error)

    try:
        cells = extract_cells(image)
    except RuntimeError:
        raise
    except Exception:  # pragma: no cover - fall back to flat text
        cells = {}

    return parse_activity_text(text, today=today, assume_miles=assume_miles, cells=cells)
