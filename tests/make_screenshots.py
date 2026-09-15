"""Render Strava-like activity screenshots for testing the OCR pipeline.

These are not pixel-perfect copies of Strava — they reproduce the things that
actually matter to an OCR parser: a grid of big numbers above small labels, a
phone-sized viewport, light and dark modes, and the metric/imperial split.

Run directly to write PNGs into tests/screenshots/ for eyeballing:
    python tests/make_screenshots.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
REGULAR = FONT_DIR / "DejaVuSans.ttf"
BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

LIGHT = {"bg": (255, 255, 255), "ink": (30, 30, 30), "muted": (120, 120, 120)}
DARK = {"bg": (24, 24, 26), "ink": (242, 242, 242), "muted": (150, 150, 150)}


def _font(path: Path, size: int):
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:  # pragma: no cover - font missing
        return ImageFont.load_default()


def _fitted(text: str, max_width: int, start_size: int, bold: bool = True):
    """Largest font size at which `text` still fits the column.

    Real apps lay out this way — a long value renders smaller rather than
    spilling into its neighbour. Reproducing that matters, because a renderer
    whose columns overflow produces an image no phone would ever show, and
    tuning the parser against it tunes it for the wrong problem.
    """
    path = BOLD if bold else REGULAR
    for size in range(start_size, 9, -1):
        font = _font(path, size)
        if font.getlength(text) <= max_width:
            return font
    return _font(path, 10)


def phone_summary(title: str, when: str, cells: list[tuple[str, str]],
                  dark: bool = False, width: int = 430, height: int = 560,
                  tight: bool = False) -> Image.Image:
    """The Strava mobile activity screen: title, date, then a grid of
    value-over-label cells.

    `tight=True` deliberately overflows the columns, to exercise the
    cross-validation safety net against values that run together.
    """
    theme = DARK if dark else LIGHT
    image = Image.new("RGB", (width, height), theme["bg"])
    draw = ImageDraw.Draw(image)

    draw.text((26, 34), title, font=_font(BOLD, 26), fill=theme["ink"])
    draw.text((26, 74), when, font=_font(REGULAR, 16), fill=theme["muted"])

    columns, x0, gap = 3, 26, (width - 52) // 3
    for index, (value, label) in enumerate(cells):
        row, column = divmod(index, columns)
        x = x0 + column * gap
        y = 140 + row * 110
        value_font = (_font(BOLD, 34) if tight
                      else _fitted(value, gap - 14, 30))
        draw.text((x, y), value, font=value_font, fill=theme["ink"])
        draw.text((x, y + 44), label, font=_font(REGULAR, 15), fill=theme["muted"])

    return image


def web_detail(title: str, when: str, pairs: list[tuple[str, str]],
               dark: bool = False, width: int = 760, height: int = 460) -> Image.Image:
    """The Strava web activity page: a label-then-value list, left aligned."""
    theme = DARK if dark else LIGHT
    image = Image.new("RGB", (width, height), theme["bg"])
    draw = ImageDraw.Draw(image)

    draw.text((30, 28), title, font=_font(BOLD, 28), fill=theme["ink"])
    draw.text((30, 68), when, font=_font(REGULAR, 16), fill=theme["muted"])

    label_font, value_font = _font(REGULAR, 18), _font(BOLD, 22)
    for index, (label, value) in enumerate(pairs):
        y = 130 + index * 52
        draw.text((30, y + 3), label, font=label_font, fill=theme["muted"])
        draw.text((300, y), value, font=value_font, fill=theme["ink"])

    return image


# The cases the tests run against.
CASES = {
    "phone_light": lambda: phone_summary(
        "Morning Run", "Today at 7:32 AM",
        [("12.41", "Distance"), ("1:06:52", "Moving Time"), ("5:23 /km", "Pace"),
         ("86 m", "Elevation"), ("148", "Avg HR"), ("742", "Calories")],
    ),
    "phone_dark": lambda: phone_summary(
        "Evening Run", "Yesterday at 6:10 PM",
        [("8.04", "Distance"), ("0:44:12", "Moving Time"), ("5:30 /km", "Pace"),
         ("42 m", "Elevation"), ("141", "Avg HR"), ("478", "Calories")],
        dark=True,
    ),
    "phone_miles": lambda: phone_summary(
        "Lunch Run", "Today at 12:40 PM",
        [("6.21", "Distance"), ("0:52:30", "Moving Time"), ("8:27 /mi", "Pace"),
         ("110 ft", "Elevation"), ("152", "Avg HR"), ("610", "Calories")],
    ),
    "web_light": lambda: web_detail(
        "Long run", "September 13, 2026",
        [("Distance", "21.48 km"), ("Moving Time", "1:59:37"), ("Pace", "5:34 /km"),
         ("Elevation Gain", "124 m"), ("Avg HR", "148 bpm")],
    ),
    "web_dark": lambda: web_detail(
        "5 x 1 km intervals", "September 10, 2026",
        [("Distance", "11.00 km"), ("Moving Time", "0:56:16"), ("Pace", "5:07 /km"),
         ("Elevation Gain", "31 m"), ("Avg HR", "163 bpm")],
        dark=True,
    ),
    # Deliberately unreadable: oversized values run into each other, the way a
    # bad crop or an odd device scaling would. Nothing is expected to parse
    # correctly here — the point is that the app says so instead of inventing
    # a plausible-looking run.
    "phone_overflow": lambda: phone_summary(
        "Tempo run", "Today at 6:02 PM",
        [("14.62", "Distance"), ("1:12:48", "Moving Time"), ("4:59 /km", "Pace")],
        tight=True,
    ),
}

# What each case should produce, for the tests to assert against.
EXPECTED = {
    "phone_light": {"distance_km": 12.41, "moving_seconds": 4012, "pace_sec_per_km": 323},
    "phone_dark": {"distance_km": 8.04, "moving_seconds": 2652, "pace_sec_per_km": 330},
    "phone_miles": {"distance_km": 9.9967, "moving_seconds": 3150, "pace_sec_per_km": 315.0},
    "web_light": {"distance_km": 21.48, "moving_seconds": 7177, "pace_sec_per_km": 334},
    "web_dark": {"distance_km": 11.00, "moving_seconds": 3376, "pace_sec_per_km": 307},
}


def render(name: str) -> Image.Image:
    return CASES[name]()


def render_bytes(name: str) -> bytes:
    import io
    buffer = io.BytesIO()
    render(name).save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    out = Path(__file__).parent / "screenshots"
    out.mkdir(exist_ok=True)
    for name in CASES:
        path = out / f"{name}.png"
        render(name).save(path)
        print(f"wrote {path}")
