"""Altair chart builders.

Altair ships with Streamlit, so nothing extra installs on deploy. The
categorical colours are a validated slot order — worst adjacent colour-vision
separation ΔE 9.2 on a white surface against a target of 8, and the first
three slots also clear the stricter all-pairs test used for scatter plots.
Take hues from the front of the list; do not reorder them.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
          "critical": "#d03b3b", "muted": "#898781"}

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

INTENSITY_ORDER = ["easy", "moderate", "hard"]
INTENSITY_LABEL = {"easy": "Easy", "moderate": "Moderate", "hard": "Hard", "unknown": "Unknown"}


def _style(chart):
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelFont=FONT, titleFont=FONT, labelColor=INK_MUTED, titleColor=INK_SECONDARY,
            labelFontSize=11, titleFontSize=11, titlePadding=10,
            gridColor=GRIDLINE, gridWidth=1, domainColor=BASELINE, tickColor=BASELINE,
        )
        .configure_legend(
            labelFont=FONT, titleFont=FONT, labelColor=INK_SECONDARY, titleColor=INK_SECONDARY,
            labelFontSize=11, titleFontSize=11, orient="top", direction="horizontal",
            offset=8, title=None, symbolType="square",
        )
        .configure_title(font=FONT, fontSize=13, color=INK_PRIMARY, anchor="start", dy=-6)
    )


def weekly_volume_chart(weekly: pd.DataFrame, runs: pd.DataFrame,
                        target_km: float | None = None, height: int = 300):
    """Weekly kilometres, split easy vs harder running.

    The split is the point: the bar height is volume, the blue share is how
    much of it was genuinely easy.
    """
    if weekly.empty or runs.empty:
        return None

    frame = runs.copy()
    frame["week"] = frame["date"].dt.to_period("W").dt.start_time
    frame["band"] = frame["intensity"].map(
        lambda i: "Easy" if i == "easy" else "Moderate / hard")
    tidy = frame.groupby(["week", "band"], as_index=False)["distance_km"].sum()

    bars = (
        alt.Chart(tidy)
        .mark_bar(stroke=SURFACE, strokeWidth=2, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("week:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)),
            y=alt.Y("distance_km:Q", title="km", stack=True, axis=alt.Axis(grid=True)),
            color=alt.Color("band:N", sort=["Easy", "Moderate / hard"],
                            scale=alt.Scale(domain=["Easy", "Moderate / hard"],
                                            range=[SERIES[0], SERIES[1]]),
                            legend=alt.Legend()),
            order=alt.Order("band:N", sort="ascending"),
            tooltip=[alt.Tooltip("week:T", title="Week of", format="%d %b %Y"),
                     alt.Tooltip("band:N", title="Intensity"),
                     alt.Tooltip("distance_km:Q", title="km", format=".1f")],
        )
    )

    layers = [bars]
    if target_km and target_km == target_km:
        rule = (
            alt.Chart(pd.DataFrame({"y": [target_km]}))
            .mark_rule(color=INK_SECONDARY, strokeDash=[4, 3], strokeWidth=1.5)
            .encode(y="y:Q")
        )
        label = (
            alt.Chart(pd.DataFrame({"y": [target_km], "text": [f"this week's target {target_km:.0f} km"]}))
            .mark_text(align="left", dx=4, dy=-6, font=FONT, fontSize=10, color=INK_SECONDARY)
            .encode(y="y:Q", text="text:N", x=alt.value(4))
        )
        layers += [rule, label]

    return _style(alt.layer(*layers).properties(title="Weekly volume", height=height))


def acwr_chart(runs: pd.DataFrame, height: int = 260):
    """Acute:chronic ratio week by week, against the 0.8–1.3 comfort band."""
    if runs.empty:
        return None

    dates = pd.date_range(runs["date"].min() + pd.Timedelta(days=28),
                          runs["date"].max(), freq="D")
    if len(dates) == 0:
        return None

    rows = []
    for day in dates:
        acute = runs.loc[(runs["date"] > day - pd.Timedelta(days=7)) & (runs["date"] <= day), "load"].sum()
        chronic = runs.loc[(runs["date"] > day - pd.Timedelta(days=28)) & (runs["date"] <= day), "load"].sum() / 4
        if chronic > 0:
            rows.append({"date": day, "ratio": acute / chronic})
    if not rows:
        return None
    tidy = pd.DataFrame(rows)

    band = (
        alt.Chart(pd.DataFrame({"low": [0.8], "high": [1.3]}))
        .mark_rect(color=SERIES[2], opacity=0.10)
        .encode(y=alt.Y("low:Q", title="acute ÷ chronic"), y2="high:Q")
    )
    line = (
        alt.Chart(tidy).mark_line(strokeWidth=2, color=SERIES[0])
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)),
            y=alt.Y("ratio:Q", title="acute ÷ chronic", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                     alt.Tooltip("ratio:Q", title="ACWR", format=".2f")],
        )
    )
    return _style(
        alt.layer(band, line)
        .properties(title="Acute : chronic workload  (shaded band = 0.8–1.3)", height=height)
    )


def pace_scatter(runs: pd.DataFrame, zones: dict[str, float], height: int = 300):
    """Every run: distance against pace, coloured by intensity, with the
    threshold and marathon pace lines for reference."""
    if runs.empty:
        return None

    tidy = runs.copy()
    tidy["Intensity"] = tidy["intensity"].map(lambda i: INTENSITY_LABEL.get(i, "Unknown"))
    tidy["pace_min"] = tidy["pace_sec_per_km"] / 60.0
    names = [INTENSITY_LABEL[k] for k in INTENSITY_ORDER]

    points = (
        alt.Chart(tidy)
        .mark_point(size=70, filled=True, opacity=0.85, stroke=SURFACE, strokeWidth=1.5)
        .encode(
            x=alt.X("distance_km:Q", title="distance (km)", scale=alt.Scale(zero=False)),
            y=alt.Y("pace_min:Q", title="pace (min/km)",
                    scale=alt.Scale(zero=False, reverse=True)),
            color=alt.Color("Intensity:N", sort=names,
                            scale=alt.Scale(domain=names, range=SERIES[:3]),
                            legend=alt.Legend()),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                     alt.Tooltip("name:N", title="Run"),
                     alt.Tooltip("distance_km:Q", title="km", format=".1f"),
                     alt.Tooltip("pace_min:Q", title="min/km", format=".2f"),
                     alt.Tooltip("Intensity:N")],
        )
    )

    reference = pd.DataFrame([
        {"pace_min": zones["threshold"] / 60.0, "label": "threshold"},
        {"pace_min": zones["marathon"] / 60.0, "label": "marathon pace"},
        {"pace_min": zones["easy"] / 60.0, "label": "easy (quick end)"},
    ])
    rules = (
        alt.Chart(reference).mark_rule(color=BASELINE, strokeDash=[4, 3], strokeWidth=1)
        .encode(y="pace_min:Q")
    )
    labels = (
        alt.Chart(reference)
        .mark_text(align="left", dx=4, dy=-5, font=FONT, fontSize=10, color=INK_MUTED)
        .encode(y="pace_min:Q", text="label:N", x=alt.value(3))
    )

    return _style(
        alt.layer(rules, labels, points)
        .properties(title="Every run — pace against distance", height=height)
    )


def projection_chart(current_vdot: float, required_vdot: float, weeks: float,
                     gain_rate: float, height: int = 260):
    """Projected fitness against what the goal requires.

    Two lines on one axis: where a typical improvement rate takes you, and the
    level the target actually needs.
    """
    if not all(isinstance(v, (int, float)) for v in (current_vdot, required_vdot, weeks)):
        return None
    if weeks != weeks or weeks <= 0:
        return None

    steps = list(range(0, int(weeks) + 1, max(1, int(weeks) // 30)))
    rows = []
    for week in steps:
        rows.append({"week": week, "series": "Projected fitness", "vdot": current_vdot + gain_rate * week})
        rows.append({"week": week, "series": "Needed for the goal", "vdot": required_vdot})
    tidy = pd.DataFrame(rows)
    names = ["Projected fitness", "Needed for the goal"]

    lines = (
        alt.Chart(tidy).mark_line(strokeWidth=2)
        .encode(
            x=alt.X("week:Q", title="weeks from now"),
            y=alt.Y("vdot:Q", title="VDOT", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", sort=names,
                            scale=alt.Scale(domain=names, range=[SERIES[0], SERIES[1]]),
                            legend=alt.Legend()),
            strokeDash=alt.StrokeDash("series:N", sort=names,
                                      scale=alt.Scale(domain=names, range=[[1, 0], [5, 4]]),
                                      legend=None),
            tooltip=[alt.Tooltip("week:Q", title="Week"),
                     alt.Tooltip("series:N", title=""),
                     alt.Tooltip("vdot:Q", title="VDOT", format=".1f")],
        )
    )
    return _style(lines.properties(title="Fitness projection against the goal", height=height))
