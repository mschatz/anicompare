"""Plotting and correlation utilities."""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

from .io import read_fasta, read_json, read_tsv, write_tsv


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _pearson(values_x: list[float], values_y: list[float]) -> float:
    mean_x = _mean(values_x)
    mean_y = _mean(values_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in values_x))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in values_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return numerator / (denom_x * denom_y)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        next_position = position + 1
        while next_position < len(indexed) and indexed[next_position][1] == indexed[position][1]:
            next_position += 1
        average_rank = (position + next_position - 1) / 2.0 + 1.0
        for original_index, _value in indexed[position:next_position]:
            ranks[original_index] = average_rank
        position = next_position
    return ranks


def _spearman(values_x: list[float], values_y: list[float]) -> float:
    return _pearson(_rank(values_x), _rank(values_y))


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    size = 3
    for pivot in range(size):
        max_row = max(range(pivot, size), key=lambda idx: abs(augmented[idx][pivot]))
        augmented[pivot], augmented[max_row] = augmented[max_row], augmented[pivot]
        pivot_value = augmented[pivot][pivot]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Singular matrix in quadratic fit")
        for column in range(pivot, size + 1):
            augmented[pivot][column] /= pivot_value
        for row_index in range(size):
            if row_index == pivot:
                continue
            factor = augmented[row_index][pivot]
            for column in range(pivot, size + 1):
                augmented[row_index][column] -= factor * augmented[pivot][column]
    return [augmented[row][size] for row in range(size)]


def _linear_fit(values_x: list[float], values_y: list[float]) -> dict[str, object]:
    mean_x = _mean(values_x)
    mean_y = _mean(values_y)
    denominator = sum((x - mean_x) ** 2 for x in values_x)
    slope = 0.0 if denominator == 0 else sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y)) / denominator
    intercept = mean_y - slope * mean_x
    predicted = [intercept + slope * x for x in values_x]
    corr = _pearson(values_y, predicted)
    return {
        "model_type": "linear",
        "coefficients": {"slope": slope, "intercept": intercept},
        "equation": f"y = {slope:.6f}x + {intercept:.6f}",
        "fit_correlation": corr,
        "r_squared": corr * corr,
    }


def _quadratic_fit(values_x: list[float], values_y: list[float]) -> dict[str, object]:
    n = len(values_x)
    sum_x = sum(values_x)
    sum_x2 = sum(x * x for x in values_x)
    sum_x3 = sum(x * x * x for x in values_x)
    sum_x4 = sum(x * x * x * x for x in values_x)
    sum_y = sum(values_y)
    sum_xy = sum(x * y for x, y in zip(values_x, values_y))
    sum_x2y = sum((x * x) * y for x, y in zip(values_x, values_y))
    matrix = [
        [sum_x4, sum_x3, sum_x2],
        [sum_x3, sum_x2, sum_x],
        [sum_x2, sum_x, n],
    ]
    vector = [sum_x2y, sum_xy, sum_y]
    a, b, c = _solve_3x3(matrix, vector)
    predicted = [a * x * x + b * x + c for x in values_x]
    corr = _pearson(values_y, predicted)
    return {
        "model_type": "quadratic",
        "coefficients": {"a": a, "b": b, "c": c},
        "equation": f"y = {a:.6f}x^2 + {b:.6f}x + {c:.6f}",
        "fit_correlation": corr,
        "r_squared": corr * corr,
    }


def _root_jaccard_fit(values_x: list[float], values_y: list[float], k: int) -> dict[str, object]:
    rooted_y = [max(y, 0.0) ** (1.0 / k) for y in values_y]
    linear = _linear_fit(values_x, rooted_y)
    slope = float(linear["coefficients"]["slope"])
    intercept = float(linear["coefficients"]["intercept"])
    predicted_root = [max(0.0, min(1.0, intercept + slope * x)) for x in values_x]
    predicted_y = [value ** k for value in predicted_root]
    corr = _pearson(values_y, predicted_y)
    return {
        "model_type": "kth_root",
        "coefficients": {"slope": slope, "intercept": intercept, "k": k},
        "equation": f"y = ({slope:.6f}x + {intercept:.6f})^{k}",
        "fit_correlation": corr,
        "r_squared": corr * corr,
    }


def _root_divergence_fit(values_x: list[float], values_y: list[float], k: int) -> dict[str, object]:
    rooted_x = [max(x, 0.0) ** (1.0 / k) for x in values_x]
    linear = _linear_fit(rooted_x, values_y)
    slope = float(linear["coefficients"]["slope"])
    intercept = float(linear["coefficients"]["intercept"])
    predicted = [intercept + slope * value for value in rooted_x]
    corr = _pearson(values_y, predicted)
    return {
        "model_type": "kth_root_divergence",
        "coefficients": {"slope": slope, "intercept": intercept, "k": k},
        "equation": f"y = {slope:.6f}x^(1/{k}) + {intercept:.6f}",
        "fit_correlation": corr,
        "r_squared": corr * corr,
    }


def compute_regressions(input_tsv: Path) -> list[dict[str, object]]:
    rows = _load_master_table(input_tsv)
    full_rows = _load_full_master_rows(input_tsv)
    k = int(full_rows[0]["k"])
    specs = [
        ("true_mutation_rate", "minimap2_ani", "linear"),
        ("true_mutation_rate", "minimap2_dv", "linear"),
        ("true_mutation_rate", "jaccard_similarity", "quadratic"),
        ("true_mutation_rate", "ref_jaccard_similarity", "quadratic"),
        ("minimap2_ani", "jaccard_similarity", "quadratic"),
        ("minimap2_ani", "ref_jaccard_similarity", "quadratic"),
        ("jaccard_similarity", "true_mutation_rate", "quadratic"),
        ("ref_jaccard_similarity", "true_mutation_rate", "quadratic"),
    ]
    regressions: list[dict[str, object]] = []
    for x_name, y_name, model_type in specs:
        pairs = [(float(row[x_name]), float(row[y_name])) for row in rows if not math.isnan(float(row[x_name])) and not math.isnan(float(row[y_name]))]
        values_x = [pair[0] for pair in pairs]
        values_y = [pair[1] for pair in pairs]
        try:
            model = _linear_fit(values_x, values_y) if model_type == "linear" else _quadratic_fit(values_x, values_y)
        except ValueError:
            continue
        regressions.append({"x_metric": x_name, "y_metric": y_name, **model})
        if y_name in {"jaccard_similarity", "ref_jaccard_similarity"}:
            try:
                regressions.append(
                    {
                        "x_metric": x_name,
                        "y_metric": y_name,
                        **_root_jaccard_fit(values_x, values_y, k),
                    }
                )
            except ValueError:
                pass
        if x_name in {"jaccard_similarity", "ref_jaccard_similarity"} and y_name == "true_mutation_rate":
            try:
                regressions.append(
                    {
                        "x_metric": x_name,
                        "y_metric": y_name,
                        **_root_divergence_fit(values_x, values_y, k),
                    }
                )
            except ValueError:
                pass
    return regressions


def _reference_metrics(input_tsv: Path) -> dict[str, object]:
    results_dir = input_tsv.parent
    config_path = results_dir / "run_config.json"
    reference_path = results_dir / "reference" / "reference.fa"

    config = read_json(config_path) if config_path.exists() else {}
    records = read_fasta(reference_path)
    total_length = sum(len(record.sequence) for record in records)
    gc_count = sum(base in {"G", "C"} for record in records for base in record.sequence.upper())
    gc_fraction = gc_count / total_length if total_length else 0.0

    source = "simulated" if config.get("simulate_length") else "reference_fasta"
    source_value = config.get("simulate_length") if config.get("simulate_length") else config.get("reference_fasta", "NA")
    return {
        "source": source,
        "source_value": source_value,
        "contigs": len(records),
        "length": total_length,
        "gc_fraction": gc_fraction,
        "reference_path": str(reference_path),
        "analysis_mode": config.get("analysis_mode", "whole_reference"),
        "chunk_length": config.get("chunk_length", ""),
    }


def _load_master_table(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float | str]] = []
    for row in read_tsv(path):
        minimap2_ani = row["minimap2_ani"].strip()
        jaccard = row["jaccard_similarity"].strip()
        rows.append(
            {
                "rate_label": row["rate_label"],
                "replicate": row["replicate"],
                "true_mutation_rate": float(row["true_mutation_rate"]),
                "minimap2_ani": float(minimap2_ani) if minimap2_ani else math.nan,
                "minimap2_dv": float(row["minimap2_dv"]) if row["minimap2_dv"].strip() else math.nan,
                "minimap2_query_coverage": float(row["minimap2_query_coverage"]),
                "jaccard_similarity": float(jaccard) if jaccard else math.nan,
                "ref_jaccard_similarity": float(row["ref_jaccard_similarity"]),
            }
        )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _load_full_master_rows(path: Path) -> list[dict[str, str]]:
    rows = list(read_tsv(path))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def compute_correlations(input_tsv: Path) -> list[dict[str, object]]:
    rows = _load_master_table(input_tsv)
    comparisons = [
        ("true_mutation_rate", "minimap2_ani"),
        ("true_mutation_rate", "minimap2_dv"),
        ("true_mutation_rate", "minimap2_query_coverage"),
        ("true_mutation_rate", "jaccard_similarity"),
        ("true_mutation_rate", "ref_jaccard_similarity"),
        ("minimap2_ani", "jaccard_similarity"),
        ("minimap2_ani", "ref_jaccard_similarity"),
        ("jaccard_similarity", "ref_jaccard_similarity"),
    ]
    output: list[dict[str, object]] = []
    for x_name, y_name in comparisons:
        pairs = [(row[x_name], row[y_name]) for row in rows if not math.isnan(row[x_name]) and not math.isnan(row[y_name])]
        values_x = [pair[0] for pair in pairs]
        values_y = [pair[1] for pair in pairs]
        output.append(
            {
                "x_metric": x_name,
                "y_metric": y_name,
                "pearson_r": _pearson(values_x, values_y) if values_x else math.nan,
                "spearman_rho": _spearman(values_x, values_y) if values_x else math.nan,
            }
        )
    return output


def write_correlations(input_tsv: Path, output_tsv: Path) -> list[dict[str, object]]:
    correlations = compute_correlations(input_tsv)
    write_tsv(correlations, output_tsv, fieldnames=["x_metric", "y_metric", "pearson_r", "spearman_rho"])
    return correlations


def _scale_value(value: float, min_value: float, max_value: float, low: float, high: float) -> float:
    if max_value == min_value:
        return (low + high) / 2.0
    return low + ((value - min_value) / (max_value - min_value)) * (high - low)


def _nice_number(value: float, round_value: bool) -> float:
    if value == 0:
        return 1.0
    exponent = math.floor(math.log10(abs(value)))
    fraction = abs(value) / (10 ** exponent)
    if round_value:
        if fraction < 1.5:
            nice_fraction = 1.0
        elif fraction < 3.0:
            nice_fraction = 2.0
        elif fraction < 7.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    else:
        if fraction <= 1.0:
            nice_fraction = 1.0
        elif fraction <= 2.0:
            nice_fraction = 2.0
        elif fraction <= 5.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    return nice_fraction * (10 ** exponent)


def _nice_axis_bounds(min_value: float, max_value: float, tick_count: int = 5) -> tuple[float, float, float]:
    if min_value == max_value:
        padding = 0.5 if min_value == 0 else abs(min_value) * 0.1
        min_value -= padding
        max_value += padding

    if 0.0 <= min_value and max_value <= 1.0:
        step = _nice_number(1.0 / max(1, tick_count - 1), round_value=True)
        return 0.0, 1.0, step

    value_range = _nice_number(max_value - min_value, round_value=False)
    step = _nice_number(value_range / max(1, tick_count - 1), round_value=True)
    nice_min = math.floor(min_value / step) * step
    nice_max = math.ceil(max_value / step) * step
    return nice_min, nice_max, step


def _format_tick(value: float, step: float) -> str:
    if abs(step) >= 1:
        decimals = 0
    else:
        decimals = max(0, int(math.ceil(-math.log10(abs(step)))) + 1)
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text


def _tooltip_text(row: dict[str, float | str], x_name: str, y_name: str) -> str:
    x_value = row[x_name]
    y_value = row[y_name]
    return (
        f"rate_label: {row['rate_label']}\n"
        f"replicate: {row['replicate']}\n"
        f"{x_name}: {x_value:.6f}\n"
        f"{y_name}: {y_value:.6f}\n"
        f"minimap2_ani: {row['minimap2_ani']:.6f}\n"
        f"minimap2_dv: {row['minimap2_dv']:.6f}\n"
        f"minimap2_chunk_coverage: {row['minimap2_query_coverage']:.6f}\n"
        f"jaccard_similarity: {row['jaccard_similarity']:.6f}\n"
        f"ref_jaccard_similarity: {row['ref_jaccard_similarity']:.6f}"
    )


def _write_svg_scatter(rows: list[dict[str, float | str]], x_name: str, y_name: str, output_path: Path) -> None:
    width = 1100
    height = 820
    left = 130
    right = 60
    top = 70
    bottom = 120

    filtered_rows = [row for row in rows if not math.isnan(float(row[x_name])) and not math.isnan(float(row[y_name]))]
    x_values = [row[x_name] for row in filtered_rows]
    y_values = [row[y_name] for row in filtered_rows]
    if x_name == "true_mutation_rate" and y_name == "minimap2_ani":
        x_min, x_max, x_step = 0.0, 0.25, 0.05
    elif x_name == "true_mutation_rate":
        x_min, x_max, x_step = 0.0, 0.20, 0.05
    else:
        x_min, x_max, x_step = _nice_axis_bounds(min(x_values), max(x_values))
    if x_name == "true_mutation_rate" and y_name == "minimap2_ani":
        y_min, y_max, y_step = 0.75, 1.0, 0.05
    else:
        y_min, y_max, y_step = _nice_axis_bounds(min(y_values), max(y_values))

    plot_left = left
    plot_right = width - right
    plot_top = top
    plot_bottom = height - bottom

    points: list[str] = []
    for row in filtered_rows:
        cx = _scale_value(float(row[x_name]), x_min, x_max, plot_left, plot_right)
        cy = _scale_value(float(row[y_name]), y_min, y_max, plot_bottom, plot_top)
        tooltip = escape(_tooltip_text(row, x_name, y_name))
        points.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7" fill="#1f77b4" fill-opacity="0.8"><title>{tooltip}</title></circle>'
        )

    x_ticks: list[str] = []
    current_x = x_min
    while current_x <= x_max + (x_step / 10):
        x_pos = _scale_value(current_x, x_min, x_max, plot_left, plot_right)
        x_ticks.append(
            "\n".join(
                [
                    f'<line x1="{x_pos:.2f}" y1="{plot_bottom}" x2="{x_pos:.2f}" y2="{plot_top}" stroke="#d6d6d6" stroke-width="1" />',
                    f'<line x1="{x_pos:.2f}" y1="{plot_bottom}" x2="{x_pos:.2f}" y2="{plot_bottom + 12}" stroke="black" stroke-width="2" />',
                    f'<text x="{x_pos:.2f}" y="{plot_bottom + 38}" text-anchor="middle" font-family="sans-serif" font-size="20">{escape(_format_tick(current_x, x_step))}</text>',
                ]
            )
        )
        current_x += x_step

    y_ticks: list[str] = []
    current_y = y_min
    while current_y <= y_max + (y_step / 10):
        y_pos = _scale_value(current_y, y_min, y_max, plot_bottom, plot_top)
        y_ticks.append(
            "\n".join(
                [
                    f'<line x1="{plot_left}" y1="{y_pos:.2f}" x2="{plot_right}" y2="{y_pos:.2f}" stroke="#d6d6d6" stroke-width="1" />',
                    f'<line x1="{plot_left - 12}" y1="{y_pos:.2f}" x2="{plot_left}" y2="{y_pos:.2f}" stroke="black" stroke-width="2" />',
                    f'<text x="{plot_left - 18}" y="{y_pos + 7:.2f}" text-anchor="end" font-family="sans-serif" font-size="20">{escape(_format_tick(current_y, y_step))}</text>',
                ]
            )
        )
        current_y += y_step

    title = f"{y_name} vs {x_name}"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.0f}" y="40" text-anchor="middle" font-family="sans-serif" font-size="28" font-weight="600">{escape(title)}</text>',
        *x_ticks,
        *y_ticks,
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="black" stroke-width="2" />',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="black" stroke-width="2" />',
        f'<text x="{width / 2:.0f}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" font-size="24">{escape(x_name)}</text>',
        (
            f'<text x="34" y="{height / 2:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="24" transform="rotate(-90 34 {height / 2:.0f})">{escape(y_name)}</text>'
        ),
        *points,
        "</svg>",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_svg_summary(
    rows: list[dict[str, float | str]],
    x_name: str,
    y_name: str,
    output_path: Path,
) -> None:
    width = 1100
    height = 820
    left = 130
    right = 60
    top = 70
    bottom = 120

    grouped: dict[float, list[float]] = {}
    for row in rows:
        x_value = float(row[x_name])
        y_value = float(row[y_name])
        if math.isnan(x_value) or math.isnan(y_value):
            continue
        grouped.setdefault(x_value, []).append(y_value)

    summary_rows: list[dict[str, float]] = []
    for x_value in sorted(grouped):
        values = sorted(grouped[x_value])
        summary_rows.append(
            {
                x_name: x_value,
                "mean": sum(values) / len(values),
                "min": values[0],
                "max": values[-1],
            }
        )

    x_values = [row[x_name] for row in summary_rows]
    if x_name == "true_mutation_rate":
        x_min, x_max, x_step = 0.0, 0.20, 0.05
    else:
        x_min, x_max, x_step = _nice_axis_bounds(min(x_values), max(x_values))

    y_values = [value for values in grouped.values() for value in values]
    y_lower = min(y_values)
    y_upper = max(y_values)
    padding = max(0.0002, (y_upper - y_lower) * 0.3)
    y_min = max(0.0, y_lower - padding)
    y_max = min(1.0, y_upper + padding)
    if y_max - y_min < 0.001:
        midpoint = (y_min + y_max) / 2.0
        y_min = max(0.0, midpoint - 0.0008)
        y_max = min(1.0, midpoint + 0.0008)
    _tmp_min, _tmp_max, y_step = _nice_axis_bounds(y_min, y_max)
    y_min, y_max = _tmp_min, _tmp_max

    plot_left = left
    plot_right = width - right
    plot_top = top
    plot_bottom = height - bottom

    x_ticks: list[str] = []
    current_x = x_min
    while current_x <= x_max + (x_step / 10):
        x_pos = _scale_value(current_x, x_min, x_max, plot_left, plot_right)
        x_ticks.append(
            "\n".join(
                [
                    f'<line x1="{x_pos:.2f}" y1="{plot_bottom}" x2="{x_pos:.2f}" y2="{plot_top}" stroke="#d6d6d6" stroke-width="1" />',
                    f'<line x1="{x_pos:.2f}" y1="{plot_bottom}" x2="{x_pos:.2f}" y2="{plot_bottom + 12}" stroke="black" stroke-width="2" />',
                    f'<text x="{x_pos:.2f}" y="{plot_bottom + 38}" text-anchor="middle" font-family="sans-serif" font-size="20">{escape(_format_tick(current_x, x_step))}</text>',
                ]
            )
        )
        current_x += x_step

    y_ticks: list[str] = []
    current_y = y_min
    while current_y <= y_max + (y_step / 10):
        y_pos = _scale_value(current_y, y_min, y_max, plot_bottom, plot_top)
        y_ticks.append(
            "\n".join(
                [
                    f'<line x1="{plot_left}" y1="{y_pos:.2f}" x2="{plot_right}" y2="{y_pos:.2f}" stroke="#d6d6d6" stroke-width="1" />',
                    f'<line x1="{plot_left - 12}" y1="{y_pos:.2f}" x2="{plot_left}" y2="{y_pos:.2f}" stroke="black" stroke-width="2" />',
                    f'<text x="{plot_left - 18}" y="{y_pos + 7:.2f}" text-anchor="end" font-family="sans-serif" font-size="20">{escape(_format_tick(current_y, y_step))}</text>',
                ]
            )
        )
        current_y += y_step

    vertical_ranges: list[str] = []
    mean_points: list[str] = []
    polyline_points: list[str] = []
    for row in summary_rows:
        x_pos = _scale_value(row[x_name], x_min, x_max, plot_left, plot_right)
        y_min_pos = _scale_value(row["min"], y_min, y_max, plot_bottom, plot_top)
        y_max_pos = _scale_value(row["max"], y_min, y_max, plot_bottom, plot_top)
        y_mean_pos = _scale_value(row["mean"], y_min, y_max, plot_bottom, plot_top)
        summary_tooltip = escape(
            "\n".join(
                [
                    f"{x_name}: {row[x_name]:.6f}",
                    f"mean_{y_name}: {row['mean']:.6f}",
                    f"min_{y_name}: {row['min']:.6f}",
                    f"max_{y_name}: {row['max']:.6f}",
                ]
            )
        )
        vertical_ranges.append(
            f'<line x1="{x_pos:.2f}" y1="{y_min_pos:.2f}" x2="{x_pos:.2f}" y2="{y_max_pos:.2f}" stroke="#7a8aa0" stroke-width="4" />'
        )
        mean_points.append(
            (
                f'<circle cx="{x_pos:.2f}" cy="{y_mean_pos:.2f}" r="8" fill="#1f77b4">'
                f'<title>{summary_tooltip}</title>'
                f'</circle>'
            )
        )
        polyline_points.append(f"{x_pos:.2f},{y_mean_pos:.2f}")

    title = f"{y_name} vs {x_name}"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.0f}" y="40" text-anchor="middle" font-family="sans-serif" font-size="28" font-weight="600">{escape(title)}</text>',
        *x_ticks,
        *y_ticks,
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="black" stroke-width="2" />',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="black" stroke-width="2" />',
        f'<text x="{width / 2:.0f}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" font-size="24">{escape(x_name)}</text>',
        (
            f'<text x="34" y="{height / 2:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="24" transform="rotate(-90 34 {height / 2:.0f})">{escape(y_name)}</text>'
        ),
        f'<polyline points="{" ".join(polyline_points)}" fill="none" stroke="#1f77b4" stroke-width="3" />',
        *vertical_ranges,
        *mean_points,
        (
            f'<text x="{plot_right}" y="{plot_top + 24}" text-anchor="end" font-family="sans-serif" '
            f'font-size="18" fill="#4d5c70">Point = mean across observations; bar = min to max</text>'
        ),
        "</svg>",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_master_table(input_tsv: Path, output_dir: Path) -> list[Path]:
    rows = _load_master_table(input_tsv)
    reference = _reference_metrics(input_tsv)
    has_jaccard = any(not math.isnan(row["jaccard_similarity"]) for row in rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    outputs: list[Path] = []
    plot_specs = [
        ("true_mutation_rate", "minimap2_ani"),
        ("true_mutation_rate", "minimap2_dv"),
        ("true_mutation_rate", "minimap2_query_coverage"),
        ("true_mutation_rate", "ref_jaccard_similarity"),
        ("minimap2_ani", "ref_jaccard_similarity"),
        ("ref_jaccard_similarity", "true_mutation_rate"),
    ]
    if has_jaccard and reference["analysis_mode"] != "reference_chunks":
        plot_specs.extend(
            [
                ("true_mutation_rate", "jaccard_similarity"),
                ("minimap2_ani", "jaccard_similarity"),
                ("jaccard_similarity", "ref_jaccard_similarity"),
                ("jaccard_similarity", "true_mutation_rate"),
            ]
        )
    for x_name, y_name in plot_specs:
        use_summary_plot = y_name == "minimap2_query_coverage"
        if plt is not None and not use_summary_plot:
            figure, axis = plt.subplots(figsize=(6, 4))
            axis.scatter([row[x_name] for row in rows], [row[y_name] for row in rows], alpha=0.8)
            axis.set_xlabel(x_name)
            axis.set_ylabel(y_name)
            axis.set_title(f"{y_name} vs {x_name}")
            plot_path = output_dir / f"{x_name}_vs_{y_name}.png"
            figure.tight_layout()
            figure.savefig(plot_path, dpi=200)
            plt.close(figure)
        else:
            plot_path = output_dir / f"{x_name}_vs_{y_name}.svg"
            if use_summary_plot:
                _write_svg_summary(rows, x_name, y_name, plot_path)
            else:
                _write_svg_scatter(rows, x_name, y_name, plot_path)
        outputs.append(plot_path)
    return outputs


def write_html_report(
    input_tsv: Path,
    plot_paths: list[Path],
    correlations: list[dict[str, object]],
    regressions: list[dict[str, object]],
    output_path: Path,
) -> Path:
    rows = _load_master_table(input_tsv)
    full_rows = _load_full_master_rows(input_tsv)
    reference = _reference_metrics(input_tsv)
    observation_label = "chunk" if reference["analysis_mode"] == "reference_chunks" else "replicate"
    observation_label_title = observation_label.title()
    has_jaccard = any(not math.isnan(row["jaccard_similarity"]) for row in rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": len(rows),
        "min_rate": min(row["true_mutation_rate"] for row in rows),
        "max_rate": max(row["true_mutation_rate"] for row in rows),
        "min_ani": min(row["minimap2_ani"] for row in rows),
        "max_ani": max(row["minimap2_ani"] for row in rows),
        "min_dv": min(row["minimap2_dv"] for row in rows if not math.isnan(row["minimap2_dv"])),
        "max_dv": max(row["minimap2_dv"] for row in rows if not math.isnan(row["minimap2_dv"])),
        "min_jaccard": min((row["jaccard_similarity"] for row in rows if not math.isnan(row["jaccard_similarity"])), default=math.nan),
        "max_jaccard": max((row["jaccard_similarity"] for row in rows if not math.isnan(row["jaccard_similarity"])), default=math.nan),
        "min_ref_jaccard": min(row["ref_jaccard_similarity"] for row in rows),
        "max_ref_jaccard": max(row["ref_jaccard_similarity"] for row in rows),
        "k": full_rows[0]["k"],
    }

    plot_specs = [
        {
            "id": "plot-ani-vs-rate",
            "title": "minimap2_ani vs true_mutation_rate",
            "kind": "scatter",
            "x": "true_mutation_rate",
            "y": "minimap2_ani",
            "xDomain": [0.0, 0.25],
            "yDomain": [0.75, 1.0],
            "xLabel": "true_mutation_rate",
            "yLabel": "minimap2_ani",
            "download": "true_mutation_rate_vs_minimap2_ani.svg",
            "regressionKey": "true_mutation_rate|minimap2_ani",
            "legendPosition": "right",
        },
        {
            "id": "plot-dv-vs-rate",
            "title": "minimap2_dv vs true_mutation_rate",
            "kind": "scatter",
            "x": "true_mutation_rate",
            "y": "minimap2_dv",
            "xDomain": [0.0, 0.25],
            "yDomain": [0.0, 0.25],
            "xLabel": "true_mutation_rate",
            "yLabel": "minimap2_dv",
            "download": "true_mutation_rate_vs_minimap2_dv.svg",
            "regressionKey": "true_mutation_rate|minimap2_dv",
            "legendPosition": "right",
            "identityLine": True,
        },
        {
            "id": "plot-coverage-vs-rate",
            "title": "minimap2_chunk_coverage vs true_mutation_rate",
            "kind": "summary",
            "x": "true_mutation_rate",
            "y": "minimap2_query_coverage",
            "xDomain": [0.0, 0.20],
            "xLabel": "true_mutation_rate",
            "yLabel": "minimap2_chunk_coverage",
            "download": "true_mutation_rate_vs_minimap2_query_coverage.svg",
        },
        {
            "id": "plot-ref-jaccard-vs-rate",
            "title": "ref_jaccard_similarity vs true_mutation_rate",
            "kind": "scatter",
            "x": "true_mutation_rate",
            "y": "ref_jaccard_similarity",
            "xDomain": [0.0, 0.20],
            "yDomain": [0.0, 1.0],
            "xLabel": "true_mutation_rate",
            "yLabel": "ref_jaccard_similarity",
            "download": "true_mutation_rate_vs_ref_jaccard_similarity.svg",
            "regressionKey": "true_mutation_rate|ref_jaccard_similarity",
            "legendPosition": "right",
        },
        {
            "id": "plot-ref-jaccard-vs-ani",
            "title": "ref_jaccard_similarity vs minimap2_ani",
            "kind": "scatter",
            "x": "minimap2_ani",
            "y": "ref_jaccard_similarity",
            "xDomain": [0.75, 1.0],
            "yDomain": [0.0, 1.0],
            "xLabel": "minimap2_ani",
            "yLabel": "ref_jaccard_similarity",
            "download": "minimap2_ani_vs_ref_jaccard_similarity.svg",
            "regressionKey": "minimap2_ani|ref_jaccard_similarity",
            "legendPosition": "left",
        },
        {
            "id": "plot-divergence-vs-ref-jaccard",
            "title": "true_mutation_rate vs ref_jaccard_similarity",
            "kind": "scatter",
            "x": "ref_jaccard_similarity",
            "y": "true_mutation_rate",
            "xDomain": [0.0, 1.0],
            "yDomain": [0.0, 0.20],
            "xLabel": "ref_jaccard_similarity",
            "yLabel": "nucleotide_divergence",
            "download": "ref_jaccard_similarity_vs_true_mutation_rate.svg",
            "regressionKey": "ref_jaccard_similarity|true_mutation_rate",
            "legendPosition": "right",
        },
    ]
    if has_jaccard and reference["analysis_mode"] != "reference_chunks":
        plot_specs[3:3] = [
            {
                "id": "plot-jaccard-vs-rate",
                "title": "jaccard_similarity vs true_mutation_rate",
                "kind": "scatter",
                "x": "true_mutation_rate",
                "y": "jaccard_similarity",
                "xDomain": [0.0, 0.20],
                "yDomain": [0.0, 1.0],
                "xLabel": "true_mutation_rate",
                "yLabel": "jaccard_similarity",
                "download": "true_mutation_rate_vs_jaccard_similarity.svg",
                "regressionKey": "true_mutation_rate|jaccard_similarity",
                "legendPosition": "right",
            }
        ]
        plot_specs[5:5] = [
            {
                "id": "plot-jaccard-vs-ani",
                "title": "jaccard_similarity vs minimap2_ani",
                "kind": "scatter",
                "x": "minimap2_ani",
                "y": "jaccard_similarity",
                "xDomain": [0.75, 1.0],
                "yDomain": [0.0, 1.0],
                "xLabel": "minimap2_ani",
                "yLabel": "jaccard_similarity",
                "download": "minimap2_ani_vs_jaccard_similarity.svg",
                "regressionKey": "minimap2_ani|jaccard_similarity",
                "legendPosition": "left",
            },
            {
                "id": "plot-jaccard-vs-ref-jaccard",
                "title": "ref_jaccard_similarity vs jaccard_similarity",
                "kind": "scatter",
                "x": "jaccard_similarity",
                "y": "ref_jaccard_similarity",
                "xDomain": [0.0, 1.0],
                "yDomain": [0.0, 1.0],
                "xLabel": "jaccard_similarity",
                "yLabel": "ref_jaccard_similarity",
                "download": "jaccard_similarity_vs_ref_jaccard_similarity.svg",
                "legendPosition": "left",
                "identityLine": True,
                "theoryCurve": "ref_from_jaccard",
            },
            {
                "id": "plot-divergence-vs-jaccard",
                "title": "true_mutation_rate vs jaccard_similarity",
                "kind": "scatter",
                "x": "jaccard_similarity",
                "y": "true_mutation_rate",
                "xDomain": [0.0, 1.0],
                "yDomain": [0.0, 0.20],
                "xLabel": "jaccard_similarity",
                "yLabel": "nucleotide_divergence",
                "download": "jaccard_similarity_vs_true_mutation_rate.svg",
                "regressionKey": "jaccard_similarity|true_mutation_rate",
                "legendPosition": "right",
            },
        ]

    regression_map: dict[str, list[dict[str, object]]] = {}
    for row in regressions:
        regression_map.setdefault(f"{row['x_metric']}|{row['y_metric']}", []).append(row)

    plot_blocks: list[str] = []
    for spec in plot_specs:
        download_path = next((path.name for path in plot_paths if path.name == spec["download"]), spec["download"])
        regressions_for_plot = regression_map.get(spec.get("regressionKey", ""), [])
        fit_block = ""
        if regressions_for_plot:
            fit_lines = []
            for regression in regressions_for_plot:
                model_label = "kth-root" if regression["model_type"] in {"kth_root", "kth_root_divergence"} else str(regression["model_type"]).title()
                fit_lines.append(
                    f"<div><strong>{escape(model_label)} fit</strong> "
                    f"<span>{escape(str(regression['equation']))}</span> "
                    f"<span>corr={float(regression['fit_correlation']):.6f}</span> "
                    f"<span>R²={float(regression['r_squared']):.6f}</span></div>"
                )
            fit_block = '<div class="fit-note">' + "".join(fit_lines) + "</div>"
        plot_blocks.append(
            "\n".join(
                [
                    '<section class="plot-card">',
                    f'  <div class="plot-head"><h3>{escape(spec["title"])}</h3><a href="{escape(download_path)}">Open static export</a></div>',
                    f'  {fit_block}',
                    f'  <div class="chart-host" id="{escape(spec["id"])}"></div>',
                    "</section>",
                ]
            )
        )

    d3_rows: list[dict[str, object]] = []
    for row in full_rows:
        minimap2_ani = row["minimap2_ani"].strip()
        d3_rows.append(
            {
                "rate_label": row["rate_label"],
                "replicate": int(row["replicate"]),
                "reference_label": row.get("reference_label", ""),
                "chunk_start": int(row.get("chunk_start", "0") or 0),
                "chunk_end": int(row.get("chunk_end", "0") or 0),
                "true_mutation_rate": float(row["true_mutation_rate"]),
                "minimap2_ani": None if not minimap2_ani else float(minimap2_ani),
                "minimap2_dv": None if not row["minimap2_dv"].strip() else float(row["minimap2_dv"]),
                "minimap2_query_coverage": float(row["minimap2_query_coverage"]),
                "jaccard_similarity": None if not row["jaccard_similarity"].strip() else float(row["jaccard_similarity"]),
                "ref_jaccard_similarity": float(row["ref_jaccard_similarity"]),
                "minimap2_preset": row["minimap2_preset"],
            }
        )

    correlation_rows = "\n".join(
        [
            (
                "<tr>"
                f"<td>{escape(str(row['x_metric']))}</td>"
                f"<td>{escape(str(row['y_metric']))}</td>"
                f"<td>{'NA' if math.isnan(float(row['pearson_r'])) else format(float(row['pearson_r']), '.6f')}</td>"
                f"<td>{'NA' if math.isnan(float(row['spearman_rho'])) else format(float(row['spearman_rho']), '.6f')}</td>"
                "</tr>"
            )
            for row in correlations
        ]
    )

    table_rows = "\n".join(
        [
            (
                "<tr>"
                f"<td>{escape(row['rate_label'])}</td>"
                f"<td>{int(row['replicate'])}</td>"
                f"<td>{escape(row.get('reference_label', ''))}</td>"
                f"<td>{float(row['true_mutation_rate']):.6f}</td>"
                f"<td>{'NA' if not row['minimap2_ani'].strip() else format(float(row['minimap2_ani']), '.6f')}</td>"
                f"<td>{'N/A' if not row['jaccard_similarity'].strip() else format(float(row['jaccard_similarity']), '.6f')}</td>"
                f"<td>{float(row['ref_jaccard_similarity']):.6f}</td>"
                f"<td>{float(row['realized_substitution_rate']):.6f}</td>"
                f"<td>{float(row['realized_insertion_rate']):.6f}</td>"
                f"<td>{float(row['realized_deletion_rate']):.6f}</td>"
                "</tr>"
            )
            for row in full_rows
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>anicompare report</title>
  <style>
    :root {{
      --bg: #f7f4ed;
      --panel: #fffdf8;
      --ink: #1b1f23;
      --muted: #5a6470;
      --line: #d8d0bf;
      --accent: #0d6e6e;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f3efe4 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
      line-height: 1.15;
    }}
    p {{
      color: var(--muted);
      max-width: 75ch;
    }}
    .hero {{
      background: radial-gradient(circle at top left, #d8ece7 0%, var(--panel) 55%);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 24px;
      margin-bottom: 22px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .stat {{
      background: rgba(255,255,255,0.8);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .stat .label {{
      display: block;
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 6px;
    }}
    .stat .value {{
      font-size: 1.3rem;
      font-weight: 700;
    }}
    .section {{
      margin-top: 26px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }}
    .plot-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: white;
    }}
    .plot-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .plot-head a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 0.95rem;
    }}
    .plot-card h3 {{
      font-size: 1.5rem;
      margin-bottom: 0;
      overflow-wrap: anywhere;
    }}
    .chart-host {{
      position: relative;
      min-height: 760px;
    }}
    .fit-note {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .fit-note div {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .chart-host svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .d3-tooltip {{
      position: fixed;
      pointer-events: none;
      opacity: 0;
      background: rgba(20, 24, 28, 0.94);
      color: white;
      padding: 10px 12px;
      border-radius: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-line;
      box-shadow: 0 10px 30px rgba(0,0,0,0.18);
      z-index: 20;
      transition: opacity 120ms ease;
      max-width: 280px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f3efe4;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    code {{
      background: #f0ece1;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>anicompare report</h1>
      <p>This report summarizes the relationship between the true simulated mutation rate, ANI estimated from <code>minimap2</code> SAM alignments, and similarity estimated from k-mer conservation using Jaccard and ref-Jaccard coefficients.</p>
      <div class="stats">
        <div class="stat"><span class="label">Runs</span><span class="value">{summary['rows']}</span></div>
        <div class="stat"><span class="label">Mutation rate range</span><span class="value">{summary['min_rate']:.4f} to {summary['max_rate']:.4f}</span></div>
        <div class="stat"><span class="label">ANI range</span><span class="value">{summary['min_ani']:.4f} to {summary['max_ani']:.4f}</span></div>
        <div class="stat"><span class="label">DV range</span><span class="value">{summary['min_dv']:.4f} to {summary['max_dv']:.4f}</span></div>
        <div class="stat"><span class="label">Jaccard range</span><span class="value">{'N/A' if math.isnan(summary['min_jaccard']) or math.isnan(summary['max_jaccard']) else f'{summary["min_jaccard"]:.4f} to {summary["max_jaccard"]:.4f}'}</span></div>
        <div class="stat"><span class="label">ref-Jaccard range</span><span class="value">{summary['min_ref_jaccard']:.4f} to {summary['max_ref_jaccard']:.4f}</span></div>
        <div class="stat"><span class="label">k-mer length</span><span class="value">{summary['k']}</span></div>
        <div class="stat"><span class="label">Observation unit</span><span class="value">{escape(observation_label)}</span></div>
      </div>
    </section>

    <section class="section">
      <h2>Reference Genome</h2>
      <div class="stats">
        <div class="stat"><span class="label">Source</span><span class="value">{escape(str(reference['source']))}</span></div>
        <div class="stat"><span class="label">Source value</span><span class="value">{escape(str(reference['source_value']))}</span></div>
        <div class="stat"><span class="label">Contigs</span><span class="value">{reference['contigs']}</span></div>
        <div class="stat"><span class="label">Total length</span><span class="value">{reference['length']}</span></div>
        <div class="stat"><span class="label">GC fraction</span><span class="value">{reference['gc_fraction']:.4f}</span></div>
        <div class="stat"><span class="label">Analysis mode</span><span class="value">{escape(str(reference['analysis_mode']))}</span></div>
        <div class="stat"><span class="label">Chunk length</span><span class="value">{escape(str(reference['chunk_length']))}</span></div>
      </div>
      <p>Reference FASTA: <code>{escape(str(reference['reference_path']))}</code></p>
    </section>

    <section class="section">
      <h2>Plots</h2>
      <div class="plot-grid">
        {' '.join(plot_blocks)}
      </div>
    </section>

    <section class="section">
      <h2>Correlations</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>X metric</th>
              <th>Y metric</th>
              <th>Pearson r</th>
              <th>Spearman rho</th>
            </tr>
          </thead>
          <tbody>
            {correlation_rows}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>Run Table</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rate label</th>
              <th>{observation_label_title}</th>
              <th>Reference label</th>
              <th>True rate</th>
              <th>minimap2 ANI</th>
              <th>Jaccard</th>
              <th>ref-Jaccard</th>
              <th>Realized substitutions</th>
              <th>Realized insertions</th>
              <th>Realized deletions</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </div>
    </section>
  </main>
  <div class="d3-tooltip" id="chart-tooltip"></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script>
    const reportData = {json.dumps(d3_rows)};
    const plotSpecs = {json.dumps(plot_specs)};
    const regressionMap = {json.dumps(regression_map)};
    const observationLabel = {json.dumps(observation_label)};

    const tooltip = d3.select("#chart-tooltip");

    function formatValue(value) {{
      if (value === null || Number.isNaN(value)) return "NA";
      return d3.format(".6f")(value);
    }}

    function showTooltip(event, lines) {{
      tooltip
        .style("opacity", 1)
        .style("left", `${{event.clientX + 16}}px`)
        .style("top", `${{event.clientY + 16}}px`)
        .text(lines.join("\\n"));
    }}

    function hideTooltip() {{
      tooltip.style("opacity", 0);
    }}

    function drawScatter(spec) {{
      const host = d3.select(`#${{spec.id}}`);
      const width = 1060;
      const height = 760;
      const margin = {{ top: 60, right: 40, bottom: 90, left: 120 }};
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const data = reportData.filter(d => d[spec.x] !== null && d[spec.y] !== null);

      const svg = host.append("svg")
        .attr("viewBox", `0 0 ${{width}} ${{height}}`);

      const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
      const x = d3.scaleLinear().domain(spec.xDomain).range([0, innerWidth]);
      const y = d3.scaleLinear().domain(spec.yDomain).range([innerHeight, 0]);

      g.append("g")
        .attr("transform", `translate(0,${{innerHeight}})`)
        .call(d3.axisBottom(x).tickFormat(d3.format(".2~f")));

      g.append("g")
        .call(d3.axisLeft(y).tickFormat(d3.format(".2~f")));

      g.append("g")
        .attr("class", "grid")
        .attr("transform", `translate(0,${{innerHeight}})`)
        .call(d3.axisBottom(x).tickSize(-innerHeight).tickFormat(""));

      g.append("g")
        .attr("class", "grid")
        .call(d3.axisLeft(y).tickSize(-innerWidth).tickFormat(""));

      g.selectAll(".grid line").attr("stroke", "#d6d6d6");
      g.selectAll(".grid path").remove();

      g.selectAll("circle.point")
        .data(data)
        .enter()
        .append("circle")
        .attr("class", "point")
        .attr("cx", d => x(d[spec.x]))
        .attr("cy", d => y(d[spec.y]))
        .attr("r", 7)
        .attr("fill", "#1f77b4")
        .attr("fill-opacity", 0.85)
        .on("mousemove", (event, d) => showTooltip(event, [
          `rate_label: ${{d.rate_label}}`,
          `${{observationLabel}}: ${{d.replicate}}`,
          `reference_label: ${{d.reference_label || "NA"}}`,
          `chunk_start: ${{d.chunk_start || "NA"}}`,
          `chunk_end: ${{d.chunk_end || "NA"}}`,
          `${{spec.x}}: ${{formatValue(d[spec.x])}}`,
          `${{spec.y}}: ${{formatValue(d[spec.y])}}`,
          `minimap2_ani: ${{formatValue(d.minimap2_ani)}}`,
          `minimap2_dv: ${{formatValue(d.minimap2_dv)}}`,
          `minimap2_chunk_coverage: ${{formatValue(d.minimap2_query_coverage)}}`,
          `jaccard_similarity: ${{formatValue(d.jaccard_similarity)}}`,
          `ref_jaccard_similarity: ${{formatValue(d.ref_jaccard_similarity)}}`,
          `minimap2_preset: ${{d.minimap2_preset}}`
        ]))
        .on("mouseleave", hideTooltip);

      if (spec.identityLine) {{
        const identityPoints = [
          {{ x: Math.max(spec.xDomain[0], spec.yDomain[0]), y: Math.max(spec.xDomain[0], spec.yDomain[0]) }},
          {{ x: Math.min(spec.xDomain[1], spec.yDomain[1]), y: Math.min(spec.xDomain[1], spec.yDomain[1]) }}
        ];
        const identityLine = d3.line()
          .x(d => x(d.x))
          .y(d => y(d.y));
        g.append("path")
          .datum(identityPoints)
          .attr("fill", "none")
          .attr("stroke", "#6c757d")
          .attr("stroke-width", 3)
          .attr("stroke-dasharray", "8,6")
          .attr("d", identityLine);
      }}

      if (spec.theoryCurve === "ref_from_jaccard") {{
        const theoryPoints = d3.range(spec.xDomain[0], spec.xDomain[1] + 0.0001, (spec.xDomain[1] - spec.xDomain[0]) / 200)
          .map(xValue => ({{ x: xValue, y: (2 * xValue) / (1 + xValue) }}))
          .filter(d => Number.isFinite(d.y));
        const theoryLine = d3.line()
          .x(d => x(d.x))
          .y(d => y(d.y));
        g.append("path")
          .datum(theoryPoints)
          .attr("fill", "none")
          .attr("stroke", "#f4a261")
          .attr("stroke-width", 4)
          .attr("stroke-dasharray", "12,7")
          .attr("d", theoryLine);
      }}

      const regressions = regressionMap[spec.regressionKey] || [];
      const colors = {{ linear: "#d1495b", quadratic: "#2a9d8f", kth_root: "#8f5bd6", kth_root_divergence: "#8f5bd6" }};
      const dashes = {{ linear: "10,6", quadratic: "14,7", kth_root: "5,5", kth_root_divergence: "5,5" }};
      for (const regression of regressions) {{
        const linePoints = d3.range(spec.xDomain[0], spec.xDomain[1] + 0.0001, (spec.xDomain[1] - spec.xDomain[0]) / 200)
          .map(xValue => {{
            let yValue = null;
            if (regression.model_type === "linear") {{
              yValue = regression.coefficients.intercept + regression.coefficients.slope * xValue;
            }} else if (regression.model_type === "quadratic") {{
              yValue = regression.coefficients.a * xValue * xValue + regression.coefficients.b * xValue + regression.coefficients.c;
            }} else if (regression.model_type === "kth_root") {{
              const rooted = regression.coefficients.intercept + regression.coefficients.slope * xValue;
              yValue = Math.pow(Math.max(0, Math.min(1, rooted)), regression.coefficients.k);
            }} else if (regression.model_type === "kth_root_divergence") {{
              yValue = regression.coefficients.intercept + regression.coefficients.slope * Math.pow(Math.max(0, xValue), 1 / regression.coefficients.k);
            }}
            return {{ x: xValue, y: yValue }};
          }})
          .filter(d => d.y !== null && Number.isFinite(d.y) && d.y >= spec.yDomain[0] - 1 && d.y <= spec.yDomain[1] + 1);

        const fitLine = d3.line()
          .x(d => x(d.x))
          .y(d => y(d.y));

        g.append("path")
          .datum(linePoints)
          .attr("fill", "none")
          .attr("stroke", colors[regression.model_type] || "#d1495b")
          .attr("stroke-width", 4)
          .attr("stroke-dasharray", dashes[regression.model_type] || "10,6")
          .attr("d", fitLine);
      }}

      const legendEntries = [
        ...(spec.identityLine ? [{{ label: "identity (y=x)", color: "#6c757d", dash: "8,6" }}] : []),
        ...(spec.theoryCurve === "ref_from_jaccard" ? [{{ label: "r=2J/(1+J)", color: "#f4a261", dash: "12,7" }}] : []),
        ...regressions.map(regression => {{
          const label = regression.model_type === "kth_root"
            ? "kth-root"
            : regression.model_type.charAt(0).toUpperCase() + regression.model_type.slice(1);
          return {{
            label,
            color: colors[regression.model_type] || "#d1495b",
            dash: dashes[regression.model_type] || "10,6",
          }};
        }})
      ];

      if (legendEntries.length > 0) {{
        const legendX = spec.legendPosition === "left" ? 132 : width - 280;
        const legend = svg.append("g").attr("transform", `translate(${{legendX}},${{88}})`);
        legend.append("rect")
          .attr("width", 240)
          .attr("height", 34 + legendEntries.length * 28)
          .attr("rx", 12)
          .attr("fill", "rgba(255,255,255,0.92)")
          .attr("stroke", "#d8d0bf");
        legend.append("text")
          .attr("x", 16)
          .attr("y", 24)
          .attr("font-size", 18)
          .attr("font-weight", 700)
          .text("Fits");
        legendEntries.forEach((entry, index) => {{
          const yPos = 46 + index * 28;
          legend.append("line")
            .attr("x1", 16)
            .attr("x2", 68)
            .attr("y1", yPos)
            .attr("y2", yPos)
            .attr("stroke", entry.color)
            .attr("stroke-width", 4)
            .attr("stroke-dasharray", entry.dash);
          legend.append("text")
            .attr("x", 80)
            .attr("y", yPos + 5)
            .attr("font-size", 16)
            .text(entry.label);
        }});
      }}

      svg.append("text")
        .attr("x", width / 2)
        .attr("y", 32)
        .attr("text-anchor", "middle")
        .attr("font-size", 26)
        .attr("font-weight", 700)
        .text(spec.title);

      svg.append("text")
        .attr("x", width / 2)
        .attr("y", height - 18)
        .attr("text-anchor", "middle")
        .attr("font-size", 22)
        .text(spec.xLabel);

      svg.append("text")
        .attr("transform", `translate(26,${{height / 2}}) rotate(-90)`)
        .attr("text-anchor", "middle")
        .attr("font-size", 22)
        .text(spec.yLabel);
    }}

    function drawCoverageSummary(spec) {{
      const host = d3.select(`#${{spec.id}}`);
      const width = 1060;
      const height = 760;
      const margin = {{ top: 60, right: 40, bottom: 90, left: 120 }};
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;

      const grouped = d3.rollups(
        reportData,
        values => ({{
          mean: d3.mean(values, d => d[spec.y]),
          min: d3.min(values, d => d[spec.y]),
          max: d3.max(values, d => d[spec.y]),
          count: values.length
        }}),
        d => d[spec.x]
      ).map(([x, stats]) => ({{ x: +x, ...stats }})).sort((a, b) => a.x - b.x);

      const yMin = Math.max(0, d3.min(grouped, d => d.min) - 0.02);
      const yMax = 1.0;

      const svg = host.append("svg")
        .attr("viewBox", `0 0 ${{width}} ${{height}}`);

      const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
      const x = d3.scaleLinear().domain(spec.xDomain).range([0, innerWidth]);
      const y = d3.scaleLinear().domain([yMin, yMax]).range([innerHeight, 0]);

      g.append("g")
        .attr("transform", `translate(0,${{innerHeight}})`)
        .call(d3.axisBottom(x).tickFormat(d3.format(".2~f")));

      g.append("g")
        .call(d3.axisLeft(y).tickFormat(d3.format(".2~f")));

      g.append("g")
        .attr("class", "grid")
        .attr("transform", `translate(0,${{innerHeight}})`)
        .call(d3.axisBottom(x).tickSize(-innerHeight).tickFormat(""));

      g.append("g")
        .attr("class", "grid")
        .call(d3.axisLeft(y).tickSize(-innerWidth).tickFormat(""));

      g.selectAll(".grid line").attr("stroke", "#d6d6d6");
      g.selectAll(".grid path").remove();

      const line = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.mean));

      g.append("path")
        .datum(grouped)
        .attr("fill", "none")
        .attr("stroke", "#1f77b4")
        .attr("stroke-width", 4)
        .attr("d", line);

      g.selectAll("line.range")
        .data(grouped)
        .enter()
        .append("line")
        .attr("x1", d => x(d.x))
        .attr("x2", d => x(d.x))
        .attr("y1", d => y(d.min))
        .attr("y2", d => y(d.max))
        .attr("stroke", "#7a8aa0")
        .attr("stroke-width", 5);

      g.selectAll("circle.mean")
        .data(grouped)
        .enter()
        .append("circle")
        .attr("cx", d => x(d.x))
        .attr("cy", d => y(d.mean))
        .attr("r", 9)
        .attr("fill", "#1f77b4")
        .on("mousemove", (event, d) => showTooltip(event, [
          `${{spec.x}}: ${{formatValue(d.x)}}`,
          `mean_${{spec.y}}: ${{formatValue(d.mean)}}`,
          `min_${{spec.y}}: ${{formatValue(d.min)}}`,
          `max_${{spec.y}}: ${{formatValue(d.max)}}`,
          `${{observationLabel}}s: ${{d.count}}`
        ]))
        .on("mouseleave", hideTooltip);

      svg.append("text")
        .attr("x", width / 2)
        .attr("y", 32)
        .attr("text-anchor", "middle")
        .attr("font-size", 26)
        .attr("font-weight", 700)
        .text(spec.title);

      svg.append("text")
        .attr("x", width / 2)
        .attr("y", height - 18)
        .attr("text-anchor", "middle")
        .attr("font-size", 22)
        .text(spec.xLabel);

      svg.append("text")
        .attr("transform", `translate(26,${{height / 2}}) rotate(-90)`)
        .attr("text-anchor", "middle")
        .attr("font-size", 22)
        .text(spec.yLabel);
    }}

    for (const spec of plotSpecs) {{
      if (spec.kind === "summary") {{
        drawCoverageSummary(spec);
      }} else {{
        drawScatter(spec);
      }}
    }}
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
