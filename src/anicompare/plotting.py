"""Plotting and correlation utilities."""

from __future__ import annotations

import math
from pathlib import Path

from .io import read_tsv, write_tsv


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


def _load_master_table(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in read_tsv(path):
        rows.append(
            {
                "true_mutation_rate": float(row["true_mutation_rate"]),
                "minimap2_ani": float(row["minimap2_ani"]),
                "jaccard_similarity": float(row["jaccard_similarity"]),
            }
        )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def compute_correlations(input_tsv: Path) -> list[dict[str, object]]:
    rows = _load_master_table(input_tsv)
    comparisons = [
        ("true_mutation_rate", "minimap2_ani"),
        ("true_mutation_rate", "jaccard_similarity"),
        ("minimap2_ani", "jaccard_similarity"),
    ]
    output: list[dict[str, object]] = []
    for x_name, y_name in comparisons:
        values_x = [row[x_name] for row in rows]
        values_y = [row[y_name] for row in rows]
        output.append(
            {
                "x_metric": x_name,
                "y_metric": y_name,
                "pearson_r": _pearson(values_x, values_y),
                "spearman_rho": _spearman(values_x, values_y),
            }
        )
    return output


def write_correlations(input_tsv: Path, output_tsv: Path) -> list[dict[str, object]]:
    correlations = compute_correlations(input_tsv)
    write_tsv(correlations, output_tsv, fieldnames=["x_metric", "y_metric", "pearson_r", "spearman_rho"])
    return correlations


def plot_master_table(input_tsv: Path, output_dir: Path) -> list[Path]:
    rows = _load_master_table(input_tsv)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    outputs: list[Path] = []
    plot_specs = [
        ("true_mutation_rate", "minimap2_ani"),
        ("true_mutation_rate", "jaccard_similarity"),
        ("minimap2_ani", "jaccard_similarity"),
    ]
    for x_name, y_name in plot_specs:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.scatter([row[x_name] for row in rows], [row[y_name] for row in rows], alpha=0.8)
        axis.set_xlabel(x_name)
        axis.set_ylabel(y_name)
        axis.set_title(f"{y_name} vs {x_name}")
        plot_path = output_dir / f"{x_name}_vs_{y_name}.png"
        figure.tight_layout()
        figure.savefig(plot_path, dpi=200)
        plt.close(figure)
        outputs.append(plot_path)
    return outputs
