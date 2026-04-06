"""Plotting and correlation utilities."""

from __future__ import annotations

import math
from html import escape
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
        minimap2_ani = row["minimap2_ani"].strip()
        rows.append(
            {
                "true_mutation_rate": float(row["true_mutation_rate"]),
                "minimap2_ani": float(minimap2_ani) if minimap2_ani else math.nan,
                "jaccard_similarity": float(row["jaccard_similarity"]),
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
        ("true_mutation_rate", "jaccard_similarity"),
        ("minimap2_ani", "jaccard_similarity"),
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


def _write_svg_scatter(rows: list[dict[str, float]], x_name: str, y_name: str, output_path: Path) -> None:
    width = 1100
    height = 820
    left = 130
    right = 60
    top = 70
    bottom = 120

    filtered_rows = [row for row in rows if not math.isnan(row[x_name]) and not math.isnan(row[y_name])]
    x_values = [row[x_name] for row in filtered_rows]
    y_values = [row[y_name] for row in filtered_rows]
    if x_name == "true_mutation_rate":
        x_min, x_max, x_step = 0.0, 0.20, 0.05
    else:
        x_min, x_max, x_step = _nice_axis_bounds(min(x_values), max(x_values))
    y_min, y_max, y_step = _nice_axis_bounds(min(y_values), max(y_values))

    plot_left = left
    plot_right = width - right
    plot_top = top
    plot_bottom = height - bottom

    points: list[str] = []
    for row in filtered_rows:
        cx = _scale_value(row[x_name], x_min, x_max, plot_left, plot_right)
        cy = _scale_value(row[y_name], y_min, y_max, plot_bottom, plot_top)
        points.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7" fill="#1f77b4" fill-opacity="0.8" />')

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


def plot_master_table(input_tsv: Path, output_dir: Path) -> list[Path]:
    rows = _load_master_table(input_tsv)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    outputs: list[Path] = []
    plot_specs = [
        ("true_mutation_rate", "minimap2_ani"),
        ("true_mutation_rate", "jaccard_similarity"),
        ("minimap2_ani", "jaccard_similarity"),
    ]
    for x_name, y_name in plot_specs:
        if plt is not None:
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
            _write_svg_scatter(rows, x_name, y_name, plot_path)
        outputs.append(plot_path)
    return outputs


def write_html_report(
    input_tsv: Path,
    plot_paths: list[Path],
    correlations: list[dict[str, object]],
    output_path: Path,
) -> Path:
    rows = _load_master_table(input_tsv)
    full_rows = _load_full_master_rows(input_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": len(rows),
        "min_rate": min(row["true_mutation_rate"] for row in rows),
        "max_rate": max(row["true_mutation_rate"] for row in rows),
        "min_ani": min(row["minimap2_ani"] for row in rows),
        "max_ani": max(row["minimap2_ani"] for row in rows),
        "min_jaccard": min(row["jaccard_similarity"] for row in rows),
        "max_jaccard": max(row["jaccard_similarity"] for row in rows),
    }

    plot_blocks: list[str] = []
    for plot_path in plot_paths:
        relative = plot_path.relative_to(output_path.parent)
        plot_blocks.append(
            "\n".join(
                [
                    '<section class="plot-card">',
                    f'  <h3>{escape(plot_path.stem)}</h3>',
                    f'  <img src="{escape(str(relative))}" alt="{escape(plot_path.stem)}" />',
                    "</section>",
                ]
            )
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
                f"<td>{float(row['true_mutation_rate']):.6f}</td>"
                f"<td>{'NA' if not row['minimap2_ani'].strip() else format(float(row['minimap2_ani']), '.6f')}</td>"
                f"<td>{float(row['jaccard_similarity']):.6f}</td>"
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
    .plot-card img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
      background: #fff;
    }}
    .plot-card h3 {{
      font-size: 1.5rem;
      margin-bottom: 14px;
      overflow-wrap: anywhere;
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
      <p>This report summarizes the relationship between the true simulated mutation rate, ANI estimated from <code>minimap2</code> SAM alignments, and similarity estimated from k-mer conservation using a Jaccard coefficient.</p>
      <div class="stats">
        <div class="stat"><span class="label">Runs</span><span class="value">{summary['rows']}</span></div>
        <div class="stat"><span class="label">Mutation rate range</span><span class="value">{summary['min_rate']:.4f} to {summary['max_rate']:.4f}</span></div>
        <div class="stat"><span class="label">ANI range</span><span class="value">{summary['min_ani']:.4f} to {summary['max_ani']:.4f}</span></div>
        <div class="stat"><span class="label">Jaccard range</span><span class="value">{summary['min_jaccard']:.4f} to {summary['max_jaccard']:.4f}</span></div>
      </div>
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
              <th>Replicate</th>
              <th>True rate</th>
              <th>minimap2 ANI</th>
              <th>Jaccard</th>
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
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
