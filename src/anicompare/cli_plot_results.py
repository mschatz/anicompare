"""CLI for plotting and correlation summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from .plotting import plot_master_table, write_correlations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot and summarize experiment results.")
    parser.add_argument("--input", type=Path, required=True, help="Master results TSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and summaries.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = plot_master_table(args.input, args.output_dir)
    correlation_path = args.output_dir / "correlations.tsv"
    write_correlations(args.input, correlation_path)
    for plot_path in plot_paths:
        print(plot_path)
    print(correlation_path)


if __name__ == "__main__":
    main()
