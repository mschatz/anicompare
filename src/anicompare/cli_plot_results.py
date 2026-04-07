"""CLI for plotting and correlation summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from .plotting import compute_regressions, plot_master_table, write_correlations, write_html_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot and summarize experiment results.")
    parser.add_argument("--input", type=Path, required=True, help="Master results TSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and summaries.")
    parser.add_argument("--report-mode", choices=["full", "observed"], default="full", help="Use 'observed' for real-genome reports that hide true mutation rate plots.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = plot_master_table(args.input, args.output_dir, report_mode=args.report_mode)
    correlation_path = args.output_dir / "correlations.tsv"
    correlations = write_correlations(args.input, correlation_path, report_mode=args.report_mode)
    regressions = compute_regressions(args.input, report_mode=args.report_mode)
    report_path = write_html_report(args.input, plot_paths, correlations, regressions, args.output_dir / "report.html", report_mode=args.report_mode)
    for plot_path in plot_paths:
        print(plot_path)
    print(correlation_path)
    print(report_path)


if __name__ == "__main__":
    main()
