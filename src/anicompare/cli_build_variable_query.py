"""CLI for building a variable-rate chunk-mutated query genome."""

from __future__ import annotations

import argparse
from pathlib import Path

from .variable_query import build_variable_chunk_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a query genome with per-chunk variable substitution rates.")
    parser.add_argument("--reference-fasta", type=Path, required=True, help="Reference FASTA path.")
    parser.add_argument("--output-fasta", type=Path, required=True, help="Output FASTA path for the mutated query genome.")
    parser.add_argument("--metadata-json", type=Path, required=True, help="Output metadata JSON path.")
    parser.add_argument("--chunk-length", type=int, default=10000, help="Chunk length used for per-chunk rate sampling.")
    parser.add_argument("--min-rate", type=float, default=0.0, help="Minimum substitution rate sampled per chunk.")
    parser.add_argument("--max-rate", type=float, default=0.2, help="Maximum substitution rate sampled per chunk.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = build_variable_chunk_query(
        args.reference_fasta,
        args.output_fasta,
        args.metadata_json,
        chunk_length=args.chunk_length,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
        seed=args.seed,
    )
    print(args.output_fasta)
    print(args.metadata_json)
    print(
        "realized_substitution_rate="
        f"{float(metadata['summary']['realized_substitution_rate']):.6f}"
    )


if __name__ == "__main__":
    main()
