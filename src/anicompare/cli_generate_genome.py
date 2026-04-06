"""CLI for random genome generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import write_fasta
from .random_genome import generate_random_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a random genome FASTA.")
    parser.add_argument("--length", type=int, required=True, help="Genome length in bases.")
    parser.add_argument("--output", type=Path, required=True, help="Output FASTA path.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--gc-content", type=float, default=0.5, help="Target GC content.")
    parser.add_argument("--name", default="random_genome", help="FASTA header name.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = generate_random_record(
        length=args.length,
        gc_content=args.gc_content,
        seed=args.seed,
        name=args.name,
    )
    write_fasta([record], args.output)


if __name__ == "__main__":
    main()
