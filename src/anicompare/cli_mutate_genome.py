"""CLI for mutating FASTA genomes."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import read_fasta, write_fasta, write_json
from .mutate import MutationRates, mutate_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply mutations to a FASTA genome.")
    parser.add_argument("--input-fasta", type=Path, required=True, help="Input FASTA.")
    parser.add_argument("--output", type=Path, required=True, help="Output FASTA.")
    parser.add_argument("--metadata-output", type=Path, default=None, help="Optional JSON metadata output.")
    parser.add_argument("--substitution-rate", type=float, default=0.0)
    parser.add_argument("--insertion-rate", type=float, default=0.0)
    parser.add_argument("--deletion-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = read_fasta(args.input_fasta)
    mutated_records, metadata = mutate_records(
        records,
        MutationRates(
            substitution_rate=args.substitution_rate,
            insertion_rate=args.insertion_rate,
            deletion_rate=args.deletion_rate,
        ),
        seed=args.seed,
    )
    write_fasta(mutated_records, args.output)
    metadata_output = args.metadata_output or args.output.with_suffix(args.output.suffix + ".metadata.json")
    write_json(metadata, metadata_output)


if __name__ == "__main__":
    main()
