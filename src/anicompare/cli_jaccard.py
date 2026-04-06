"""CLI for genome similarity based on k-mers or modimizers."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import read_fasta, write_tsv
from .kmers import exact_kmer_set, jaccard_similarity
from .modimizers import modimizer_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute genome similarity from FASTA inputs.")
    parser.add_argument("--fasta-a", type=Path, required=True)
    parser.add_argument("--fasta-b", type=Path, required=True)
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--mode", choices=["exact", "modimizer"], default="exact")
    parser.add_argument("--modimizer-modulus", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None, help="Optional TSV output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records_a = read_fasta(args.fasta_a)
    records_b = read_fasta(args.fasta_b)

    if args.mode == "exact":
        set_a = exact_kmer_set(records_a, k=args.k)
        set_b = exact_kmer_set(records_b, k=args.k)
    else:
        set_a = modimizer_set(records_a, k=args.k, modulus=args.modimizer_modulus)
        set_b = modimizer_set(records_b, k=args.k, modulus=args.modimizer_modulus)

    score = jaccard_similarity(set_a, set_b)
    row = {
        "comparison_mode": args.mode,
        "k": args.k,
        "modimizer_modulus": args.modimizer_modulus,
        "reference_features": len(set_a),
        "query_features": len(set_b),
        "jaccard_similarity": score,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_tsv([row], args.output, fieldnames=list(row.keys()))
    else:
        for key, value in row.items():
            print(f"{key}\t{value}")


if __name__ == "__main__":
    main()
