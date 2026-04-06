"""CLI for the end-to-end experiment runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import DEFAULT_MUTATION_RATES, ExperimentConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ANI vs Jaccard comparison experiments.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--reference-fasta", type=Path, help="Reference FASTA path.")
    source_group.add_argument("--simulate-length", type=int, help="Simulate a random reference genome of this length.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Results directory.")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--mutation-rates", type=float, nargs="+", default=DEFAULT_MUTATION_RATES)
    parser.add_argument("--substitution-scale", type=float, default=1.0)
    parser.add_argument("--insertion-scale", type=float, default=0.0)
    parser.add_argument("--deletion-scale", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--sketch-mode", choices=["exact", "modimizer"], default="exact")
    parser.add_argument("--modimizer-modulus", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--minimap2-threads", type=int, default=1)
    parser.add_argument("--minimap2-executable", default="minimap2")
    parser.add_argument("--minimap2-preset", default="asm5")
    parser.add_argument("--gc-content", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Allow overwriting a mismatched run config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        output_dir=args.output_dir,
        reference_fasta=args.reference_fasta,
        simulate_length=args.simulate_length,
        replicates=args.replicates,
        mutation_rates=tuple(args.mutation_rates),
        substitution_scale=args.substitution_scale,
        insertion_scale=args.insertion_scale,
        deletion_scale=args.deletion_scale,
        k=args.k,
        sketch_mode=args.sketch_mode,
        modimizer_modulus=args.modimizer_modulus,
        workers=args.workers,
        minimap2_threads=args.minimap2_threads,
        minimap2_executable=args.minimap2_executable,
        minimap2_preset=args.minimap2_preset,
        gc_content=args.gc_content,
        seed=args.seed,
        force=args.force,
    )
    master_path = run_experiment(config)
    print(master_path)


if __name__ == "__main__":
    main()
