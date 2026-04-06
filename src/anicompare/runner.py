"""High-level experiment orchestration."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .io import FastaRecord, read_fasta, read_json, write_fasta, write_json, write_tsv
from .kmers import exact_kmer_set, jaccard_similarity
from .minimap2_ani import choose_minimap2_preset, parse_sam_for_ani, run_minimap2
from .modimizers import modimizer_set
from .mutate import MutationRates, mutate_records
from .random_genome import generate_random_record


DEFAULT_MUTATION_RATES = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15]
MASTER_FIELDS = [
    "rate_label",
    "replicate",
    "true_mutation_rate",
    "substitution_rate",
    "insertion_rate",
    "deletion_rate",
    "realized_substitution_rate",
    "realized_insertion_rate",
    "realized_deletion_rate",
    "minimap2_preset",
    "minimap2_ani",
    "minimap2_aligned_bases",
    "minimap2_edit_distance",
    "minimap2_query_bases",
    "minimap2_query_coverage",
    "minimap2_mapped_records",
    "jaccard_similarity",
    "comparison_mode",
    "k",
]


@dataclass(frozen=True)
class SketchConfig:
    mode: str = "exact"
    modulus: int = 100


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path
    reference_fasta: Path | None
    simulate_length: int | None
    replicates: int
    mutation_rates: tuple[float, ...]
    substitution_scale: float
    insertion_scale: float
    deletion_scale: float
    k: int
    sketch_mode: str
    modimizer_modulus: int
    workers: int
    minimap2_threads: int
    minimap2_executable: str
    minimap2_preset: str
    gc_content: float
    seed: int | None
    force: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "reference_fasta": str(self.reference_fasta) if self.reference_fasta else None,
            "simulate_length": self.simulate_length,
            "replicates": self.replicates,
            "mutation_rates": list(self.mutation_rates),
            "substitution_scale": self.substitution_scale,
            "insertion_scale": self.insertion_scale,
            "deletion_scale": self.deletion_scale,
            "k": self.k,
            "sketch_mode": self.sketch_mode,
            "modimizer_modulus": self.modimizer_modulus,
            "workers": self.workers,
            "minimap2_threads": self.minimap2_threads,
            "minimap2_executable": self.minimap2_executable,
            "minimap2_preset": self.minimap2_preset,
            "gc_content": self.gc_content,
            "seed": self.seed,
            "config_version": 1,
        }


def _rate_label(rate: float) -> str:
    return f"{rate * 100:g}pct".replace(".", "p")


def _replicate_dir(output_dir: Path, rate: float, replicate: int) -> Path:
    return output_dir / f"rate_{_rate_label(rate)}" / f"replicate_{replicate:02d}"


def _reference_dir(output_dir: Path) -> Path:
    return output_dir / "reference"


def _reference_path(output_dir: Path) -> Path:
    return _reference_dir(output_dir) / "reference.fa"


def _ensure_reference(config: ExperimentConfig) -> Path:
    ref_dir = _reference_dir(config.output_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = _reference_path(config.output_dir)
    if ref_path.exists():
        return ref_path
    if config.reference_fasta is not None:
        shutil.copy2(config.reference_fasta, ref_path)
        return ref_path
    if config.simulate_length is None:
        raise ValueError("Either reference_fasta or simulate_length must be set")
    record = generate_random_record(
        length=config.simulate_length,
        gc_content=config.gc_content,
        seed=config.seed,
        name="reference",
    )
    write_fasta([record], ref_path)
    return ref_path


def _comparison_set(records: list[FastaRecord], k: int, sketch_config: SketchConfig) -> set[str]:
    if sketch_config.mode == "exact":
        return exact_kmer_set(records, k=k)
    if sketch_config.mode == "modimizer":
        return modimizer_set(records, k=k, modulus=sketch_config.modulus)
    raise ValueError(f"Unsupported sketch mode: {sketch_config.mode}")


def _write_log(path: Path, lines: Iterable[str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line.rstrip("\n") + "\n")


def _run_single_job(job: dict[str, object]) -> dict[str, object]:
    replicate_dir = Path(str(job["replicate_dir"]))
    replicate_dir.mkdir(parents=True, exist_ok=True)

    reference_path = Path(str(job["reference_path"]))
    mutated_path = replicate_dir / "mutated.fa"
    metadata_path = replicate_dir / "mutation_metadata.json"
    sam_path = replicate_dir / "minimap2.sam"
    minimap_metrics_path = replicate_dir / "minimap2_metrics.tsv"
    jaccard_metrics_path = replicate_dir / "jaccard_metrics.tsv"
    summary_path = replicate_dir / "run_summary.tsv"
    log_path = replicate_dir / "run.log"

    if not (replicate_dir / "reference.fa").exists():
        os.symlink(os.path.relpath(reference_path, start=replicate_dir), replicate_dir / "reference.fa")

    rate = float(job["rate"])
    substitution_scale = float(job["substitution_scale"])
    insertion_scale = float(job["insertion_scale"])
    deletion_scale = float(job["deletion_scale"])
    rates = MutationRates(
        substitution_rate=rate * substitution_scale,
        insertion_rate=rate * insertion_scale,
        deletion_rate=rate * deletion_scale,
    )

    if not mutated_path.exists() or not metadata_path.exists():
        reference_records = read_fasta(reference_path)
        mutated_records, metadata = mutate_records(reference_records, rates, seed=int(job["seed"]))
        write_fasta(mutated_records, mutated_path)
        write_json(metadata, metadata_path)
        _write_log(
            log_path,
            [
                f"Generated mutated genome for rate {rate}",
                f"Requested rates: sub={rates.substitution_rate}, ins={rates.insertion_rate}, del={rates.deletion_rate}",
            ],
        )
    else:
        metadata = read_json(metadata_path)
        _write_log(log_path, [f"Reused existing mutated genome for rate {rate}"])

    if not sam_path.exists():
        preset = str(job["minimap2_preset"])
        if preset == "auto":
            preset = choose_minimap2_preset(float(job["rate"]))
        run_minimap2(
            reference_fasta=reference_path,
            query_fasta=mutated_path,
            sam_path=sam_path,
            executable=str(job["minimap2_executable"]),
            threads=int(job["minimap2_threads"]),
            preset=preset,
        )
        _write_log(log_path, ["Completed minimap2 alignment"])
    else:
        preset = str(job["minimap2_preset"])
        if preset == "auto":
            preset = choose_minimap2_preset(float(job["rate"]))

    sam_metrics = parse_sam_for_ani(sam_path)
    mutated_records = read_fasta(mutated_path)
    total_query_bases = sum(len(record.sequence) for record in mutated_records)
    query_coverage = sam_metrics.aligned_bases / total_query_bases if total_query_bases else 0.0
    write_tsv(
        [
            {
                "mapped_records": sam_metrics.mapped_records,
                "aligned_bases": sam_metrics.aligned_bases,
                "edit_distance": sam_metrics.edit_distance,
                "query_bases": sam_metrics.query_bases,
                "query_coverage": query_coverage,
                "ani": "" if sam_metrics.ani is None else sam_metrics.ani,
            }
        ],
        minimap_metrics_path,
        fieldnames=["mapped_records", "aligned_bases", "edit_distance", "query_bases", "query_coverage", "ani"],
    )

    reference_records = read_fasta(reference_path)
    sketch_config = SketchConfig(mode=str(job["sketch_mode"]), modulus=int(job["modimizer_modulus"]))
    reference_set = _comparison_set(reference_records, k=int(job["k"]), sketch_config=sketch_config)
    mutated_set = _comparison_set(mutated_records, k=int(job["k"]), sketch_config=sketch_config)
    jaccard = jaccard_similarity(reference_set, mutated_set)

    write_tsv(
        [
            {
                "comparison_mode": sketch_config.mode,
                "k": int(job["k"]),
                "modimizer_modulus": int(job["modimizer_modulus"]),
                "reference_features": len(reference_set),
                "query_features": len(mutated_set),
                "jaccard_similarity": jaccard,
            }
        ],
        jaccard_metrics_path,
        fieldnames=["comparison_mode", "k", "modimizer_modulus", "reference_features", "query_features", "jaccard_similarity"],
    )

    metadata_summary = dict(metadata["summary"])
    summary_row = {
        "rate_label": str(job["rate_label"]),
        "replicate": int(job["replicate"]),
        "true_mutation_rate": rate,
        "substitution_rate": rates.substitution_rate,
        "insertion_rate": rates.insertion_rate,
        "deletion_rate": rates.deletion_rate,
        "realized_substitution_rate": metadata_summary["realized_substitution_rate"],
        "realized_insertion_rate": metadata_summary["realized_insertion_rate"],
        "realized_deletion_rate": metadata_summary["realized_deletion_rate"],
        "minimap2_preset": preset,
        "minimap2_ani": "" if sam_metrics.ani is None else sam_metrics.ani,
        "minimap2_aligned_bases": sam_metrics.aligned_bases,
        "minimap2_edit_distance": sam_metrics.edit_distance,
        "minimap2_query_bases": total_query_bases,
        "minimap2_query_coverage": query_coverage,
        "minimap2_mapped_records": sam_metrics.mapped_records,
        "jaccard_similarity": jaccard,
        "comparison_mode": sketch_config.mode,
        "k": int(job["k"]),
    }
    write_tsv([summary_row], summary_path, fieldnames=MASTER_FIELDS)
    _write_log(log_path, [f"Computed jaccard={jaccard:.6f}", f"Computed minimap2 ANI={sam_metrics.ani:.6f}"])
    return summary_row


def _write_config(config: ExperimentConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.output_dir / "run_config.json"
    existing = read_json(config_path) if config_path.exists() else None
    current = config.as_dict()
    if existing is not None and existing != current and not config.force:
        raise ValueError(
            f"Existing configuration at {config_path} does not match current parameters. "
            "Use a new output directory or rerun with force enabled."
        )
    write_json(current, config_path)


def run_experiment(config: ExperimentConfig) -> Path:
    _write_config(config)
    reference_path = _ensure_reference(config)

    jobs: list[dict[str, object]] = []
    seed_base = config.seed if config.seed is not None else 0
    for rate_index, rate in enumerate(config.mutation_rates):
        for replicate in range(1, config.replicates + 1):
            replicate_dir = _replicate_dir(config.output_dir, rate, replicate)
            jobs.append(
                {
                    "replicate_dir": str(replicate_dir),
                    "reference_path": str(reference_path),
                    "rate": rate,
                    "rate_label": _rate_label(rate),
                    "replicate": replicate,
                    "substitution_scale": config.substitution_scale,
                    "insertion_scale": config.insertion_scale,
                    "deletion_scale": config.deletion_scale,
                    "k": config.k,
                    "sketch_mode": config.sketch_mode,
                    "modimizer_modulus": config.modimizer_modulus,
                    "minimap2_threads": config.minimap2_threads,
                    "minimap2_executable": config.minimap2_executable,
                    "minimap2_preset": config.minimap2_preset,
                    "seed": seed_base + (rate_index * 1000) + replicate,
                }
            )

    if config.workers <= 1:
        results = [_run_single_job(job) for job in jobs]
    else:
        try:
            with ProcessPoolExecutor(max_workers=config.workers) as executor:
                results = list(executor.map(_run_single_job, jobs))
        except PermissionError:
            with ThreadPoolExecutor(max_workers=config.workers) as executor:
                results = list(executor.map(_run_single_job, jobs))

    master_path = config.output_dir / "master_results.tsv"
    results.sort(key=lambda row: (float(row["true_mutation_rate"]), int(row["replicate"])))
    write_tsv(results, master_path, fieldnames=MASTER_FIELDS)
    return master_path
