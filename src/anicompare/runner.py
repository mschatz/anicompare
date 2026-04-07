"""High-level experiment orchestration."""

from __future__ import annotations

import gzip
import os
import shutil
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .io import FastaRecord, read_fasta, read_json, read_tsv, write_fasta, write_json, write_tsv
from .kmers import exact_kmer_set, jaccard_similarity, reference_jaccard_similarity
from .minimap2_ani import SamMetrics, choose_minimap2_preset, parse_sam_for_ani, run_minimap2
from .modimizers import modimizer_set
from .mutate import MutationRates, mutate_records
from .variable_query import build_variable_chunk_query
from .random_genome import generate_random_record


DEFAULT_MUTATION_RATES = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15]
MASTER_FIELDS = [
    "rate_label",
    "replicate",
    "reference_label",
    "chunk_start",
    "chunk_end",
    "true_mutation_rate",
    "substitution_rate",
    "insertion_rate",
    "deletion_rate",
    "realized_substitution_rate",
    "realized_insertion_rate",
    "realized_deletion_rate",
    "minimap2_preset",
    "minimap2_ani",
    "minimap2_dv",
    "minimap2_dv_source",
    "minimap2_aligned_bases",
    "minimap2_edit_distance",
    "minimap2_query_bases",
    "minimap2_query_coverage",
    "minimap2_mapped_records",
    "jaccard_similarity",
    "ref_jaccard_similarity",
    "comparison_mode",
    "k",
    "timing_mutation_setup_seconds",
    "timing_minimap2_alignment_seconds",
    "timing_minimap2_metrics_seconds",
    "timing_ref_jaccard_seconds",
    "timing_summary_write_seconds",
    "timing_total_job_seconds",
]
TIMING_FIELDS = [
    "timing_mutation_setup_seconds",
    "timing_minimap2_alignment_seconds",
    "timing_minimap2_metrics_seconds",
    "timing_ref_jaccard_seconds",
    "timing_summary_write_seconds",
    "timing_total_job_seconds",
]


@dataclass(frozen=True)
class SketchConfig:
    mode: str = "exact"
    modulus: int = 100


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path
    reference_fasta: Path | None
    query_fasta: Path | None
    simulate_length: int | None
    replicates: int
    analysis_mode: str
    chunk_length: int
    variable_chunk_mutation: bool
    variable_chunk_min_rate: float
    variable_chunk_max_rate: float
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
            "query_fasta": str(self.query_fasta) if self.query_fasta else None,
            "simulate_length": self.simulate_length,
            "replicates": self.replicates,
            "analysis_mode": self.analysis_mode,
            "chunk_length": self.chunk_length,
            "variable_chunk_mutation": self.variable_chunk_mutation,
            "variable_chunk_min_rate": self.variable_chunk_min_rate,
            "variable_chunk_max_rate": self.variable_chunk_max_rate,
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
            "config_version": 3,
        }


def _rate_label(rate: float) -> str:
    return f"{rate * 100:g}pct".replace(".", "p")


def _replicate_dir(output_dir: Path, rate: float, replicate: int) -> Path:
    return output_dir / f"rate_{_rate_label(rate)}" / f"replicate_{replicate:02d}"


def _reference_dir(output_dir: Path) -> Path:
    return output_dir / "reference"


def _reference_path(output_dir: Path) -> Path:
    return _reference_dir(output_dir) / "reference.fa"


def _reference_chunk_dir(output_dir: Path) -> Path:
    return _reference_dir(output_dir) / "chunks"


def _query_dir(output_dir: Path) -> Path:
    return output_dir / "query"


def _query_path(output_dir: Path) -> Path:
    return _query_dir(output_dir) / "query.fa"


def _query_metadata_path(output_dir: Path) -> Path:
    return _query_dir(output_dir) / "query_metadata.json"


def _chunk_manifest_path(output_dir: Path) -> Path:
    return _reference_dir(output_dir) / "chunk_manifest.tsv"


def _rate_dir(output_dir: Path, rate: float) -> Path:
    return output_dir / f"rate_{_rate_label(rate)}"


def _ensure_reference(config: ExperimentConfig) -> Path:
    ref_dir = _reference_dir(config.output_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = _reference_path(config.output_dir)
    if ref_path.exists():
        return ref_path
    if config.reference_fasta is not None:
        reference_records = read_fasta(config.reference_fasta)
        write_fasta(reference_records, ref_path)
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


def _chunk_records(records: list[FastaRecord], chunk_length: int) -> list[dict[str, object]]:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    chunks: list[dict[str, object]] = []
    chunk_index = 1
    for record in records:
        sequence = record.sequence
        for start in range(0, len(sequence), chunk_length):
            end = min(start + chunk_length, len(sequence))
            chunk_header = f"{record.header}|chunk_{chunk_index:04d}|{start + 1}-{end}"
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "record_header": record.header,
                    "chunk_header": chunk_header,
                    "chunk_start": start + 1,
                    "chunk_end": end,
                    "record": FastaRecord(header=chunk_header, sequence=sequence[start:end]),
                }
            )
            chunk_index += 1
    return chunks


def _ensure_reference_chunks(config: ExperimentConfig, reference_path: Path) -> list[dict[str, object]]:
    chunk_dir = _reference_chunk_dir(config.output_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    reference_records = read_fasta(reference_path)
    chunks = _chunk_records(reference_records, config.chunk_length)
    for chunk in chunks:
        chunk_path = chunk_dir / f"chunk_{int(chunk['chunk_index']):04d}.fa"
        if not chunk_path.exists():
            write_fasta([chunk["record"]], chunk_path)
        chunk["chunk_path"] = chunk_path
        chunk["chunk_length"] = len(chunk["record"].sequence)
    write_tsv(
        [
            {
                "chunk_index": int(chunk["chunk_index"]),
                "record_header": str(chunk["record_header"]),
                "chunk_header": str(chunk["chunk_header"]),
                "chunk_start": int(chunk["chunk_start"]),
                "chunk_end": int(chunk["chunk_end"]),
                "chunk_length": int(chunk["chunk_length"]),
                "chunk_path": str(chunk["chunk_path"]),
            }
            for chunk in chunks
        ],
        _chunk_manifest_path(config.output_dir),
        fieldnames=[
            "chunk_index",
            "record_header",
            "chunk_header",
            "chunk_start",
            "chunk_end",
            "chunk_length",
            "chunk_path",
        ],
    )
    return chunks


def _ensure_rate_mutation(
    config: ExperimentConfig,
    reference_path: Path,
    rate: float,
    seed: int,
) -> dict[str, object]:
    rate_dir = _rate_dir(config.output_dir, rate)
    rate_dir.mkdir(parents=True, exist_ok=True)
    mutated_path = rate_dir / "mutated.fa"
    metadata_path = rate_dir / "mutation_metadata.json"
    log_path = rate_dir / "mutation.log"

    rates = MutationRates(
        substitution_rate=rate * config.substitution_scale,
        insertion_rate=rate * config.insertion_scale,
        deletion_rate=rate * config.deletion_scale,
    )

    if not mutated_path.exists() or not metadata_path.exists():
        reference_records = read_fasta(reference_path)
        mutated_records, metadata = mutate_records(reference_records, rates, seed=seed)
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
    return {
        "mutated_path": mutated_path,
        "metadata_path": metadata_path,
    }


def _ensure_query_input(config: ExperimentConfig) -> dict[str, object] | None:
    if config.query_fasta is None:
        return None
    query_dir = _query_dir(config.output_dir)
    query_dir.mkdir(parents=True, exist_ok=True)
    query_path = _query_path(config.output_dir)
    metadata_path = _query_metadata_path(config.output_dir)
    if not query_path.exists():
        query_records = read_fasta(config.query_fasta)
        write_fasta(query_records, query_path)
    if not metadata_path.exists():
        write_json(
            {
                "query_source": str(config.query_fasta),
                "summary": {
                    "realized_substitution_rate": 0.0,
                    "realized_insertion_rate": 0.0,
                    "realized_deletion_rate": 0.0,
                },
            },
            metadata_path,
        )
    return {"mutated_path": query_path, "metadata_path": metadata_path}


def _ensure_variable_chunk_query(
    config: ExperimentConfig,
    reference_path: Path,
) -> dict[str, object] | None:
    if not config.variable_chunk_mutation:
        return None
    if config.analysis_mode != "reference_chunks":
        raise ValueError("variable_chunk_mutation requires analysis_mode=reference_chunks")

    query_dir = _query_dir(config.output_dir)
    query_dir.mkdir(parents=True, exist_ok=True)
    query_path = _query_path(config.output_dir)
    metadata_path = _query_metadata_path(config.output_dir)
    if query_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        return {"mutated_path": query_path, "metadata_path": metadata_path, "metadata": metadata}

    metadata = build_variable_chunk_query(
        reference_path,
        query_path,
        metadata_path,
        chunk_length=config.chunk_length,
        min_rate=config.variable_chunk_min_rate,
        max_rate=config.variable_chunk_max_rate,
        seed=config.seed,
    )
    return {"mutated_path": query_path, "metadata_path": metadata_path, "metadata": metadata}


def _sam_gz_path(replicate_dir: Path) -> Path:
    return replicate_dir / "minimap2.sam.gz"


def _legacy_sam_path(replicate_dir: Path) -> Path:
    return replicate_dir / "minimap2.sam"


def _compress_sam_file(source_path: Path, target_path: Path) -> Path:
    with source_path.open("rb") as source_handle, gzip.open(target_path, "wb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
    source_path.unlink()
    return target_path


def _ensure_compressed_sam(replicate_dir: Path) -> Path | None:
    sam_gz_path = _sam_gz_path(replicate_dir)
    legacy_sam_path = _legacy_sam_path(replicate_dir)
    if sam_gz_path.exists():
        return sam_gz_path
    if legacy_sam_path.exists():
        return _compress_sam_file(legacy_sam_path, sam_gz_path)
    return None


def _read_single_tsv_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    rows = list(read_tsv(path))
    return rows[0] if rows else None


def _parse_optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _effective_expected_divergence(job: dict[str, object]) -> float:
    override = job.get("true_mutation_rate_override")
    if override is not None:
        return float(override)
    return float(job["rate"])


def _read_cached_summary(summary_path: Path) -> dict[str, object] | None:
    cached = _read_single_tsv_row(summary_path)
    if cached is None:
        return None
    summary: dict[str, object] = {}
    for field in MASTER_FIELDS:
        value = cached.get(field, "")
        if field in {
            "replicate",
            "chunk_start",
            "chunk_end",
            "minimap2_aligned_bases",
            "minimap2_edit_distance",
            "minimap2_query_bases",
            "minimap2_mapped_records",
            "k",
        }:
            summary[field] = int(value) if value.strip() else 0
        elif field in {
            "true_mutation_rate",
            "substitution_rate",
            "insertion_rate",
            "deletion_rate",
            "realized_substitution_rate",
            "realized_insertion_rate",
            "realized_deletion_rate",
            "minimap2_ani",
            "minimap2_dv",
            "minimap2_query_coverage",
            "jaccard_similarity",
            "ref_jaccard_similarity",
            "timing_mutation_setup_seconds",
            "timing_minimap2_alignment_seconds",
            "timing_minimap2_metrics_seconds",
            "timing_ref_jaccard_seconds",
            "timing_summary_write_seconds",
            "timing_total_job_seconds",
        }:
            summary[field] = value if not value.strip() else float(value)
        else:
            summary[field] = value
    return summary


def _timing_summary_path(output_dir: Path) -> Path:
    return output_dir / "timing_summary.tsv"


def _aggregate_timing_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregates: list[dict[str, object]] = []
    for field in TIMING_FIELDS:
        values = [float(row.get(field, 0.0) or 0.0) for row in results]
        if not values:
            continue
        aggregates.append(
            {
                "metric": field,
                "sum_seconds": sum(values),
                "mean_seconds": sum(values) / len(values),
                "max_seconds": max(values),
                "job_count": len(values),
            }
        )
    return aggregates


def _print_timing_summary(results: list[dict[str, object]]) -> None:
    aggregates = _aggregate_timing_rows(results)
    if not aggregates:
        return
    metric_parts = []
    for row in aggregates:
        metric_name = str(row["metric"]).removeprefix("timing_").removesuffix("_seconds")
        metric_parts.append(f"{metric_name}={float(row['sum_seconds']):.2f}s")
    print(f"[timing] {' '.join(metric_parts)}", flush=True)


def _write_restart_script(config: ExperimentConfig) -> Path:
    root_dir = Path(__file__).resolve().parents[2]
    script_path = config.output_dir / "restart.sh"
    lines = [
        "#!/usr/bin/env bash",
        "",
        "set -euo pipefail",
        "",
        f'ROOT_DIR="{root_dir}"',
        'export PYTHONPATH="$ROOT_DIR/src"',
        "",
        "python3 -m anicompare.cli_run_experiment \\",
    ]
    if config.reference_fasta is not None:
        lines.append(f'  --reference-fasta "{config.reference_fasta}" \\')
    else:
        lines.append(f"  --simulate-length {config.simulate_length} \\")
    if config.query_fasta is not None:
        lines.append(f'  --query-fasta "{config.query_fasta}" \\')
    run_lines = [
        f'  --output-dir "{config.output_dir}" \\',
        f"  --replicates {config.replicates} \\",
        f"  --analysis-mode {config.analysis_mode} \\",
        f"  --chunk-length {config.chunk_length} \\",
        f"  --variable-chunk-min-rate {config.variable_chunk_min_rate} \\",
        f"  --variable-chunk-max-rate {config.variable_chunk_max_rate} \\",
        "  --mutation-rates " + " ".join(str(rate) for rate in config.mutation_rates) + " \\",
        f"  --substitution-scale {config.substitution_scale} \\",
        f"  --insertion-scale {config.insertion_scale} \\",
        f"  --deletion-scale {config.deletion_scale} \\",
        f"  --k {config.k} \\",
        f"  --sketch-mode {config.sketch_mode} \\",
        f"  --modimizer-modulus {config.modimizer_modulus} \\",
        f"  --workers {config.workers} \\",
        f"  --minimap2-threads {config.minimap2_threads} \\",
        f'  --minimap2-executable "{config.minimap2_executable}" \\',
        f"  --minimap2-preset {config.minimap2_preset} \\",
        f"  --gc-content {config.gc_content} \\",
    ]
    if config.seed is not None:
        run_lines.append(f"  --seed {config.seed} \\")
    if config.variable_chunk_mutation:
        run_lines.append("  --variable-chunk-mutation \\")
    run_lines.append("  --force")
    lines.extend(
        run_lines
        + [
            "",
            "python3 -m anicompare.cli_plot_results \\",
            f'  --input "{config.output_dir / "master_results.tsv"}" \\',
            f'  --output-dir "{config.output_dir / "plots"}"',
            "",
        ]
    )
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _job_status(job: dict[str, object]) -> tuple[bool, bool, bool, bool]:
    replicate_dir = Path(str(job["replicate_dir"]))
    sam_present = _ensure_compressed_sam(replicate_dir) is not None
    metrics_present = (replicate_dir / "minimap2_metrics.tsv").exists()
    comparison_present = (replicate_dir / "jaccard_metrics.tsv").exists()
    summary_present = (replicate_dir / "run_summary.tsv").exists()
    return sam_present, metrics_present, comparison_present, comparison_present and summary_present


def _print_status_snapshot(jobs: list[dict[str, object]]) -> None:
    total = len(jobs)
    complete = 0
    sam_ready = 0
    metrics_ready = 0
    comparison_ready = 0
    for job in jobs:
        sam_present, metrics_present, comparison_present, summary_present = _job_status(job)
        sam_ready += int(sam_present)
        metrics_ready += int(metrics_present)
        comparison_ready += int(comparison_present)
        complete += int(summary_present)
    pending = total - complete
    print(
        f"[status] total_jobs={total} completed={complete} pending={pending} "
        f"sam_ready={sam_ready} metrics_ready={metrics_ready} ref_jaccard_ready={comparison_ready}",
        flush=True,
    )


def _write_summary_row(summary_path: Path, summary_row: dict[str, object]) -> None:
    write_tsv([summary_row], summary_path, fieldnames=MASTER_FIELDS)


def _compute_chunked_ref_jaccard_for_rate(rate_jobs: list[dict[str, object]]) -> None:
    if not rate_jobs:
        return

    sketch_config = SketchConfig(
        mode=str(rate_jobs[0]["sketch_mode"]),
        modulus=int(rate_jobs[0]["modimizer_modulus"]),
    )
    k = int(rate_jobs[0]["k"])
    mutated_source = Path(str(rate_jobs[0]["mutated_source_path"]))
    build_start = time.perf_counter()
    mutated_records = read_fasta(mutated_source)
    mutated_set = _comparison_set(mutated_records, k=k, sketch_config=sketch_config)
    shared_build_seconds = time.perf_counter() - build_start
    shared_per_job_seconds = shared_build_seconds / len(rate_jobs)

    completed = 0
    for job in rate_jobs:
        replicate_dir = Path(str(job["replicate_dir"]))
        jaccard_metrics_path = replicate_dir / "jaccard_metrics.tsv"
        summary_path = replicate_dir / "run_summary.tsv"
        log_path = replicate_dir / "run.log"

        cached_comparison = _read_single_tsv_row(jaccard_metrics_path)
        summary_row = _read_cached_summary(summary_path)
        per_job_ref_seconds = shared_per_job_seconds

        if cached_comparison is None:
            ref_start = time.perf_counter()
            reference_records = read_fasta(Path(str(job["reference_path"])))
            reference_set = _comparison_set(reference_records, k=k, sketch_config=sketch_config)
            ref_jaccard = reference_jaccard_similarity(reference_set, mutated_set)
            per_job_ref_seconds += time.perf_counter() - ref_start
            write_tsv(
                [
                    {
                        "comparison_mode": sketch_config.mode,
                        "k": k,
                        "modimizer_modulus": int(job["modimizer_modulus"]),
                        "reference_features": len(reference_set),
                        "query_features": len(mutated_set),
                        "jaccard_similarity": "",
                        "ref_jaccard_similarity": ref_jaccard,
                    }
                ],
                jaccard_metrics_path,
                fieldnames=[
                    "comparison_mode",
                    "k",
                    "modimizer_modulus",
                    "reference_features",
                    "query_features",
                    "jaccard_similarity",
                    "ref_jaccard_similarity",
                ],
            )
            _write_log(log_path, [f"Computed ref-jaccard metrics in per-rate batch pass ({per_job_ref_seconds:.3f}s)"])
        else:
            ref_jaccard = float(cached_comparison["ref_jaccard_similarity"])
            _write_log(log_path, [f"Reused existing ref-jaccard metrics TSV in per-rate batch pass (+{shared_per_job_seconds:.3f}s shared setup)"])

        if summary_row is None:
            raise ValueError(f"Expected summary TSV before ref-jaccard batch pass: {summary_path}")
        summary_row["jaccard_similarity"] = ""
        summary_row["ref_jaccard_similarity"] = ref_jaccard
        summary_row["timing_ref_jaccard_seconds"] = per_job_ref_seconds
        _write_summary_row(summary_path, summary_row)
        completed += 1
        print(
            f"[ref-jaccard] completed={completed}/{len(rate_jobs)} rate={job['rate_label']} chunk={job['replicate']}",
            flush=True,
        )


def _run_single_job(job: dict[str, object]) -> dict[str, object]:
    job_start = time.perf_counter()
    mutation_setup_seconds = 0.0
    minimap2_alignment_seconds = 0.0
    minimap2_metrics_seconds = 0.0
    ref_jaccard_seconds = 0.0
    summary_write_seconds = 0.0
    replicate_dir = Path(str(job["replicate_dir"]))
    replicate_dir.mkdir(parents=True, exist_ok=True)

    reference_path = Path(str(job["reference_path"]))
    mutated_source_path = Path(str(job["mutated_source_path"])) if job.get("mutated_source_path") else None
    metadata_source_path = Path(str(job["metadata_source_path"])) if job.get("metadata_source_path") else None
    mutated_path = replicate_dir / "mutated.fa"
    metadata_path = replicate_dir / "mutation_metadata.json"
    sam_path = _sam_gz_path(replicate_dir)
    minimap_metrics_path = replicate_dir / "minimap2_metrics.tsv"
    jaccard_metrics_path = replicate_dir / "jaccard_metrics.tsv"
    summary_path = replicate_dir / "run_summary.tsv"
    log_path = replicate_dir / "run.log"

    cached_summary = _read_cached_summary(summary_path)
    if cached_summary is not None:
        _write_log(log_path, ["Reused existing run summary"])
        return cached_summary

    if not (replicate_dir / "reference.fa").exists():
        os.symlink(os.path.relpath(reference_path, start=replicate_dir), replicate_dir / "reference.fa")

    rate = float(job["rate"])
    true_mutation_rate_override = job.get("true_mutation_rate_override")
    realized_substitution_rate_override = job.get("realized_substitution_rate_override")
    realized_insertion_rate_override = job.get("realized_insertion_rate_override")
    realized_deletion_rate_override = job.get("realized_deletion_rate_override")
    true_mutation_rate = rate if true_mutation_rate_override is None else float(true_mutation_rate_override)
    substitution_scale = float(job["substitution_scale"])
    insertion_scale = float(job["insertion_scale"])
    deletion_scale = float(job["deletion_scale"])
    rates = MutationRates(
        substitution_rate=rate * substitution_scale,
        insertion_rate=rate * insertion_scale,
        deletion_rate=rate * deletion_scale,
    )

    if mutated_source_path is not None and metadata_source_path is not None:
        mutation_setup_start = time.perf_counter()
        if not mutated_path.exists():
            os.symlink(os.path.relpath(mutated_source_path, start=replicate_dir), mutated_path)
        if not metadata_path.exists():
            os.symlink(os.path.relpath(metadata_source_path, start=replicate_dir), metadata_path)
        metadata = read_json(metadata_source_path)
        _write_log(log_path, [f"Reused shared mutated genome for rate {rate}"])
        mutation_setup_seconds += time.perf_counter() - mutation_setup_start
    elif not mutated_path.exists() or not metadata_path.exists():
        mutation_setup_start = time.perf_counter()
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
        mutation_setup_seconds += time.perf_counter() - mutation_setup_start
    else:
        mutation_setup_start = time.perf_counter()
        metadata = read_json(metadata_path)
        _write_log(log_path, [f"Reused existing mutated genome for rate {rate}"])
        mutation_setup_seconds += time.perf_counter() - mutation_setup_start

    sam_source_path = _ensure_compressed_sam(replicate_dir)
    if sam_source_path is None:
        preset = str(job["minimap2_preset"])
        if preset == "auto":
            preset = choose_minimap2_preset(_effective_expected_divergence(job))
        legacy_sam_path = _legacy_sam_path(replicate_dir)
        minimap2_alignment_start = time.perf_counter()
        run_minimap2(
            reference_fasta=reference_path,
            query_fasta=mutated_path,
            sam_path=legacy_sam_path,
            executable=str(job["minimap2_executable"]),
            threads=int(job["minimap2_threads"]),
            preset=preset,
        )
        sam_source_path = _compress_sam_file(legacy_sam_path, sam_path)
        minimap2_alignment_seconds += time.perf_counter() - minimap2_alignment_start
        _write_log(log_path, ["Completed minimap2 alignment"])
    else:
        preset = str(job["minimap2_preset"])
        if preset == "auto":
            preset = choose_minimap2_preset(_effective_expected_divergence(job))
        _write_log(log_path, ["Reused existing compressed SAM"])

    cached_metrics = _read_single_tsv_row(minimap_metrics_path)
    mutated_records = read_fasta(mutated_path)
    total_query_bases = sum(len(record.sequence) for record in mutated_records)
    reference_records = read_fasta(reference_path)
    total_reference_bases = sum(len(record.sequence) for record in reference_records)
    if cached_metrics is None:
        minimap2_metrics_start = time.perf_counter()
        sam_metrics = parse_sam_for_ani(sam_source_path)
        query_coverage = sam_metrics.reference_covered_bases / total_reference_bases if total_reference_bases else 0.0
        write_tsv(
            [
                {
                    "mapped_records": sam_metrics.mapped_records,
                    "aligned_bases": sam_metrics.aligned_bases,
                    "edit_distance": sam_metrics.edit_distance,
                    "query_bases": sam_metrics.query_bases,
                    "query_coverage": query_coverage,
                    "ani": "" if sam_metrics.ani is None else sam_metrics.ani,
                    "dv": "" if sam_metrics.divergence_estimate is None else sam_metrics.divergence_estimate,
                    "dv_source": "" if sam_metrics.divergence_source is None else sam_metrics.divergence_source,
                }
            ],
            minimap_metrics_path,
            fieldnames=[
                "mapped_records",
                "aligned_bases",
                "edit_distance",
                "query_bases",
                "query_coverage",
                "ani",
                "dv",
                "dv_source",
            ],
        )
        minimap2_metrics_seconds += time.perf_counter() - minimap2_metrics_start
        _write_log(log_path, ["Computed minimap2 metrics"])
    else:
        minimap2_metrics_start = time.perf_counter()
        query_coverage = float(cached_metrics["query_coverage"]) if cached_metrics["query_coverage"].strip() else 0.0
        sam_metrics = SamMetrics(
            mapped_records=int(cached_metrics["mapped_records"]),
            aligned_bases=int(cached_metrics["aligned_bases"]),
            edit_distance=int(cached_metrics["edit_distance"]),
            query_bases=int(cached_metrics["query_bases"]),
            reference_covered_bases=int(round(query_coverage * total_reference_bases)),
            ani=_parse_optional_float(cached_metrics.get("ani", "")),
            divergence_estimate=_parse_optional_float(cached_metrics.get("dv", "")),
            divergence_source=cached_metrics.get("dv_source", "").strip() or None,
        )
        minimap2_metrics_seconds += time.perf_counter() - minimap2_metrics_start
        _write_log(log_path, ["Reused existing minimap2 metrics TSV"])

    sketch_config = SketchConfig(mode=str(job["sketch_mode"]), modulus=int(job["modimizer_modulus"]))
    cached_comparison = _read_single_tsv_row(jaccard_metrics_path)
    if str(job.get("analysis_mode", "whole_reference")) == "reference_chunks":
        if cached_comparison is not None:
            ref_jaccard_start = time.perf_counter()
            jaccard = _parse_optional_float(cached_comparison.get("jaccard_similarity", "")) or math.nan
            ref_jaccard = float(cached_comparison["ref_jaccard_similarity"])
            ref_jaccard_seconds += time.perf_counter() - ref_jaccard_start
            _write_log(log_path, ["Reused existing ref-jaccard metrics TSV"])
        else:
            jaccard = math.nan
            ref_jaccard = ""
            _write_log(log_path, ["Deferred ref-jaccard metrics to per-rate batch pass"])
    elif cached_comparison is None:
        ref_jaccard_start = time.perf_counter()
        reference_set = _comparison_set(reference_records, k=int(job["k"]), sketch_config=sketch_config)
        mutated_set = _comparison_set(mutated_records, k=int(job["k"]), sketch_config=sketch_config)
        jaccard = jaccard_similarity(reference_set, mutated_set)
        ref_jaccard = reference_jaccard_similarity(reference_set, mutated_set)

        write_tsv(
            [
                {
                    "comparison_mode": sketch_config.mode,
                    "k": int(job["k"]),
                    "modimizer_modulus": int(job["modimizer_modulus"]),
                    "reference_features": len(reference_set),
                    "query_features": len(mutated_set),
                    "jaccard_similarity": jaccard,
                    "ref_jaccard_similarity": ref_jaccard,
                }
            ],
            jaccard_metrics_path,
            fieldnames=[
                "comparison_mode",
                "k",
                "modimizer_modulus",
                "reference_features",
                "query_features",
                "jaccard_similarity",
                "ref_jaccard_similarity",
            ],
        )
        ref_jaccard_seconds += time.perf_counter() - ref_jaccard_start
        _write_log(log_path, ["Computed ref-jaccard metrics"])
    else:
        ref_jaccard_start = time.perf_counter()
        jaccard = _parse_optional_float(cached_comparison.get("jaccard_similarity", "")) or math.nan
        ref_jaccard = float(cached_comparison["ref_jaccard_similarity"])
        ref_jaccard_seconds += time.perf_counter() - ref_jaccard_start
        _write_log(log_path, ["Reused existing ref-jaccard metrics TSV"])

    metadata_summary = dict(metadata["summary"])
    summary_row = {
        "rate_label": str(job["rate_label"]),
        "replicate": int(job["replicate"]),
        "reference_label": str(job.get("reference_label", "")),
        "chunk_start": int(job.get("chunk_start", 0)),
        "chunk_end": int(job.get("chunk_end", 0)),
        "true_mutation_rate": true_mutation_rate,
        "substitution_rate": rates.substitution_rate,
        "insertion_rate": rates.insertion_rate,
        "deletion_rate": rates.deletion_rate,
        "realized_substitution_rate": (
            metadata_summary["realized_substitution_rate"]
            if realized_substitution_rate_override is None
            else float(realized_substitution_rate_override)
        ),
        "realized_insertion_rate": (
            metadata_summary["realized_insertion_rate"]
            if realized_insertion_rate_override is None
            else float(realized_insertion_rate_override)
        ),
        "realized_deletion_rate": (
            metadata_summary["realized_deletion_rate"]
            if realized_deletion_rate_override is None
            else float(realized_deletion_rate_override)
        ),
        "minimap2_preset": preset,
        "minimap2_ani": "" if sam_metrics.ani is None else sam_metrics.ani,
        "minimap2_dv": "" if sam_metrics.divergence_estimate is None else sam_metrics.divergence_estimate,
        "minimap2_dv_source": "" if sam_metrics.divergence_source is None else sam_metrics.divergence_source,
        "minimap2_aligned_bases": sam_metrics.aligned_bases,
        "minimap2_edit_distance": sam_metrics.edit_distance,
        "minimap2_query_bases": total_query_bases,
        "minimap2_query_coverage": query_coverage,
        "minimap2_mapped_records": sam_metrics.mapped_records,
        "jaccard_similarity": "" if math.isnan(jaccard) else jaccard,
        "ref_jaccard_similarity": ref_jaccard,
        "comparison_mode": sketch_config.mode,
        "k": int(job["k"]),
        "timing_mutation_setup_seconds": mutation_setup_seconds,
        "timing_minimap2_alignment_seconds": minimap2_alignment_seconds,
        "timing_minimap2_metrics_seconds": minimap2_metrics_seconds,
        "timing_ref_jaccard_seconds": ref_jaccard_seconds,
        "timing_summary_write_seconds": 0.0,
        "timing_total_job_seconds": 0.0,
    }
    summary_write_start = time.perf_counter()
    write_tsv([summary_row], summary_path, fieldnames=MASTER_FIELDS)
    summary_write_seconds = time.perf_counter() - summary_write_start
    total_job_seconds = time.perf_counter() - job_start
    summary_row["timing_summary_write_seconds"] = summary_write_seconds
    summary_row["timing_total_job_seconds"] = total_job_seconds
    write_tsv([summary_row], summary_path, fieldnames=MASTER_FIELDS)
    _write_log(
        log_path,
        [
            ("Computed jaccard=NA" if math.isnan(jaccard) else f"Computed jaccard={jaccard:.6f}"),
            (
                "Computed ref-jaccard=DEFERRED"
                if ref_jaccard == ""
                else f"Computed ref-jaccard={float(ref_jaccard):.6f}"
            ),
            (
                f"Computed minimap2 ANI={sam_metrics.ani:.6f}"
                if sam_metrics.ani is not None
                else "Computed minimap2 ANI=NA"
            ),
            (
                "Computed minimap2 divergence estimate="
                f"{sam_metrics.divergence_estimate:.6f} ({sam_metrics.divergence_source})"
                if sam_metrics.divergence_estimate is not None
                else "Computed minimap2 divergence estimate=NA"
            ),
            (
                "Timing seconds: "
                f"mutation_setup={mutation_setup_seconds:.3f} "
                f"minimap2_alignment={minimap2_alignment_seconds:.3f} "
                f"minimap2_metrics={minimap2_metrics_seconds:.3f} "
                f"ref_jaccard={ref_jaccard_seconds:.3f} "
                f"summary_write={summary_write_seconds:.3f} "
                f"total={total_job_seconds:.3f}"
            ),
        ],
    )
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
    _write_restart_script(config)
    reference_path = _ensure_reference(config)
    query_input = _ensure_query_input(config)
    variable_query_input = _ensure_variable_chunk_query(config, reference_path)
    if query_input is not None and variable_query_input is not None:
        raise ValueError("query_fasta and variable_chunk_mutation cannot be used together")
    if variable_query_input is not None:
        query_input = variable_query_input

    jobs: list[dict[str, object]] = []
    seed_base = config.seed if config.seed is not None else 0
    if config.analysis_mode == "reference_chunks":
        chunks = _ensure_reference_chunks(config, reference_path)
        chunk_metadata_by_index: dict[int, dict[str, object]] = {}
        if variable_query_input is not None:
            for chunk_entry in variable_query_input["metadata"]["chunks"]:
                chunk_metadata_by_index[int(chunk_entry["chunk_index"])] = dict(chunk_entry)
        job_rates = config.mutation_rates if query_input is None else (0.0,)
        for rate_index, rate in enumerate(job_rates):
            shared_mutation = (
                query_input
                if query_input is not None
                else _ensure_rate_mutation(
                    config=config,
                    reference_path=reference_path,
                    rate=rate,
                    seed=seed_base + (rate_index * 1000) + 1,
                )
            )
            for chunk in chunks:
                replicate_dir = _rate_dir(config.output_dir, rate) / f"chunk_{int(chunk['chunk_index']):04d}"
                jobs.append(
                    {
                        "replicate_dir": str(replicate_dir),
                        "reference_path": str(chunk["chunk_path"]),
                        "mutated_source_path": str(shared_mutation["mutated_path"]),
                        "metadata_source_path": str(shared_mutation["metadata_path"]),
                        "rate": rate,
                        "rate_label": _rate_label(rate) if query_input is None else "pairwise",
                        "replicate": int(chunk["chunk_index"]),
                        "reference_label": str(chunk["chunk_header"]),
                        "chunk_start": int(chunk["chunk_start"]),
                        "chunk_end": int(chunk["chunk_end"]),
                        "substitution_scale": config.substitution_scale,
                        "insertion_scale": config.insertion_scale,
                        "deletion_scale": config.deletion_scale,
                        "k": config.k,
                        "sketch_mode": config.sketch_mode,
                        "modimizer_modulus": config.modimizer_modulus,
                        "analysis_mode": config.analysis_mode,
                        "minimap2_threads": config.minimap2_threads,
                        "minimap2_executable": config.minimap2_executable,
                        "minimap2_preset": config.minimap2_preset,
                        "seed": seed_base + (rate_index * 1000) + int(chunk["chunk_index"]),
                        "true_mutation_rate_override": (
                            chunk_metadata_by_index.get(int(chunk["chunk_index"]), {}).get("requested_substitution_rate")
                            if variable_query_input is not None
                            else None
                        ),
                        "realized_substitution_rate_override": (
                            chunk_metadata_by_index.get(int(chunk["chunk_index"]), {}).get("realized_substitution_rate")
                            if variable_query_input is not None
                            else None
                        ),
                        "realized_insertion_rate_override": (
                            chunk_metadata_by_index.get(int(chunk["chunk_index"]), {}).get("realized_insertion_rate")
                            if variable_query_input is not None
                            else None
                        ),
                        "realized_deletion_rate_override": (
                            chunk_metadata_by_index.get(int(chunk["chunk_index"]), {}).get("realized_deletion_rate")
                            if variable_query_input is not None
                            else None
                        ),
                    }
                )
    else:
        job_rates = config.mutation_rates if query_input is None else (0.0,)
        for rate_index, rate in enumerate(job_rates):
            for replicate in range(1, config.replicates + 1):
                replicate_dir = _replicate_dir(config.output_dir, rate, replicate)
                jobs.append(
                    {
                        "replicate_dir": str(replicate_dir),
                        "reference_path": str(reference_path),
                        "mutated_source_path": "" if query_input is None else str(query_input["mutated_path"]),
                        "metadata_source_path": "" if query_input is None else str(query_input["metadata_path"]),
                        "rate": rate,
                        "rate_label": _rate_label(rate) if query_input is None else "pairwise",
                        "replicate": replicate,
                        "reference_label": "",
                        "chunk_start": 0,
                        "chunk_end": 0,
                        "substitution_scale": config.substitution_scale,
                        "insertion_scale": config.insertion_scale,
                        "deletion_scale": config.deletion_scale,
                        "k": config.k,
                        "sketch_mode": config.sketch_mode,
                        "modimizer_modulus": config.modimizer_modulus,
                        "analysis_mode": config.analysis_mode,
                        "minimap2_threads": config.minimap2_threads,
                        "minimap2_executable": config.minimap2_executable,
                        "minimap2_preset": config.minimap2_preset,
                        "seed": seed_base + (rate_index * 1000) + replicate,
                    }
                )

    _print_status_snapshot(jobs)
    if config.workers <= 1:
        results = []
        for index, job in enumerate(jobs, start=1):
            result = _run_single_job(job)
            results.append(result)
            print(
                f"[progress] completed={index}/{len(jobs)} rate={job['rate_label']} observation={job['replicate']}",
                flush=True,
            )
    else:
        try:
            with ProcessPoolExecutor(max_workers=config.workers) as executor:
                futures = {executor.submit(_run_single_job, job): job for job in jobs}
                results = []
                for index, future in enumerate(as_completed(futures), start=1):
                    job = futures[future]
                    results.append(future.result())
                    print(
                        f"[progress] completed={index}/{len(jobs)} rate={job['rate_label']} observation={job['replicate']}",
                        flush=True,
                    )
        except PermissionError:
            with ThreadPoolExecutor(max_workers=config.workers) as executor:
                futures = {executor.submit(_run_single_job, job): job for job in jobs}
                results = []
                for index, future in enumerate(as_completed(futures), start=1):
                    job = futures[future]
                    results.append(future.result())
                    print(
                        f"[progress] completed={index}/{len(jobs)} rate={job['rate_label']} observation={job['replicate']}",
                        flush=True,
                    )

    if config.analysis_mode == "reference_chunks":
        rate_groups: dict[str, list[dict[str, object]]] = {}
        for job in jobs:
            rate_groups.setdefault(str(job["rate_label"]), []).append(job)
        print(f"[ref-jaccard] starting batched per-rate pass for {len(rate_groups)} rates", flush=True)
        for rate_label in sorted(rate_groups, key=lambda value: next(job["rate"] for job in jobs if job["rate_label"] == value)):
            rate_jobs = sorted(rate_groups[rate_label], key=lambda job: int(job["replicate"]))
            _compute_chunked_ref_jaccard_for_rate(rate_jobs)
        results = []
        for job in jobs:
            summary = _read_cached_summary(Path(str(job["replicate_dir"])) / "run_summary.tsv")
            if summary is None:
                raise ValueError(f"Missing run summary after batched ref-jaccard pass: {job['replicate_dir']}")
            results.append(summary)

    master_path = config.output_dir / "master_results.tsv"
    results.sort(key=lambda row: (float(row["true_mutation_rate"]), int(row["replicate"])))
    write_tsv(results, master_path, fieldnames=MASTER_FIELDS)
    write_tsv(
        _aggregate_timing_rows(results),
        _timing_summary_path(config.output_dir),
        fieldnames=["metric", "sum_seconds", "mean_seconds", "max_seconds", "job_count"],
    )
    _print_timing_summary(results)
    print(f"[done] master_table={master_path}", flush=True)
    return master_path
