"""Helpers for building variable-rate chunk-mutated query genomes."""

from __future__ import annotations

import random
from pathlib import Path

from .io import FastaRecord, read_fasta, write_fasta, write_json
from .mutate import MutationRates, mutate_records


def build_variable_chunk_query(
    reference_fasta: Path,
    output_fasta: Path,
    metadata_path: Path,
    *,
    chunk_length: int,
    min_rate: float,
    max_rate: float,
    seed: int | None,
) -> dict[str, object]:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if min_rate < 0.0 or max_rate < 0.0 or min_rate > 1.0 or max_rate > 1.0:
        raise ValueError("min_rate and max_rate must be between 0 and 1")
    if min_rate > max_rate:
        raise ValueError("min_rate must be less than or equal to max_rate")

    rng = random.Random(0 if seed is None else seed)
    reference_records = read_fasta(reference_fasta)
    mutated_records: list[FastaRecord] = []
    chunk_summaries: list[dict[str, object]] = []
    total_original = 0
    total_mutated = 0
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    chunk_index = 1

    for record in reference_records:
        mutated_parts: list[str] = []
        sequence = record.sequence
        for start in range(0, len(sequence), chunk_length):
            end = min(start + chunk_length, len(sequence))
            chunk_sequence = sequence[start:end]
            substitution_rate = rng.uniform(min_rate, max_rate)
            rates = MutationRates(
                substitution_rate=substitution_rate,
                insertion_rate=0.0,
                deletion_rate=0.0,
            )
            mutated_chunk_records, metadata = mutate_records(
                [FastaRecord(header=f"{record.header}|chunk_{chunk_index:04d}", sequence=chunk_sequence)],
                rates,
                seed=rng.randint(0, 2**31 - 1),
            )
            mutated_parts.append(mutated_chunk_records[0].sequence)
            chunk_summary = dict(metadata["summary"])
            chunk_summary.update(
                {
                    "chunk_index": chunk_index,
                    "record_header": record.header,
                    "chunk_start": start + 1,
                    "chunk_end": end,
                    "requested_substitution_rate": substitution_rate,
                }
            )
            chunk_summaries.append(chunk_summary)
            total_original += int(chunk_summary["original_length"])
            total_mutated += int(chunk_summary["mutated_length"])
            total_substitutions += int(chunk_summary["substitutions"])
            total_insertions += int(chunk_summary["insertions"])
            total_deletions += int(chunk_summary["deletions"])
            chunk_index += 1
        mutated_records.append(FastaRecord(header=record.header, sequence="".join(mutated_parts)))

    metadata = {
        "query_source": "variable_chunk_mutation",
        "requested_rates": {
            "substitution_rate": None,
            "insertion_rate": 0.0,
            "deletion_rate": 0.0,
            "variable_chunk_min_rate": min_rate,
            "variable_chunk_max_rate": max_rate,
        },
        "summary": {
            "original_length": total_original,
            "mutated_length": total_mutated,
            "substitutions": total_substitutions,
            "insertions": total_insertions,
            "deletions": total_deletions,
            "realized_substitution_rate": total_substitutions / total_original if total_original else 0.0,
            "realized_insertion_rate": total_insertions / total_original if total_original else 0.0,
            "realized_deletion_rate": total_deletions / total_original if total_original else 0.0,
        },
        "chunks": chunk_summaries,
    }
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(mutated_records, output_fasta)
    write_json(metadata, metadata_path)
    return metadata
