"""Run minimap2 and estimate ANI from SAM alignments."""

from __future__ import annotations

import gzip
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


SAM_FLAG_UNMAPPED = 0x4
SAM_FLAG_SECONDARY = 0x100
SAM_FLAG_SUPPLEMENTARY = 0x800


_CIGAR_PATTERN = re.compile(r"(\d+)([MIDNSHP=X])")


@dataclass(frozen=True)
class SamMetrics:
    mapped_records: int
    aligned_bases: int
    edit_distance: int
    query_bases: int
    reference_covered_bases: int
    ani: float | None
    divergence_estimate: float | None
    divergence_source: str | None


def ensure_minimap2_available(executable: str = "minimap2") -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"Could not find minimap2 executable: {executable}")
    return resolved


def run_minimap2(
    reference_fasta: Path,
    query_fasta: Path,
    sam_path: Path,
    executable: str = "minimap2",
    threads: int = 1,
    preset: str = "asm5",
) -> None:
    ensure_minimap2_available(executable)
    command = [
        executable,
        "-a",
        "-x",
        preset,
        "-t",
        str(max(1, threads)),
        str(reference_fasta),
        str(query_fasta),
    ]
    with sam_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, check=False, stdout=handle, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"minimap2 failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )


def choose_minimap2_preset(expected_divergence: float) -> str:
    if expected_divergence <= 0.03:
        return "asm5"
    if expected_divergence < 0.10:
        return "asm10"
    return "asm20"


def _aligned_bases_from_cigar(cigar: str) -> int:
    aligned = 0
    for length_text, op in _CIGAR_PATTERN.findall(cigar):
        length = int(length_text)
        if op in {"M", "=", "X", "I", "D"}:
            aligned += length
    return aligned


def _query_bases_from_cigar(cigar: str) -> int:
    total = 0
    for length_text, op in _CIGAR_PATTERN.findall(cigar):
        length = int(length_text)
        if op in {"M", "=", "X", "I", "S"}:
            total += length
    return total


def _reference_bases_from_cigar(cigar: str) -> int:
    total = 0
    for length_text, op in _CIGAR_PATTERN.findall(cigar):
        length = int(length_text)
        if op in {"M", "=", "X", "D", "N"}:
            total += length
    return total


def parse_sam_for_ani(sam_path: Path) -> SamMetrics:
    mapped_records = 0
    aligned_bases = 0
    edit_distance = 0
    query_bases = 0
    weighted_divergence = 0.0
    divergence_weight_bases = 0
    divergence_source: str | None = None
    reference_intervals: dict[str, list[tuple[int, int]]] = {}

    opener = gzip.open if sam_path.suffix == ".gz" else Path.open
    if sam_path.suffix == ".gz":
        handle_context = opener(sam_path, "rt", encoding="utf-8")
    else:
        handle_context = sam_path.open("r", encoding="utf-8")
    with handle_context as handle:
        for line in handle:
            if not line or line.startswith("@"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue
            flag = int(fields[1])
            if flag & SAM_FLAG_UNMAPPED:
                continue
            if flag & SAM_FLAG_SECONDARY:
                continue
            if flag & SAM_FLAG_SUPPLEMENTARY:
                continue

            cigar = fields[5]
            reference_name = fields[2]
            reference_start = int(fields[3]) - 1
            nm_value = None
            divergence_value = None
            local_source = None
            for field in fields[11:]:
                if field.startswith("NM:i:"):
                    nm_value = int(field[5:])
                elif field.startswith("dv:f:"):
                    divergence_value = float(field[5:])
                    local_source = "dv"
                elif field.startswith("de:f:") and divergence_value is None:
                    divergence_value = float(field[5:])
                    local_source = "de"
            if nm_value is None:
                continue

            local_aligned_bases = _aligned_bases_from_cigar(cigar)
            reference_span = _reference_bases_from_cigar(cigar)
            mapped_records += 1
            aligned_bases += local_aligned_bases
            query_bases += _query_bases_from_cigar(cigar)
            edit_distance += nm_value
            if reference_span > 0:
                reference_intervals.setdefault(reference_name, []).append(
                    (reference_start, reference_start + reference_span)
                )
            if divergence_value is not None and local_aligned_bases > 0:
                weighted_divergence += divergence_value * local_aligned_bases
                divergence_weight_bases += local_aligned_bases
                if divergence_source is None or divergence_source == "de":
                    divergence_source = local_source

    ani = None
    if aligned_bases > 0:
        ani = max(0.0, 1.0 - (edit_distance / aligned_bases))
    divergence_estimate = None
    if divergence_weight_bases > 0:
        divergence_estimate = weighted_divergence / divergence_weight_bases
    reference_covered_bases = 0
    for intervals in reference_intervals.values():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        reference_covered_bases += sum(end - start for start, end in merged)
    return SamMetrics(
        mapped_records=mapped_records,
        aligned_bases=aligned_bases,
        edit_distance=edit_distance,
        query_bases=query_bases,
        reference_covered_bases=reference_covered_bases,
        ani=ani,
        divergence_estimate=divergence_estimate,
        divergence_source=divergence_source,
    )
