"""Run minimap2 and estimate ANI from SAM alignments."""

from __future__ import annotations

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
    ani: float | None


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


def parse_sam_for_ani(sam_path: Path) -> SamMetrics:
    mapped_records = 0
    aligned_bases = 0
    edit_distance = 0
    query_bases = 0

    with sam_path.open("r", encoding="utf-8") as handle:
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
            nm_value = None
            for field in fields[11:]:
                if field.startswith("NM:i:"):
                    nm_value = int(field[5:])
                    break
            if nm_value is None:
                continue

            mapped_records += 1
            aligned_bases += _aligned_bases_from_cigar(cigar)
            query_bases += _query_bases_from_cigar(cigar)
            edit_distance += nm_value

    ani = None
    if aligned_bases > 0:
        ani = max(0.0, 1.0 - (edit_distance / aligned_bases))
    return SamMetrics(
        mapped_records=mapped_records,
        aligned_bases=aligned_bases,
        edit_distance=edit_distance,
        query_bases=query_bases,
        ani=ani,
    )
