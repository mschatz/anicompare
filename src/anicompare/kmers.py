"""Exact k-mer comparison."""

from __future__ import annotations

from typing import Iterable

from .io import FastaRecord


_TRANSLATION = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_TRANSLATION)[::-1]


def canonical_kmer(kmer: str) -> str:
    revcomp = reverse_complement(kmer)
    return kmer if kmer <= revcomp else revcomp


def iter_canonical_kmers(records: Iterable[FastaRecord], k: int) -> Iterable[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    for record in records:
        sequence = record.sequence.upper()
        for start in range(0, len(sequence) - k + 1):
            kmer = sequence[start:start + k]
            if "N" in kmer:
                continue
            yield canonical_kmer(kmer)


def exact_kmer_set(records: Iterable[FastaRecord], k: int) -> set[str]:
    return set(iter_canonical_kmers(records, k=k))


def jaccard_similarity(values_a: set[str], values_b: set[str]) -> float:
    union = values_a | values_b
    if not union:
        return 1.0
    return len(values_a & values_b) / len(union)


def reference_jaccard_similarity(reference_values: set[str], query_values: set[str]) -> float:
    if not reference_values:
        return 1.0
    return len(reference_values & query_values) / len(reference_values)
