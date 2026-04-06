"""Simple modimizer sketching."""

from __future__ import annotations

import hashlib
from typing import Iterable

from .io import FastaRecord
from .kmers import iter_canonical_kmers


def _stable_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode("ascii")).hexdigest(), 16)


def modimizer_set(records: Iterable[FastaRecord], k: int, modulus: int = 100) -> set[str]:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    selected: set[str] = set()
    for kmer in iter_canonical_kmers(records, k=k):
        if _stable_hash(kmer) % modulus == 0:
            selected.add(kmer)
    return selected
