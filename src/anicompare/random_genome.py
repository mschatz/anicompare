"""Random genome generation."""

from __future__ import annotations

import random

from .io import FastaRecord


DNA_ALPHABET = ("A", "C", "G", "T")


def generate_random_sequence(length: int, gc_content: float = 0.5, seed: int | None = None) -> str:
    if length <= 0:
        raise ValueError("Genome length must be positive")
    if not 0.0 <= gc_content <= 1.0:
        raise ValueError("GC content must be between 0 and 1")

    rng = random.Random(seed)
    gc_weight = gc_content / 2.0
    at_weight = (1.0 - gc_content) / 2.0
    bases = ["A", "C", "G", "T"]
    weights = [at_weight, gc_weight, gc_weight, at_weight]
    return "".join(rng.choices(bases, weights=weights, k=length))


def generate_random_record(
    length: int,
    gc_content: float = 0.5,
    seed: int | None = None,
    name: str = "random_genome",
) -> FastaRecord:
    return FastaRecord(header=name, sequence=generate_random_sequence(length, gc_content=gc_content, seed=seed))
