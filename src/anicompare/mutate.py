"""Genome mutation helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .io import FastaRecord


DNA_ALPHABET = ("A", "C", "G", "T")


@dataclass(frozen=True)
class MutationRates:
    substitution_rate: float
    insertion_rate: float
    deletion_rate: float

    def validate(self) -> None:
        for name, value in (
            ("substitution_rate", self.substitution_rate),
            ("insertion_rate", self.insertion_rate),
            ("deletion_rate", self.deletion_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class MutationSummary:
    original_length: int
    mutated_length: int
    substitutions: int
    insertions: int
    deletions: int

    @property
    def total_events(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def realized_substitution_rate(self) -> float:
        return self.substitutions / self.original_length if self.original_length else 0.0

    @property
    def realized_insertion_rate(self) -> float:
        return self.insertions / self.original_length if self.original_length else 0.0

    @property
    def realized_deletion_rate(self) -> float:
        return self.deletions / self.original_length if self.original_length else 0.0


def _mutate_sequence(sequence: str, rates: MutationRates, rng: random.Random) -> tuple[str, MutationSummary]:
    rates.validate()
    mutated: list[str] = []
    substitutions = 0
    insertions = 0
    deletions = 0

    for base in sequence.upper():
        if rng.random() < rates.insertion_rate:
            mutated.append(rng.choice(DNA_ALPHABET))
            insertions += 1

        if rng.random() < rates.deletion_rate:
            deletions += 1
            continue

        if rng.random() < rates.substitution_rate:
            alternatives = [candidate for candidate in DNA_ALPHABET if candidate != base]
            mutated.append(rng.choice(alternatives))
            substitutions += 1
        else:
            mutated.append(base)

    summary = MutationSummary(
        original_length=len(sequence),
        mutated_length=len(mutated),
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
    )
    return "".join(mutated), summary


def mutate_records(
    records: list[FastaRecord],
    rates: MutationRates,
    seed: int | None = None,
) -> tuple[list[FastaRecord], dict[str, object]]:
    rng = random.Random(seed)
    mutated_records: list[FastaRecord] = []
    per_record: list[dict[str, object]] = []

    total_original = 0
    total_mutated = 0
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0

    for index, record in enumerate(records):
        record_seed = rng.randint(0, 2**31 - 1)
        mutated_sequence, summary = _mutate_sequence(record.sequence, rates, random.Random(record_seed))
        mutated_records.append(FastaRecord(header=record.header, sequence=mutated_sequence))
        per_record.append(
            {
                "record_index": index,
                "header": record.header,
                "original_length": summary.original_length,
                "mutated_length": summary.mutated_length,
                "substitutions": summary.substitutions,
                "insertions": summary.insertions,
                "deletions": summary.deletions,
                "realized_substitution_rate": summary.realized_substitution_rate,
                "realized_insertion_rate": summary.realized_insertion_rate,
                "realized_deletion_rate": summary.realized_deletion_rate,
            }
        )
        total_original += summary.original_length
        total_mutated += summary.mutated_length
        total_substitutions += summary.substitutions
        total_insertions += summary.insertions
        total_deletions += summary.deletions

    metadata = {
        "requested_rates": {
            "substitution_rate": rates.substitution_rate,
            "insertion_rate": rates.insertion_rate,
            "deletion_rate": rates.deletion_rate,
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
        "records": per_record,
    }
    return mutated_records, metadata
