from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from random import Random
from typing import Iterable, Mapping, Sequence


RULE_NAMES: tuple[str, ...] = (
    "RULE_EMIT_X",
    "RULE_EMIT_Y",
    "RULE_ZERO_2",
    "RULE_ADD0_IF1",
    "RULE_DECAY1",
    "RULE_OUTPUT_IF1Z",
    "RULE_INHIBIT1Z",
    "RULE_COPY0_3",
    "RULE_COPY1_3",
    "RULE_ADD3_IF1",
    "RULE_DECAY3",
    "RULE_THRESH1",
    "RULE_THRESH3",
    "SEND",
    "RECV",
)


@dataclass(frozen=True, slots=True)
class Codon:
    value: int

    def mutate(self, rng: Random, modulus: int = 256) -> "Codon":
        step = rng.choice((-7, -3, -1, 1, 3, 7, 11))
        return Codon((self.value + step) % modulus)


@dataclass(frozen=True)
class CodonTable:
    codon_to_op: dict[int, str]

    @property
    def ops(self) -> list[str]:
        return [self.codon_to_op[i] for i in range(len(self.codon_to_op))]

    @property
    def op_to_codons(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        for codon, op in self.codon_to_op.items():
            out[op].append(codon)
        return dict(out)

    def op(self, codon: int) -> str:
        return self.codon_to_op[codon]

    def random_codon(self, op: str) -> int:
        from random import choice

        return choice(self.op_to_codons[op])

    def encode_ops(self, ops: list[str]) -> list[int]:
        return [self.random_codon(op) for op in ops]

    def decode(self, genome: list[int]) -> list[str]:
        return [self.op(codon) for codon in genome]


def build_codon_table(
    allocation: Mapping[str, int],
    *,
    table_size: int = 64,
    fill_op: str = "NOOP",
) -> CodonTable:
    if table_size < len(allocation):
        raise ValueError("table_size must be at least the number of distinct ops")

    codon_to_op: dict[int, str] = {}
    idx = 0

    for op in allocation:
        codon_to_op[idx] = op
        idx += 1

    for op, count in allocation.items():
        for _ in range(max(0, count - 1)):
            if idx >= table_size:
                break
            codon_to_op[idx] = op
            idx += 1
        if idx >= table_size:
            break

    while idx < table_size:
        codon_to_op[idx] = fill_op
        idx += 1

    return CodonTable(codon_to_op)


def default_codon_table() -> CodonTable:
    allocation = {
        "START": 3,
        "STOP": 3,
        "NOOP": 4,
        "SEND": 2,
        "RECV": 2,
        "RULE_EMIT_X": 5,
        "RULE_EMIT_Y": 5,
        "RULE_ZERO_2": 4,
        "RULE_ADD0_IF1": 6,
        "RULE_DECAY1": 6,
        "RULE_OUTPUT_IF1Z": 5,
        "RULE_INHIBIT1Z": 3,
        "RULE_COPY0_3": 3,
        "RULE_COPY1_3": 3,
        "RULE_ADD3_IF1": 3,
        "RULE_DECAY3": 3,
        "RULE_THRESH1": 2,
        "RULE_THRESH3": 2,
    }
    return build_codon_table(allocation, table_size=64, fill_op="NOOP")


def decode_rule_name(value: int) -> str:
    return RULE_NAMES[value % len(RULE_NAMES)]


def encode_rule_name(rule_name: str) -> int:
    try:
        return RULE_NAMES.index(rule_name)
    except ValueError as exc:
        raise KeyError(rule_name) from exc


def codons_from_rule_names(rule_names: Sequence[str]) -> tuple[int, ...]:
    return tuple(encode_rule_name(rule_name) for rule_name in rule_names)


def random_codons(rng: Random, length: int, modulus: int = 256) -> tuple[int, ...]:
    return tuple(rng.randrange(modulus) for _ in range(length))


def mutate_codons(
    codons: Sequence[int],
    rng: Random,
    *,
    mutation_rate: float = 0.08,
    modulus: int = 256,
) -> tuple[int, ...]:
    mutated = [codon % modulus for codon in codons]
    for index, value in enumerate(mutated):
        if rng.random() < mutation_rate:
            mutated[index] = (value + rng.choice((-7, -3, -1, 1, 3, 7))) % modulus
    return tuple(mutated)


def insert_delete_codons(
    codons: Sequence[int],
    rng: Random,
    *,
    insertion_rate: float = 0.05,
    deletion_rate: float = 0.03,
    modulus: int = 256,
    max_length: int = 32,
) -> tuple[int, ...]:
    mutated = list(codons)
    if mutated and rng.random() < deletion_rate:
        del mutated[rng.randrange(len(mutated))]
    if len(mutated) < max_length and rng.random() < insertion_rate:
        mutated.insert(rng.randrange(len(mutated) + 1), rng.randrange(modulus))
    return tuple(mutated)


def crossover_codons(
    left: Sequence[int],
    right: Sequence[int],
    rng: Random,
) -> tuple[int, ...]:
    if not left:
        return tuple(right)
    if not right:
        return tuple(left)
    pivot_left = rng.randrange(len(left))
    pivot_right = rng.randrange(len(right))
    return tuple(left[:pivot_left]) + tuple(right[pivot_right:])


def unique_ordered(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)
