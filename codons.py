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

REGULATORY_OP_NAMES: tuple[str, ...] = (
    "PROMOTER",
    "GATE",
    "IF_S0_GT",
    "IF_S1_GT",
    "IF_S2_GT",
    "IF_S2_LT",
    "IF_S3_GT",
    "IF_S3_LT",
    "IF_S4_GT",
    "IF_S4_LT",
    "IF_S0_LT",
    "IF_S1_LT",
    "SENSE_PEER_0",
    "SENSE_PEER_1",
    "SENSE_PEER_2",
    "SENSE_PEER_2_TO_3",
    "SENSE_PEER_1_TO_3",
    "RULE_COPY1_2",
    "SCALE_BY_S0",
    "SCALE_BY_S1",
    "SCALE_BY_S2",
    "SCALE_BY_S3",
    "END_BLOCK",
    "GENE_START",
    "GENE_ID_0",
    "GENE_ID_1",
    "GENE_ID_2",
    "GENE_ID_3",
    "GENE_ID_4",
    "GENE_ID_5",
    "GENE_ID_6",
    "GENE_ID_7",
    "GENE_END",
    "CALL_0",
    "CALL_1",
    "CALL_2",
    "CALL_3",
    "CALL_4",
    "CALL_5",
    "CALL_6",
    "CALL_7",
)

GENE_BLOCK_CODON_MAP: dict[int, str] = {
    128: "GENE_START",
    129: "GENE_START",
    130: "GENE_START",
    131: "GENE_ID_0",
    132: "GENE_ID_0",
    133: "GENE_ID_0",
    134: "GENE_ID_1",
    135: "GENE_ID_1",
    136: "GENE_ID_1",
    137: "GENE_ID_2",
    138: "GENE_ID_2",
    139: "GENE_ID_2",
    140: "GENE_ID_3",
    141: "GENE_ID_3",
    142: "GENE_ID_3",
    143: "GENE_ID_4",
    144: "GENE_ID_4",
    145: "GENE_ID_4",
    146: "GENE_ID_5",
    147: "GENE_ID_5",
    148: "GENE_ID_5",
    149: "GENE_ID_6",
    150: "GENE_ID_6",
    151: "GENE_ID_6",
    152: "GENE_ID_7",
    153: "GENE_ID_7",
    154: "GENE_ID_7",
    155: "GENE_END",
    156: "GENE_END",
    157: "GENE_END",
    158: "CALL_0",
    159: "CALL_0",
    160: "CALL_0",
    161: "CALL_1",
    162: "CALL_1",
    163: "CALL_1",
    164: "CALL_2",
    165: "CALL_2",
    166: "CALL_2",
    167: "CALL_3",
    168: "CALL_3",
    169: "CALL_3",
    170: "CALL_4",
    171: "CALL_4",
    172: "CALL_4",
    173: "CALL_5",
    174: "CALL_5",
    175: "CALL_5",
    176: "CALL_6",
    177: "CALL_6",
    178: "CALL_6",
    179: "CALL_7",
    180: "CALL_7",
    181: "CALL_7",
}

REGULATORY_CODON_MAP: dict[int, str] = {
    200: "SENSE_PEER_0",
    201: "SENSE_PEER_0",
    202: "SENSE_PEER_0",
    203: "SENSE_PEER_1",
    204: "SENSE_PEER_1",
    205: "SENSE_PEER_1",
    206: "SENSE_PEER_2",
    207: "SENSE_PEER_2",
    208: "SENSE_PEER_2",
    209: "SENSE_PEER_2_TO_3",
    210: "SENSE_PEER_2_TO_3",
    211: "SENSE_PEER_2_TO_3",
    212: "SENSE_PEER_1_TO_3",
    213: "SENSE_PEER_1_TO_3",
    214: "RULE_COPY1_2",
    215: "RULE_COPY1_2",
    216: "RULE_COPY1_2",
    271: "SCALE_BY_S0",
    272: "SCALE_BY_S0",
    273: "SCALE_BY_S0",
    274: "SCALE_BY_S1",
    275: "SCALE_BY_S1",
    276: "SCALE_BY_S1",
    277: "SCALE_BY_S2",
    278: "SCALE_BY_S2",
    279: "SCALE_BY_S2",
    280: "SCALE_BY_S3",
    281: "SCALE_BY_S3",
    282: "SCALE_BY_S3",
    231: "PROMOTER",
    232: "PROMOTER",
    233: "PROMOTER",
    234: "GATE",
    235: "GATE",
    236: "GATE",
    237: "IF_S0_GT",
    238: "IF_S0_GT",
    239: "IF_S0_GT",
    240: "IF_S1_GT",
    241: "IF_S1_GT",
    242: "IF_S1_GT",
    243: "IF_S2_GT",
    244: "IF_S2_GT",
    245: "IF_S2_GT",
    246: "IF_S2_LT",
    247: "IF_S2_LT",
    248: "IF_S2_LT",
    249: "IF_S3_GT",
    250: "IF_S3_GT",
    251: "IF_S3_GT",
    252: "IF_S3_LT",
    253: "IF_S3_LT",
    254: "IF_S3_LT",
    255: "IF_S4_GT",
    256: "IF_S4_GT",
    257: "IF_S4_GT",
    258: "IF_S4_LT",
    259: "IF_S4_LT",
    260: "IF_S4_LT",
    261: "IF_S0_LT",
    262: "IF_S0_LT",
    263: "IF_S0_LT",
    264: "IF_S1_LT",
    265: "IF_S1_LT",
    266: "IF_S1_LT",
    267: "END_BLOCK",
    268: "END_BLOCK",
    269: "END_BLOCK",
    270: "END_BLOCK",
    **GENE_BLOCK_CODON_MAP,
}


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

    def synonymous_codons(self, codon: int, *, modulus: int = 256) -> tuple[int, ...]:
        table_size = len(self.codon_to_op)
        op = self.op(codon % table_size)
        return tuple(candidate for candidate in self.op_to_codons[op] if candidate % modulus == candidate)

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
    table = build_codon_table(allocation, table_size=231, fill_op="NOOP")
    codon_to_op = dict(table.codon_to_op)
    codon_to_op.update(REGULATORY_CODON_MAP)
    return CodonTable(codon_to_op)


def decode_rule_name(value: int) -> str:
    return RULE_NAMES[value % len(RULE_NAMES)]


def decode_codon_op(value: int) -> str:
    return default_codon_table().op(value)


def encode_rule_name(rule_name: str) -> int:
    try:
        return RULE_NAMES.index(rule_name)
    except ValueError as exc:
        raise KeyError(rule_name) from exc


def codons_from_rule_names(rule_names: Sequence[str]) -> tuple[int, ...]:
    return tuple(encode_rule_name(rule_name) for rule_name in rule_names)


def random_codons(
    rng: Random,
    length: int,
    modulus: int = 256,
    *,
    exclude: Sequence[int] | None = tuple(REGULATORY_CODON_MAP),
) -> tuple[int, ...]:
    forbidden = set(exclude or ())
    if not forbidden:
        return tuple(rng.randrange(modulus) for _ in range(length))
    candidates = [value for value in range(modulus) if value not in forbidden]
    return tuple(rng.choice(candidates) for _ in range(length))


def mutate_codons(
    codons: Sequence[int],
    rng: Random,
    *,
    mutation_rate: float = 0.08,
    modulus: int = 256,
    exclude: Sequence[int] | None = tuple(REGULATORY_CODON_MAP),
) -> tuple[int, ...]:
    mutated = [codon % modulus for codon in codons]
    forbidden = set(exclude or ())
    for index, value in enumerate(mutated):
        if rng.random() < mutation_rate:
            candidate = (value + rng.choice((-7, -3, -1, 1, 3, 7))) % modulus
            while candidate in forbidden:
                candidate = (candidate + 1) % modulus
            mutated[index] = candidate
    return tuple(mutated)


def synonymous_codons(
    codon: int,
    *,
    modulus: int = 256,
    synonym_modulus: int = len(RULE_NAMES),
) -> tuple[int, ...]:
    rule_index = codon % synonym_modulus
    synonyms = tuple(candidate for candidate in range(modulus) if candidate % synonym_modulus == rule_index)
    return synonyms


def mutate_codons_neutral(
    codons: Sequence[int],
    rng: Random,
    *,
    mutation_rate: float = 0.08,
    synonym_rate: float = 0.65,
    modulus: int = 256,
    exclude: Sequence[int] | None = tuple(REGULATORY_CODON_MAP),
) -> tuple[int, ...]:
    mutated = [codon % modulus for codon in codons]
    forbidden = set(exclude or ())
    for index, value in enumerate(mutated):
        if rng.random() >= mutation_rate:
            continue
        if rng.random() < synonym_rate:
            synonyms = tuple(
                candidate
                for candidate in synonymous_codons(value, modulus=modulus)
                if candidate != value and candidate not in forbidden
            )
            if synonyms:
                mutated[index] = rng.choice(synonyms)
                continue
        candidate = (value + rng.choice((-7, -3, -1, 1, 3, 7))) % modulus
        while candidate in forbidden:
            candidate = (candidate + 1) % modulus
        mutated[index] = candidate
    return tuple(mutated)


def mutate_codons_with_table(
    codons: Sequence[int],
    rng: Random,
    table: CodonTable,
    *,
    mutation_rate: float = 0.08,
    synonym_rate: float = 0.65,
    modulus: int = 256,
) -> tuple[int, ...]:
    table_size = len(table.codon_to_op)
    effective_modulus = max(modulus, table_size)
    mutated = [codon % effective_modulus for codon in codons]
    for index, value in enumerate(mutated):
        if rng.random() >= mutation_rate:
            continue
        if rng.random() < synonym_rate:
            synonyms = [candidate for candidate in table.synonymous_codons(value, modulus=effective_modulus) if candidate != value]
            if synonyms:
                mutated[index] = rng.choice(synonyms)
                continue
        mutated[index] = (value + rng.choice((-7, -3, -1, 1, 3, 7))) % effective_modulus
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
