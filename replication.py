from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Sequence

from codons import (
    build_codon_table,
    crossover_codons,
    insert_delete_codons,
    mutate_codons_with_table,
    random_codons,
)
from tracing import ExperimentReport, lineage_tree_text


REPLICATION_OPS: tuple[str, ...] = (
    "START",
    "STOP",
    "NOOP",
    "PUSH",
    "READ",
    "WRITE",
    "DEC",
    "JNZ",
    "DIVIDE",
)

REPLICATION_CODON_ALLOCATION: dict[str, int] = {
    "START": 2,
    "STOP": 2,
    "NOOP": 8,
    "PUSH": 6,
    "READ": 8,
    "WRITE": 8,
    "DEC": 6,
    "JNZ": 8,
    "DIVIDE": 4,
}

REPLICATION_CODON_TABLE = build_codon_table(REPLICATION_CODON_ALLOCATION, table_size=64, fill_op="NOOP")


def _literal(value: int) -> int:
    return value % 256


def decode_signed_literal(value: int) -> int:
    return value if value < 128 else value - 256


def encode_signed_literal(value: int) -> int:
    return value % 256


def encode_replication_op(op_name: str) -> int:
    try:
        return REPLICATION_CODON_TABLE.op_to_codons[op_name][0]
    except KeyError as exc:
        raise KeyError(op_name) from exc


def random_replication_codon(op_name: str, rng: Random) -> int:
    return rng.choice(REPLICATION_CODON_TABLE.op_to_codons[op_name])


def codons_from_replication_ops(ops: Sequence[str], rng: Random | None = None) -> tuple[int, ...]:
    if rng is None:
        return tuple(encode_replication_op(op) for op in ops)
    return tuple(random_replication_codon(op, rng) for op in ops)


@dataclass(frozen=True, slots=True)
class ReplicationGenome:
    codons: tuple[int, ...]
    lineage_id: str = "replicator"

    @property
    def length(self) -> int:
        return len(self.codons)

    @classmethod
    def from_ops(
        cls,
        ops: Sequence[str],
        *,
        lineage_id: str = "replicator",
        rng: Random | None = None,
    ) -> "ReplicationGenome":
        return cls(codons=codons_from_replication_ops(ops, rng), lineage_id=lineage_id)

    def mutate(
        self,
        rng: Random,
        *,
        mutation_rate: float = 0.08,
        synonym_rate: float = 0.65,
        insertion_rate: float = 0.05,
        deletion_rate: float = 0.03,
        max_length: int = 32,
    ) -> "ReplicationGenome":
        codons = mutate_codons_with_table(
            self.codons,
            rng,
            REPLICATION_CODON_TABLE,
            mutation_rate=mutation_rate,
            synonym_rate=synonym_rate,
        )
        codons = insert_delete_codons(
            codons,
            rng,
            insertion_rate=insertion_rate,
            deletion_rate=deletion_rate,
            max_length=max_length,
        )
        return ReplicationGenome(codons=codons, lineage_id=self.lineage_id)

    def with_lineage(self, lineage_id: str) -> "ReplicationGenome":
        return ReplicationGenome(codons=self.codons, lineage_id=lineage_id)


@dataclass(slots=True)
class ReplicationState:
    genome: ReplicationGenome
    pc: int = 0
    stack: list[int] = field(default_factory=list)
    read_cursor: int = 0
    offspring_buffer: list[int] = field(default_factory=list)
    offspring: tuple[int, ...] | None = None
    halted: bool = False
    trace: list[dict[str, object]] = field(default_factory=list)

    def snapshot(self) -> dict[str, object]:
        return {
            "pc": self.pc,
            "stack": list(self.stack),
            "read_cursor": self.read_cursor,
            "offspring_buffer": list(self.offspring_buffer),
            "offspring": self.offspring,
            "halted": self.halted,
        }


@dataclass(frozen=True, slots=True)
class ReplicationEvaluation:
    genome: ReplicationGenome
    score: float
    exact_replication: bool
    offspring_len: int
    mismatch_count: int
    prefix_match: int
    trace_examples: tuple[str, ...]
    neutrality_estimate: float = 0.0


@dataclass(frozen=True, slots=True)
class ReplicationSearchConfig:
    population_size: int = 24
    restarts: int = 8
    generations: int = 40
    survivor_count: int = 6
    mutation_rate: float = 0.08
    synonym_rate: float = 0.65
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    crossover_rate: float = 0.35
    founder_fraction: float = 0.5
    founder_bias_fraction: float = 0.7
    max_length: int = 24


SAFE_REPLICATION_MOTIFS: tuple[tuple[str, ...], ...] = (
    ("PUSH", "READ", "WRITE", "DEC", "JNZ", "DIVIDE", "STOP"),
    ("PUSH", "READ", "WRITE"),
    ("DEC", "JNZ"),
    ("DIVIDE", "STOP"),
    ("READ", "WRITE", "NOOP"),
    ("PUSH", "NOOP", "DEC", "JNZ"),
)


def build_biased_replication_founder(*, lineage_id: str, rng: Random) -> ReplicationGenome:
    base_ops = list(rng.choice(SAFE_REPLICATION_MOTIFS))
    codons: list[int] = []
    target_len = rng.randrange(9, 15)
    for op in base_ops:
        codons.append(random_replication_codon(op, rng))
        if op == "PUSH":
            codons.append(encode_signed_literal(rng.randrange(1, 24)))
        elif op == "JNZ":
            codons.append(encode_signed_literal(rng.choice((-6, -5, -4, -3, -2, -1))))
    if base_ops == list(SAFE_REPLICATION_MOTIFS[0]):
        codons[1] = encode_signed_literal(rng.randrange(4, 16))
        codons[6] = encode_signed_literal(rng.choice((-6, -5, -4)))
    if len(codons) < 9:
        codons.append(random_replication_codon("NOOP", rng))
    while len(codons) < target_len:
        codons.insert(rng.randrange(len(codons) + 1), random_replication_codon("NOOP", rng))
    return ReplicationGenome(codons=tuple(codons), lineage_id=lineage_id)


def build_random_replication_genome(*, lineage_id: str, rng: Random, length: int = 12) -> ReplicationGenome:
    return ReplicationGenome(codons=random_codons(rng, length), lineage_id=lineage_id)


def build_initial_replication_population(
    rng: Random,
    *,
    size: int,
    lineage_prefix: str = "R",
    founder_fraction: float = 0.5,
    founder_bias_fraction: float = 0.7,
) -> list[ReplicationGenome]:
    population: list[ReplicationGenome] = []
    founder_count = max(1, min(size, int(round(size * founder_fraction))))
    biased_count = max(1, int(round(founder_count * founder_bias_fraction)))
    for index in range(biased_count):
        population.append(
            build_biased_replication_founder(
                lineage_id=f"{lineage_prefix}B{index + 1}",
                rng=rng,
            )
        )
    while len(population) < founder_count:
        population.append(
            build_random_replication_genome(
                lineage_id=f"{lineage_prefix}F{len(population) + 1}",
                rng=rng,
                length=rng.randrange(9, 16),
            )
        )
    while len(population) < size:
        population.append(
            build_random_replication_genome(
                lineage_id=f"{lineage_prefix}R{len(population) + 1}",
                rng=rng,
                length=rng.randrange(9, 20),
            )
        )
    return population


def _hamming_or_length_penalty(left: Sequence[int], right: Sequence[int]) -> tuple[int, int]:
    mismatch = sum(1 for a, b in zip(left, right, strict=False) if a != b)
    length_diff = abs(len(left) - len(right))
    return mismatch, length_diff


def _shared_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    prefix = 0
    for left_value, right_value in zip(left, right, strict=False):
        if left_value != right_value:
            break
        prefix += 1
    return prefix


def evaluate_replication_genome(genome: ReplicationGenome) -> ReplicationEvaluation:
    state = run_replication_program(genome)
    offspring = state.offspring or ()
    mismatch_count, length_diff = _hamming_or_length_penalty(offspring, genome.codons)
    prefix_match = _shared_prefix_length(offspring, genome.codons)
    has_offspring = bool(offspring)
    exact = offspring == genome.codons
    score = float(length_diff * 5 + mismatch_count * 10)
    score -= float(prefix_match * 4)
    if not has_offspring:
        score += 500.0
    if not state.halted:
        score += 100.0
    if not exact:
        score += 25.0
    if offspring and len(offspring) != len(genome.codons):
        score += 15.0
    if not any(event["op"] == "READ" for event in state.trace):
        score += 50.0
    if not any(event["op"] == "WRITE" for event in state.trace):
        score += 50.0
    if not any(event["op"] == "DIVIDE" for event in state.trace):
        score += 50.0
    trace_examples = tuple(
        f"s{event['step']}:{event['op']} {event['note']}"
        for event in state.trace[:8]
    )
    neutrality_rng = Random(sum(genome.codons) + len(genome.codons))
    neutrality_trials = 6
    neutral = 0
    for trial in range(neutrality_trials):
        candidate = genome.mutate(
            neutrality_rng,
            mutation_rate=0.08,
            synonym_rate=0.65,
            insertion_rate=0.0,
            deletion_rate=0.0,
            max_length=max(32, len(genome.codons)),
        )
        candidate_eval = run_replication_program(candidate)
        candidate_offspring = candidate_eval.offspring or ()
        if candidate_offspring == offspring:
            neutral += 1
    return ReplicationEvaluation(
        genome=genome,
        score=score,
        exact_replication=exact,
        offspring_len=len(offspring),
        mismatch_count=mismatch_count + length_diff,
        prefix_match=prefix_match,
        trace_examples=trace_examples,
        neutrality_estimate=neutral / neutrality_trials,
    )


def _mate(left: ReplicationGenome, right: ReplicationGenome, rng: Random, *, lineage_id: str) -> ReplicationGenome:
    return ReplicationGenome(
        codons=crossover_codons(left.codons, right.codons, rng),
        lineage_id=lineage_id,
    )


def _weighted_survivor_index(rng: Random, survivor_count: int) -> int:
    weights = [survivor_count - index for index in range(survivor_count)]
    return rng.choices(range(survivor_count), weights=weights, k=1)[0]


def run_replication_experiment(
    *,
    experiment_name: str,
    seed: int,
    config: ReplicationSearchConfig | None = None,
    require_exact: bool = False,
) -> ExperimentReport:
    config = config or ReplicationSearchConfig()
    best_eval: ReplicationEvaluation | None = None
    best_genome: ReplicationGenome | None = None
    lineage_parents: dict[str, tuple[str, ...]] = {}

    for restart in range(config.restarts):
        rng = Random(seed + restart * 101)
        population = build_initial_replication_population(
            rng,
            size=config.population_size,
            lineage_prefix=f"R{restart}",
            founder_fraction=config.founder_fraction,
            founder_bias_fraction=config.founder_bias_fraction,
        )

        for generation in range(config.generations):
            evaluations = [evaluate_replication_genome(genome) for genome in population]
            ranked = sorted(zip(population, evaluations), key=lambda pair: pair[1].score)
            generation_best_genome, generation_best_eval = ranked[0]
            if best_eval is None or generation_best_eval.score < best_eval.score:
                best_eval = generation_best_eval
                best_genome = generation_best_genome
            if generation_best_eval.exact_replication:
                best_eval = generation_best_eval
                best_genome = generation_best_genome
                break

            survivors = [genome for genome, _evaluation in ranked[: config.survivor_count]]
            next_population: list[ReplicationGenome] = []
            for _index in range(config.population_size):
                child_lineage = f"R{restart}G{generation}.{len(next_population)}"
                parent = survivors[_weighted_survivor_index(rng, len(survivors))]
                if len(survivors) >= 2 and rng.random() < config.crossover_rate:
                    partner = survivors[_weighted_survivor_index(rng, len(survivors))]
                    child = _mate(parent, partner, rng, lineage_id=child_lineage)
                    lineage_parents[child_lineage] = (parent.lineage_id, partner.lineage_id)
                else:
                    child = parent.mutate(
                        rng,
                        mutation_rate=config.mutation_rate,
                        synonym_rate=config.synonym_rate,
                        insertion_rate=config.insertion_rate,
                        deletion_rate=config.deletion_rate,
                        max_length=config.max_length,
                    ).with_lineage(child_lineage)
                    lineage_parents[child_lineage] = (parent.lineage_id,)
                next_population.append(child)
            population = next_population

        final_evaluations = [evaluate_replication_genome(genome) for genome in population]
        restart_best_genome, restart_best_eval = min(zip(population, final_evaluations), key=lambda pair: pair[1].score)
        if best_eval is None or restart_best_eval.score < best_eval.score:
            best_eval = restart_best_eval
            best_genome = restart_best_genome
        if best_eval.exact_replication:
            break

    assert best_eval is not None
    assert best_genome is not None

    report = ExperimentReport(
        experiment=experiment_name,
        train_error=float(best_eval.mismatch_count),
        full_validation_error=0.0 if best_eval.exact_replication else float(best_eval.score),
        cell_count=1,
        active_rules=tuple(decode_op(codon) for codon in best_genome.codons),
        lineage_tree=lineage_tree_text(tuple(sorted(lineage_parents.items()))),
        motifs_per_cell=(),
        shortcut_hits=0,
        trace_examples=best_eval.trace_examples,
        extra={
            "genome_codons": best_genome.codons,
            "offspring_len": best_eval.offspring_len,
            "exact_replication": best_eval.exact_replication,
            "score": best_eval.score,
            "prefix_match": best_eval.prefix_match,
            "neutrality_estimate": best_eval.neutrality_estimate,
            "restarts": config.restarts,
            "generations": config.generations,
            "population_size": config.population_size,
            "founder_fraction": config.founder_fraction,
            "founder_bias_fraction": config.founder_bias_fraction,
        },
    )
    if require_exact and not best_eval.exact_replication:
        raise SystemExit(1)
    return report


def build_replication_program(length: int) -> ReplicationGenome:
    if length < 9:
        raise ValueError("replication program must be at least 9 codons")
    codons = [
        encode_replication_op("PUSH"),
        0,  # literal length, filled after padding
        encode_replication_op("READ"),
        encode_replication_op("WRITE"),
        encode_replication_op("DEC"),
        encode_replication_op("JNZ"),
        encode_signed_literal(-5),
        encode_replication_op("DIVIDE"),
        encode_replication_op("STOP"),
    ]
    while len(codons) < length:
        codons.insert(-2, encode_replication_op("NOOP"))  # NOOP padding before DIVIDE/STOP
    codons[1] = encode_signed_literal(len(codons))
    return ReplicationGenome(codons=tuple(codons), lineage_id=f"selfcopy-{length}")


def decode_op(codon: int) -> str:
    return REPLICATION_CODON_TABLE.op(codon % len(REPLICATION_CODON_TABLE.ops))


def run_replication_program(
    genome: ReplicationGenome,
    *,
    max_steps: int = 512,
) -> ReplicationState:
    state = ReplicationState(genome=genome)
    steps = 0

    while not state.halted and 0 <= state.pc < len(genome.codons) and steps < max_steps:
        codon = genome.codons[state.pc]
        op = decode_op(codon)
        before = state.snapshot()
        state.pc += 1

        if op == "START" or op == "NOOP":
            note = op.lower()
        elif op == "STOP":
            state.halted = True
            note = "stop"
        elif op == "PUSH":
            if state.pc >= len(genome.codons):
                state.halted = True
                note = "push_missing_literal"
            else:
                literal = decode_signed_literal(genome.codons[state.pc])
                state.stack.append(literal)
                state.pc += 1
                note = f"push {literal}"
        elif op == "READ":
            if state.read_cursor < len(genome.codons):
                state.stack.append(genome.codons[state.read_cursor])
                state.read_cursor += 1
                note = "read"
            else:
                note = "read_eof"
        elif op == "WRITE":
            if state.stack:
                state.offspring_buffer.append(state.stack.pop())
                note = "write"
            else:
                note = "write_empty"
        elif op == "DEC":
            if state.stack:
                state.stack[-1] -= 1
                note = f"dec->{state.stack[-1]}"
            else:
                note = "dec_empty"
        elif op == "JNZ":
            if state.pc >= len(genome.codons):
                state.halted = True
                note = "jnz_missing_offset"
            else:
                offset = decode_signed_literal(genome.codons[state.pc])
                state.pc += 1
                cond = state.stack[-1] if state.stack else 0
                if cond != 0:
                    state.pc += offset
                    note = f"jnz {offset}"
                else:
                    note = "jnz_skip"
        elif op == "DIVIDE":
            state.offspring = tuple(state.offspring_buffer)
            state.halted = True
            note = "divide"
        else:
            note = op.lower()

        state.trace.append(
            {
                "step": steps,
                "op": op,
                "before": before,
                "after": state.snapshot(),
                "note": note,
            }
        )
        steps += 1
        if state.offspring is not None and state.halted:
            break

    if state.offspring is None and state.offspring_buffer:
        state.offspring = tuple(state.offspring_buffer)
    return state


def build_exact_copy_program() -> ReplicationGenome:
    return build_replication_program(9)


def offspring_matches_parent(state: ReplicationState) -> bool:
    return state.offspring == state.genome.codons
