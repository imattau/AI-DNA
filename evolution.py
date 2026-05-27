from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence

from codons import crossover_codons, random_codons
from genome import CellGenome, Motif, extract_motif_from_rules, unique_genomes_by_signature


RULE_PRIOR_SETS: tuple[tuple[str, ...], ...] = (
    ("RULE_EMIT_X", "RULE_COPY0_3", "RULE_ADD0_IF1", "RULE_DECAY1", "RULE_OUTPUT_IF1Z"),
    ("RULE_EMIT_Y", "RULE_COPY1_3", "RULE_ADD0_IF1", "RULE_DECAY1", "RULE_OUTPUT_IF1Z"),
    ("RULE_COPY0_3", "RULE_ADD0_IF1", "RULE_DECAY1", "RULE_OUTPUT_IF1Z"),
    ("RULE_COPY1_3", "RULE_ADD0_IF1", "RULE_DECAY1", "RULE_OUTPUT_IF1Z"),
    ("RULE_THRESH3", "RULE_ADD0_IF1", "RULE_DECAY1", "RULE_OUTPUT_IF1Z"),
    ("RULE_EMIT_X", "RULE_ADD3_IF1", "RULE_DECAY3", "RULE_OUTPUT_IF1Z"),
)


@dataclass(frozen=True, slots=True)
class Evaluation:
    genome: CellGenome
    train_error: float
    validation_error: float
    shortcut_hits: int
    trace_examples: tuple[str, ...]

    @property
    def score(self) -> float:
        return self.validation_error + self.train_error


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    mutation_rate: float = 0.08
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    motif_mutation_rate: float = 0.2
    crossover_rate: float = 0.35
    immigrant_rate: float = 0.1


def random_genome(
    rng: Random,
    *,
    lineage_id: str,
    length: int = 4,
) -> CellGenome:
    return CellGenome(
        codons=random_codons(rng, length),
        local_motifs=(),
        lineage_id=lineage_id,
    )


def rule_prior_genome(
    rng: Random,
    *,
    lineage_id: str,
    prior_index: int | None = None,
) -> CellGenome:
    prior = RULE_PRIOR_SETS[prior_index if prior_index is not None else rng.randrange(len(RULE_PRIOR_SETS))]
    codons = list(CellGenome.from_rule_names(prior, lineage_id=lineage_id).codons)
    codons.extend(random_codons(rng, rng.randrange(1, 4)))
    return CellGenome(codons=tuple(codons), local_motifs=(), lineage_id=lineage_id)


def mixed_initial_population(
    rng: Random,
    *,
    size: int,
    lineage_prefix: str = "R",
    prior_fraction: float = 0.5,
) -> list[CellGenome]:
    population: list[CellGenome] = []
    prior_count = max(1, min(size, int(round(size * prior_fraction))))
    for index in range(prior_count):
        population.append(rule_prior_genome(rng, lineage_id=f"{lineage_prefix}P{index + 1}"))
    while len(population) < size:
        population.append(random_genome(rng, lineage_id=f"{lineage_prefix}R{len(population) + 1}", length=rng.randrange(4, 8)))
    return population


def crossover_genomes(left: CellGenome, right: CellGenome, rng: Random, *, lineage_id: str) -> CellGenome:
    return CellGenome(
        codons=crossover_codons(left.codons, right.codons, rng),
        local_motifs=left.local_motifs + right.local_motifs,
        lineage_id=lineage_id,
    )


def mutate_genome(
    genome: CellGenome,
    rng: Random,
    *,
    lineage_id: str,
    mutation_rate: float = 0.08,
    insertion_rate: float = 0.05,
    deletion_rate: float = 0.03,
    motif_mutation_rate: float = 0.2,
) -> CellGenome:
    mutated = genome.mutate(
        rng,
        mutation_rate=mutation_rate,
        insertion_rate=insertion_rate,
        deletion_rate=deletion_rate,
        motif_mutation_rate=motif_mutation_rate,
    )
    return mutated.with_lineage(lineage_id)


def spawn_sibling_variants(
    survivor: CellGenome,
    rng: Random,
    *,
    lineage_prefix: str,
    count: int,
    config: EvolutionConfig | None = None,
) -> tuple[CellGenome, ...]:
    config = config or EvolutionConfig()
    variants = []
    for index in range(count):
        variants.append(
            mutate_genome(
                survivor,
                rng,
                lineage_id=f"{lineage_prefix}.{index + 1}",
                mutation_rate=config.mutation_rate,
                insertion_rate=config.insertion_rate,
                deletion_rate=config.deletion_rate,
                motif_mutation_rate=config.motif_mutation_rate,
            )
        )
    return tuple(variants)


def select_top(evaluations: Sequence[Evaluation], *, k: int) -> tuple[Evaluation, ...]:
    return tuple(sorted(evaluations, key=lambda evaluation: evaluation.score)[:k])


def select_top_diverse(evaluations: Sequence[Evaluation], *, k: int) -> tuple[Evaluation, ...]:
    ranked = sorted(evaluations, key=lambda evaluation: evaluation.score)
    diverse: list[Evaluation] = []
    seen: set[tuple[int, ...]] = set()
    for evaluation in ranked:
        signature = evaluation.genome.signature()
        if signature in seen:
            continue
        seen.add(signature)
        diverse.append(evaluation)
        if len(diverse) >= k:
            break
    if len(diverse) < k:
        for evaluation in ranked:
            if evaluation in diverse:
                continue
            diverse.append(evaluation)
            if len(diverse) >= k:
                break
    return tuple(diverse)


def attach_success_motif(genome: CellGenome, task_name: str) -> CellGenome:
    motif = extract_motif_from_rules(genome.declare_rules(), origin_lineage=genome.lineage_id, origin_task=task_name)
    return genome.attach_motif(motif)


def lineage_edges(genomes: Iterable[CellGenome]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((genome.lineage_id, tuple(motif.origin_lineage for motif in genome.local_motifs)) for genome in genomes)


def diversify_genomes(genomes: Sequence[CellGenome]) -> tuple[CellGenome, ...]:
    return unique_genomes_by_signature(genomes)
