from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Callable, Sequence

from evolution import (
    EvolutionConfig,
    Evaluation,
    attach_success_motif,
    crossover_genomes,
    diversify_genomes,
    mutate_genome,
    random_genome,
    select_top_diverse,
)
from genome import CellGenome


@dataclass(slots=True)
class Colony:
    members: list[CellGenome]
    rng: Random
    generation: int = 0
    lineage_parents: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def evaluate(self, scorer: Callable[[CellGenome], Evaluation]) -> tuple[Evaluation, ...]:
        return tuple(scorer(member) for member in self.members)

    def advance(
        self,
        scorer: Callable[[CellGenome], Evaluation],
        *,
        survivor_count: int = 2,
        siblings_per_survivor: int = 3,
        task_name: str = "multiply",
        config: EvolutionConfig | None = None,
    ) -> tuple[Evaluation, ...]:
        config = config or EvolutionConfig()
        evaluations = self.evaluate(scorer)
        survivors = select_top_diverse(evaluations, k=survivor_count)
        target_size = len(self.members)
        next_members: list[CellGenome] = []
        for index, survivor in enumerate(survivors, start=1):
            elite = attach_success_motif(survivor.genome, task_name).with_lineage(
                f"{survivor.genome.lineage_id}.{self.generation + 1}.{index}"
            )
            next_members.append(elite)
            self.lineage_parents[elite.lineage_id] = (survivor.genome.lineage_id,)
            for sibling_index in range(siblings_per_survivor):
                if len(next_members) >= target_size:
                    break
                sibling = mutate_genome(
                    elite,
                    self.rng,
                    lineage_id=f"{elite.lineage_id}.s{sibling_index + 1}",
                    mutation_rate=config.mutation_rate,
                    insertion_rate=config.insertion_rate,
                    deletion_rate=config.deletion_rate,
                    motif_mutation_rate=config.motif_mutation_rate,
                )
                next_members.append(sibling)
                self.lineage_parents[sibling.lineage_id] = (elite.lineage_id,)

        parent_pool = [survivor.genome for survivor in survivors]
        child_index = 0
        while len(next_members) < target_size:
            parent = parent_pool[child_index % len(parent_pool)]
            child_index += 1
            if self.rng.random() < config.immigrant_rate:
                child = random_genome(
                    self.rng,
                    lineage_id=f"{parent.lineage_id}.{self.generation + 1}.i{child_index}",
                    length=self.rng.randrange(4, 8),
                )
            elif len(parent_pool) > 1 and self.rng.random() < config.crossover_rate:
                mate = parent_pool[self.rng.randrange(len(parent_pool))]
                if mate.lineage_id == parent.lineage_id and len(parent_pool) > 2:
                    mate = parent_pool[(parent_pool.index(parent) + 1) % len(parent_pool)]
                child = crossover_genomes(
                    parent,
                    mate,
                    self.rng,
                    lineage_id=f"{parent.lineage_id}.{self.generation + 1}.x{child_index}",
                )
                child = mutate_genome(
                    child,
                    self.rng,
                    lineage_id=child.lineage_id,
                    mutation_rate=config.mutation_rate,
                    insertion_rate=config.insertion_rate,
                    deletion_rate=config.deletion_rate,
                    motif_mutation_rate=config.motif_mutation_rate,
                )
            else:
                child = mutate_genome(
                    parent,
                    self.rng,
                    lineage_id=f"{parent.lineage_id}.{self.generation + 1}.{child_index}",
                    mutation_rate=config.mutation_rate,
                    insertion_rate=config.insertion_rate,
                    deletion_rate=config.deletion_rate,
                    motif_mutation_rate=config.motif_mutation_rate,
                )
            next_members.append(child)
            self.lineage_parents[child.lineage_id] = (parent.lineage_id,)

        next_members = list(diversify_genomes(next_members))
        while len(next_members) < target_size:
            parent = parent_pool[self.rng.randrange(len(parent_pool))]
            immigrant = mutate_genome(
                parent,
                self.rng,
                lineage_id=f"{parent.lineage_id}.{self.generation + 1}.i{len(next_members)}",
                mutation_rate=config.mutation_rate,
                insertion_rate=config.insertion_rate,
                deletion_rate=config.deletion_rate,
                motif_mutation_rate=config.motif_mutation_rate,
            )
            next_members.append(immigrant)
            self.lineage_parents[immigrant.lineage_id] = (parent.lineage_id,)

        self.members = next_members[:target_size]
        self.generation += 1
        return evaluations
