from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import Iterable, Sequence

from codons import RULE_NAMES, codons_from_rule_names, decode_rule_name, insert_delete_codons, mutate_codons, unique_ordered


@dataclass(frozen=True, slots=True)
class Motif:
    pattern: tuple[str, ...]
    origin_lineage: str
    origin_task: str
    reuse_count: int = 0

    def inherit(self) -> "Motif":
        return replace(self, reuse_count=self.reuse_count + 1)


@dataclass(frozen=True, slots=True)
class CellGenome:
    codons: tuple[int, ...]
    local_motifs: tuple[Motif, ...]
    lineage_id: str

    @classmethod
    def from_rule_names(
        cls,
        rule_names: Sequence[str],
        *,
        lineage_id: str,
        local_motifs: Sequence[Motif] | None = None,
    ) -> "CellGenome":
        return cls(
            codons=codons_from_rule_names(rule_names),
            local_motifs=tuple(local_motifs or ()),
            lineage_id=lineage_id,
        )

    def declare_rules(self) -> tuple[str, ...]:
        declared = [decode_rule_name(codon) for codon in self.codons]
        for motif in self.local_motifs:
            declared.extend(motif.pattern)
        return unique_ordered(declared)

    def with_lineage(self, lineage_id: str) -> "CellGenome":
        return replace(self, lineage_id=lineage_id)

    def mutate(
        self,
        rng: Random,
        *,
        mutation_rate: float = 0.08,
        insertion_rate: float = 0.05,
        deletion_rate: float = 0.03,
        motif_mutation_rate: float = 0.2,
    ) -> "CellGenome":
        codons = mutate_codons(self.codons, rng, mutation_rate=mutation_rate)
        codons = insert_delete_codons(
            codons,
            rng,
            insertion_rate=insertion_rate,
            deletion_rate=deletion_rate,
        )
        motifs = list(self.local_motifs)
        if motifs and rng.random() < motif_mutation_rate:
            motifs[rng.randrange(len(motifs))] = motifs[rng.randrange(len(motifs))].inherit()
        return replace(self, codons=codons, local_motifs=tuple(motifs))

    def attach_motif(self, motif: Motif) -> "CellGenome":
        return replace(self, local_motifs=self.local_motifs + (motif,))

    def signature(self) -> tuple[int, ...]:
        return self.codons


def extract_motif_from_rules(
    declared_rules: Sequence[str],
    *,
    origin_lineage: str,
    origin_task: str,
) -> Motif:
    return Motif(
        pattern=unique_ordered(declared_rules),
        origin_lineage=origin_lineage,
        origin_task=origin_task,
    )


def is_subsequence(candidate: Sequence[str], reference: Sequence[str]) -> bool:
    if not candidate:
        return True
    it = iter(reference)
    return all(any(rule == value for value in it) for rule in candidate)


def unique_genomes_by_signature(genomes: Sequence[CellGenome]) -> tuple[CellGenome, ...]:
    seen: set[tuple[int, ...]] = set()
    unique: list[CellGenome] = []
    for genome in genomes:
        signature = genome.signature()
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(genome)
    return tuple(unique)
