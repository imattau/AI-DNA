from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

from codons import crossover_codons, random_codons, unique_ordered
from genome import CellGenome, format_motif, motif_statistics
from spatial3d import (
    SPATIAL3D_OPS,
    build_spatial3d_genome,
    encode_spatial3d_op,
    decode_spatial3d_op,
    run_spatial3d_development,
)
from tracing import lineage_tree_text


@dataclass(frozen=True, slots=True)
class SpatialTargetCell:
    position: tuple[int, int, int]
    type_id: int = 0


@dataclass(frozen=True, slots=True)
class SpatialBodyPlanTarget:
    width: int
    height: int
    depth: int
    target_cells: tuple[SpatialTargetCell, ...]
    steps: int = 10

    @property
    def occupied_positions(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(cell.position for cell in self.target_cells)


@dataclass(frozen=True, slots=True)
class SpatialBodyPlanEvaluation:
    genome: CellGenome
    score: float
    exact_match: bool
    position_error: float
    missing_cells: int
    extra_cells: int
    type_error: float
    morphogen_peak: float
    occupied_positions: tuple[tuple[int, int, int], ...]
    trace_examples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpatialBodyPlanSearchConfig:
    population_size: int = 18
    restarts: int = 6
    generations: int = 18
    survivor_count: int = 5
    siblings_per_survivor: int = 3
    mutation_rate: float = 0.1
    synonym_rate: float = 0.7
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    crossover_rate: float = 0.35
    founder_fraction: float = 0.5
    founder_bias_fraction: float = 0.6
    width: int = 4
    height: int = 4
    depth: int = 4
    steps: int = 10
    mutation_steps: int = 1
    random_length_min: int = 7
    random_length_max: int = 13


SAFE_SPATIAL_BODY_MOTIFS: tuple[tuple[str, ...], ...] = (
    ("GET_X", "EMIT_0", "DIVIDE_EAST", "HALT"),
    ("GET_Y", "EMIT_0", "DIVIDE_SOUTH", "HALT"),
    ("GET_Z", "EMIT_0", "DIVIDE_UP", "HALT"),
    ("SENSE_0", "SET_TYPE_HIGH", "DIVIDE_EAST", "HALT"),
    ("SENSE_GRAD_X_0", "SENSE_0", "DIVIDE_SOUTH", "HALT"),
    ("SENSE_GRAD_Z_0", "SET_TYPE_LOW", "DIVIDE_UP", "HALT"),
    ("GET_X", "GET_Y", "EMIT_0", "HALT"),
)


@dataclass(frozen=True, slots=True)
class SpatialBodyPlanReport:
    experiment: str
    target_positions: tuple[tuple[int, int, int], ...]
    best_score: float
    exact_match: bool
    best_lineage: str
    occupied_positions: tuple[tuple[int, int, int], ...]
    morphogen_peak: float
    active_ops: tuple[str, ...]
    motifs_per_cell: tuple[str, ...]
    trace_examples: tuple[str, ...]
    generations: int
    restarts: int
    extra: dict[str, object]

    def format_text(self) -> str:
        lines = [
            f"experiment: {self.experiment}",
            f"best_score: {self.best_score:.6f}",
            f"exact_match: {self.exact_match}",
            f"best_lineage: {self.best_lineage}",
            f"target_positions: {self.target_positions}",
            f"occupied_positions: {self.occupied_positions}",
            f"morphogen_peak: {self.morphogen_peak:.6f}",
            f"active_ops: {', '.join(self.active_ops) if self.active_ops else '<none>'}",
            f"motifs_per_cell: {', '.join(self.motifs_per_cell) if self.motifs_per_cell else '<none>'}",
            f"generations: {self.generations}",
            f"restarts: {self.restarts}",
            "trace_examples:",
        ]
        lines.extend(f"  - {trace}" for trace in self.trace_examples or ("<none>",))
        if self.extra:
            lines.append("extra:")
            for key, value in self.extra.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def build_default_body_plan_target(*, width: int, height: int, depth: int) -> SpatialBodyPlanTarget:
    center = (width // 2, height // 2, depth // 2)
    target_cells = (
        SpatialTargetCell(position=center),
        SpatialTargetCell(position=(min(width - 1, center[0] + 1), center[1], center[2])),
        SpatialTargetCell(position=(center[0], min(height - 1, center[1] + 1), center[2])),
        SpatialTargetCell(position=(center[0], center[1], min(depth - 1, center[2] + 1))),
    )
    return SpatialBodyPlanTarget(width=width, height=height, depth=depth, target_cells=target_cells)


def _manhattan(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return sum(abs(a - b) for a, b in zip(left, right, strict=False))


def _target_lookup(target: SpatialBodyPlanTarget) -> dict[tuple[int, int, int], SpatialTargetCell]:
    return {cell.position: cell for cell in target.target_cells}


def _build_trace_examples(report) -> tuple[str, ...]:
    examples: list[str] = []
    for cell in report.cells[:3]:
        for entry in cell.trace[:2]:
            position = entry.get("position")
            examples.append(
                f"{cell.lineage_id}@{position}:{entry['op']} {list(entry['before'])} -> {list(entry['after'])}"
                + (f" ({entry['note']})" if entry.get("note") else "")
            )
    return tuple(examples[:8])


def evaluate_spatial_body_plan(
    genome: CellGenome,
    target: SpatialBodyPlanTarget,
    *,
    seed: int,
) -> SpatialBodyPlanEvaluation:
    report = run_spatial3d_development(
        width=target.width,
        height=target.height,
        depth=target.depth,
        steps=target.steps,
        seed=seed,
        genome=genome,
    )
    occupied_positions = report.occupied_positions
    occupied_set = set(occupied_positions)
    target_lookup = _target_lookup(target)
    occupied_lookup = {(cell.x, cell.y, cell.z): cell for cell in report.cells}

    if occupied_positions:
        position_error = sum(
            min(_manhattan(target_cell.position, occupied) for occupied in occupied_positions)
            for target_cell in target.target_cells
        )
    else:
        position_error = float(len(target.target_cells) * max(target.width, target.height, target.depth))

    missing_cells = sum(1 for target_cell in target.target_cells if target_cell.position not in occupied_set)
    extra_cells = max(0, len(occupied_positions) - len(target.target_cells))
    type_error = 0.0
    for target_cell in target.target_cells:
        cell = occupied_lookup.get(target_cell.position)
        if cell is None:
            type_error += 1.0
            continue
        if cell.type_id != target_cell.type_id:
            type_error += 1.0

    score = position_error + 1.5 * missing_cells + 0.75 * extra_cells + type_error
    exact_match = score == 0.0
    return SpatialBodyPlanEvaluation(
        genome=genome,
        score=score,
        exact_match=exact_match,
        position_error=position_error,
        missing_cells=missing_cells,
        extra_cells=extra_cells,
        type_error=type_error,
        morphogen_peak=report.morphogen_peak,
        occupied_positions=occupied_positions,
        trace_examples=_build_trace_examples(report),
    )


def _build_biased_spatial_founder(*, lineage_id: str, rng: Random, target: SpatialBodyPlanTarget) -> CellGenome:
    motif = list(rng.choice(SAFE_SPATIAL_BODY_MOTIFS))
    if rng.random() < 0.5:
        motif.insert(rng.randrange(len(motif) + 1), "NOOP")
    if rng.random() < 0.5:
        motif.append(rng.choice(("GET_X", "GET_Y", "GET_Z", "SENSE_0")))
    codons = tuple(encode_spatial3d_op(op_name) for op_name in motif)
    if len(codons) < 6:
        codons = codons + tuple(random_codons(rng, 6 - len(codons), modulus=len(SPATIAL3D_OPS)))
    return CellGenome(codons=codons, local_motifs=(), lineage_id=lineage_id)


def _mutate_spatial_genome(
    genome: CellGenome,
    rng: Random,
    *,
    lineage_id: str,
    mutation_rate: float,
    synonym_rate: float,
    insertion_rate: float,
    deletion_rate: float,
) -> CellGenome:
    mutated = genome.mutate(
        rng,
        mutation_rate=mutation_rate,
        synonym_rate=synonym_rate,
        insertion_rate=insertion_rate,
        deletion_rate=deletion_rate,
        motif_mutation_rate=0.1,
    )
    return mutated.with_lineage(lineage_id)


def _crossover_spatial_genomes(left: CellGenome, right: CellGenome, rng: Random, *, lineage_id: str) -> CellGenome:
    return CellGenome(
        codons=crossover_codons(left.codons, right.codons, rng),
        local_motifs=left.local_motifs + right.local_motifs,
        lineage_id=lineage_id,
    )


def _format_active_ops(genome: CellGenome) -> tuple[str, ...]:
    return unique_ordered(decode_spatial3d_op(codon) for codon in genome.codons)


def run_spatial_body_plan_search(
    *,
    experiment_name: str,
    seed: int,
    target: SpatialBodyPlanTarget | None = None,
    config: SpatialBodyPlanSearchConfig | None = None,
) -> SpatialBodyPlanReport:
    config = config or SpatialBodyPlanSearchConfig()
    target = target or build_default_body_plan_target(width=config.width, height=config.height, depth=config.depth)
    rng = Random(seed)

    best_evaluation: SpatialBodyPlanEvaluation | None = None
    best_genome: CellGenome | None = None

    for restart in range(config.restarts):
        population: list[CellGenome] = []
        founder_count = max(1, min(config.population_size, int(round(config.population_size * config.founder_fraction))))
        biased_count = max(1, min(founder_count, int(round(founder_count * config.founder_bias_fraction))))
        for index in range(biased_count):
            population.append(
                _build_biased_spatial_founder(
                    lineage_id=f"V{restart}B{index + 1}",
                    rng=rng,
                    target=target,
                )
            )
        while len(population) < founder_count:
            population.append(
                build_spatial3d_genome(
                    tuple(rng.choice(SPATIAL3D_OPS) for _ in range(rng.randrange(config.random_length_min, config.random_length_max))),
                    lineage_id=f"V{restart}F{len(population) + 1}",
                )
            )
        while len(population) < config.population_size:
            population.append(
                build_spatial3d_genome(
                    tuple(rng.choice(SPATIAL3D_OPS) for _ in range(rng.randrange(config.random_length_min, config.random_length_max))),
                    lineage_id=f"V{restart}R{len(population) + 1}",
                )
            )

        for generation in range(config.generations):
            evaluations = [
                evaluate_spatial_body_plan(
                    genome,
                    target,
                    seed=seed + restart * 101 + generation,
                )
                for genome in population
            ]
            generation_best = min(evaluations, key=lambda evaluation: evaluation.score)
            if best_evaluation is None or generation_best.score < best_evaluation.score:
                best_evaluation = generation_best
                best_genome = generation_best.genome
            if generation_best.exact_match:
                best_evaluation = generation_best
                best_genome = generation_best.genome
                break

            survivors = [evaluation.genome for evaluation in sorted(evaluations, key=lambda evaluation: evaluation.score)[: config.survivor_count]]
            next_population: list[CellGenome] = []
            next_population.extend(survivors[: max(1, len(survivors) // 2)])
            while len(next_population) < config.population_size:
                parent = rng.choice(survivors)
                if len(survivors) > 1 and rng.random() < config.crossover_rate:
                    partner = rng.choice([candidate for candidate in survivors if candidate.lineage_id != parent.lineage_id] or survivors)
                    child = _crossover_spatial_genomes(
                        parent,
                        partner,
                        rng,
                        lineage_id=f"{parent.lineage_id}.x{generation + 1}.{len(next_population) + 1}",
                    )
                    child = _mutate_spatial_genome(
                        child,
                        rng,
                        lineage_id=child.lineage_id,
                        mutation_rate=config.mutation_rate * 0.7,
                        synonym_rate=config.synonym_rate,
                        insertion_rate=config.insertion_rate,
                        deletion_rate=config.deletion_rate,
                    )
                else:
                    child = _mutate_spatial_genome(
                        parent,
                        rng,
                        lineage_id=f"{parent.lineage_id}.m{generation + 1}.{len(next_population) + 1}",
                        mutation_rate=config.mutation_rate,
                        synonym_rate=config.synonym_rate,
                        insertion_rate=config.insertion_rate,
                        deletion_rate=config.deletion_rate,
                    )
                next_population.append(child)
            population = next_population

    assert best_evaluation is not None
    assert best_genome is not None
    return SpatialBodyPlanReport(
        experiment=experiment_name,
        target_positions=target.occupied_positions,
        best_score=best_evaluation.score,
        exact_match=best_evaluation.exact_match,
        best_lineage=best_genome.lineage_id,
        occupied_positions=best_evaluation.occupied_positions,
        morphogen_peak=best_evaluation.morphogen_peak,
        active_ops=_format_active_ops(best_genome),
        motifs_per_cell=tuple(format_motif(motif) for motif in best_genome.local_motifs),
        trace_examples=best_evaluation.trace_examples,
        generations=config.generations,
        restarts=config.restarts,
        extra={
            "position_error": best_evaluation.position_error,
            "missing_cells": best_evaluation.missing_cells,
            "extra_cells": best_evaluation.extra_cells,
            "type_error": best_evaluation.type_error,
            "target_shape": "center_plus_axes",
            "score_is_exact": best_evaluation.exact_match,
            **motif_statistics(best_genome.local_motifs),
        },
    )
