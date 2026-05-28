from __future__ import annotations

from dataclasses import dataclass
from random import Random

from codons import crossover_codons, random_codons, unique_ordered
from genome import CellGenome, format_motif, motif_statistics
from spatial import (
    SPATIAL_OPS,
    SpatialArena,
    SpatialCell,
    build_spatial_genome,
    decode_spatial_op,
    run_spatial_development,
)


@dataclass(frozen=True, slots=True)
class SpatialRoutingTarget:
    width: int
    height: int
    target_cells: tuple[tuple[int, int], ...]
    source_position: tuple[int, int]
    sink_position: tuple[int, int]
    development_steps: int = 6
    route_steps: int = 3
    signal_threshold: float = 0.5

    @property
    def occupied_positions(self) -> tuple[tuple[int, int], ...]:
        return self.target_cells


@dataclass(frozen=True, slots=True)
class SpatialRoutingEvaluation:
    genome: CellGenome
    score: float
    exact_match: bool
    layout_error: float
    missing_cells: int
    extra_cells: int
    signal_strength: float
    signal_error: float
    occupied_positions: tuple[tuple[int, int], ...]
    trace_examples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpatialRoutingSearchConfig:
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
    width: int = 6
    height: int = 5
    development_steps: int = 6
    route_steps: int = 3
    mutation_steps: int = 1
    random_length_min: int = 6
    random_length_max: int = 12


SAFE_SPATIAL_ROUTE_MOTIFS: tuple[tuple[str, ...], ...] = (
    ("GET_X", "EMIT_0", "DIVIDE_EAST", "HALT"),
    ("GET_Y", "EMIT_0", "DIVIDE_EAST", "HALT"),
    ("GET_X", "SENSE_0", "EMIT_0", "DIVIDE_EAST"),
    ("SENSE_0", "ADHERE", "EMIT_0", "HALT"),
    ("GET_X", "GET_Y", "EMIT_0", "HALT"),
    ("SENSE_0", "SET_TYPE_HIGH", "DIVIDE_EAST", "HALT"),
)


@dataclass(frozen=True, slots=True)
class SpatialRoutingReport:
    experiment: str
    target_positions: tuple[tuple[int, int], ...]
    source_position: tuple[int, int]
    sink_position: tuple[int, int]
    best_score: float
    exact_match: bool
    best_lineage: str
    occupied_positions: tuple[tuple[int, int], ...]
    signal_strength: float
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
            f"source_position: {self.source_position}",
            f"sink_position: {self.sink_position}",
            f"signal_strength: {self.signal_strength:.6f}",
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


def build_default_routing_target(*, width: int, height: int) -> SpatialRoutingTarget:
    y = height // 2
    target_cells = tuple((x, y) for x in range(1, width - 1))
    return SpatialRoutingTarget(
        width=width,
        height=height,
        target_cells=target_cells,
        source_position=target_cells[0],
        sink_position=target_cells[-1],
    )


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


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


def evaluate_spatial_routing(
    genome: CellGenome,
    target: SpatialRoutingTarget,
    *,
    seed: int,
) -> SpatialRoutingEvaluation:
    report = run_spatial_development(
        width=target.width,
        height=target.height,
        steps=target.development_steps,
        seed=seed,
        genome=genome,
    )
    arena = SpatialArena(width=target.width, height=target.height)
    for cell in report.cells:
        arena.place(
            SpatialCell(
                genome=cell.genome,
                x=cell.x,
                y=cell.y,
                pc=cell.pc,
                registers=list(cell.registers),
                type_id=cell.type_id,
                output=cell.output,
                alive=cell.alive,
                halted=cell.halted,
                age=cell.age,
                trace=list(cell.trace),
            )
        )
    arena.morphogen_field.emit(*target.source_position, 1.0)
    for _ in range(target.route_steps):
        arena.step()
    routed_signal = arena.morphogen_field.sense(*target.sink_position)
    occupied_positions = tuple(sorted(arena.cells))
    occupied_set = set(occupied_positions)
    if occupied_positions:
        layout_error = sum(
            min(_manhattan(target_position, occupied) for occupied in occupied_positions)
            for target_position in target.target_cells
        )
    else:
        layout_error = float(len(target.target_cells) * max(target.width, target.height))
    missing_cells = sum(1 for target_position in target.target_cells if target_position not in occupied_set)
    extra_cells = max(0, len(occupied_positions) - len(target.target_cells))
    signal_error = max(0.0, target.signal_threshold - routed_signal)
    score = layout_error + 1.5 * missing_cells + 0.75 * extra_cells + 4.0 * signal_error
    exact_match = score == 0.0
    return SpatialRoutingEvaluation(
        genome=genome,
        score=score,
        exact_match=exact_match,
        layout_error=layout_error,
        missing_cells=missing_cells,
        extra_cells=extra_cells,
        signal_strength=routed_signal,
        signal_error=signal_error,
        occupied_positions=occupied_positions,
        trace_examples=_build_trace_examples(report),
    )


def _build_biased_routing_founder(*, lineage_id: str, rng: Random) -> CellGenome:
    motif = list(rng.choice(SAFE_SPATIAL_ROUTE_MOTIFS))
    if rng.random() < 0.5:
        motif.insert(rng.randrange(len(motif) + 1), "NOOP")
    if rng.random() < 0.5:
        motif.append(rng.choice(("GET_X", "GET_Y", "SENSE_0")))
    codons = tuple(SPATIAL_OPS.index(op_name) for op_name in motif)
    if len(codons) < 6:
        codons = codons + tuple(random_codons(rng, 6 - len(codons), modulus=len(SPATIAL_OPS)))
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
    return unique_ordered(decode_spatial_op(codon) for codon in genome.codons)


def build_spatial_routing_demo_genome(*, lineage_id: str = "Q") -> CellGenome:
    return build_spatial_genome(
        (
            "SENSE_0",
            "EMIT_0",
            "DIVIDE_EAST",
            "DIVIDE_WEST",
            "HALT",
        ),
        lineage_id=lineage_id,
    )


def run_spatial_routing_search(
    *,
    experiment_name: str,
    seed: int,
    target: SpatialRoutingTarget | None = None,
    config: SpatialRoutingSearchConfig | None = None,
) -> SpatialRoutingReport:
    config = config or SpatialRoutingSearchConfig()
    target = target or build_default_routing_target(width=config.width, height=config.height)
    rng = Random(seed)

    best_evaluation: SpatialRoutingEvaluation | None = None
    best_genome: CellGenome | None = None

    for restart in range(config.restarts):
        population: list[CellGenome] = []
        founder_count = max(1, min(config.population_size, int(round(config.population_size * config.founder_fraction))))
        biased_count = max(1, min(founder_count, int(round(founder_count * config.founder_bias_fraction))))
        for index in range(biased_count):
            population.append(_build_biased_routing_founder(lineage_id=f"Q{restart}B{index + 1}", rng=rng))
        while len(population) < founder_count:
            population.append(
                build_spatial_genome(
                    tuple(rng.choice(SPATIAL_OPS) for _ in range(rng.randrange(config.random_length_min, config.random_length_max))),
                    lineage_id=f"Q{restart}F{len(population) + 1}",
                )
            )
        while len(population) < config.population_size:
            population.append(
                build_spatial_genome(
                    tuple(rng.choice(SPATIAL_OPS) for _ in range(rng.randrange(config.random_length_min, config.random_length_max))),
                    lineage_id=f"Q{restart}R{len(population) + 1}",
                )
            )

        for generation in range(config.generations):
            evaluations = [
                evaluate_spatial_routing(
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
    return SpatialRoutingReport(
        experiment=experiment_name,
        target_positions=target.occupied_positions,
        source_position=target.source_position,
        sink_position=target.sink_position,
        best_score=best_evaluation.score,
        exact_match=best_evaluation.exact_match,
        best_lineage=best_genome.lineage_id,
        occupied_positions=best_evaluation.occupied_positions,
        signal_strength=best_evaluation.signal_strength,
        active_ops=_format_active_ops(best_genome),
        motifs_per_cell=tuple(format_motif(motif) for motif in best_genome.local_motifs),
        trace_examples=best_evaluation.trace_examples,
        generations=config.generations,
        restarts=config.restarts,
        extra={
            "layout_error": best_evaluation.layout_error,
            "missing_cells": best_evaluation.missing_cells,
            "extra_cells": best_evaluation.extra_cells,
            "signal_error": best_evaluation.signal_error,
            "signal_threshold": target.signal_threshold,
            **motif_statistics(best_genome.local_motifs),
        },
    )
