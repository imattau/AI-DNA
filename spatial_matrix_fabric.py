from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import Sequence

from codons import crossover_codons, random_codons, unique_ordered
from evolution import attach_success_motif
from genome import CellGenome, Motif, format_motif, motif_statistics, extract_motif_from_rules
from spatial import build_spatial_genome, run_spatial_development


@dataclass(frozen=True, slots=True)
class MatrixCase:
    a: tuple[tuple[float, ...], ...]
    b: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class MatrixFabricTarget:
    grid_size: int = 16
    fabric_size: int = 4
    origin_x: int = 6
    origin_y: int = 6
    steps: int = 12
    test_cases: tuple[MatrixCase, ...] = (
        MatrixCase(
            a=((1.0, 2.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0), (2.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 2.0)),
            b=((1.0, 0.0, 1.0, 2.0), (0.0, 1.0, 2.0, 1.0), (1.0, 1.0, 0.0, 1.0), (2.0, 1.0, 1.0, 0.0)),
        ),
        MatrixCase(
            a=((2.0, 1.0, 0.0, 0.0), (1.0, 0.0, 2.0, 1.0), (0.0, 1.0, 1.0, 2.0), (1.0, 2.0, 1.0, 0.0)),
            b=((0.0, 1.0, 2.0, 1.0), (1.0, 0.0, 1.0, 2.0), (2.0, 1.0, 0.0, 1.0), (1.0, 2.0, 1.0, 0.0)),
        ),
    )

    @property
    def target_positions(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (self.origin_x + dx, self.origin_y + dy)
            for dy in range(self.fabric_size)
            for dx in range(self.fabric_size)
        )

    @property
    def edge_positions(self) -> tuple[tuple[int, int], ...]:
        rows = tuple((0, self.origin_y + dy) for dy in range(self.fabric_size))
        cols = tuple((self.origin_x + dx, 0) for dx in range(self.fabric_size))
        return rows + cols

    @property
    def input_positions(self) -> tuple[tuple[int, int], ...]:
        return self.edge_positions

    @property
    def output_positions(self) -> tuple[tuple[int, int], ...]:
        return self.target_positions

    @property
    def role_map(self) -> dict[tuple[int, int], str]:
        roles: dict[tuple[int, int], str] = {}
        for position in self.input_positions:
            roles[position] = "input"
        for position in self.output_positions:
            roles[position] = "output"
        return roles


@dataclass(frozen=True, slots=True)
class MatrixFabricEvaluation:
    genome: CellGenome
    score: float
    exact_match: bool
    layout_error: float
    input_error: float
    output_error: float
    role_error: float
    matrix_error: float
    correct_output_count: int
    missing_cells: int
    extra_cells: int
    dropout_robustness: float
    occupied_positions: tuple[tuple[int, int], ...]
    outputs: tuple[tuple[tuple[float, ...], ...], ...]
    trace_examples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MatrixFabricSearchConfig:
    population_size: int = 18
    restarts: int = 5
    generations: int = 16
    curriculum_generations: int = 6
    survivor_count: int = 5
    siblings_per_survivor: int = 3
    mutation_rate: float = 0.1
    synonym_rate: float = 0.7
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    crossover_rate: float = 0.35
    founder_fraction: float = 0.5
    founder_bias_fraction: float = 0.6
    founder_rail_bias_fraction: float = 0.7


@dataclass(frozen=True, slots=True)
class MatrixFabricReport:
    experiment: str
    best_score: float
    exact_match: bool
    best_lineage: str
    target_positions: tuple[tuple[int, int], ...]
    edge_positions: tuple[tuple[int, int], ...]
    occupied_positions: tuple[tuple[int, int], ...]
    outputs: tuple[tuple[tuple[float, ...], ...], ...]
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
            f"edge_positions: {self.edge_positions}",
            f"occupied_positions: {self.occupied_positions}",
            f"active_ops: {', '.join(self.active_ops) if self.active_ops else '<none>'}",
            f"motifs_per_cell: {', '.join(self.motifs_per_cell) if self.motifs_per_cell else '<none>'}",
            f"generations: {self.generations}",
            f"restarts: {self.restarts}",
            "trace_examples:",
        ]
        lines.extend(f"  - {trace}" for trace in self.trace_examples or ("<none>",))
        lines.append(f"outputs: {self.outputs}")
        if self.extra:
            lines.append("extra:")
            for key, value in self.extra.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MatrixFabricStreamTask:
    label: str
    target: MatrixFabricTarget


@dataclass(frozen=True, slots=True)
class MatrixFabricStreamConfig:
    population_size: int = 14
    generations_per_task: int = 3
    survivor_count: int = 4
    mutation_rate: float = 0.1
    synonym_rate: float = 0.7
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    crossover_rate: float = 0.35
    founder_fraction: float = 0.5
    founder_bias_fraction: float = 0.6
    founder_rail_bias_fraction: float = 0.7
    motif_transfer_limit: int = 2


@dataclass(frozen=True, slots=True)
class MatrixFabricStreamEpisode:
    episode_index: int
    target_label: str
    best_score: float
    exact_match: bool
    mean_score: float
    best_lineage: str
    target_positions: tuple[tuple[int, int], ...]
    edge_positions: tuple[tuple[int, int], ...]
    occupied_positions: tuple[tuple[int, int], ...]
    outputs: tuple[tuple[tuple[float, ...], ...], ...]
    active_ops: tuple[str, ...]
    motifs_per_cell: tuple[str, ...]
    trace_examples: tuple[str, ...]
    extra: dict[str, object]


@dataclass(frozen=True, slots=True)
class MatrixFabricStreamReport:
    experiment: str
    episodes: tuple[MatrixFabricStreamEpisode, ...]
    best_score: float
    exact_match: bool
    best_lineage: str
    final_target_label: str
    final_target_positions: tuple[tuple[int, int], ...]
    final_edge_positions: tuple[tuple[int, int], ...]
    final_occupied_positions: tuple[tuple[int, int], ...]
    outputs: tuple[tuple[tuple[float, ...], ...], ...]
    active_ops: tuple[str, ...]
    motifs_per_cell: tuple[str, ...]
    generations_per_task: int
    task_count: int
    extra: dict[str, object]

    def format_text(self) -> str:
        lines = [
            f"stream: {self.experiment}",
            f"best_score: {self.best_score:.6f}",
            f"exact_match: {self.exact_match}",
            f"best_lineage: {self.best_lineage}",
            f"final_target_label: {self.final_target_label}",
            f"final_target_positions: {self.final_target_positions}",
            f"final_edge_positions: {self.final_edge_positions}",
            f"final_occupied_positions: {self.final_occupied_positions}",
            f"active_ops: {', '.join(self.active_ops) if self.active_ops else '<none>'}",
            f"motifs_per_cell: {', '.join(self.motifs_per_cell) if self.motifs_per_cell else '<none>'}",
            f"generations_per_task: {self.generations_per_task}",
            f"task_count: {self.task_count}",
            "episodes:",
        ]
        for episode in self.episodes:
            lines.append(
                f"  - episode {episode.episode_index} {episode.target_label}: best_score={episode.best_score:.6f} "
                f"mean_score={episode.mean_score:.6f} exact_match={episode.exact_match} "
                f"best_lineage={episode.best_lineage} "
                f"target_positions={episode.target_positions} "
                f"edge_positions={episode.edge_positions} "
                f"occupied_positions={episode.occupied_positions} "
                f"motifs={len(episode.motifs_per_cell)} reuse_total={episode.extra.get('motif_reuse_total', 0)} "
                f"transfer={episode.extra.get('motif_transfer_count', 0)}/{len(episode.motifs_per_cell) if episode.motifs_per_cell else 0}"
            )
        lines.append(f"outputs: {self.outputs}")
        if self.extra:
            lines.append("extra:")
            for key, value in self.extra.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


SAFE_FABRIC_MOTIFS: tuple[tuple[str, ...], ...] = (
    ("GET_X", "EMIT_0", "DIVIDE_EAST", "DIVIDE_SOUTH", "HALT"),
    ("GET_Y", "EMIT_0", "DIVIDE_SOUTH", "DIVIDE_EAST", "HALT"),
    ("SENSE_0", "SET_TYPE_HIGH", "DIVIDE_EAST", "DIVIDE_SOUTH", "HALT"),
    ("GET_X", "GET_Y", "EMIT_0", "HALT"),
)

FABRIC_RAIL_MOTIFS: tuple[tuple[str, ...], ...] = (
    ("GET_X", "DIVIDE_EAST", "HALT"),
    ("GET_Y", "DIVIDE_SOUTH", "HALT"),
    ("GET_X", "SENSE_0", "HALT"),
    ("GET_Y", "SENSE_0", "HALT"),
    ("GET_X", "EMIT_0", "HALT"),
    ("GET_Y", "EMIT_0", "HALT"),
    ("SENSE_0", "SET_TYPE_HIGH", "HALT"),
)

FABRIC_MOTIF_LIMIT = 8
FABRIC_SHARED_MOTIF_LIMIT = 128


def _matrix_multiply(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    size = len(a)
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(size)) for j in range(size))
        for i in range(size)
    )


def _matrix_columns(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    size = len(matrix)
    return tuple(tuple(matrix[row][col] for row in range(size)) for col in range(size))


def _fabric_trace_examples(report) -> tuple[str, ...]:
    examples: list[str] = []
    for cell in report.cells[:3]:
        for entry in cell.trace[:2]:
            examples.append(
                f"{cell.lineage_id}@{(cell.x, cell.y)}:{entry['op']} {list(entry['before'])} -> {list(entry['after'])}"
                + (f" ({entry['note']})" if entry.get("note") else "")
            )
    return tuple(examples[:8])


def _normalize_motifs(genome: CellGenome, *, limit: int = FABRIC_MOTIF_LIMIT) -> CellGenome:
    motifs: list[Motif] = []
    seen: set[tuple[str, ...]] = set()
    for motif in genome.local_motifs:
        if motif.pattern in seen:
            continue
        seen.add(motif.pattern)
        motifs.append(motif)
    if len(motifs) > limit:
        motifs = motifs[-limit:]
    return replace(genome, local_motifs=tuple(motifs))


def _normalize_motif_pool(motifs: Sequence[Motif], *, limit: int = FABRIC_SHARED_MOTIF_LIMIT) -> list[Motif]:
    pool: list[Motif] = []
    seen: set[tuple[str, ...]] = set()
    for motif in motifs:
        if motif.pattern in seen:
            continue
        seen.add(motif.pattern)
        pool.append(motif)
    if len(pool) > limit:
        pool = pool[-limit:]
    return pool


def _fabric_compute_case(
    report,
    target: MatrixFabricTarget,
    case: MatrixCase,
) -> tuple[tuple[tuple[float, ...], ...], float, int]:
    size = len(case.a)
    occupied_set = set(report.occupied_positions)
    cell_types = {(cell.x, cell.y): cell.type_id for cell in report.cells}
    expected = _matrix_multiply(case.a, case.b)
    columns = _matrix_columns(case.b)
    computed_rows: list[tuple[float, ...]] = []
    matrix_error = 0.0
    correct_output_count = 0

    for row_index in range(size):
        row_pos = (0, target.origin_y + row_index)
        row_ready = row_pos in occupied_set and cell_types.get(row_pos) == 1
        row_vector = tuple(case.a[row_index]) if row_ready else tuple(0.0 for _ in range(size))
        computed_row: list[float] = []
        for col_index in range(size):
            col_pos = (target.origin_x + col_index, 0)
            output_pos = (target.origin_x + col_index, target.origin_y + row_index)
            column_ready = col_pos in occupied_set and cell_types.get(col_pos) == 1
            output_ready = output_pos in occupied_set and cell_types.get(output_pos) == 0
            column_vector = columns[col_index] if column_ready else tuple(0.0 for _ in range(size))
            predicted = sum(row_vector[k] * column_vector[k] for k in range(size)) if output_ready else 0.0
            expected_value = expected[row_index][col_index]
            computed_row.append(predicted)
            matrix_error += abs(predicted - expected_value)
            if abs(predicted - expected_value) < 1e-9:
                correct_output_count += 1
        computed_rows.append(tuple(computed_row))

    return tuple(computed_rows), matrix_error, correct_output_count


def _layout_error(occupied: Sequence[tuple[int, int]], target: MatrixFabricTarget) -> tuple[float, int, int]:
    occupied_set = set(occupied)
    missing = sum(1 for position in target.output_positions if position not in occupied_set)
    extra = max(0, len(occupied) - len(target.target_positions))
    if occupied:
        distance = sum(
            min(abs(tx - ox) + abs(ty - oy) for ox, oy in occupied)
            for tx, ty in target.output_positions
        )
    else:
        distance = float(len(target.output_positions) * target.grid_size)
    return float(distance), missing, extra


def _input_error(occupied: Sequence[tuple[int, int]], target: MatrixFabricTarget) -> tuple[float, int]:
    occupied_set = set(occupied)
    missing = sum(1 for position in target.input_positions if position not in occupied_set)
    distance = sum(
        min(abs(ix - ox) + abs(iy - oy) for ox, oy in occupied)
        for ix, iy in target.input_positions
    ) if occupied else float(len(target.input_positions) * target.grid_size)
    return float(distance), missing


def _role_errors(cell_types: dict[tuple[int, int], int], target: MatrixFabricTarget) -> tuple[float, float, float]:
    input_penalty = 0.0
    output_penalty = 0.0
    for position in target.input_positions:
        if position in cell_types and cell_types[position] != 1:
            input_penalty += 0.5
    for position in target.output_positions:
        if position in cell_types and cell_types[position] != 0:
            output_penalty += 0.25
    return input_penalty + output_penalty, input_penalty, output_penalty


def _dropout_robustness(occupied: Sequence[tuple[int, int]], target: MatrixFabricTarget) -> float:
    occupied_set = set(occupied)
    if not occupied_set:
        return 0.0
    target_set = set(target.target_positions)
    scores: list[float] = []
    for removed in occupied_set:
        remaining = occupied_set - {removed}
        coverage = len(target_set & remaining) / max(1, len(target_set))
        scores.append(coverage)
    return sum(scores) / len(scores)


def _fabric_score(
    *,
    layout_error: float,
    input_error: float,
    role_error: float,
    matrix_error: float,
    correct_output_count: int,
    missing_cells: int,
    extra_cells: int,
    missing_inputs: int,
    motif_reuse_total: int,
    dropout_robustness: float,
    curriculum_phase: str,
) -> float:
    motif_bonus = 0.25 * motif_reuse_total
    if curriculum_phase == "structure":
        matrix_weight = 0.0
        output_weight = 0.0
        layout_weight = 1.25
        input_weight = 1.25
        role_weight = 1.25
        missing_weight = 1.5
        extra_weight = 0.5
        dropout_weight = 0.75
    else:
        matrix_weight = 0.75
        output_weight = 0.25
        layout_weight = 1.0
        input_weight = 1.0
        role_weight = 1.0
        missing_weight = 1.25
        extra_weight = 0.5
        dropout_weight = 0.5
    return (
        matrix_weight * matrix_error
        - output_weight * correct_output_count
        + layout_weight * layout_error
        + input_weight * input_error
        + role_weight * role_error
        + missing_weight * missing_cells
        + extra_weight * extra_cells
        + missing_weight * missing_inputs
        - motif_bonus
        - dropout_weight * dropout_robustness
    )


def evaluate_spatial_matrix_fabric(
    genome: CellGenome,
    target: MatrixFabricTarget,
    *,
    seed: int,
    curriculum_phase: str = "full",
) -> MatrixFabricEvaluation:
    report = run_spatial_development(
        width=target.grid_size,
        height=target.grid_size,
        steps=target.steps,
        seed=seed,
        genome=genome,
    )
    layout_error, missing_cells, extra_cells = _layout_error(report.occupied_positions, target)
    input_error, missing_inputs = _input_error(report.occupied_positions, target)
    cell_types = {(cell.x, cell.y): cell.type_id for cell in report.cells}
    role_error, input_role_error, output_role_error = _role_errors(cell_types, target)
    dropout_robustness = _dropout_robustness(report.occupied_positions, target)
    outputs: list[tuple[tuple[float, ...], ...]] = []
    matrix_error = 0.0
    correct_output_count = 0
    for case in target.test_cases:
        computed, case_error, case_correct = _fabric_compute_case(report, target, case)
        outputs.append(computed)
        matrix_error += case_error
        correct_output_count += case_correct
    motif_reuse_total = sum(motif.reuse_count for motif in genome.local_motifs)
    score = _fabric_score(
        layout_error=layout_error,
        input_error=input_error,
        role_error=role_error,
        matrix_error=matrix_error,
        correct_output_count=correct_output_count,
        missing_cells=missing_cells,
        extra_cells=extra_cells,
        missing_inputs=missing_inputs,
        motif_reuse_total=motif_reuse_total,
        dropout_robustness=dropout_robustness,
        curriculum_phase=curriculum_phase,
    )
    exact_match = (
        matrix_error == 0.0
        and layout_error == 0.0
        and input_error == 0.0
        and missing_cells == 0
        and extra_cells == 0
        and role_error == 0.0
    )
    return MatrixFabricEvaluation(
        genome=genome,
        score=score,
        exact_match=exact_match,
        layout_error=layout_error,
        input_error=input_error,
        output_error=output_role_error,
        role_error=role_error,
        matrix_error=matrix_error,
        correct_output_count=correct_output_count,
        missing_cells=missing_cells,
        extra_cells=extra_cells,
        dropout_robustness=dropout_robustness,
        occupied_positions=report.occupied_positions,
        outputs=tuple(outputs),
        trace_examples=_fabric_trace_examples(report),
    )


def _build_fabric_founder(
    *,
    lineage_id: str,
    rng: Random,
    rail_bias_fraction: float = 0.7,
) -> CellGenome:
    motif_source = FABRIC_RAIL_MOTIFS if rng.random() < rail_bias_fraction else SAFE_FABRIC_MOTIFS
    motif = list(rng.choice(motif_source))
    if rng.random() < 0.5:
        motif.insert(rng.randrange(len(motif) + 1), "NOOP")
    codons = tuple(
        {
            "NOOP": 2,
            "GET_X": 3,
            "GET_Y": 4,
            "SENSE_0": 13,
            "SET_TYPE_HIGH": 14,
            "DIVIDE_EAST": 9,
            "DIVIDE_SOUTH": 12,
            "HALT": 15,
            "EMIT_0": 8,
        }[op]
        for op in motif
    )
    if len(codons) < 6:
        codons = codons + tuple(random_codons(rng, 6 - len(codons), modulus=16))
    motif_record = extract_motif_from_rules(motif, origin_lineage=lineage_id, origin_task="spatial_matrix_fabric")
    return _normalize_motifs(CellGenome(codons=codons, local_motifs=(motif_record,), lineage_id=lineage_id), limit=4)


def _attach_restart_motifs(genome: CellGenome, motifs: Sequence[Motif], *, limit: int = 2) -> CellGenome:
    attached = genome
    for motif in motifs[:limit]:
        attached = attached.attach_motif(motif.inherit())
    return _normalize_motifs(attached)


def _attach_shared_motifs(genome: CellGenome, motifs: Sequence[Motif], *, limit: int = 2) -> CellGenome:
    return _attach_restart_motifs(genome, motifs, limit=limit)


def _mutate_fabric_genome(
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
    return _normalize_motifs(mutated.with_lineage(lineage_id))


def _format_active_ops(genome: CellGenome) -> tuple[str, ...]:
    op_names = []
    lookup = {2: "NOOP", 3: "GET_X", 4: "GET_Y", 8: "EMIT_0", 9: "DIVIDE_EAST", 12: "DIVIDE_SOUTH", 13: "SENSE_0", 14: "SET_TYPE_HIGH", 15: "HALT"}
    for codon in genome.codons:
        op_names.append(lookup.get(codon % 16, "NOOP"))
    return unique_ordered(op_names)


def run_spatial_matrix_fabric_search(
    *,
    experiment_name: str,
    seed: int,
    target: MatrixFabricTarget | None = None,
    config: MatrixFabricSearchConfig | None = None,
) -> MatrixFabricReport:
    target = target or MatrixFabricTarget()
    config = config or MatrixFabricSearchConfig()
    rng = Random(seed)
    best: MatrixFabricEvaluation | None = None
    best_genome: CellGenome | None = None
    shared_motifs: list[Motif] = []
    restart_motif_transfer_count = 0
    restart_motif_pool_size = 0

    for restart in range(config.restarts):
        population: list[CellGenome] = []
        founder_count = max(1, min(config.population_size, int(round(config.population_size * config.founder_fraction))))
        biased_count = max(1, min(founder_count, int(round(founder_count * config.founder_bias_fraction))))
        for index in range(biased_count):
            founder = _build_fabric_founder(
                lineage_id=f"F{restart}B{index + 1}",
                rng=rng,
                rail_bias_fraction=config.founder_rail_bias_fraction,
            )
            if shared_motifs:
                transferred = min(2, len(shared_motifs))
                founder = _attach_restart_motifs(founder, rng.sample(shared_motifs, k=transferred), limit=transferred)
                restart_motif_transfer_count += transferred
            population.append(founder)
        while len(population) < founder_count:
            founder = build_spatial_genome(
                tuple(rng.choice(("NOOP", "GET_X", "GET_Y", "SENSE_0", "DIVIDE_EAST", "DIVIDE_SOUTH", "HALT")) for _ in range(rng.randrange(5, 10))),
                lineage_id=f"F{restart}F{len(population) + 1}",
            )
            if shared_motifs:
                transferred = min(1, len(shared_motifs))
                founder = _attach_restart_motifs(founder, rng.sample(shared_motifs, k=transferred), limit=transferred)
                restart_motif_transfer_count += transferred
            population.append(founder)
        while len(population) < config.population_size:
            founder = build_spatial_genome(
                tuple(rng.choice(("NOOP", "GET_X", "GET_Y", "SENSE_0", "DIVIDE_EAST", "DIVIDE_SOUTH", "HALT")) for _ in range(rng.randrange(5, 10))),
                lineage_id=f"F{restart}R{len(population) + 1}",
            )
            if shared_motifs:
                transferred = min(1, len(shared_motifs))
                founder = _attach_restart_motifs(founder, rng.sample(shared_motifs, k=transferred), limit=transferred)
                restart_motif_transfer_count += transferred
            population.append(founder)

        for generation in range(config.generations):
            curriculum_phase = "structure" if generation < config.curriculum_generations else "full"
            evaluations = [
                evaluate_spatial_matrix_fabric(
                    genome,
                    target,
                    seed=seed + restart * 101 + generation,
                    curriculum_phase=curriculum_phase,
                )
                for genome in population
            ]
            generation_best = min(evaluations, key=lambda evaluation: evaluation.score)
            if best is None or generation_best.score < best.score:
                best = generation_best
                best_genome = generation_best.genome
            if generation_best.exact_match:
                best = generation_best
                best_genome = generation_best.genome
                break
            survivors = [evaluation.genome for evaluation in sorted(evaluations, key=lambda evaluation: evaluation.score)[: config.survivor_count]]
            next_population: list[CellGenome] = survivors[: max(1, len(survivors) // 2)]
            while len(next_population) < config.population_size:
                parent = rng.choice(survivors)
                child = _mutate_fabric_genome(
                    parent,
                    rng,
                    lineage_id=f"{parent.lineage_id}.m{generation + 1}.{len(next_population) + 1}",
                    mutation_rate=config.mutation_rate,
                    synonym_rate=config.synonym_rate,
                    insertion_rate=config.insertion_rate,
                    deletion_rate=config.deletion_rate,
                )
                if len(survivors) > 1 and rng.random() < config.crossover_rate:
                    partner = rng.choice([candidate for candidate in survivors if candidate.lineage_id != parent.lineage_id] or survivors)
                    child = _mutate_fabric_genome(
                        CellGenome(
                            codons=crossover_codons(parent.codons, partner.codons, rng),
                            local_motifs=parent.local_motifs + partner.local_motifs,
                            lineage_id=child.lineage_id,
                        ),
                        rng,
                        lineage_id=child.lineage_id,
                        mutation_rate=config.mutation_rate * 0.7,
                        synonym_rate=config.synonym_rate,
                        insertion_rate=config.insertion_rate,
                        deletion_rate=config.deletion_rate,
                    )
                next_population.append(child)
            population = next_population

        if best_genome is not None and best_genome.local_motifs:
            shared_motifs = _normalize_motif_pool(shared_motifs + list(best_genome.local_motifs))
            restart_motif_pool_size = len(shared_motifs)

    assert best is not None
    assert best_genome is not None
    return MatrixFabricReport(
        experiment=experiment_name,
        best_score=best.score,
        exact_match=best.exact_match,
        best_lineage=best_genome.lineage_id,
        target_positions=target.target_positions,
        edge_positions=target.edge_positions,
        occupied_positions=best.occupied_positions,
        outputs=best.outputs,
        active_ops=_format_active_ops(best_genome),
        motifs_per_cell=tuple(format_motif(motif) for motif in best_genome.local_motifs),
        trace_examples=best.trace_examples,
        generations=config.generations,
        restarts=config.restarts,
        extra={
            "layout_error": best.layout_error,
            "input_error": best.input_error,
            "output_error": best.output_error,
            "role_error": best.role_error,
            "matrix_error": best.matrix_error,
            "correct_output_count": best.correct_output_count,
            "missing_cells": best.missing_cells,
            "extra_cells": best.extra_cells,
            "dropout_robustness": best.dropout_robustness,
            "restart_motif_transfer_count": restart_motif_transfer_count,
            "restart_motif_pool_size": restart_motif_pool_size,
            "curriculum_generations": config.curriculum_generations,
            "founder_rail_bias_fraction": config.founder_rail_bias_fraction,
            **motif_statistics(best_genome.local_motifs),
        },
    )


def _select_diverse_survivors(evaluations: Sequence[MatrixFabricEvaluation], *, k: int) -> tuple[CellGenome, ...]:
    ranked = sorted(evaluations, key=lambda evaluation: evaluation.score)
    survivors: list[CellGenome] = []
    seen: set[tuple[int, ...]] = set()
    for evaluation in ranked:
        signature = evaluation.genome.signature()
        if signature in seen:
            continue
        seen.add(signature)
        survivors.append(evaluation.genome)
        if len(survivors) >= k:
            break
    if len(survivors) < k:
        for evaluation in ranked:
            if evaluation.genome in survivors:
                continue
            survivors.append(evaluation.genome)
            if len(survivors) >= k:
                break
    return tuple(survivors)


def run_spatial_matrix_fabric_stream(
    *,
    experiment_name: str,
    seed: int,
    tasks: Sequence[MatrixFabricStreamTask],
    config: MatrixFabricStreamConfig | None = None,
) -> MatrixFabricStreamReport:
    if not tasks:
        raise ValueError("tasks must not be empty")
    config = config or MatrixFabricStreamConfig()
    rng = Random(seed)

    founder_count = max(1, min(config.population_size, int(round(config.population_size * config.founder_fraction))))
    biased_count = max(1, min(founder_count, int(round(founder_count * config.founder_bias_fraction))))
    population: list[CellGenome] = []
    for index in range(biased_count):
        population.append(
            _build_fabric_founder(
                lineage_id=f"F0B{index + 1}",
                rng=rng,
                rail_bias_fraction=config.founder_rail_bias_fraction,
            )
        )
    while len(population) < founder_count:
        population.append(
            build_spatial_genome(
                tuple(rng.choice(("NOOP", "GET_X", "GET_Y", "SENSE_0", "DIVIDE_EAST", "DIVIDE_SOUTH", "HALT")) for _ in range(rng.randrange(5, 10))),
                lineage_id=f"F0R{len(population) + 1}",
            )
        )
    while len(population) < config.population_size:
        population.append(
            build_spatial_genome(
                tuple(rng.choice(("NOOP", "GET_X", "GET_Y", "SENSE_0", "DIVIDE_EAST", "DIVIDE_SOUTH", "HALT")) for _ in range(rng.randrange(5, 10))),
                lineage_id=f"F0X{len(population) + 1}",
            )
        )

    best: MatrixFabricEvaluation | None = None
    best_genome: CellGenome | None = None
    shared_motifs: list[Motif] = []
    shared_motif_transfer_count = 0
    episodes: list[MatrixFabricStreamEpisode] = []

    for episode_index, task in enumerate(tasks):
        episode_best: MatrixFabricEvaluation | None = None
        episode_mean_score = 0.0
        for generation in range(config.generations_per_task):
            evaluations = [
                evaluate_spatial_matrix_fabric(
                    genome,
                    task.target,
                    seed=seed + episode_index * 101 + generation,
                )
                for genome in population
            ]
            generation_best = min(evaluations, key=lambda evaluation: evaluation.score)
            episode_mean_score = sum(evaluation.score for evaluation in evaluations) / max(1, len(evaluations))
            if episode_best is None or generation_best.score < episode_best.score:
                episode_best = generation_best
            if best is None or generation_best.score < best.score:
                best = generation_best
                best_genome = generation_best.genome
            survivors = _select_diverse_survivors(evaluations, k=config.survivor_count)
            if survivors:
                survivors = list(survivors)
                survivors[0] = _normalize_motifs(attach_success_motif(survivors[0], task.label))
            next_population: list[CellGenome] = []
            while len(next_population) < config.population_size and survivors:
                parent = rng.choice(survivors)
                child = _mutate_fabric_genome(
                    parent,
                    rng,
                    lineage_id=f"{parent.lineage_id}.t{episode_index + 1}.{len(next_population) + 1}",
                    mutation_rate=config.mutation_rate,
                    synonym_rate=config.synonym_rate,
                    insertion_rate=config.insertion_rate,
                    deletion_rate=config.deletion_rate,
                )
                if len(survivors) > 1 and rng.random() < config.crossover_rate:
                    partner = rng.choice([candidate for candidate in survivors if candidate.lineage_id != parent.lineage_id] or survivors)
                    child = _mutate_fabric_genome(
                        CellGenome(
                            codons=crossover_codons(parent.codons, partner.codons, rng),
                            local_motifs=parent.local_motifs + partner.local_motifs,
                            lineage_id=child.lineage_id,
                        ),
                        rng,
                        lineage_id=child.lineage_id,
                        mutation_rate=config.mutation_rate * 0.7,
                        synonym_rate=config.synonym_rate,
                        insertion_rate=config.insertion_rate,
                        deletion_rate=config.deletion_rate,
                    )
                if shared_motifs and rng.random() < 0.5:
                    transferred = min(config.motif_transfer_limit, len(shared_motifs))
                    if transferred > 0:
                        child = _attach_shared_motifs(child, rng.sample(shared_motifs, k=transferred), limit=transferred)
                        shared_motif_transfer_count += transferred
                child = _normalize_motifs(child)
                next_population.append(child)
            if next_population:
                population = next_population
            if episode_best is not None and episode_best.exact_match:
                break

        assert episode_best is not None
        if episode_best.genome.local_motifs:
            shared_motifs = _normalize_motif_pool(shared_motifs + list(episode_best.genome.local_motifs))
        episodes.append(
            MatrixFabricStreamEpisode(
                episode_index=episode_index + 1,
                target_label=task.label,
                best_score=episode_best.score,
                exact_match=episode_best.exact_match,
                mean_score=episode_mean_score,
                best_lineage=episode_best.genome.lineage_id,
                target_positions=task.target.target_positions,
                edge_positions=task.target.edge_positions,
                occupied_positions=episode_best.occupied_positions,
                outputs=episode_best.outputs,
                active_ops=_format_active_ops(episode_best.genome),
                motifs_per_cell=tuple(format_motif(motif) for motif in episode_best.genome.local_motifs),
                trace_examples=episode_best.trace_examples,
                extra={
                    "layout_error": episode_best.layout_error,
                    "input_error": episode_best.input_error,
                    "output_error": episode_best.output_error,
                    "role_error": episode_best.role_error,
                    "matrix_error": episode_best.matrix_error,
                    "correct_output_count": episode_best.correct_output_count,
                    "missing_cells": episode_best.missing_cells,
                    "extra_cells": episode_best.extra_cells,
                    "dropout_robustness": episode_best.dropout_robustness,
                    "motif_count": len(episode_best.genome.local_motifs),
                    "motif_reuse_total": sum(motif.reuse_count for motif in episode_best.genome.local_motifs),
                    "motif_transfer_count": sum(1 for motif in episode_best.genome.local_motifs if motif.origin_task != task.label),
                    "motif_transfer_ratio": (
                        sum(1 for motif in episode_best.genome.local_motifs if motif.origin_task != task.label)
                        / max(1, len(episode_best.genome.local_motifs))
                    ),
                    "motif_origin_tasks": tuple(sorted({motif.origin_task for motif in episode_best.genome.local_motifs})),
                    "motif_origin_lineages": tuple(sorted({motif.origin_lineage for motif in episode_best.genome.local_motifs})),
                },
            )
        )

    assert best is not None
    assert best_genome is not None
    final_episode = episodes[-1]
    return MatrixFabricStreamReport(
        experiment=experiment_name,
        episodes=tuple(episodes),
        best_score=best.score,
        exact_match=best.exact_match,
        best_lineage=best_genome.lineage_id,
        final_target_label=final_episode.target_label,
        final_target_positions=final_episode.target_positions,
        final_edge_positions=final_episode.edge_positions,
        final_occupied_positions=final_episode.occupied_positions,
        outputs=final_episode.outputs,
        active_ops=final_episode.active_ops,
        motifs_per_cell=final_episode.motifs_per_cell,
        generations_per_task=config.generations_per_task,
        task_count=len(tasks),
        extra={
            "shared_motif_pool_size": len(shared_motifs),
            "shared_motif_transfer_count": shared_motif_transfer_count,
            "best_layout_error": best.layout_error,
            "best_input_error": best.input_error,
            "best_output_error": best.output_error,
            "best_role_error": best.role_error,
            "best_dropout_robustness": best.dropout_robustness,
            "founder_rail_bias_fraction": config.founder_rail_bias_fraction,
            **motif_statistics(best_genome.local_motifs),
        },
    )
