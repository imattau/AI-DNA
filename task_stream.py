from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Callable, Sequence

from cell import CellState
from chemistry import ChemistrySystem
from evolution import Evaluation, EvolutionConfig, mixed_initial_population, mutate_genome, random_genome
from evolution import attach_success_motif, estimate_neutrality
from genome import CellGenome, format_motif, motif_statistics
from tasks import ContextualTask, TaskBundle, TaskContext, TaskCase, materialize_contextual_bundle


@dataclass(slots=True)
class StreamCell:
    genome: CellGenome
    energy: float
    lineage_id: str
    age: int = 0
    alive: bool = True
    parent_lineage: str | None = None


@dataclass(slots=True)
class StreamTaskResult:
    task_name: str
    context_label: str
    step_index: int
    mean_error: float
    best_error: float
    live_cells: int
    births: int
    deaths: int
    energy_total: float
    max_cells: int
    resource_pool: float
    energy_efficiency: float = 0.0
    lineage_efficiency: float = 0.0
    lineage_efficiency_count: int = 0
    lineage_root: str = ""
    lineage_transfer_scores: tuple[tuple[str, float], ...] = ()
    forgetting_delta: float = 0.0
    motif_reappearance_count: int = 0
    motif_reappearance_by_root: tuple[tuple[str, int], ...] = ()
    resource_regen: float = 0.0
    motif_count: int = 0
    motif_reuse_total: int = 0
    motif_transfer_count: int = 0
    motif_transfer_ratio: float = 0.0
    motif_origin_tasks: tuple[str, ...] = ()
    motif_origin_lineages: tuple[str, ...] = ()
    phenotype_diversity: float = 0.0
    recovery_steps: int = 0


@dataclass(slots=True)
class ArchiveSnapshot:
    generation: int
    current_task: str
    seen_tasks: tuple[str, ...]
    mean_archive_error: float
    per_task_error: tuple[tuple[str, float], ...]
    best_lineage: str
    energy_efficiency: float = 0.0
    lineage_efficiency: float = 0.0
    lineage_efficiency_count: int = 0
    lineage_root: str = ""
    lineage_transfer_scores: tuple[tuple[str, float], ...] = ()
    forgetting_delta: float = 0.0
    motif_reappearance_count: int = 0
    motif_reappearance_by_root: tuple[tuple[str, int], ...] = ()
    resource_regen: float = 0.0
    motif_count: int = 0
    motif_reuse_total: int = 0
    motif_transfer_count: int = 0
    motif_transfer_ratio: float = 0.0
    motif_origin_tasks: tuple[str, ...] = ()
    motif_origin_lineages: tuple[str, ...] = ()
    neutral_drift_rate: float = 0.0


@dataclass(slots=True)
class StreamReport:
    sequence_name: str
    results: tuple[StreamTaskResult, ...]
    archive_snapshots: tuple[ArchiveSnapshot, ...]
    lineage_edges: tuple[tuple[str, tuple[str, ...]], ...]
    final_population: tuple[StreamCell, ...]
    config: "StreamConfig"

    def format_text(self) -> str:
        lines = [f"stream: {self.sequence_name}", f"config: {self.config}"]
        for result in self.results:
            context_suffix = f" [{result.context_label}]" if result.context_label else ""
            lines.append(
                f"- step {result.step_index} {result.task_name}{context_suffix}: mean_error={result.mean_error:.6f} "
                f"best_error={result.best_error:.6f} live_cells={result.live_cells} "
                f"births={result.births} deaths={result.deaths} energy_total={result.energy_total:.2f} "
                f"max_cells={result.max_cells} resource_pool={result.resource_pool:.2f} "
                f"energy_efficiency={result.energy_efficiency:.4f} "
                f"lineage_efficiency={result.lineage_efficiency:.4f} "
                f"lineage_root={result.lineage_root or '<none>'} "
                f"forgetting_delta={result.forgetting_delta:.6f} "
                f"transfer_scores={result.lineage_transfer_scores or '<none>'} "
                f"resource_regen={result.resource_regen:.2f} "
                f"phenotype_diversity={result.phenotype_diversity:.4f} "
                f"recovery_steps={result.recovery_steps} "
                f"motifs={result.motif_count} reuse_total={result.motif_reuse_total} "
                f"transfer={result.motif_transfer_count}/{result.motif_count if result.motif_count else 0} "
                f"motif_reappearance={result.motif_reappearance_count} "
                f"motif_tasks={','.join(result.motif_origin_tasks) if result.motif_origin_tasks else '<none>'} "
                f"motif_lineages={','.join(result.motif_origin_lineages) if result.motif_origin_lineages else '<none>'}"
            )
        if self.archive_snapshots:
            lines.append("archive:")
            for snapshot in self.archive_snapshots:
                lines.append(
                f"  - gen {snapshot.generation} task={snapshot.current_task} "
                f"mean_archive_error={snapshot.mean_archive_error:.6f} best={snapshot.best_lineage} "
                f"energy_efficiency={snapshot.energy_efficiency:.4f} "
                f"lineage_efficiency={snapshot.lineage_efficiency:.4f} "
                f"lineage_root={snapshot.lineage_root or '<none>'} "
                f"forgetting_delta={snapshot.forgetting_delta:.6f} "
                f"transfer_scores={snapshot.lineage_transfer_scores or '<none>'} "
                f"resource_regen={snapshot.resource_regen:.2f} "
                f"neutral_drift_rate={snapshot.neutral_drift_rate:.4f} "
                f"motifs={snapshot.motif_count} reuse_total={snapshot.motif_reuse_total} "
                f"transfer={snapshot.motif_transfer_count}/{snapshot.motif_count if snapshot.motif_count else 0} "
                f"motif_reappearance={snapshot.motif_reappearance_count} "
                f"motif_tasks={','.join(snapshot.motif_origin_tasks) if snapshot.motif_origin_tasks else '<none>'} "
                f"motif_lineages={','.join(snapshot.motif_origin_lineages) if snapshot.motif_origin_lineages else '<none>'}"
                )
        lines.append(f"lineage_edges: {self.lineage_edges}")
        lines.append(f"final_cells: {len(self.final_population)}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class StreamConfig:
    initial_population_size: int = 8
    initial_energy: float = 6.0
    maintenance_cost: float = 0.75
    spawn_cost: float = 2.0
    reproduction_threshold: float = 7.0
    reward_scale: float = 4.0
    resource_pool: float = 8.0
    resource_regen: float = 1.5
    resource_capacity: float = 20.0
    death_threshold: float = 0.0
    max_steps_per_task: int = 8
    archive_interval: int = 10
    chemistry_max_time: float = 32.0
    chemistry_dt: float = 1.0
    survivor_count: int = 3
    mutation_rate: float = 0.08
    synonym_rate: float = 0.65
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    motif_mutation_rate: float = 0.2
    crossover_rate: float = 0.35
    immigrant_rate: float = 0.12
    resource_regen_floor: float = 0.5
    resource_regen_ceiling: float = 3.0
    regen_error_sensitivity: float = 0.4
    regen_birth_bonus: float = 0.15
    lineage_efficiency_sensitivity: float = 0.35
    estimate_neutrality_trials: int = 8


def _probe_prediction(
    genome: CellGenome,
    *,
    x: float = 2.0,
    y: float = 3.0,
    chemistry_max_time: float,
    chemistry_dt: float,
) -> float:
    chemistry = ChemistrySystem(max_time=chemistry_max_time, dt=chemistry_dt)
    cell = CellState(active_rules=list(genome.declare_rules()))
    probe = TaskCase(x=x, y=y, target=x * y, task_name="probe")
    cell.reset(x=x, y=y)
    chemistry.run(cell, probe)
    return float(cell.output if cell.output is not None else cell.signals[2])


def _probe_signature(
    genome: CellGenome,
    *,
    chemistry_max_time: float,
    chemistry_dt: float,
) -> tuple[float, ...]:
    probe_points = ((2.0, 3.0), (5.0, 8.0), (8.0, 5.0), (11.0, 4.0))
    return tuple(
        _probe_prediction(
            genome,
            x=x,
            y=y,
            chemistry_max_time=chemistry_max_time,
            chemistry_dt=chemistry_dt,
        )
        for x, y in probe_points
    )


def _phenotype_diversity(
    cells: Sequence[StreamCell],
    *,
    chemistry_max_time: float,
    chemistry_dt: float,
) -> float:
    live = [cell for cell in cells if cell.alive]
    if len(live) < 2:
        return 0.0
    signatures = [
        _probe_signature(
            cell.genome,
            chemistry_max_time=chemistry_max_time,
            chemistry_dt=chemistry_dt,
        )
        for cell in live
    ]
    pairwise_sum = 0.0
    pairwise_count = 0
    for index, left in enumerate(signatures):
        for right in signatures[index + 1 :]:
            pairwise_sum += sum(abs(left_value - right_value) for left_value, right_value in zip(left, right, strict=False))
            pairwise_count += 1
    if pairwise_count == 0:
        return 0.0
    scale = max(1.0, max(abs(output) for signature in signatures for output in signature))
    return min(1.0, (pairwise_sum / pairwise_count / max(1, len(signatures[0]))) / scale)


@dataclass(slots=True)
class TaskStreamColony:
    rng: Random
    members: list[StreamCell]
    lineage_parents: dict[str, tuple[str, ...]] = field(default_factory=dict)
    archive: list[CellGenome] = field(default_factory=list)
    max_cells_seen: int = 0

    @classmethod
    def from_initial_population(
        cls,
        rng: Random,
        *,
        size: int,
        initial_energy: float,
        lineage_prefix: str = "S",
    ) -> "TaskStreamColony":
        genomes = mixed_initial_population(rng, size=size, lineage_prefix=lineage_prefix, prior_fraction=0.5)
        cells = [
            StreamCell(genome=genome, energy=initial_energy, lineage_id=genome.lineage_id)
            for genome in genomes
        ]
        return cls(rng=rng, members=cells, max_cells_seen=len(cells))

    def live_members(self) -> list[StreamCell]:
        return [cell for cell in self.members if cell.alive]

    def prune_dead(self) -> int:
        before = len(self.members)
        self.members = [cell for cell in self.members if cell.alive]
        return before - len(self.members)

    def archive_genomes(self, cells: Sequence[StreamCell], *, limit: int) -> None:
        ranked = sorted(cells, key=lambda cell: cell.energy, reverse=True)
        for cell in ranked[:limit]:
            self.archive.append(cell.genome)
        if len(self.archive) > limit * 6:
            self.archive = self.archive[-limit * 6 :]

    def rescue_if_empty(self, *, config: StreamConfig) -> int:
        if self.live_members():
            return 0
        source_pool = self.archive or [cell.genome for cell in self.members] or mixed_initial_population(
            self.rng,
            size=max(1, config.initial_population_size // 2),
            lineage_prefix="S",
            prior_fraction=0.5,
        )
        rescue_count = max(1, min(config.survivor_count, len(source_pool)))
        for index in range(rescue_count):
            genome = source_pool[index % len(source_pool)]
            child_genome = mutate_genome(
                genome,
                self.rng,
                lineage_id=f"{genome.lineage_id}.r{len(self.members) + index + 1}",
                mutation_rate=config.mutation_rate,
                synonym_rate=config.synonym_rate,
                insertion_rate=config.insertion_rate,
                deletion_rate=config.deletion_rate,
                motif_mutation_rate=config.motif_mutation_rate,
            )
            self.members.append(
                StreamCell(
                    genome=child_genome,
                    energy=config.initial_energy,
                    lineage_id=child_genome.lineage_id,
                    parent_lineage=genome.lineage_id,
                )
            )
            self.lineage_parents[child_genome.lineage_id] = (genome.lineage_id,)
        self.max_cells_seen = max(self.max_cells_seen, len(self.members))
        return rescue_count

    def spawn_child(
        self,
        parent: StreamCell,
        *,
        config: StreamConfig,
        lineage_suffix: str,
        force_random: bool = False,
    ) -> StreamCell:
        if force_random or self.rng.random() < config.immigrant_rate:
            genome = random_genome(
                self.rng,
                lineage_id=f"{parent.lineage_id}.{lineage_suffix}",
                length=self.rng.randrange(4, 8),
            )
        else:
            genome = mutate_genome(
                parent.genome,
                self.rng,
                lineage_id=f"{parent.lineage_id}.{lineage_suffix}",
                mutation_rate=config.mutation_rate,
                synonym_rate=config.synonym_rate,
                insertion_rate=config.insertion_rate,
                deletion_rate=config.deletion_rate,
                motif_mutation_rate=config.motif_mutation_rate,
            )
        child = StreamCell(
            genome=genome,
            energy=config.initial_energy,
            lineage_id=genome.lineage_id,
            parent_lineage=parent.lineage_id,
        )
        self.lineage_parents[child.lineage_id] = (parent.lineage_id,)
        return child

    def reproduce_from_survivors(
        self,
        parents: Sequence[StreamCell],
        config: StreamConfig,
        *,
        resource_pool: float,
    ) -> tuple[int, int, float]:
        births = 0
        deaths = 0
        if not parents:
            return births, deaths, resource_pool

        ordered_parents = list(parents)
        while resource_pool >= config.spawn_cost:
            spawned_this_round = False
            for parent in ordered_parents:
                if not parent.alive:
                    continue
                if parent.energy <= config.reproduction_threshold + config.spawn_cost:
                    continue
                if parent.energy <= config.spawn_cost:
                    continue
                parent.energy -= config.spawn_cost
                child = self.spawn_child(
                    parent,
                    config=config,
                    lineage_suffix=f"r{births + 1}",
                )
                self.members.append(child)
                births += 1
                resource_pool -= config.spawn_cost
                spawned_this_round = True
                break
            if not spawned_this_round:
                break

        for cell in self.members:
            if not cell.alive:
                deaths += 1
        deaths += self.prune_dead()
        self.max_cells_seen = max(self.max_cells_seen, len(self.members))
        return births, deaths, resource_pool


def evaluate_stream_cell(
    genome: CellGenome,
    bundle: TaskBundle,
    *,
    chemistry_max_time: float,
    chemistry_dt: float,
    include_neutrality: bool = False,
) -> Evaluation:
    from experiments.runner import evaluate_genome

    return evaluate_genome(
        genome,
        bundle,
        max_time=chemistry_max_time,
        dt=chemistry_dt,
        include_neutrality=include_neutrality,
    )


def task_reward(error: float, *, reward_scale: float) -> float:
    return max(-reward_scale, reward_scale * (1.0 - error))


def _lineage_root(lineage_id: str) -> str:
    return lineage_id.split(".", 1)[0]


def _lineage_efficiency_summary(cells: Sequence[StreamCell]) -> tuple[str, float, int]:
    lineage_buckets: dict[str, list[StreamCell]] = {}
    for cell in cells:
        if not cell.alive:
            continue
        root = _lineage_root(cell.lineage_id)
        lineage_buckets.setdefault(root, []).append(cell)
    if not lineage_buckets:
        return "", 0.0, 0
    best_root = ""
    best_efficiency = float("-inf")
    best_size = 0
    for root, members in lineage_buckets.items():
        total_energy = sum(cell.energy for cell in members)
        efficiency = total_energy / max(1, len(members))
        if efficiency > best_efficiency:
            best_root = root
            best_efficiency = efficiency
            best_size = len(members)
    return best_root, best_efficiency, best_size


def _best_lineage_representatives(cells: Sequence[StreamCell]) -> dict[str, StreamCell]:
    best: dict[str, StreamCell] = {}
    for cell in cells:
        if not cell.alive:
            continue
        root = _lineage_root(cell.lineage_id)
        current = best.get(root)
        if current is None or cell.energy > current.energy:
            best[root] = cell
    return best


def _lineage_transfer_scores(
    representatives: dict[str, StreamCell],
    bundles: Sequence[TaskBundle],
    *,
    chemistry_max_time: float,
    chemistry_dt: float,
) -> tuple[tuple[str, float], ...]:
    scores: list[tuple[str, float]] = []
    for root, cell in sorted(representatives.items()):
        if not bundles:
            scores.append((root, 0.0))
            continue
        bundle_scores = [
            evaluate_stream_cell(
                cell.genome,
                bundle,
                chemistry_max_time=chemistry_max_time,
                chemistry_dt=chemistry_dt,
            ).score
            for bundle in bundles
        ]
        mean_error = sum(bundle_scores) / max(1, len(bundle_scores))
        transfer_score = 1.0 / (1.0 + mean_error)
        scores.append((root, transfer_score))
    return tuple(scores)


def _motif_reappearance_summary(
    cells: Sequence[StreamCell],
    seen_patterns: set[tuple[str, ...]],
) -> tuple[int, tuple[tuple[str, int], ...]]:
    counts: dict[str, int] = {}
    total = 0
    for cell in cells:
        if not cell.alive:
            continue
        root = _lineage_root(cell.lineage_id)
        for motif in cell.genome.local_motifs:
            if motif.pattern in seen_patterns:
                total += 1
                counts[root] = counts.get(root, 0) + 1
    return total, tuple(sorted(counts.items()))


def _adaptive_regen(
    config: StreamConfig,
    *,
    mean_error: float,
    births: int,
    lineage_efficiency: float,
) -> float:
    error_term = max(0.0, 1.0 - min(1.0, mean_error / max(1e-9, config.reward_scale)))
    births_term = min(1.0, births * config.regen_birth_bonus)
    lineage_term = min(1.0, max(0.0, lineage_efficiency / max(1e-9, config.initial_energy)))
    adaptive = (
        config.resource_regen
        + config.regen_error_sensitivity * (1.0 - error_term)
        + births_term
        + config.lineage_efficiency_sensitivity * lineage_term
    )
    return max(config.resource_regen_floor, min(config.resource_regen_ceiling, adaptive))


def _motif_summary(motifs, seen_patterns: set[tuple[str, ...]]) -> dict[str, object]:
    stats = motif_statistics(motifs)
    patterns = tuple(motif.pattern for motif in motifs)
    transfer_count = sum(1 for pattern in patterns if pattern in seen_patterns)
    motif_count = int(stats["motif_count"])
    transfer_ratio = transfer_count / motif_count if motif_count else 0.0
    return {
        **stats,
        "motif_transfer_count": transfer_count,
        "motif_transfer_ratio": transfer_ratio,
        "motif_patterns": patterns,
    }


def _motif_result_fields(motifs, seen_patterns: set[tuple[str, ...]]) -> dict[str, object]:
    summary = _motif_summary(motifs, seen_patterns)
    summary.pop("motif_patterns", None)
    return summary


def _contextual_overrides(config: StreamConfig, context: TaskContext) -> tuple[float, float, float, float, float]:
    reward_scale = context.reward_scale if context.reward_scale is not None else config.reward_scale
    resource_pool = context.resource_pool if context.resource_pool is not None else config.resource_pool
    resource_regen = context.resource_regen if context.resource_regen is not None else config.resource_regen
    chemistry_max_time = context.chemistry_max_time if context.chemistry_max_time is not None else config.chemistry_max_time
    chemistry_dt = context.chemistry_dt if context.chemistry_dt is not None else config.chemistry_dt
    return reward_scale, resource_pool, resource_regen, chemistry_max_time, chemistry_dt


def run_task_stream(
    *,
    sequence_name: str,
    tasks: Sequence[TaskBundle],
    seed: int,
    config: StreamConfig | None = None,
) -> StreamReport:
    config = config or StreamConfig()
    rng = Random(seed)
    colony = TaskStreamColony.from_initial_population(
        rng,
        size=config.initial_population_size,
        initial_energy=config.initial_energy,
        lineage_prefix="S",
    )
    resource_pool = config.resource_pool

    results: list[StreamTaskResult] = []
    archive_snapshots: list[ArchiveSnapshot] = []
    seen_motif_patterns: set[tuple[str, ...]] = set()
    previous_transfer_scores: dict[str, float] = {}
    previous_task_best_error: float | None = None
    global_generation = 0
    adaptive_regen = config.resource_regen
    for step_index, bundle in enumerate(tasks):
        chemistry_max_time = config.chemistry_max_time
        chemistry_dt = config.chemistry_dt
        errors: list[float] = []
        births = 0
        deaths = 0
        lineage_efficiency = 0.0
        lineage_root = ""
        lineage_efficiency_count = 0
        phenotype_diversity = 0.0
        recovery_steps = 0

        for _local_step in range(config.max_steps_per_task):
            resource_pool = min(config.resource_capacity, resource_pool + adaptive_regen)
            evaluations: list[tuple[StreamCell, Evaluation, float]] = []
            for cell in colony.live_members():
                evaluation = evaluate_stream_cell(
                    cell.genome,
                    bundle,
                    chemistry_max_time=chemistry_max_time,
                    chemistry_dt=chemistry_dt,
                )
                error = evaluation.score
                reward = task_reward(error, reward_scale=config.reward_scale)
                cell.energy += reward - config.maintenance_cost
                cell.age += 1
                if cell.energy <= config.death_threshold:
                    cell.alive = False
                evaluations.append((cell, evaluation, error))

            live = [cell for cell, _, _ in evaluations if cell.alive]
            live.sort(key=lambda cell: cell.energy, reverse=True)
            survivors = live
            phenotype_diversity = _phenotype_diversity(
                survivors,
                chemistry_max_time=chemistry_max_time,
                chemistry_dt=chemistry_dt,
            )
            lineage_root, lineage_efficiency, lineage_efficiency_count = _lineage_efficiency_summary(survivors)
            representatives = _best_lineage_representatives(survivors)
            seen_bundles = tasks[: step_index + 1]
            lineage_transfer_scores = _lineage_transfer_scores(
                representatives,
                seen_bundles,
                chemistry_max_time=chemistry_max_time,
                chemistry_dt=chemistry_dt,
            )
            best_transfer_score = max((score for _, score in lineage_transfer_scores), default=0.0)
            previous_best_transfer_score = previous_transfer_scores.get(lineage_root, 0.0) if lineage_root else 0.0
            forgetting_delta = previous_best_transfer_score - best_transfer_score
            motif_reappearance_count, motif_reappearance_by_root = _motif_reappearance_summary(survivors, seen_motif_patterns)
            if previous_task_best_error is not None and not recovery_steps and evaluations:
                step_best_error = min(error for _, _, error in evaluations)
                if step_best_error <= previous_task_best_error:
                    recovery_steps = _local_step + 1

            if survivors:
                top_survivor = survivors[0]
                top_survivor.genome = attach_success_motif(top_survivor.genome, bundle.name)

            if survivors:
                step_births, step_deaths, resource_pool = colony.reproduce_from_survivors(
                    survivors,
                    config,
                    resource_pool=resource_pool,
                )
                births += step_births
                deaths += step_deaths
            deaths += colony.prune_dead()
            colony.archive_genomes((survivors[: config.survivor_count] if survivors else [cell for cell, _, _ in evaluations]), limit=config.survivor_count)
            births += colony.rescue_if_empty(config=config)
            colony.max_cells_seen = max(colony.max_cells_seen, len(colony.members))
            errors.extend(error for _, _, error in evaluations)

            if config.archive_interval > 0 and global_generation % config.archive_interval == 0:
                representative = min(
                    (evaluation for _, evaluation, _ in evaluations),
                    key=lambda evaluation: evaluation.score,
                    default=None,
                )
                if representative is not None:
                    motif_summary = _motif_summary(representative.genome.local_motifs, seen_motif_patterns)
                    neutral_drift_rate = 0.0
                    if config.estimate_neutrality_trials > 0:
                        neutral_drift_rate = estimate_neutrality(
                            representative.genome,
                            lambda candidate: evaluate_stream_cell(
                                candidate,
                                bundle,
                                chemistry_max_time=chemistry_max_time,
                                chemistry_dt=chemistry_dt,
                                include_neutrality=False,
                            ),
                            rng,
                            trials=config.estimate_neutrality_trials,
                        )
                    per_task_errors: list[tuple[str, float]] = []
                    for seen_bundle in seen_bundles:
                        archive_eval = evaluate_stream_cell(
                            representative.genome,
                            seen_bundle,
                            chemistry_max_time=chemistry_max_time,
                            chemistry_dt=chemistry_dt,
                        )
                        per_task_errors.append((seen_bundle.name, archive_eval.score))
                    mean_archive_error = sum(error for _, error in per_task_errors) / max(1, len(per_task_errors))
                    archive_snapshots.append(
                        ArchiveSnapshot(
                            generation=global_generation,
                            current_task=bundle.name,
                            seen_tasks=tuple(seen_bundle.name for seen_bundle in seen_bundles),
                            mean_archive_error=mean_archive_error,
                            per_task_error=tuple(per_task_errors),
                            best_lineage=representative.genome.lineage_id,
                            energy_efficiency=(sum(cell.energy for cell in colony.members) / max(1, len(colony.members))) if colony.members else 0.0,
                            lineage_efficiency=lineage_efficiency,
                            lineage_efficiency_count=lineage_efficiency_count,
                            lineage_root=lineage_root,
                            lineage_transfer_scores=lineage_transfer_scores,
                            forgetting_delta=forgetting_delta,
                            motif_reappearance_count=motif_reappearance_count,
                            motif_reappearance_by_root=motif_reappearance_by_root,
                            resource_regen=adaptive_regen,
                            neutral_drift_rate=neutral_drift_rate,
                            motif_count=motif_summary["motif_count"],
                            motif_reuse_total=motif_summary["motif_reuse_total"],
                            motif_transfer_count=motif_summary["motif_transfer_count"],
                            motif_transfer_ratio=motif_summary["motif_transfer_ratio"],
                            motif_origin_tasks=motif_summary["motif_origin_tasks"],
                            motif_origin_lineages=motif_summary["motif_origin_lineages"],
                        )
                    )
                    seen_motif_patterns.update(motif_summary["motif_patterns"])
            global_generation += 1

        results.append(
            StreamTaskResult(
                task_name=bundle.name,
                context_label="",
                step_index=step_index,
                mean_error=sum(errors) / max(1, len(errors)),
                best_error=min(errors) if errors else 0.0,
                live_cells=len(colony.members),
                births=births,
                deaths=deaths,
                energy_total=sum(cell.energy for cell in colony.members),
                max_cells=colony.max_cells_seen,
                resource_pool=resource_pool,
                energy_efficiency=(sum(cell.energy for cell in colony.members) / max(1, births + len(colony.members))) if colony.members else 0.0,
                lineage_efficiency=lineage_efficiency,
                lineage_efficiency_count=lineage_efficiency_count,
                lineage_root=lineage_root,
                lineage_transfer_scores=lineage_transfer_scores,
                forgetting_delta=forgetting_delta,
                motif_reappearance_count=motif_reappearance_count,
                motif_reappearance_by_root=motif_reappearance_by_root,
                resource_regen=adaptive_regen,
                phenotype_diversity=phenotype_diversity,
                recovery_steps=recovery_steps if previous_task_best_error is not None else 0,
                **_motif_result_fields(
                    next(
                        (cell.genome.local_motifs for cell in colony.members if cell.alive and cell.genome.local_motifs),
                        tuple(),
                    ),
                    seen_motif_patterns,
                ),
            )
        )
        if previous_task_best_error is not None and results[-1].recovery_steps == 0:
            results[-1].recovery_steps = config.max_steps_per_task

        adaptive_regen = _adaptive_regen(
            config,
            mean_error=results[-1].mean_error,
            births=births,
            lineage_efficiency=lineage_efficiency,
        )
        previous_task_best_error = results[-1].best_error
        for root, score in lineage_transfer_scores:
            previous_transfer_scores[root] = score

        for cell in colony.members:
            if cell.energy > config.reproduction_threshold + config.spawn_cost:
                cell.energy -= config.spawn_cost * 0.5

    lineage_edges = tuple(
        (cell.lineage_id, (cell.parent_lineage,) if cell.parent_lineage else ())
        for cell in colony.members
    )
    return StreamReport(
        sequence_name=sequence_name,
        results=tuple(results),
        archive_snapshots=tuple(archive_snapshots),
        lineage_edges=lineage_edges,
        final_population=tuple(colony.members),
        config=config,
    )


def run_contextual_task_stream(
    *,
    sequence_name: str,
    tasks: Sequence[ContextualTask],
    seed: int,
    config: StreamConfig | None = None,
) -> StreamReport:
    config = config or StreamConfig()
    rng = Random(seed)
    colony = TaskStreamColony.from_initial_population(
        rng,
        size=config.initial_population_size,
        initial_energy=config.initial_energy,
        lineage_prefix="S",
    )
    resource_pool = config.resource_pool

    results: list[StreamTaskResult] = []
    archive_snapshots: list[ArchiveSnapshot] = []
    seen_motif_patterns: set[tuple[str, ...]] = set()
    previous_transfer_scores: dict[str, float] = {}
    previous_task_best_error: float | None = None
    global_generation = 0
    adaptive_regen = config.resource_regen
    for step_index, contextual_task in enumerate(tasks):
        bundle = materialize_contextual_bundle(contextual_task.bundle, contextual_task.context, rng)
        reward_scale, _step_resource_pool, resource_regen, chemistry_max_time, chemistry_dt = _contextual_overrides(
            config,
            contextual_task.context,
        )
        if contextual_task.context.resource_pool is not None:
            resource_pool = min(config.resource_capacity, contextual_task.context.resource_pool)
        errors: list[float] = []
        births = 0
        deaths = 0
        lineage_efficiency = 0.0
        lineage_root = ""
        lineage_efficiency_count = 0
        phenotype_diversity = 0.0
        recovery_steps = 0

        for _local_step in range(config.max_steps_per_task):
            resource_pool = min(config.resource_capacity, resource_pool + adaptive_regen)
            evaluations: list[tuple[StreamCell, Evaluation, float]] = []
            for cell in colony.live_members():
                evaluation = evaluate_stream_cell(
                    cell.genome,
                    bundle,
                    chemistry_max_time=chemistry_max_time,
                    chemistry_dt=chemistry_dt,
                )
                error = evaluation.score
                reward = task_reward(error, reward_scale=reward_scale)
                cell.energy += reward - config.maintenance_cost
                cell.age += 1
                if cell.energy <= config.death_threshold:
                    cell.alive = False
                evaluations.append((cell, evaluation, error))

            live = [cell for cell, _, _ in evaluations if cell.alive]
            live.sort(key=lambda cell: cell.energy, reverse=True)
            survivors = live
            phenotype_diversity = _phenotype_diversity(
                survivors,
                chemistry_max_time=chemistry_max_time,
                chemistry_dt=chemistry_dt,
            )
            lineage_root, lineage_efficiency, lineage_efficiency_count = _lineage_efficiency_summary(survivors)
            representatives = _best_lineage_representatives(survivors)
            seen_tasks = tasks[: step_index + 1]
            seen_bundles = [
                materialize_contextual_bundle(seen_task.bundle, seen_task.context, rng)
                for seen_task in seen_tasks
            ]
            lineage_transfer_scores = _lineage_transfer_scores(
                representatives,
                seen_bundles,
                chemistry_max_time=chemistry_max_time,
                chemistry_dt=chemistry_dt,
            )
            best_transfer_score = max((score for _, score in lineage_transfer_scores), default=0.0)
            previous_best_transfer_score = previous_transfer_scores.get(lineage_root, 0.0) if lineage_root else 0.0
            forgetting_delta = previous_best_transfer_score - best_transfer_score
            motif_reappearance_count, motif_reappearance_by_root = _motif_reappearance_summary(survivors, seen_motif_patterns)
            if previous_task_best_error is not None and not recovery_steps and evaluations:
                step_best_error = min(error for _, _, error in evaluations)
                if step_best_error <= previous_task_best_error:
                    recovery_steps = _local_step + 1

            if survivors:
                top_survivor = survivors[0]
                top_survivor.genome = attach_success_motif(top_survivor.genome, bundle.name)

            if survivors:
                step_births, step_deaths, resource_pool = colony.reproduce_from_survivors(
                    survivors,
                    config,
                    resource_pool=resource_pool,
                )
                births += step_births
                deaths += step_deaths
            deaths += colony.prune_dead()
            colony.archive_genomes((survivors[: config.survivor_count] if survivors else [cell for cell, _, _ in evaluations]), limit=config.survivor_count)
            births += colony.rescue_if_empty(config=config)
            colony.max_cells_seen = max(colony.max_cells_seen, len(colony.members))
            errors.extend(error for _, _, error in evaluations)

            if config.archive_interval > 0 and global_generation % config.archive_interval == 0:
                representative = min(
                    (evaluation for _, evaluation, _ in evaluations),
                    key=lambda evaluation: evaluation.score,
                    default=None,
                )
                if representative is not None:
                    motif_summary = _motif_summary(representative.genome.local_motifs, seen_motif_patterns)
                    neutral_drift_rate = 0.0
                    if config.estimate_neutrality_trials > 0:
                        neutral_drift_rate = estimate_neutrality(
                            representative.genome,
                            lambda candidate: evaluate_stream_cell(
                                candidate,
                                bundle,
                                chemistry_max_time=chemistry_max_time,
                                chemistry_dt=chemistry_dt,
                                include_neutrality=False,
                            ),
                            rng,
                            trials=config.estimate_neutrality_trials,
                        )
                    per_task_errors: list[tuple[str, float]] = []
                    for seen_bundle in seen_bundles:
                        archive_eval = evaluate_stream_cell(
                            representative.genome,
                            seen_bundle,
                            chemistry_max_time=chemistry_max_time,
                            chemistry_dt=chemistry_dt,
                        )
                        per_task_errors.append((seen_bundle.name, archive_eval.score))
                    mean_archive_error = sum(error for _, error in per_task_errors) / max(1, len(per_task_errors))
                    archive_snapshots.append(
                        ArchiveSnapshot(
                            generation=global_generation,
                            current_task=bundle.name,
                            seen_tasks=tuple(seen_bundle.name for seen_bundle in seen_bundles),
                            mean_archive_error=mean_archive_error,
                            per_task_error=tuple(per_task_errors),
                            best_lineage=representative.genome.lineage_id,
                            energy_efficiency=(sum(cell.energy for cell in colony.members) / max(1, len(colony.members))) if colony.members else 0.0,
                            lineage_efficiency=lineage_efficiency,
                            lineage_efficiency_count=lineage_efficiency_count,
                            lineage_root=lineage_root,
                            lineage_transfer_scores=lineage_transfer_scores,
                            forgetting_delta=forgetting_delta,
                            motif_reappearance_count=motif_reappearance_count,
                            motif_reappearance_by_root=motif_reappearance_by_root,
                            resource_regen=adaptive_regen,
                            neutral_drift_rate=neutral_drift_rate,
                            motif_count=motif_summary["motif_count"],
                            motif_reuse_total=motif_summary["motif_reuse_total"],
                            motif_transfer_count=motif_summary["motif_transfer_count"],
                            motif_transfer_ratio=motif_summary["motif_transfer_ratio"],
                            motif_origin_tasks=motif_summary["motif_origin_tasks"],
                            motif_origin_lineages=motif_summary["motif_origin_lineages"],
                        )
                    )
                    seen_motif_patterns.update(motif_summary["motif_patterns"])
            global_generation += 1

        results.append(
            StreamTaskResult(
                task_name=contextual_task.bundle.name,
                context_label=contextual_task.context.label,
                step_index=step_index,
                mean_error=sum(errors) / max(1, len(errors)),
                best_error=min(errors) if errors else 0.0,
                live_cells=len(colony.members),
                births=births,
                deaths=deaths,
                energy_total=sum(cell.energy for cell in colony.members),
                max_cells=colony.max_cells_seen,
                resource_pool=resource_pool,
                energy_efficiency=(sum(cell.energy for cell in colony.members) / max(1, births + len(colony.members))) if colony.members else 0.0,
                lineage_efficiency=lineage_efficiency,
                lineage_efficiency_count=lineage_efficiency_count,
                lineage_root=lineage_root,
                lineage_transfer_scores=lineage_transfer_scores,
                forgetting_delta=forgetting_delta,
                motif_reappearance_count=motif_reappearance_count,
                motif_reappearance_by_root=motif_reappearance_by_root,
                resource_regen=adaptive_regen,
                phenotype_diversity=phenotype_diversity,
                recovery_steps=recovery_steps if previous_task_best_error is not None else 0,
                **_motif_result_fields(
                    next(
                        (cell.genome.local_motifs for cell in colony.members if cell.alive and cell.genome.local_motifs),
                        tuple(),
                    ),
                    seen_motif_patterns,
                ),
            )
        )
        if previous_task_best_error is not None and results[-1].recovery_steps == 0:
            results[-1].recovery_steps = config.max_steps_per_task

        adaptive_regen = _adaptive_regen(
            config,
            mean_error=results[-1].mean_error,
            births=births,
            lineage_efficiency=lineage_efficiency,
        )
        previous_task_best_error = results[-1].best_error
        for root, score in lineage_transfer_scores:
            previous_transfer_scores[root] = score

        for cell in colony.members:
            if cell.energy > config.reproduction_threshold + config.spawn_cost:
                cell.energy -= config.spawn_cost * 0.5

    lineage_edges = tuple(
        (cell.lineage_id, (cell.parent_lineage,) if cell.parent_lineage else ())
        for cell in colony.members
    )
    return StreamReport(
        sequence_name=sequence_name,
        results=tuple(results),
        archive_snapshots=tuple(archive_snapshots),
        lineage_edges=lineage_edges,
        final_population=tuple(colony.members),
        config=config,
    )


TaskSelector = Callable[[int, StreamTaskResult | None, Random], ContextualTask]


def run_adaptive_contextual_task_stream(
    *,
    sequence_name: str,
    episodes: int,
    task_selector: TaskSelector,
    seed: int,
    config: StreamConfig | None = None,
) -> StreamReport:
    config = config or StreamConfig()
    rng = Random(seed)
    colony = TaskStreamColony.from_initial_population(
        rng,
        size=config.initial_population_size,
        initial_energy=config.initial_energy,
        lineage_prefix="S",
    )
    resource_pool = config.resource_pool

    results: list[StreamTaskResult] = []
    archive_snapshots: list[ArchiveSnapshot] = []
    seen_motif_patterns: set[tuple[str, ...]] = set()
    previous_transfer_scores: dict[str, float] = {}
    global_generation = 0
    adaptive_regen = config.resource_regen
    previous_result: StreamTaskResult | None = None
    generated_tasks: list[ContextualTask] = []

    for step_index in range(episodes):
        contextual_task = task_selector(step_index, previous_result, rng)
        generated_tasks.append(contextual_task)
        bundle = materialize_contextual_bundle(contextual_task.bundle, contextual_task.context, rng)
        reward_scale, _step_resource_pool, resource_regen, chemistry_max_time, chemistry_dt = _contextual_overrides(
            config,
            contextual_task.context,
        )
        if contextual_task.context.resource_pool is not None:
            resource_pool = min(config.resource_capacity, contextual_task.context.resource_pool)
        errors: list[float] = []
        births = 0
        deaths = 0
        lineage_efficiency = 0.0
        lineage_root = ""
        lineage_efficiency_count = 0

        for _local_step in range(config.max_steps_per_task):
            resource_pool = min(config.resource_capacity, resource_pool + adaptive_regen)
            evaluations: list[tuple[StreamCell, Evaluation, float]] = []
            for cell in colony.live_members():
                evaluation = evaluate_stream_cell(
                    cell.genome,
                    bundle,
                    chemistry_max_time=chemistry_max_time,
                    chemistry_dt=chemistry_dt,
                )
                error = evaluation.score
                reward = task_reward(error, reward_scale=reward_scale)
                cell.energy += reward - config.maintenance_cost
                cell.age += 1
                if cell.energy <= config.death_threshold:
                    cell.alive = False
                evaluations.append((cell, evaluation, error))

            live = [cell for cell, _, _ in evaluations if cell.alive]
            live.sort(key=lambda cell: cell.energy, reverse=True)
            survivors = live
            lineage_root, lineage_efficiency, lineage_efficiency_count = _lineage_efficiency_summary(survivors)
            representatives = _best_lineage_representatives(survivors)
            seen_tasks = generated_tasks[: step_index + 1]
            seen_bundles = [
                materialize_contextual_bundle(seen_task.bundle, seen_task.context, rng)
                for seen_task in seen_tasks
            ]
            lineage_transfer_scores = _lineage_transfer_scores(
                representatives,
                seen_bundles,
                chemistry_max_time=chemistry_max_time,
                chemistry_dt=chemistry_dt,
            )
            best_transfer_score = max((score for _, score in lineage_transfer_scores), default=0.0)
            previous_best_transfer_score = previous_transfer_scores.get(lineage_root, 0.0) if lineage_root else 0.0
            forgetting_delta = previous_best_transfer_score - best_transfer_score
            motif_reappearance_count, motif_reappearance_by_root = _motif_reappearance_summary(survivors, seen_motif_patterns)

            if survivors:
                top_survivor = survivors[0]
                top_survivor.genome = attach_success_motif(top_survivor.genome, bundle.name)

            if survivors:
                step_births, step_deaths, resource_pool = colony.reproduce_from_survivors(
                    survivors,
                    config,
                    resource_pool=resource_pool,
                )
                births += step_births
                deaths += step_deaths
            deaths += colony.prune_dead()
            colony.archive_genomes((survivors[: config.survivor_count] if survivors else [cell for cell, _, _ in evaluations]), limit=config.survivor_count)
            births += colony.rescue_if_empty(config=config)
            colony.max_cells_seen = max(colony.max_cells_seen, len(colony.members))
            errors.extend(error for _, _, error in evaluations)

            if config.archive_interval > 0 and global_generation % config.archive_interval == 0:
                representative = min(
                    (evaluation for _, evaluation, _ in evaluations),
                    key=lambda evaluation: evaluation.score,
                    default=None,
                )
                if representative is not None:
                    motif_summary = _motif_summary(representative.genome.local_motifs, seen_motif_patterns)
                    per_task_errors: list[tuple[str, float]] = []
                    for seen_bundle in seen_bundles:
                        archive_eval = evaluate_stream_cell(
                            representative.genome,
                            seen_bundle,
                            chemistry_max_time=chemistry_max_time,
                            chemistry_dt=chemistry_dt,
                        )
                        per_task_errors.append((seen_bundle.name, archive_eval.score))
                    mean_archive_error = sum(error for _, error in per_task_errors) / max(1, len(per_task_errors))
                    archive_snapshots.append(
                        ArchiveSnapshot(
                            generation=global_generation,
                            current_task=bundle.name,
                            seen_tasks=tuple(seen_bundle.name for seen_bundle in seen_bundles),
                            mean_archive_error=mean_archive_error,
                            per_task_error=tuple(per_task_errors),
                            best_lineage=representative.genome.lineage_id,
                            energy_efficiency=(sum(cell.energy for cell in colony.members) / max(1, len(colony.members))) if colony.members else 0.0,
                            lineage_efficiency=lineage_efficiency,
                            lineage_efficiency_count=lineage_efficiency_count,
                            lineage_root=lineage_root,
                            lineage_transfer_scores=lineage_transfer_scores,
                            forgetting_delta=forgetting_delta,
                            motif_reappearance_count=motif_reappearance_count,
                            motif_reappearance_by_root=motif_reappearance_by_root,
                            resource_regen=adaptive_regen,
                            motif_count=motif_summary["motif_count"],
                            motif_reuse_total=motif_summary["motif_reuse_total"],
                            motif_transfer_count=motif_summary["motif_transfer_count"],
                            motif_transfer_ratio=motif_summary["motif_transfer_ratio"],
                            motif_origin_tasks=motif_summary["motif_origin_tasks"],
                            motif_origin_lineages=motif_summary["motif_origin_lineages"],
                        )
                    )
                    seen_motif_patterns.update(motif_summary["motif_patterns"])
            global_generation += 1

        current_result = StreamTaskResult(
            task_name=bundle.name,
            context_label=contextual_task.context.label,
            step_index=step_index,
            mean_error=sum(errors) / max(1, len(errors)),
            best_error=min(errors) if errors else 0.0,
            live_cells=len(colony.members),
            births=births,
            deaths=deaths,
            energy_total=sum(cell.energy for cell in colony.members),
            max_cells=colony.max_cells_seen,
            resource_pool=resource_pool,
            energy_efficiency=(sum(cell.energy for cell in colony.members) / max(1, births + len(colony.members))) if colony.members else 0.0,
            lineage_efficiency=lineage_efficiency,
            lineage_efficiency_count=lineage_efficiency_count,
            lineage_root=lineage_root,
            lineage_transfer_scores=lineage_transfer_scores,
            forgetting_delta=forgetting_delta,
            motif_reappearance_count=motif_reappearance_count,
            motif_reappearance_by_root=motif_reappearance_by_root,
            resource_regen=adaptive_regen,
            **_motif_result_fields(
                next(
                    (cell.genome.local_motifs for cell in colony.members if cell.alive and cell.genome.local_motifs),
                    tuple(),
                ),
                seen_motif_patterns,
            ),
        )
        results.append(current_result)
        previous_result = current_result

        adaptive_regen = _adaptive_regen(
            config,
            mean_error=current_result.mean_error,
            births=births,
            lineage_efficiency=lineage_efficiency,
        )
        for root, score in lineage_transfer_scores:
            previous_transfer_scores[root] = score

        for cell in colony.members:
            if cell.energy > config.reproduction_threshold + config.spawn_cost:
                cell.energy -= config.spawn_cost * 0.5

    lineage_edges = tuple(
        (cell.lineage_id, (cell.parent_lineage,) if cell.parent_lineage else ())
        for cell in colony.members
    )
    return StreamReport(
        sequence_name=sequence_name,
        results=tuple(results),
        archive_snapshots=tuple(archive_snapshots),
        lineage_edges=lineage_edges,
        final_population=tuple(colony.members),
        config=config,
    )
