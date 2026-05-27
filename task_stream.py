from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Sequence

from evolution import Evaluation, EvolutionConfig, mixed_initial_population, mutate_genome, random_genome
from genome import CellGenome
from tasks import ContextualTask, TaskBundle, TaskContext, materialize_contextual_bundle


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


@dataclass(slots=True)
class ArchiveSnapshot:
    generation: int
    current_task: str
    seen_tasks: tuple[str, ...]
    mean_archive_error: float
    per_task_error: tuple[tuple[str, float], ...]
    best_lineage: str


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
                f"max_cells={result.max_cells} resource_pool={result.resource_pool:.2f}"
            )
        if self.archive_snapshots:
            lines.append("archive:")
            for snapshot in self.archive_snapshots:
                lines.append(
                    f"  - gen {snapshot.generation} task={snapshot.current_task} "
                    f"mean_archive_error={snapshot.mean_archive_error:.6f} best={snapshot.best_lineage}"
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
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    motif_mutation_rate: float = 0.2
    crossover_rate: float = 0.35
    immigrant_rate: float = 0.12


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
) -> Evaluation:
    from experiments.runner import evaluate_genome

    return evaluate_genome(genome, bundle, max_time=chemistry_max_time, dt=chemistry_dt)


def task_reward(error: float, *, reward_scale: float) -> float:
    return max(-reward_scale, reward_scale * (1.0 - error))


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
    global_generation = 0
    for step_index, bundle in enumerate(tasks):
        chemistry_max_time = config.chemistry_max_time
        chemistry_dt = config.chemistry_dt
        errors: list[float] = []
        births = 0
        deaths = 0

        for _local_step in range(config.max_steps_per_task):
            resource_pool = min(config.resource_capacity, resource_pool + config.resource_regen)
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
                    seen_bundles = tasks[: step_index + 1]
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
                        )
                    )
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
            )
        )

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
    global_generation = 0
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

        for _local_step in range(config.max_steps_per_task):
            resource_pool = min(config.resource_capacity, resource_pool + resource_regen)
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
                    seen_tasks = tasks[: step_index + 1]
                    seen_bundles = [
                        materialize_contextual_bundle(seen_task.bundle, seen_task.context, rng)
                        for seen_task in seen_tasks
                    ]
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
                        )
                    )
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
            )
        )

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
