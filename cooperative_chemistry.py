from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from random import Random
from statistics import fmean, pvariance
from typing import Any, Iterable, Mapping, Sequence

from cell import CellState
from chemistry import ChemistryContext, ChemistrySystem
from codons import decode_codon_op, decode_rule_name, random_codons
from evolution import Evaluation, EvolutionConfig, crossover_genomes, mutate_genome, select_top_diverse
from genome import CellGenome
from tasks import TaskBundle, TaskCase
from tracing import TraceEvent, format_trace, lineage_tree_text


@dataclass(slots=True)
class CooperativeRunResult:
    cells: tuple[CellState, ...]
    context: ChemistryContext
    steps: int
    messages_delivered: int


@dataclass(slots=True)
class CooperativeSample:
    mean_error: float
    outputs: tuple[float, ...]
    signals: tuple[tuple[float, ...], ...]
    messages_delivered: float
    trace_examples: tuple[str, ...] = ()


@dataclass(slots=True)
class SpecialisationMetrics:
    solo_mean_error: float
    cooperative_mean_error: float
    solo_output_variance: float
    cooperative_output_variance: float
    specialisation_index: float
    dominant_channel: str
    dominant_channel_ratio: float
    gene_usage: tuple[tuple[str, int], ...]


@dataclass(slots=True)
class CooperativePairScorer:
    system: "CooperativeChemistrySystem"
    task_bundle: TaskBundle

    def _cases(self) -> tuple[TaskCase, ...]:
        return tuple(self.task_bundle.train) + tuple(self.task_bundle.validation)

    def _prepare_cell(self, item: CellState | CellGenome, case: TaskCase) -> CellState:
        if isinstance(item, CellGenome):
            cell = CellState(active_rules=[])
            cell.reset(x=case.x, y=case.y)
            cell.active_rules = list(item.declare_rules(signals=cell.signals))
            return cell
        cell = CellState(active_rules=list(item.active_rules))
        cell.reset(x=case.x, y=case.y)
        return cell

    def _format_trace(self, cell_index: int, trace: Sequence[dict[str, Any]]) -> tuple[str, ...]:
        events = [
            TraceEvent(
                round_index=int(entry["round"]),
                rule_name=str(entry["rule"]),
                before=tuple(entry["before"]),
                after=tuple(entry["after"]),
                note=str(entry["note"]),
            )
            for entry in trace
        ]
        return tuple(f"cell{cell_index}:{text}" for text in format_trace(events))

    def sample_solo(self, item: CellState | CellGenome) -> CooperativeSample:
        outputs: list[float] = []
        signals: list[tuple[float, ...]] = []
        traces: list[str] = []
        total_error = 0.0
        messages = 0.0
        for case in self._cases():
            cell = self._prepare_cell(item, case)
            self.system.chemistry.run(cell, case)
            output = float(cell.output if cell.output is not None else cell.signals[2])
            outputs.append(output)
            signals.append(tuple(cell.signals[:5]))
            total_error += abs(output - case.target)
            if len(traces) < 8:
                traces.extend(self._format_trace(0, cell.trace[:4]))
            messages += float(len([entry for entry in cell.trace if entry["rule"] == "SEND"]))
        count = max(1, len(self._cases()))
        return CooperativeSample(
            mean_error=total_error / count,
            outputs=tuple(outputs),
            signals=tuple(signals),
            messages_delivered=messages / count,
            trace_examples=tuple(traces[:8]),
        )

    def sample_pair(self, item_a: CellState | CellGenome, item_b: CellState | CellGenome) -> CooperativeSample:
        outputs: list[float] = []
        signals: list[tuple[float, ...]] = []
        traces: list[str] = []
        total_error = 0.0
        messages = 0.0
        for case in self._cases():
            cell_a = self._prepare_cell(item_a, case)
            cell_b = self._prepare_cell(item_b, case)
            result = self.system.run([cell_a, cell_b], case)
            a_output = cell_a.output if cell_a.output is not None else cell_a.signals[2]
            b_output = cell_b.output if cell_b.output is not None else cell_b.signals[2]
            if cell_a.output is not None and cell_b.output is not None:
                combined = (float(a_output) + float(b_output)) / 2.0
            elif cell_a.output is not None:
                combined = float(a_output)
            elif cell_b.output is not None:
                combined = float(b_output)
            else:
                combined = 0.0
            outputs.append(combined)
            signals.append(tuple((cell_a.signals[i] + cell_b.signals[i]) / 2.0 for i in range(5)))
            total_error += abs(combined - case.target)
            messages += float(result.messages_delivered)
            if len(traces) < 8:
                traces.extend(self._format_trace(0, cell_a.trace[:2]))
                traces.extend(self._format_trace(1, cell_b.trace[:2]))
        count = max(1, len(self._cases()))
        return CooperativeSample(
            mean_error=total_error / count,
            outputs=tuple(outputs),
            signals=tuple(signals),
            messages_delivered=messages / count,
            trace_examples=tuple(traces[:8]),
        )

    def score_pair(self, item_a: CellState | CellGenome, item_b: CellState | CellGenome) -> tuple[float, float]:
        sample = self.sample_pair(item_a, item_b)
        return sample.mean_error, sample.messages_delivered

    def score_solo(self, item: CellState | CellGenome) -> float:
        return self.sample_solo(item).mean_error

    def to_evaluation(
        self,
        genome: object,
        error: float,
        trace_examples: tuple[str, ...],
    ) -> Evaluation:
        return Evaluation(
            genome=genome,
            train_error=error,
            validation_error=error,
            shortcut_hits=0,
            trace_examples=trace_examples,
            neutrality_estimate=0.0,
        )


@dataclass(slots=True)
class CooperativeEvolutionConfig:
    population_size: int = 12
    generations: int = 8
    survivor_count: int = 4
    siblings_per_survivor: int = 2
    cooperative_fraction: float = 1.0
    mutation_rate: float = 0.08
    synonym_rate: float = 0.65
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    motif_mutation_rate: float = 0.2
    crossover_rate: float = 0.35
    immigrant_rate: float = 0.1
    sharing_radius: float = 0.0
    sharing_strength: float = 1.0
    motif_crossover_bias: float = 0.0
    founder_min_length: int = 5
    founder_max_length: int = 10


@dataclass(slots=True)
class CooperativeGenerationResult:
    generation: int
    solo_mean_error: float
    cooperative_mean_error: float
    specialisation_index: float
    dominant_channel: str
    dominant_channel_ratio: float
    top_gene_usage: tuple[tuple[str, int], ...]
    messages_delivered: float


@dataclass(slots=True)
class CooperativeEvolutionReport:
    experiment: str
    generation_results: tuple[CooperativeGenerationResult, ...]
    lineage_tree: str
    trace_examples: tuple[str, ...]
    best_pair_lineages: tuple[str, str]
    extra: dict[str, Any] = field(default_factory=dict)

    def format_text(self) -> str:
        lines = [
            f"cooperative_stream: {self.experiment}",
            f"best_pair_lineages: {self.best_pair_lineages[0]}, {self.best_pair_lineages[1]}",
            f"lineage_tree: {self.lineage_tree}",
            "generation_results:",
        ]
        for result in self.generation_results:
            lines.append(
                "  - "
                f"gen={result.generation} solo_mean_error={result.solo_mean_error:.6f} "
                f"cooperative_mean_error={result.cooperative_mean_error:.6f} "
                f"specialisation_index={result.specialisation_index:.6f} "
                f"dominant_channel={result.dominant_channel} "
                f"dominant_ratio={result.dominant_channel_ratio:.6f} "
                f"messages_delivered={result.messages_delivered:.2f}"
            )
            if result.top_gene_usage:
                lines.append(
                    f"    top_gene_usage: {', '.join(f'{name}={count}' for name, count in result.top_gene_usage)}"
                )
        lines.append("trace_examples:")
        lines.extend(f"  - {trace}" for trace in self.trace_examples or ("<none>",))
        if self.extra:
            lines.append("extra:")
            for key, value in self.extra.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass(slots=True)
class CooperativeChemistrySystem:
    chemistry: ChemistrySystem = field(default_factory=ChemistrySystem)

    def _deliver_messages(self, context: ChemistryContext, *, broadcast: bool = True) -> int:
        delivered = len(context.outbox)
        if not context.outbox:
            return 0
        if broadcast:
            context.inbox.extend(context.outbox)
        else:
            context.inbox.append(context.outbox[-1])
        context.outbox.clear()
        return delivered

    def run(
        self,
        cells: Sequence[CellState],
        task: TaskCase,
        *,
        context: ChemistryContext | None = None,
        max_time: float | None = None,
    ) -> CooperativeRunResult:
        context = context or ChemistryContext()
        time_limit = max_time if max_time is not None else self.chemistry.max_time
        event_index = 0
        quiet_ticks = 0
        messages_delivered = 0
        step_count = 0
        cell_list = list(cells)

        while context.time <= time_limit:
            changed = False
            context.peer_vectors = {cell_index: cell.to_vector() for cell_index, cell in enumerate(cell_list)}
            messages_delivered += self._deliver_messages(context, broadcast=True)
            for cell_index, cell in enumerate(cell_list):
                context.self_cell_index = cell_index
                cell_changed, event_index = self.chemistry.step(cell, task, context, event_index_start=event_index)
                changed = changed or cell_changed
                context.peer_vectors[cell_index] = cell.to_vector()
                messages_delivered += self._deliver_messages(context, broadcast=True)
            if context.stage_increment > 0.0:
                quiet_ticks = 0
            elif all(cell.output is not None for cell in cell_list) and not changed:
                quiet_ticks += 1
            else:
                quiet_ticks = 0
            if quiet_ticks >= self.chemistry.quiescence_steps:
                break
            context.time += self.chemistry.dt
            step_count += 1

        for cell in cell_list:
            if cell.output is None:
                cell.output = cell.signals[2]
        return CooperativeRunResult(
            cells=tuple(cell_list),
            context=context,
            steps=step_count,
            messages_delivered=messages_delivered,
        )


def _variance(values: Sequence[float]) -> float:
    data = tuple(values)
    if len(data) < 2:
        return 0.0
    return pvariance(data)


def _flatten_signal_samples(samples: Sequence[CooperativeSample], slot: int) -> tuple[float, ...]:
    flattened: list[float] = []
    for sample in samples:
        for signal_row in sample.signals:
            if slot < len(signal_row):
                flattened.append(float(signal_row[slot]))
    return tuple(flattened)


def compute_specialisation_metrics(
    solo_samples: Sequence[CooperativeSample],
    cooperative_samples: Sequence[CooperativeSample],
    gene_usage: Mapping[str, int],
) -> SpecialisationMetrics:
    solo_outputs = [output for sample in solo_samples for output in sample.outputs]
    cooperative_outputs = [output for sample in cooperative_samples for output in sample.outputs]
    solo_output_variance = _variance(solo_outputs)
    cooperative_output_variance = _variance(cooperative_outputs)
    specialisation_index = solo_output_variance / max(cooperative_output_variance, 1e-9)

    best_channel = "signal[0]"
    best_ratio = 0.0
    for slot in range(5):
        solo_variance = _variance(_flatten_signal_samples(solo_samples, slot))
        cooperative_variance = _variance(_flatten_signal_samples(cooperative_samples, slot))
        ratio = cooperative_variance / max(solo_variance, 1e-9)
        if ratio > best_ratio:
            best_ratio = ratio
            best_channel = f"signal[{slot}]"

    top_usage = tuple(sorted(gene_usage.items(), key=lambda item: (-item[1], item[0]))[:2])
    solo_mean_error = fmean(sample.mean_error for sample in solo_samples) if solo_samples else 0.0
    cooperative_mean_error = fmean(sample.mean_error for sample in cooperative_samples) if cooperative_samples else 0.0
    return SpecialisationMetrics(
        solo_mean_error=solo_mean_error,
        cooperative_mean_error=cooperative_mean_error,
        solo_output_variance=solo_output_variance,
        cooperative_output_variance=cooperative_output_variance,
        specialisation_index=specialisation_index,
        dominant_channel=best_channel,
        dominant_channel_ratio=best_ratio,
        gene_usage=top_usage,
    )


def _random_table_genome(rng: Random, *, lineage_id: str, length: int = 6) -> CellGenome:
    return CellGenome(
        codons=random_codons(rng, length, exclude=()),
        local_motifs=(),
        lineage_id=lineage_id,
        encoding="table",
    )


def _gene_usage_from_genome(genome: CellGenome) -> Counter[str]:
    if genome.encoding == "table":
        decoded = [decode_codon_op(codon) for codon in genome.codons]
    else:
        decoded = [decode_rule_name(codon) for codon in genome.codons]
    usage: Counter[str] = Counter()
    for op in decoded:
        if op.startswith("CALL_") or op.startswith("GENE_"):
            usage[op] += 1
    return usage


def run_cooperative_evolution(
    *,
    experiment_name: str,
    task_bundle: Any,
    seed: int,
    config: CooperativeEvolutionConfig | None = None,
) -> CooperativeEvolutionReport:
    config = config or CooperativeEvolutionConfig()
    rng = Random(seed)
    system = CooperativeChemistrySystem()
    scorer = CooperativePairScorer(system=system, task_bundle=task_bundle)
    population = [
        _random_table_genome(
            rng,
            lineage_id=f"C{i + 1}",
            length=rng.randrange(config.founder_min_length, config.founder_max_length + 1),
        )
        for i in range(config.population_size)
    ]
    lineage_parents: dict[str, tuple[str, ...]] = {genome.lineage_id: tuple() for genome in population}
    generation_results: list[CooperativeGenerationResult] = []
    best_pair_lineages = (population[0].lineage_id, population[1].lineage_id if len(population) > 1 else population[0].lineage_id)
    best_pair_error = float("inf")
    best_trace_examples: tuple[str, ...] = ()

    for generation in range(config.generations):
        solo_samples: list[CooperativeSample] = []
        cooperative_samples: list[CooperativeSample] = []
        evaluations: list[Evaluation] = []
        gene_usage: Counter[str] = Counter()

        for genome in population:
            solo_sample = scorer.sample_solo(genome)
            solo_samples.append(solo_sample)
            gene_usage.update(_gene_usage_from_genome(genome))

        shuffled = population[:]
        rng.shuffle(shuffled)
        for index in range(0, len(shuffled), 2):
            left = shuffled[index]
            right = shuffled[index + 1] if index + 1 < len(shuffled) else None
            if right is None:
                pair_sample = scorer.sample_solo(left)
                cooperative_samples.append(pair_sample)
                fitness = pair_sample.mean_error
                trace_examples = pair_sample.trace_examples
                if pair_sample.mean_error < best_pair_error:
                    best_pair_error = pair_sample.mean_error
                    best_pair_lineages = (left.lineage_id, left.lineage_id)
                    best_trace_examples = pair_sample.trace_examples
            else:
                pair_sample = scorer.sample_pair(left, right)
                cooperative_samples.append(pair_sample)
                fitness = pair_sample.mean_error
                trace_examples = pair_sample.trace_examples
                if pair_sample.mean_error < best_pair_error:
                    best_pair_error = pair_sample.mean_error
                    best_pair_lineages = (left.lineage_id, right.lineage_id)
                    best_trace_examples = pair_sample.trace_examples

            if config.cooperative_fraction <= 0.0:
                left_fitness = solo_samples[population.index(left)].mean_error
                right_fitness = solo_samples[population.index(right)].mean_error if right is not None else fitness
            elif config.cooperative_fraction >= 1.0:
                left_fitness = fitness
                right_fitness = fitness
            else:
                left_solo = solo_samples[population.index(left)].mean_error
                right_solo = solo_samples[population.index(right)].mean_error if right is not None else fitness
                left_fitness = (config.cooperative_fraction * fitness) + ((1.0 - config.cooperative_fraction) * left_solo)
                right_fitness = (config.cooperative_fraction * fitness) + ((1.0 - config.cooperative_fraction) * right_solo)

            evaluations.append(
                Evaluation(
                    genome=left,
                    train_error=left_fitness / 2.0,
                    validation_error=left_fitness / 2.0,
                    shortcut_hits=0,
                    trace_examples=trace_examples[:4],
                )
            )
            if right is not None:
                evaluations.append(
                    Evaluation(
                        genome=right,
                        train_error=right_fitness / 2.0,
                        validation_error=right_fitness / 2.0,
                        shortcut_hits=0,
                        trace_examples=trace_examples[:4],
                    )
                )

        metrics = compute_specialisation_metrics(solo_samples, cooperative_samples, gene_usage)
        generation_results.append(
            CooperativeGenerationResult(
                generation=generation,
                solo_mean_error=metrics.solo_mean_error,
                cooperative_mean_error=metrics.cooperative_mean_error,
                specialisation_index=metrics.specialisation_index,
                dominant_channel=metrics.dominant_channel,
                dominant_channel_ratio=metrics.dominant_channel_ratio,
                top_gene_usage=metrics.gene_usage,
                messages_delivered=fmean(sample.messages_delivered for sample in cooperative_samples) if cooperative_samples else 0.0,
            )
        )

        survivors = select_top_diverse(
            evaluations,
            k=min(config.survivor_count, len(evaluations)),
            sharing_radius=config.sharing_radius,
            sharing_strength=config.sharing_strength,
        )
        next_population: list[CellGenome] = []
        for survivor_index, survivor in enumerate(survivors, start=1):
            child = mutate_genome(
                survivor.genome,
                rng,
                lineage_id=f"{survivor.genome.lineage_id}.{generation + 1}.{survivor_index}",
                mutation_rate=config.mutation_rate,
                synonym_rate=config.synonym_rate,
                insertion_rate=config.insertion_rate,
                deletion_rate=config.deletion_rate,
                motif_mutation_rate=config.motif_mutation_rate,
            )
            next_population.append(child)
            lineage_parents[child.lineage_id] = (survivor.genome.lineage_id,)
            for sibling_index in range(config.siblings_per_survivor):
                if len(next_population) >= config.population_size:
                    break
                mate = survivors[(survivor_index + sibling_index) % len(survivors)].genome
                if rng.random() < config.crossover_rate and len(survivors) > 1:
                    child = crossover_genomes(
                        child,
                        mate,
                        rng,
                        lineage_id=f"{survivor.genome.lineage_id}.{generation + 1}.x{sibling_index + 1}",
                        motif_crossover_bias=config.motif_crossover_bias,
                    )
                else:
                    child = mutate_genome(
                        child,
                        rng,
                        lineage_id=f"{survivor.genome.lineage_id}.{generation + 1}.s{sibling_index + 1}",
                        mutation_rate=config.mutation_rate,
                        synonym_rate=config.synonym_rate,
                        insertion_rate=config.insertion_rate,
                        deletion_rate=config.deletion_rate,
                        motif_mutation_rate=config.motif_mutation_rate,
                    )
                next_population.append(child)
                lineage_parents[child.lineage_id] = (survivor.genome.lineage_id,)

        while len(next_population) < config.population_size:
            parent = survivors[len(next_population) % len(survivors)].genome
            immigrant = mutate_genome(
                parent,
                rng,
                lineage_id=f"{parent.lineage_id}.{generation + 1}.i{len(next_population)}",
                mutation_rate=config.mutation_rate,
                synonym_rate=config.synonym_rate,
                insertion_rate=config.insertion_rate,
                deletion_rate=config.deletion_rate,
                motif_mutation_rate=config.motif_mutation_rate,
            )
            next_population.append(immigrant)
            lineage_parents[immigrant.lineage_id] = (parent.lineage_id,)

        population = next_population[: config.population_size]

    lineage_tree = lineage_tree_text(sorted(lineage_parents.items()))
    return CooperativeEvolutionReport(
        experiment=experiment_name,
        generation_results=tuple(generation_results),
        lineage_tree=lineage_tree,
        trace_examples=best_trace_examples,
        best_pair_lineages=best_pair_lineages,
        extra={
            "population_size": config.population_size,
            "generations": config.generations,
            "cooperative_fraction": config.cooperative_fraction,
            "best_pair_error": best_pair_error,
            "solved": best_pair_error == 0.0,
        },
    )
