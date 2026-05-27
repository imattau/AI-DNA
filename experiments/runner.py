from __future__ import annotations

from random import Random

from cell import CellState
from chemistry import ChemistrySystem
from colony import Colony
from evolution import Evaluation, EvolutionConfig, estimate_neutrality, mixed_initial_population
from genome import CellGenome, format_motif, motif_statistics
from tasks import TaskBundle, build_task_sequence, shortcut_hit
from tracing import ExperimentReport, format_trace, lineage_tree_text


def _to_trace_events(raw_trace):
    from tracing import TraceEvent

    return [
        TraceEvent(
            round_index=entry["round"],
            rule_name=entry["rule"],
            before=tuple(entry["before"]),
            after=tuple(entry["after"]),
            note=entry["note"],
        )
        for entry in raw_trace
    ]


def evaluate_genome(
    genome: CellGenome,
    bundle: TaskBundle,
    *,
    max_time: float = 32.0,
    dt: float = 1.0,
    include_neutrality: bool = True,
) -> Evaluation:
    chemistry = ChemistrySystem(max_time=max_time, dt=dt)
    active_rules = list(genome.declare_rules())

    train_error = 0.0
    validation_error = 0.0
    shortcut_hits = 0
    traces: list[str] = []
    showcase_cases = [case for case in bundle.validation if case.x > 0 or case.y > 0][:3]

    def run_case(case):
        cell = CellState(active_rules=list(active_rules))
        cell.reset(x=case.x, y=case.y)
        chemistry.run(cell, case)
        prediction = float(cell.output if cell.output is not None else cell.signals[2])
        return cell, prediction

    for case in bundle.train:
        _, prediction = run_case(case)
        train_error += abs(prediction - case.target)
    for case in showcase_cases:
        cell, _ = run_case(case)
        traces.extend(format_trace(_to_trace_events(cell.trace)))

    for case in bundle.validation:
        _, prediction = run_case(case)
        validation_error += abs(prediction - case.target)

    for case in bundle.anti_shortcuts:
        _, prediction = run_case(case)
        if shortcut_hit(case, prediction, bundle.shortcut_checks):
            shortcut_hits += 1

    neutrality_estimate = 0.0
    if include_neutrality:
        neutrality_rng = Random(sum(genome.codons) + len(genome.codons))
        neutrality_estimate = estimate_neutrality(
            genome,
            lambda candidate: evaluate_genome(candidate, bundle, max_time=max_time, dt=dt, include_neutrality=False),
            neutrality_rng,
            trials=6,
        )

    return Evaluation(
        genome=genome,
        train_error=train_error / max(1, len(bundle.train)),
        validation_error=validation_error / max(1, len(bundle.validation)),
        shortcut_hits=shortcut_hits,
        trace_examples=tuple(traces[:8]),
        neutrality_estimate=neutrality_estimate,
    )


def build_lineage_report(colony: Colony) -> str:
    nodes = sorted(colony.lineage_parents.items())
    return lineage_tree_text(nodes)


def clone_colony(colony: Colony) -> Colony:
    return Colony(
        members=colony.members[:],
        rng=colony.rng,
        generation=colony.generation,
        lineage_parents=dict(colony.lineage_parents),
    )


def run_experiment(
    *,
    experiment_name: str,
    bundle: TaskBundle,
    seed: int,
    restarts: int = 5,
    population_size: int = 10,
    prior_fraction: float = 0.4,
    generations: int = 6,
    survivor_count: int = 3,
    siblings_per_survivor: int = 3,
    task_sequence: tuple[str, ...] | None = None,
    require_perfect: bool = False,
    chemistry_max_time: float = 32.0,
    chemistry_dt: float = 1.0,
    evolution_config: EvolutionConfig | None = None,
) -> ExperimentReport:
    evolution_config = evolution_config or EvolutionConfig()
    best_evaluation: Evaluation | None = None
    best_colony: Colony | None = None

    for restart in range(restarts):
        rng = Random(seed + restart * 101)
        colony = Colony(
            mixed_initial_population(
                rng,
                size=population_size,
                lineage_prefix=f"R{restart}",
                prior_fraction=prior_fraction,
            ),
            rng,
        )

        for _generation in range(generations):
            evaluations = colony.advance(
                lambda genome: evaluate_genome(genome, bundle, max_time=chemistry_max_time, dt=chemistry_dt),
                survivor_count=survivor_count,
                siblings_per_survivor=siblings_per_survivor,
                task_name=bundle.name,
                config=evolution_config,
            )
            generation_best = min(evaluations, key=lambda evaluation: evaluation.score)
            if best_evaluation is None or generation_best.score < best_evaluation.score:
                best_evaluation = generation_best
                best_colony = clone_colony(colony)
            if generation_best.validation_error == 0.0 and generation_best.train_error == 0.0:
                best_evaluation = generation_best
                best_colony = clone_colony(colony)
                break

        final_evaluations = [evaluate_genome(member, bundle, max_time=chemistry_max_time, dt=chemistry_dt) for member in colony.members]
        restart_best = min(final_evaluations, key=lambda evaluation: evaluation.score)
        if best_evaluation is None or restart_best.score < best_evaluation.score:
            best_evaluation = restart_best
            best_colony = clone_colony(colony)
        if best_evaluation.validation_error == 0.0 and best_evaluation.train_error == 0.0:
            break

    assert best_evaluation is not None
    assert best_colony is not None

    report = ExperimentReport(
        experiment=experiment_name,
        train_error=best_evaluation.train_error,
        full_validation_error=best_evaluation.validation_error,
        cell_count=1,
        active_rules=tuple(best_evaluation.genome.declare_rules()),
        lineage_tree=build_lineage_report(best_colony),
        motifs_per_cell=tuple(format_motif(motif) for motif in best_evaluation.genome.local_motifs),
        shortcut_hits=best_evaluation.shortcut_hits,
        trace_examples=best_evaluation.trace_examples,
        extra={
            "genome_codons": best_evaluation.genome.signature(),
            "task_sequence": " -> ".join(task_sequence or build_task_sequence()),
            "solved": best_evaluation.validation_error == 0.0 and best_evaluation.train_error == 0.0,
            "chemistry_max_time": chemistry_max_time,
            "chemistry_dt": chemistry_dt,
            "evolution_config": evolution_config,
            "neutrality_estimate": best_evaluation.neutrality_estimate,
            **motif_statistics(best_evaluation.genome.local_motifs),
        },
    )
    if require_perfect and best_evaluation.validation_error != 0.0:
        raise SystemExit(1)
    return report
