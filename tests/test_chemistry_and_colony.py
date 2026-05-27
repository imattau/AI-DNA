from __future__ import annotations

from random import Random

from cell import CellState
from chemistry import ChemistryContext, ChemistrySystem
from colony import Colony
from evolution import EvolutionConfig, mixed_initial_population
from genome import CellGenome
from experiments.runner import evaluate_genome
from tasks import TaskCase
from tasks import build_multiply_bundle


def test_inhibit_rule_clears_signal_three() -> None:
    cell = CellGenome.from_rule_names(
        ["RULE_EMIT_X", "RULE_COPY0_3", "RULE_INHIBIT1Z"],
        lineage_id="test",
    )
    system = ChemistrySystem(max_time=2.0, dt=1.0)
    state = system.run(
        CellState(active_rules=list(cell.declare_rules())),
        TaskCase(3.0, 0.0, 0.0, "demo"),
    )
    assert state.signals[3] == 0.0


def test_send_is_deduplicated() -> None:
    cell = CellState(active_rules=["SEND"])
    cell.reset(x=1.0, y=2.0)
    context = ChemistryContext()
    system = ChemistrySystem(max_time=4.0, dt=1.0)
    system.run(cell, TaskCase(1.0, 2.0, 0.0, "demo"), context=context)
    sends = [entry for entry in cell.trace if entry["rule"] == "SEND"]
    assert len(sends) == 1
    assert any(event["kind"] == "send" for event in context.events)


def test_colony_advance_keeps_population_size_constant() -> None:
    rng = Random(3)
    members = mixed_initial_population(rng, size=8, lineage_prefix="T", prior_fraction=0.5)
    colony = Colony(members, rng)
    colony.advance(
        lambda genome: evaluate_genome(genome, build_multiply_bundle()),
        survivor_count=3,
        siblings_per_survivor=2,
        task_name="multiply",
        config=EvolutionConfig(),
    )
    assert len(colony.members) == 8


def test_evaluate_genome_reports_neutrality_estimate() -> None:
    genome = CellGenome.from_rule_names(
        ["RULE_EMIT_X", "RULE_COPY0_3", "RULE_ADD0_IF1"],
        lineage_id="neutrality",
    )
    evaluation = evaluate_genome(genome, build_multiply_bundle())
    assert 0.0 <= evaluation.neutrality_estimate <= 1.0
