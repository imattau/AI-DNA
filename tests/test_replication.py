from __future__ import annotations

from replication import build_exact_copy_program, offspring_matches_parent, run_replication_program
from replication import ReplicationSearchConfig, run_replication_experiment


def test_exact_copy_program_replicates_self() -> None:
    genome = build_exact_copy_program()
    state = run_replication_program(genome)
    assert state.offspring == genome.codons
    assert offspring_matches_parent(state)
    assert state.halted


def test_evolving_replication_experiment_runs() -> None:
    report = run_replication_experiment(
        experiment_name="smoke_self_replication",
        seed=17,
        config=ReplicationSearchConfig(
            population_size=10,
            restarts=2,
            generations=6,
            survivor_count=3,
            founder_fraction=0.5,
            founder_bias_fraction=0.7,
            max_length=18,
        ),
        require_exact=False,
    )
    assert report.experiment == "smoke_self_replication"
    assert report.active_rules
    assert "exact_replication" in report.extra
