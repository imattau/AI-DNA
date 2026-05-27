from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from replication import ReplicationSearchConfig, run_replication_experiment


def main() -> None:
    report = run_replication_experiment(
        experiment_name="10_evolve_self_replication",
        seed=97,
        config=ReplicationSearchConfig(
            population_size=28,
            restarts=10,
            generations=60,
            survivor_count=8,
            mutation_rate=0.05,
            insertion_rate=0.03,
            deletion_rate=0.02,
            crossover_rate=0.25,
            founder_fraction=0.8,
            founder_bias_fraction=0.9,
            max_length=24,
        ),
        require_exact=False,
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
