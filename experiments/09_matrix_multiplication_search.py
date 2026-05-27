from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matrix_experiment import MatrixSearchConfig, run_matrix_experiment


def main() -> None:
    report = run_matrix_experiment(
        experiment_name="09_matrix_multiplication_search",
        seed=83,
        config=MatrixSearchConfig(
            population_size=12,
            restarts=6,
            generations=30,
            survivor_count=4,
            siblings_per_survivor=4,
            island_count=4,
            migration_interval=3,
            migration_size=1,
            basis_phase_generations=10,
        ),
        require_perfect=False,
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
