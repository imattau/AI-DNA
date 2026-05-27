from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_matrix_fabric import MatrixFabricSearchConfig, run_spatial_matrix_fabric_search


def main() -> None:
    report = run_spatial_matrix_fabric_search(
        experiment_name="21_spatial_matrix_fabric_solve",
        seed=71,
        config=MatrixFabricSearchConfig(
            population_size=20,
            restarts=8,
            generations=24,
            curriculum_generations=10,
            survivor_count=6,
            siblings_per_survivor=4,
            founder_fraction=0.65,
            founder_bias_fraction=0.75,
            founder_rail_bias_fraction=0.9,
        ),
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
