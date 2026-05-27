from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_matrix_fabric import MatrixFabricSearchConfig, run_spatial_matrix_fabric_search


def main() -> None:
    report = run_spatial_matrix_fabric_search(
        experiment_name="18_spatial_matrix_fabric",
        seed=53,
        config=MatrixFabricSearchConfig(
            population_size=14,
            restarts=4,
            generations=10,
            survivor_count=4,
            siblings_per_survivor=3,
        ),
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
