from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_matrix_fabric import (
    MatrixFabricStreamConfig,
    MatrixFabricStreamTask,
    MatrixFabricTarget,
    run_spatial_matrix_fabric_stream,
)


def main() -> None:
    tasks = (
        MatrixFabricStreamTask(
            label="shift_a",
            target=MatrixFabricTarget(origin_x=5, origin_y=5, steps=10),
        ),
        MatrixFabricStreamTask(
            label="shift_b",
            target=MatrixFabricTarget(origin_x=7, origin_y=4, steps=11),
        ),
        MatrixFabricStreamTask(
            label="shift_c",
            target=MatrixFabricTarget(origin_x=4, origin_y=7, steps=9),
        ),
    )
    report = run_spatial_matrix_fabric_stream(
        experiment_name="19_spatial_matrix_fabric_stream",
        seed=61,
        tasks=tasks,
        config=MatrixFabricStreamConfig(
            population_size=12,
            generations_per_task=3,
            survivor_count=4,
        ),
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
