from __future__ import annotations

from spatial_matrix_fabric import (
    MatrixFabricStreamConfig,
    MatrixFabricStreamTask,
    MatrixFabricTarget,
    run_spatial_matrix_fabric_stream,
)


def test_spatial_matrix_fabric_stream_runs_across_changing_targets() -> None:
    tasks = (
        MatrixFabricStreamTask(label="shift_a", target=MatrixFabricTarget(origin_x=5, origin_y=5, steps=8)),
        MatrixFabricStreamTask(label="shift_b", target=MatrixFabricTarget(origin_x=7, origin_y=4, steps=8)),
    )
    report = run_spatial_matrix_fabric_stream(
        experiment_name="test_spatial_matrix_fabric_stream",
        seed=19,
        tasks=tasks,
        config=MatrixFabricStreamConfig(
            population_size=8,
            generations_per_task=2,
            survivor_count=3,
        ),
    )
    assert report.task_count == 2
    assert report.episodes
    assert report.final_target_label in {"shift_a", "shift_b"}
    assert report.best_score >= 0.0
    assert "shared_motif_pool_size" in report.extra
    assert "shared_motif_transfer_count" in report.extra
    assert report.format_text().startswith("stream:")
