from __future__ import annotations

from spatial import build_spatial_genome
from spatial_matrix_fabric import (
    MatrixFabricSearchConfig,
    MatrixFabricTarget,
    evaluate_spatial_matrix_fabric,
    run_spatial_matrix_fabric_search,
)


def test_matrix_fabric_target_exposes_block_and_edges() -> None:
    target = MatrixFabricTarget()
    assert len(target.target_positions) == 16
    assert len(target.edge_positions) == 8
    assert target.input_positions == target.edge_positions
    assert len(target.output_positions) == 16
    assert target.target_positions[0] == (6, 6)


def test_matrix_fabric_evaluator_runs_on_a_spatial_genome() -> None:
    target = MatrixFabricTarget()
    genome = build_spatial_genome(("GET_X", "EMIT_0", "DIVIDE_EAST", "DIVIDE_SOUTH", "HALT"), lineage_id="M1")
    evaluation = evaluate_spatial_matrix_fabric(genome, target, seed=13)
    assert evaluation.score >= 0.0
    assert evaluation.outputs
    assert evaluation.input_error >= 0.0
    assert evaluation.output_error >= 0.0
    assert evaluation.dropout_robustness >= 0.0
    assert len(evaluation.trace_examples) <= 8


def test_matrix_fabric_search_report_renders() -> None:
    report = run_spatial_matrix_fabric_search(
        experiment_name="test_spatial_matrix_fabric_search",
        seed=53,
        config=MatrixFabricSearchConfig(
            population_size=6,
            restarts=2,
            generations=2,
            survivor_count=2,
            siblings_per_survivor=2,
        ),
    )
    assert report.target_positions
    assert report.edge_positions
    assert "input_error" in report.extra
    assert "output_error" in report.extra
    assert "dropout_robustness" in report.extra
    assert "role_error" in report.extra
    assert "restart_motif_transfer_count" in report.extra
    assert "restart_motif_pool_size" in report.extra
    assert report.format_text().startswith("experiment:")
