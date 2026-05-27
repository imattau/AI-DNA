from __future__ import annotations

from spatial_matrix_fabric import MatrixFabricSearchConfig, run_spatial_matrix_fabric_search


def test_spatial_matrix_fabric_solve_attempt_reports_matrix_error() -> None:
    report = run_spatial_matrix_fabric_search(
        experiment_name="test_spatial_matrix_fabric_solve_attempt",
        seed=71,
        config=MatrixFabricSearchConfig(
            population_size=6,
            restarts=1,
            generations=2,
            survivor_count=2,
            siblings_per_survivor=2,
        ),
    )
    assert report.best_score >= 0.0
    assert "matrix_error" in report.extra
    assert "best_layout_error" in report.extra or "layout_error" in report.extra
    assert report.format_text().startswith("experiment:")
