from __future__ import annotations

from spatial3d import build_spatial3d_demo_genome
from spatial_body_plan import (
    SpatialBodyPlanSearchConfig,
    build_default_body_plan_target,
    evaluate_spatial_body_plan,
    run_spatial_body_plan_search,
)


def test_default_body_plan_target_uses_center_and_axes() -> None:
    target = build_default_body_plan_target(width=4, height=4, depth=4)
    assert target.occupied_positions == ((2, 2, 2), (3, 2, 2), (2, 3, 2), (2, 2, 3))


def test_body_plan_evaluator_accepts_known_demo_genome() -> None:
    target = build_default_body_plan_target(width=4, height=4, depth=4)
    evaluation = evaluate_spatial_body_plan(build_spatial3d_demo_genome(), target, seed=29)
    assert evaluation.exact_match
    assert evaluation.score == 0.0
    assert set(evaluation.occupied_positions) == set(target.occupied_positions)


def test_body_plan_search_report_renders() -> None:
    report = run_spatial_body_plan_search(
        experiment_name="test_spatial_body_plan_search",
        seed=41,
        config=SpatialBodyPlanSearchConfig(
            population_size=6,
            restarts=1,
            generations=2,
            survivor_count=2,
            siblings_per_survivor=2,
            width=4,
            height=4,
            depth=4,
            steps=8,
        ),
    )
    assert report.target_positions
    assert report.best_score >= 0.0
    assert report.format_text().startswith("experiment:")
