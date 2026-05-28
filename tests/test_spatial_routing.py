from __future__ import annotations

from spatial_routing import (
    SpatialRoutingSearchConfig,
    build_default_routing_target,
    build_spatial_routing_demo_genome,
    evaluate_spatial_routing,
    run_spatial_routing_search,
)


def test_spatial_routing_demo_genome_reports_signal() -> None:
    target = build_default_routing_target(width=5, height=5)
    genome = build_spatial_routing_demo_genome()
    report = evaluate_spatial_routing(genome, target, seed=41)
    assert report.occupied_positions
    assert report.signal_strength >= 0.0
    assert report.score >= 0.0
    assert report.layout_error >= 0.0
    assert report.trace_examples


def test_spatial_routing_search_runs() -> None:
    report = run_spatial_routing_search(
        experiment_name="routing-test",
        seed=41,
        config=SpatialRoutingSearchConfig(
            population_size=8,
            restarts=1,
            generations=2,
            survivor_count=3,
            width=5,
            height=5,
            development_steps=4,
            route_steps=2,
        ),
    )
    assert report.target_positions
    assert report.source_position
    assert report.sink_position
    assert report.format_text().startswith("experiment: routing-test")
