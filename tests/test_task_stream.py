from __future__ import annotations

from task_stream import StreamConfig, run_task_stream
from tasks import build_max_bundle, build_multiply_bundle


def test_task_stream_runs_multiple_tasks_and_grows() -> None:
    tasks = (build_multiply_bundle(), build_max_bundle(), build_multiply_bundle())
    report = run_task_stream(
        sequence_name="smoke",
        tasks=tasks,
        seed=53,
        config=StreamConfig(
            initial_population_size=6,
            initial_energy=6.0,
            maintenance_cost=0.6,
            spawn_cost=1.5,
            reproduction_threshold=5.5,
            reward_scale=4.0,
            max_steps_per_task=4,
            chemistry_max_time=16.0,
            chemistry_dt=1.0,
            survivor_count=3,
            immigrant_rate=0.1,
        ),
    )
    assert len(report.results) == len(tasks)
    assert max(result.max_cells for result in report.results) >= 6
    assert report.final_population
    assert any(result.births > 0 for result in report.results)
    assert report.archive_snapshots
    assert report.archive_snapshots[0].mean_archive_error >= 0.0
