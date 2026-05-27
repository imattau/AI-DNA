from __future__ import annotations

from task_stream import StreamConfig, run_contextual_task_stream
from tasks import ContextualTask, TaskContext, build_abs_bundle, build_max_bundle, build_multiply_bundle


def test_contextual_task_stream_runs_with_varying_contexts() -> None:
    tasks = (
        ContextualTask(
            bundle=build_multiply_bundle(),
            context=TaskContext(label="baseline", reward_scale=4.0, resource_pool=8.0, resource_regen=1.5),
        ),
        ContextualTask(
            bundle=build_max_bundle(),
            context=TaskContext(label="noisy", target_noise=0.25, reward_scale=3.5, resource_pool=6.0, resource_regen=1.0),
        ),
        ContextualTask(
            bundle=build_abs_bundle(),
            context=TaskContext(label="scaled", x_scale=1.5, y_scale=0.75, x_shift=1.0, reward_scale=4.5, resource_pool=10.0),
        ),
    )
    report = run_contextual_task_stream(
        sequence_name="smoke_contextual",
        tasks=tasks,
        seed=31,
        config=StreamConfig(
            initial_population_size=6,
            initial_energy=6.0,
            maintenance_cost=0.6,
            spawn_cost=1.5,
            reproduction_threshold=5.5,
            max_steps_per_task=4,
            archive_interval=2,
            chemistry_max_time=16.0,
            chemistry_dt=1.0,
            survivor_count=3,
            immigrant_rate=0.1,
        ),
    )
    assert len(report.results) == len(tasks)
    assert report.results[0].context_label == "baseline"
    assert report.results[1].context_label == "noisy"
    assert report.results[2].context_label == "scaled"
    assert report.archive_snapshots
    assert any(snapshot.current_task.endswith("[noisy]") for snapshot in report.archive_snapshots)
    assert report.final_population
