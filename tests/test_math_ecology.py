from __future__ import annotations

from math_ecology import MathEcologyConfig, build_math_task_ecology


def test_math_ecology_varies_tasks_and_contexts() -> None:
    tasks = build_math_task_ecology(seed=19, config=MathEcologyConfig(episodes=10))
    assert len(tasks) == 10
    assert len({task.bundle.name for task in tasks}) >= 2
    assert len({task.context.label for task in tasks}) >= 2
    assert any(task.context.target_noise > 0.0 for task in tasks)
    assert any(task.context.resource_pool != tasks[0].context.resource_pool for task in tasks[1:])
    assert any(task.bundle.name == "linear_solve" for task in tasks)
