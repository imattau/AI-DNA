from __future__ import annotations

from types import SimpleNamespace

from math_ecology import (
    AdaptiveMathCurriculumConfig,
    MathEcologyConfig,
    adaptive_math_task_selector,
    build_math_task_ecology,
)


def test_math_ecology_varies_tasks_and_contexts() -> None:
    tasks = build_math_task_ecology(seed=19, config=MathEcologyConfig(episodes=10))
    assert len(tasks) == 10
    assert len({task.bundle.name for task in tasks}) >= 2
    assert len({task.context.label for task in tasks}) >= 2
    assert any(task.context.target_noise > 0.0 for task in tasks)
    assert any(task.context.resource_pool != tasks[0].context.resource_pool for task in tasks[1:])
    assert any(task.bundle.name == "matrix2_det" for task in tasks)
    assert any(task.bundle.name == "symbolic_equation" for task in tasks)


def test_adaptive_math_curriculum_switches_families() -> None:
    selector = adaptive_math_task_selector(seed=23, config=AdaptiveMathCurriculumConfig(task_episodes=4))
    first = selector(0, None, None)
    hard = selector(1, SimpleNamespace(mean_error=0.1), None)
    easy = selector(2, SimpleNamespace(mean_error=5.0), None)
    assert first.bundle.name in {"multiply", "max", "abs"}
    assert hard.bundle.name in {"linear_solve", "matrix2_det", "symbolic_equation"}
    assert easy.bundle.name in {"multiply", "max", "abs"}
