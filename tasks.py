from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class TaskCase:
    x: float
    y: float
    target: float
    task_name: str


@dataclass(frozen=True, slots=True)
class TaskBundle:
    name: str
    train: tuple[TaskCase, ...]
    validation: tuple[TaskCase, ...]
    anti_shortcuts: tuple[TaskCase, ...]
    shortcut_checks: tuple[Callable[[TaskCase], float], ...]


@dataclass(frozen=True, slots=True)
class TaskContext:
    label: str
    x_scale: float = 1.0
    y_scale: float = 1.0
    x_shift: float = 0.0
    y_shift: float = 0.0
    target_noise: float = 0.0
    reward_scale: float | None = None
    resource_pool: float | None = None
    resource_regen: float | None = None
    chemistry_max_time: float | None = None
    chemistry_dt: float | None = None


@dataclass(frozen=True, slots=True)
class ContextualTask:
    bundle: TaskBundle
    context: TaskContext


def multiply_target(x: float, y: float) -> float:
    return x * y


def _grid_cases(
    *,
    name: str,
    values: range,
    target_fn: Callable[[float, float], float],
) -> tuple[TaskCase, ...]:
    return tuple(
        TaskCase(float(x), float(y), target_fn(float(x), float(y)), name)
        for x in values
        for y in values
    )


def _bundle(
    *,
    name: str,
    train_values: range,
    validation_values: range,
    target_fn: Callable[[float, float], float],
    shortcut_checks: tuple[Callable[[TaskCase], float], ...],
    anti_shortcut_points: tuple[tuple[int, int], ...],
) -> TaskBundle:
    train = _grid_cases(name=name, values=train_values, target_fn=target_fn)
    validation = _grid_cases(name=name, values=validation_values, target_fn=target_fn)
    anti_shortcuts = tuple(
        TaskCase(float(x), float(y), target_fn(float(x), float(y)), name)
        for x, y in anti_shortcut_points
    )
    return TaskBundle(
        name=name,
        train=train,
        validation=validation,
        anti_shortcuts=anti_shortcuts,
        shortcut_checks=shortcut_checks,
    )


def _transform_case(case: TaskCase, *, context: TaskContext, rng: Random) -> TaskCase:
    x = case.x * context.x_scale + context.x_shift
    y = case.y * context.y_scale + context.y_shift
    target = case.target + rng.uniform(-context.target_noise, context.target_noise)
    return replace(case, x=x, y=y, target=target)


def materialize_contextual_bundle(bundle: TaskBundle, context: TaskContext, rng: Random) -> TaskBundle:
    transformed_name = f"{bundle.name}[{context.label}]"
    train = tuple(_transform_case(case, context=context, rng=rng) for case in bundle.train)
    validation = tuple(_transform_case(case, context=context, rng=rng) for case in bundle.validation)
    anti_shortcuts = tuple(_transform_case(case, context=context, rng=rng) for case in bundle.anti_shortcuts)
    return TaskBundle(
        name=transformed_name,
        train=train,
        validation=validation,
        anti_shortcuts=anti_shortcuts,
        shortcut_checks=bundle.shortcut_checks,
    )


def build_multiply_bundle(
    *,
    train_values: range = range(0, 5),
    validation_values: range = range(0, 8),
) -> TaskBundle:
    return _bundle(
        name="multiply",
        train_values=train_values,
        validation_values=validation_values,
        target_fn=multiply_target,
        shortcut_checks=(
            lambda case: case.x + case.y,
            lambda case: 2 * case.x,
            lambda case: case.x + 2 * case.y,
            lambda case: case.x,
            lambda case: case.y,
        ),
        anti_shortcut_points=(
            (2, 7),
            (3, 6),
            (4, 5),
            (5, 4),
            (6, 3),
            (7, 2),
            (8, 3),
            (3, 8),
        ),
    )


def build_task_sequence() -> tuple[str, ...]:
    return (
        "multiplication",
        "max(x,t)",
        "abs(x-t)",
        "conditional: if x <= t then x else 2x",
        "exponentiation: x^y",
    )


def build_max_bundle(
    *,
    train_values: range = range(0, 6),
    validation_values: range = range(0, 8),
) -> TaskBundle:
    return _bundle(
        name="max",
        train_values=train_values,
        validation_values=validation_values,
        target_fn=lambda x, y: max(x, y),
        shortcut_checks=(
            lambda case: case.x + case.y,
            lambda case: case.x,
            lambda case: case.y,
            lambda case: 2 * case.x,
            lambda case: 2 * case.y,
        ),
        anti_shortcut_points=(
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 1),
            (4, 2),
            (5, 3),
            (6, 4),
            (7, 2),
        ),
    )


def build_abs_bundle(
    *,
    train_values: range = range(0, 6),
    validation_values: range = range(0, 8),
) -> TaskBundle:
    return _bundle(
        name="abs",
        train_values=train_values,
        validation_values=validation_values,
        target_fn=lambda x, y: abs(x - y),
        shortcut_checks=(
            lambda case: case.x + case.y,
            lambda case: case.x - case.y,
            lambda case: case.y - case.x,
            lambda case: case.x,
            lambda case: case.y,
            lambda case: 0.0,
        ),
        anti_shortcut_points=(
            (0, 5),
            (1, 4),
            (2, 7),
            (3, 6),
            (4, 1),
            (5, 0),
            (7, 2),
            (8, 3),
        ),
    )


def build_conditional_bundle(
    *,
    train_values: range = range(0, 6),
    validation_values: range = range(0, 8),
) -> TaskBundle:
    def target_fn(x: float, y: float) -> float:
        return x if x <= y else 2 * x

    return _bundle(
        name="conditional",
        train_values=train_values,
        validation_values=validation_values,
        target_fn=target_fn,
        shortcut_checks=(
            lambda case: case.x,
            lambda case: 2 * case.x,
            lambda case: case.y,
            lambda case: max(case.x, case.y),
            lambda case: case.x + case.y,
        ),
        anti_shortcut_points=(
            (0, 5),
            (1, 4),
            (2, 7),
            (3, 1),
            (4, 2),
            (5, 3),
            (6, 4),
            (7, 2),
        ),
    )


def build_exponentiation_bundle(
    *,
    train_values: range = range(0, 5),
    validation_values: range = range(0, 6),
) -> TaskBundle:
    def target_fn(x: float, y: float) -> float:
        return x**y

    return _bundle(
        name="exponentiation",
        train_values=train_values,
        validation_values=validation_values,
        target_fn=target_fn,
        shortcut_checks=(
            lambda case: case.x * case.y,
            lambda case: case.x + case.y,
            lambda case: case.x,
            lambda case: case.y,
            lambda case: 2 * case.x,
        ),
        anti_shortcut_points=(
            (0, 5),
            (1, 4),
            (2, 5),
            (3, 3),
            (4, 2),
            (5, 1),
            (2, 7),
            (3, 6),
        ),
    )


def build_linear_solve_bundle(
    *,
    train_values: range = range(1, 6),
    validation_values: range = range(1, 8),
) -> TaskBundle:
    def target_fn(x: float, y: float) -> float:
        return y / x

    return _bundle(
        name="linear_solve",
        train_values=train_values,
        validation_values=validation_values,
        target_fn=target_fn,
        shortcut_checks=(
            lambda case: case.x + case.y,
            lambda case: case.y - case.x,
            lambda case: case.x,
            lambda case: case.y,
            lambda case: 2 * case.y,
        ),
        anti_shortcut_points=(
            (1, 2),
            (2, 5),
            (3, 7),
            (4, 9),
            (5, 11),
            (6, 13),
            (7, 15),
            (8, 17),
        ),
    )


def build_task_suite() -> tuple[TaskBundle, ...]:
    return (
        build_multiply_bundle(),
        build_max_bundle(),
        build_abs_bundle(),
        build_conditional_bundle(),
        build_exponentiation_bundle(),
        build_linear_solve_bundle(),
    )


def shortcut_hit(case: TaskCase, prediction: float, shortcut_checks: Sequence[Callable[[TaskCase], float]]) -> bool:
    if abs(prediction - case.target) < 1e-9:
        return False
    return any(abs(prediction - check(case)) < 1e-9 for check in shortcut_checks)
