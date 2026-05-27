from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Sequence

from tasks import (
    ContextualTask,
    TaskBundle,
    TaskContext,
    build_abs_bundle,
    build_conditional_bundle,
    build_exponentiation_bundle,
    build_linear_solve_bundle,
    build_matrix2_det_bundle,
    build_max_bundle,
    build_multiply_bundle,
    build_symbolic_equation_bundle,
)


@dataclass(frozen=True, slots=True)
class MathEcologyConfig:
    episodes: int = 12
    noise_probability: float = 0.35
    context_shift_probability: float = 0.55
    reward_scale_min: float = 3.5
    reward_scale_max: float = 4.5
    resource_pool_min: float = 5.0
    resource_pool_max: float = 11.0
    resource_regen_min: float = 0.75
    resource_regen_max: float = 2.25
    chemistry_time_min: float = 12.0
    chemistry_time_max: float = 28.0
    scale_choices: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5)
    shift_choices: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


@dataclass(frozen=True, slots=True)
class AdaptiveMathEcologyConfig:
    episodes: int = 12
    low_error_threshold: float = 0.8
    high_error_threshold: float = 2.5
    low_band: tuple[str, ...] = ("multiply", "max", "abs")
    mid_band: tuple[str, ...] = ("conditional", "exponentiation")
    high_band: tuple[str, ...] = ("linear_solve", "matrix2_det", "symbolic_equation")
    noise_probability: float = 0.35
    context_shift_probability: float = 0.55
    reward_scale_min: float = 3.5
    reward_scale_max: float = 4.5
    resource_pool_min: float = 5.0
    resource_pool_max: float = 11.0
    resource_regen_min: float = 0.75
    resource_regen_max: float = 2.25
    chemistry_time_min: float = 12.0
    chemistry_time_max: float = 28.0
    scale_choices: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5)
    shift_choices: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


@dataclass(frozen=True, slots=True)
class AdaptiveMathCurriculumConfig:
    task_episodes: int = 12
    low_error_threshold: float = 0.8
    high_error_threshold: float = 2.5
    easy_families: tuple[str, ...] = ("multiply", "max", "abs")
    medium_families: tuple[str, ...] = ("conditional", "exponentiation")
    hard_families: tuple[str, ...] = ("linear_solve", "matrix2_det", "symbolic_equation")
    context: AdaptiveMathEcologyConfig = field(default_factory=AdaptiveMathEcologyConfig)


def build_math_families() -> tuple[TaskBundle, ...]:
    return (
        build_multiply_bundle(),
        build_max_bundle(),
        build_abs_bundle(),
        build_conditional_bundle(),
        build_exponentiation_bundle(),
        build_linear_solve_bundle(),
        build_matrix2_det_bundle(),
        build_symbolic_equation_bundle(),
    )


def build_math_family_map() -> dict[str, TaskBundle]:
    return {bundle.name: bundle for bundle in build_math_families()}


def sample_math_context(rng: Random, index: int, bundle: TaskBundle, config: MathEcologyConfig) -> TaskContext:
    return _sample_context(rng, index, bundle, config)


def choose_adaptive_math_bundle(
    rng: Random,
    *,
    previous_error: float | None,
    config: AdaptiveMathEcologyConfig | None = None,
) -> TaskBundle:
    config = config or AdaptiveMathEcologyConfig()
    family_map = build_math_family_map()
    if previous_error is None:
        chosen_name = rng.choice(config.low_band)
    elif previous_error <= config.low_error_threshold:
        chosen_name = rng.choice(config.high_band)
    elif previous_error >= config.high_error_threshold:
        chosen_name = rng.choice(config.low_band)
    else:
        chosen_name = rng.choice(config.mid_band)
    return family_map[chosen_name]


def adaptive_math_task_selector(
    *,
    seed: int,
    config: AdaptiveMathCurriculumConfig | None = None,
):
    config = config or AdaptiveMathCurriculumConfig()
    rng = Random(seed)
    family_map = build_math_family_map()

    def selector(step_index: int, previous_result, _rng: Random) -> ContextualTask:
        previous_error = None if previous_result is None else previous_result.mean_error
        if previous_error is None:
            family_name = rng.choice(config.easy_families)
        elif previous_error <= config.low_error_threshold:
            family_name = rng.choice(config.hard_families)
        elif previous_error >= config.high_error_threshold:
            family_name = rng.choice(config.easy_families)
        else:
            family_name = rng.choice(config.medium_families)
        bundle = family_map[family_name]
        context = sample_adaptive_math_context(rng, step_index, bundle, config.context)
        return ContextualTask(bundle=bundle, context=context)

    return selector


def sample_adaptive_math_context(
    rng: Random,
    index: int,
    bundle: TaskBundle,
    config: AdaptiveMathEcologyConfig | None = None,
) -> TaskContext:
    adaptive = config or AdaptiveMathEcologyConfig()
    base = MathEcologyConfig(
        episodes=adaptive.episodes,
        noise_probability=adaptive.noise_probability,
        context_shift_probability=adaptive.context_shift_probability,
        reward_scale_min=adaptive.reward_scale_min,
        reward_scale_max=adaptive.reward_scale_max,
        resource_pool_min=adaptive.resource_pool_min,
        resource_pool_max=adaptive.resource_pool_max,
        resource_regen_min=adaptive.resource_regen_min,
        resource_regen_max=adaptive.resource_regen_max,
        chemistry_time_min=adaptive.chemistry_time_min,
        chemistry_time_max=adaptive.chemistry_time_max,
        scale_choices=adaptive.scale_choices,
        shift_choices=adaptive.shift_choices,
    )
    return _sample_context(rng, index, bundle, base)


def _sample_context(rng: Random, index: int, bundle: TaskBundle, config: MathEcologyConfig) -> TaskContext:
    x_scale = rng.choice(config.scale_choices)
    y_scale = rng.choice(config.scale_choices)
    x_shift = rng.choice(config.shift_choices)
    y_shift = rng.choice(config.shift_choices)
    target_noise = 0.25 if rng.random() < config.noise_probability else 0.0
    reward_scale = rng.uniform(config.reward_scale_min, config.reward_scale_max)
    resource_pool = rng.uniform(config.resource_pool_min, config.resource_pool_max)
    resource_regen = rng.uniform(config.resource_regen_min, config.resource_regen_max)
    chemistry_max_time = rng.uniform(config.chemistry_time_min, config.chemistry_time_max)
    chemistry_dt = 1.0 if rng.random() < 0.85 else 0.5
    label = f"e{index}:{bundle.name}"
    if rng.random() < config.context_shift_probability:
        label = f"{label}:shift"
    if target_noise > 0.0:
        label = f"{label}:noise"
    return TaskContext(
        label=label,
        x_scale=x_scale,
        y_scale=y_scale,
        x_shift=x_shift,
        y_shift=y_shift,
        target_noise=target_noise,
        reward_scale=reward_scale,
        resource_pool=resource_pool,
        resource_regen=resource_regen,
        chemistry_max_time=chemistry_max_time,
        chemistry_dt=chemistry_dt,
    )


def build_math_task_ecology(
    *,
    seed: int,
    config: MathEcologyConfig | None = None,
    families: Sequence[TaskBundle] | None = None,
) -> tuple[ContextualTask, ...]:
    config = config or MathEcologyConfig()
    families = tuple(families or build_math_families())
    rng = Random(seed)
    tasks: list[ContextualTask] = []
    for index in range(config.episodes):
        bundle = rng.choice(families)
        context = _sample_context(rng, index, bundle, config)
        tasks.append(ContextualTask(bundle=bundle, context=context))
    return tuple(tasks)
