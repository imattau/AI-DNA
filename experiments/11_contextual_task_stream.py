from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_stream import StreamConfig, run_contextual_task_stream
from tasks import ContextualTask, TaskContext, build_abs_bundle, build_max_bundle, build_multiply_bundle


def main() -> None:
    tasks = (
        ContextualTask(
            bundle=build_multiply_bundle(),
            context=TaskContext(
                label="baseline",
                reward_scale=4.0,
                resource_pool=8.0,
                resource_regen=1.5,
                chemistry_max_time=16.0,
            ),
        ),
        ContextualTask(
            bundle=build_max_bundle(),
            context=TaskContext(
                label="noisy",
                target_noise=0.25,
                reward_scale=3.5,
                resource_pool=6.0,
                resource_regen=1.0,
                chemistry_max_time=20.0,
            ),
        ),
        ContextualTask(
            bundle=build_abs_bundle(),
            context=TaskContext(
                label="scaled",
                x_scale=1.5,
                y_scale=0.75,
                x_shift=1.0,
                reward_scale=4.5,
                resource_pool=10.0,
                resource_regen=2.0,
                chemistry_max_time=24.0,
            ),
        ),
    )
    report = run_contextual_task_stream(
        sequence_name="contextual_stream",
        tasks=tasks,
        seed=83,
        config=StreamConfig(
            initial_population_size=8,
            initial_energy=6.0,
            maintenance_cost=0.7,
            spawn_cost=1.8,
            reproduction_threshold=6.5,
            max_steps_per_task=8,
            archive_interval=4,
            chemistry_max_time=24.0,
            chemistry_dt=1.0,
            survivor_count=3,
            immigrant_rate=0.15,
        ),
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
