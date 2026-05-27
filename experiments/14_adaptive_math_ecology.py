from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_ecology import AdaptiveMathCurriculumConfig, adaptive_math_task_selector
from task_stream import StreamConfig, run_adaptive_contextual_task_stream


def main() -> None:
    config = AdaptiveMathCurriculumConfig(task_episodes=6)
    selector = adaptive_math_task_selector(seed=113, config=config)
    report = run_adaptive_contextual_task_stream(
        sequence_name="adaptive_math_task_ecology",
        episodes=config.task_episodes,
        task_selector=selector,
        seed=113,
        config=StreamConfig(
            initial_population_size=6,
            initial_energy=6.0,
            maintenance_cost=0.7,
            spawn_cost=1.8,
            reproduction_threshold=6.5,
            max_steps_per_task=4,
            archive_interval=2,
            chemistry_max_time=18.0,
            chemistry_dt=1.0,
            survivor_count=2,
            immigrant_rate=0.15,
        ),
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
