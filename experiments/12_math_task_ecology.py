from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_ecology import MathEcologyConfig, build_math_task_ecology
from task_stream import StreamConfig, run_contextual_task_stream


def main() -> None:
    tasks = build_math_task_ecology(
        seed=113,
        config=MathEcologyConfig(episodes=12),
    )
    report = run_contextual_task_stream(
        sequence_name="math_task_ecology",
        tasks=tasks,
        seed=113,
        config=StreamConfig(
            initial_population_size=8,
            initial_energy=6.0,
            maintenance_cost=0.7,
            spawn_cost=1.8,
            reproduction_threshold=6.5,
            max_steps_per_task=6,
            archive_interval=3,
            chemistry_max_time=24.0,
            chemistry_dt=1.0,
            survivor_count=3,
            immigrant_rate=0.15,
        ),
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
