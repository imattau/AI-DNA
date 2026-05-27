from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_stream import StreamConfig, run_task_stream
from tasks import build_task_suite


def main() -> None:
    suite = build_task_suite()
    task_sequence = suite + suite[::-1]
    report = run_task_stream(
        sequence_name="curriculum_cycle",
        tasks=task_sequence,
        seed=53,
        config=StreamConfig(
            initial_population_size=8,
            initial_energy=6.0,
            maintenance_cost=0.7,
            spawn_cost=1.8,
            reproduction_threshold=6.5,
            reward_scale=4.0,
            max_steps_per_task=8,
            chemistry_max_time=32.0,
            chemistry_dt=1.0,
            survivor_count=3,
            mutation_rate=0.08,
            insertion_rate=0.05,
            deletion_rate=0.03,
            motif_mutation_rate=0.2,
            crossover_rate=0.35,
            immigrant_rate=0.15,
        ),
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
