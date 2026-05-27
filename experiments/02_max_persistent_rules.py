from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runner import run_experiment
from tasks import build_max_bundle


def main() -> None:
    report = run_experiment(
        experiment_name="02_max_persistent_rules",
        bundle=build_max_bundle(),
        seed=23,
        restarts=4,
        population_size=10,
        prior_fraction=0.5,
        generations=5,
        survivor_count=3,
        siblings_per_survivor=3,
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
