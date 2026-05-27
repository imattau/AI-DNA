from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks import build_multiply_bundle
from experiments.runner import run_experiment


def main() -> None:
    report = run_experiment(
        experiment_name="01_multiply_persistent_rules",
        bundle=build_multiply_bundle(),
        seed=17,
        restarts=5,
        population_size=10,
        prior_fraction=0.4,
        generations=6,
        survivor_count=3,
        siblings_per_survivor=3,
        require_perfect=True,
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
