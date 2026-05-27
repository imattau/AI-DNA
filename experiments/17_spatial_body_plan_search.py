from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_body_plan import SpatialBodyPlanSearchConfig, run_spatial_body_plan_search


def main() -> None:
    report = run_spatial_body_plan_search(
        experiment_name="17_spatial_body_plan_search",
        seed=41,
        config=SpatialBodyPlanSearchConfig(
            population_size=16,
            restarts=5,
            generations=12,
            survivor_count=4,
            siblings_per_survivor=3,
            width=4,
            height=4,
            depth=4,
            steps=10,
        ),
    )
    print(report.format_text())


if __name__ == "__main__":
    main()
