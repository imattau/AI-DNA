from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_routing import run_spatial_routing_search


def main() -> int:
    report = run_spatial_routing_search(
        experiment_name="24_spatial_routing",
        seed=41,
        config=None,
    )
    print(report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
