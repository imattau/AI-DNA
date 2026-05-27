from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial import run_spatial_development


def main() -> None:
    report = run_spatial_development(width=6, height=6, steps=8, seed=17)
    print(report.format_text())


if __name__ == "__main__":
    main()
