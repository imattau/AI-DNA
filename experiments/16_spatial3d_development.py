from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial3d import run_spatial3d_development


def main() -> None:
    report = run_spatial3d_development(width=4, height=4, depth=4, steps=10, seed=29)
    print(report.format_text())


if __name__ == "__main__":
    main()
