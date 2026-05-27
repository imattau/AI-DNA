from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_multiply_experiment_runs_cleanly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "experiments" / "01_multiply_persistent_rules.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "full_validation_error: 0.000000" in completed.stdout
    assert "no hard-coded unrolled cascade" not in completed.stdout
