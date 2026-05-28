from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_remaining_experiments_run() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        "02_max_persistent_rules.py",
        "03_abs_persistent_rules.py",
        "04_conditional_persistent_rules.py",
        "05_exponentiation_persistent_rules.py",
        "06_task_stream_adaptation.py",
        "07_self_replication.py",
        "08_rotating_selective_niches.py",
        "09_matrix_multiplication_search.py",
        "10_evolve_self_replication.py",
        "11_contextual_task_stream.py",
        "12_math_task_ecology.py",
        "13_cooperative_chemistry.py",
        "14_adaptive_math_ecology.py",
        "15_spatial_development.py",
        "16_spatial3d_development.py",
        "17_spatial_body_plan_search.py",
        "18_spatial_matrix_fabric.py",
        "19_spatial_matrix_fabric_stream.py",
        "20_spatial_self_repair.py",
        "22_spatial_roaming.py",
        "23_spatial_adhesion.py",
        "24_spatial_routing.py",
    ]
    for script_name in scripts:
        completed = subprocess.run(
            [sys.executable, str(repo_root / "experiments" / script_name)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
        assert any(
            prefix in completed.stdout
            for prefix in ("experiment:", "stream:", "spatial_development:", "spatial3d_development:")
        ), f"{script_name} produced no recognised output prefix:\n{completed.stdout}"
