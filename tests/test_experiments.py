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
        "25_spatial_task_stream.py",
        "26_spatial_ecology_stream.py",
        "27_intercell_gene_expression.py",
        "28_temporal_unfolding.py",
        "29_reading_layer.py",
        "30_writing_layer.py",
        "31_epistasis.py",
        "30_epistasis_colony.py",
        "31_epistasis_colony2.py",
        "32_epistasis_colony3.py",
        "33_epistasis_colony4.py",
        "34_epistasis_colony5.py",
        "35_epistasis_colony6.py",
        "36_epistasis_bc_gate.py",
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
            for prefix in (
                "experiment:",
                "stream:",
                "cooperative_stream:",
                "spatial_development:",
                "spatial3d_development:",
                "spatial_stream:",
                "spatial_ecology_stream:",
                "temporal_unfolding:",
                "reading_layer:",
                "epistasis_colony:",
                "epistasis_colony2:",
                "epistasis_colony3:",
                "epistasis_colony4:",
                "epistasis_colony5:",
                "epistasis_colony6:",
                "epistasis_bc_gate:",
                "writing_layer:",
            )
        ), f"{script_name} produced no recognised output prefix:\n{completed.stdout}"
