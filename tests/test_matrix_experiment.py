from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from matrix_experiment import build_basis_matrix_bundle, build_biased_founder, build_matrix_bundle, build_reference_genome, evaluate_genome, symbolic_verify_matrix_program


def test_reference_matrix_program_is_exact() -> None:
    rng_seed = 11
    from random import Random

    genome = build_reference_genome(lineage_id="seed", rng=Random(rng_seed))
    bundle = build_matrix_bundle()
    evaluation = evaluate_genome(genome, bundle)
    assert evaluation.train_error == 0.0
    assert evaluation.validation_error == 0.0
    assert evaluation.exact_output_positions == 9
    assert evaluation.symbolic_verified is True
    assert evaluation.shortcut_hits == 0
    verified, _outputs = symbolic_verify_matrix_program(genome)
    assert verified is True


def test_basis_matrix_bundle_is_smaller_and_structured() -> None:
    basis = build_basis_matrix_bundle()
    full = build_matrix_bundle()
    assert basis.name == "matrix_multiplication_basis"
    assert len(basis.train) == 18
    assert len(basis.validation) == 9
    assert len(full.train) == 4
    assert basis.anti_shortcuts


def test_biased_founder_has_safe_motifs() -> None:
    from random import Random

    genome = build_biased_founder(lineage_id="bias", rng=Random(7))
    assert genome.local_motifs
    for motif in genome.local_motifs:
        assert motif.origin_task == "matrix_founder_bias"
        assert motif.pattern


def test_matrix_experiment_script_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repo_root / "experiments" / "09_matrix_multiplication_search.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "experiment:" in completed.stdout
