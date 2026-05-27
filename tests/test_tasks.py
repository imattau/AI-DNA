from __future__ import annotations

from tasks import (
    build_abs_bundle,
    build_conditional_bundle,
    build_exponentiation_bundle,
    build_linear_solve_bundle,
    build_max_bundle,
    build_multiply_bundle,
    build_task_suite,
)


def test_task_suite_contains_six_bundles() -> None:
    suite = build_task_suite()
    assert len(suite) == 6
    assert [bundle.name for bundle in suite] == [
        "multiply",
        "max",
        "abs",
        "conditional",
        "exponentiation",
        "linear_solve",
    ]


def test_bundles_have_validation_and_shortcuts() -> None:
    for bundle in (
        build_multiply_bundle(),
        build_max_bundle(),
        build_abs_bundle(),
        build_conditional_bundle(),
        build_exponentiation_bundle(),
        build_linear_solve_bundle(),
    ):
        assert bundle.train
        assert bundle.validation
        assert bundle.anti_shortcuts
        assert bundle.shortcut_checks
