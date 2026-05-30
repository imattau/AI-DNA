from __future__ import annotations

from pathlib import Path

import pytest

from genome import Motif
from motif_store import MotifStore


def _make_motif(
    pattern: tuple[str, ...] = ("RULE_EMIT_X", "RULE_EMIT_Y"),
    lineage: str = "A1",
    task: str = "multiply_2cell",
    signals: tuple[float, ...] = (0.5, 0.3, 0.1),
) -> Motif:
    return Motif(
        pattern=pattern,
        origin_lineage=lineage,
        origin_task=task,
        origin_signals=signals,
    )


@pytest.fixture
def store(tmp_path: Path) -> MotifStore:
    return MotifStore(db_path=tmp_path / "test_motifs.db")


def test_save_and_query_by_role(store: MotifStore) -> None:
    motif = _make_motif(lineage="A1")
    store.save(motif, role="gate", task="multiply_2cell", gate_err=0.03, generation=42, experiment="exp33")
    results = store.query(role="gate")
    assert len(results) == 1
    assert results[0].origin_lineage == "A1"
    assert results[0].pattern == ("RULE_EMIT_X", "RULE_EMIT_Y")


def test_query_returns_best_first(store: MotifStore) -> None:
    worse = _make_motif(lineage="A_worse")
    better = _make_motif(lineage="A_better")
    store.save(worse, role="gate", task="multiply_2cell", gate_err=0.08, generation=10, experiment="exp33")
    store.save(better, role="gate", task="multiply_2cell", gate_err=0.02, generation=20, experiment="exp33")
    results = store.query(role="gate")
    assert len(results) == 2
    assert results[0].origin_lineage == "A_better"
    assert results[1].origin_lineage == "A_worse"


def test_query_filters_by_task(store: MotifStore) -> None:
    motif_a = _make_motif(lineage="A1", task="multiply_2cell")
    motif_b = _make_motif(lineage="B1", task="other_task")
    store.save(motif_a, role="gate", task="multiply_2cell", gate_err=0.03, generation=5, experiment="exp33")
    store.save(motif_b, role="gate", task="other_task", gate_err=0.01, generation=5, experiment="exp99")
    results = store.query(task="multiply_2cell")
    assert len(results) == 1
    assert results[0].origin_lineage == "A1"


def test_pattern_roundtrips(store: MotifStore) -> None:
    pattern = ("RULE_EMIT_X", "RULE_ZERO_2", "RULE_ADD0_IF1", "RULE_DECAY1", "RULE_OUTPUT_IF1Z")
    motif = _make_motif(pattern=pattern, lineage="C1")
    store.save(motif, role="echo", task="multiply_2cell", gate_err=0.04, generation=100, experiment="exp33")
    results = store.query(role="echo")
    assert len(results) == 1
    assert results[0].pattern == pattern


def test_query_top_k_limits_results(store: MotifStore) -> None:
    for i in range(6):
        m = _make_motif(lineage=f"L{i}")
        store.save(m, role="gate", task="multiply_2cell", gate_err=0.01 * (i + 1), generation=i, experiment="exp33")
    results = store.query(role="gate", top_k=3)
    assert len(results) == 3


def test_query_role_and_task_combined(store: MotifStore) -> None:
    store.save(_make_motif(lineage="X1"), role="gate", task="multiply_2cell", gate_err=0.02, generation=1, experiment="exp33")
    store.save(_make_motif(lineage="X2"), role="echo", task="multiply_2cell", gate_err=0.02, generation=1, experiment="exp33")
    store.save(_make_motif(lineage="X3"), role="gate", task="other_task",    gate_err=0.01, generation=1, experiment="exp99")
    results = store.query(role="gate", task="multiply_2cell")
    assert len(results) == 1
    assert results[0].origin_lineage == "X1"
