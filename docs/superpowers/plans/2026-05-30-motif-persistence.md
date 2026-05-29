# Motif Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist evolved circuits as reusable motifs in a SQLite database. When `experiments/33_epistasis_colony4.py` detects a cooperation breakthrough (gate_err < 0.05 for the first time), it captures the best cell1 and cell2 genomes as role-tagged motifs and writes them to `data/motifs.db`. Future experiments can query and seed populations from this library.

**Architecture:** New thin SQLite wrapper `motif_store.py` with `MotifStore` class. Modified `experiments/33_epistasis_colony4.py` to detect first threshold crossing and capture two motifs (role="gate" and role="echo"). `data/` directory created at runtime, excluded from git.

**Tech Stack:** Python stdlib only — `sqlite3`, `json`, `pathlib`. No new dependencies.

---

## Task 1 — Create `motif_store.py`

- [ ] Create `/home/mattthomson/workspace/AI-DNA/motif_store.py` with the following content:

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from genome import Motif
from codons import default_codon_table


class MotifStore:
    """Thin SQLite wrapper for persisting and querying Motif objects."""

    def __init__(self, db_path: str | Path = "data/motifs.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS motifs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                task TEXT NOT NULL,
                experiment TEXT NOT NULL,
                generation INTEGER NOT NULL,
                gate_err REAL NOT NULL,
                lineage_id TEXT NOT NULL,
                codons TEXT NOT NULL,
                origin_signals TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def save(
        self,
        motif: Motif,
        role: str,
        task: str,
        gate_err: float,
        generation: int,
        experiment: str,
    ) -> None:
        """Insert one motif row."""
        self._conn.execute(
            "INSERT INTO motifs "
            "(role, task, experiment, generation, gate_err, lineage_id, codons, origin_signals) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                role,
                task,
                experiment,
                generation,
                gate_err,
                motif.origin_lineage,
                json.dumps(list(motif.pattern)),
                json.dumps(list(motif.origin_signals)),
            ),
        )
        self._conn.commit()

    def query(
        self,
        role: str | None = None,
        task: str | None = None,
        top_k: int = 4,
    ) -> list[Motif]:
        """Return up to top_k Motif objects ordered by gate_err ASC."""
        clauses: list[str] = []
        params: list[object] = []
        if role is not None:
            clauses.append("role = ?")
            params.append(role)
        if task is not None:
            clauses.append("task = ?")
            params.append(task)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT codons, origin_signals, lineage_id, task FROM motifs "
            f"{where} ORDER BY gate_err ASC LIMIT ?",
            params + [top_k],
        ).fetchall()
        result: list[Motif] = []
        for codons_json, signals_json, lineage_id, origin_task in rows:
            pattern = tuple(json.loads(codons_json))
            origin_signals = tuple(json.loads(signals_json))
            result.append(
                Motif(
                    pattern=pattern,
                    origin_lineage=lineage_id,
                    origin_task=origin_task,
                    origin_signals=origin_signals,
                )
            )
        return result

    def close(self) -> None:
        self._conn.close()
```

**Verification:** `Motif.pattern` is `tuple[str, ...]` (rule name strings). `Motif.origin_lineage`, `origin_task`, `origin_signals` confirmed from `genome.py`. `codons` column stores JSON array of rule name strings (the motif pattern), not raw genome integer codons — this is what `GenomeWriter.compose()` consumes via `motif.pattern`.

---

## Task 2 — Create `tests/test_motif_store.py`

- [ ] Create `/home/mattthomson/workspace/AI-DNA/tests/test_motif_store.py` with the following content:

```python
from __future__ import annotations

import pytest
from pathlib import Path

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
```

---

## Task 3 — Modify `experiments/33_epistasis_colony4.py`

- [ ] Add imports for `MotifStore` and `extract_motif_from_rules` at the top of the import block (after the existing imports)
- [ ] Add `store`, `motif_captured`, breakthrough detection after `gen_best_gate` computation in the gate loop

**Exact diff to apply:**

After line 16 (`from tasks import TaskCase`), add:

```python
from genome import extract_motif_from_rules
from motif_store import MotifStore
```

In `main()`, after `rng = Random(33)` (line 194), add:

```python
    store = MotifStore()
    motif_captured = False
```

In the gate loop, after line 221 (the `print(...)` call inside the loop), add the following block — it goes after the existing `print(...)` and before the `if generation % 20 == 0:` block:

```python
        if not motif_captured and gen_best_gate < 0.05:
            best_a = scores_a[0][1]
            best_b = scores_b[0][1]
            probe_case = _make_case(2, 3)

            cell1 = CellState(active_rules=list(best_a.declare_rules()))
            cell1.signals = [2 / 3.0, 0.0, 0.0, 0.0, 0.0]
            cell2 = CellState(active_rules=list(best_b.declare_rules()))
            cell2.signals = [0.0, 1.0, 0.0, 0.0, 0.0]
            context = ChemistryContext()
            system.run([cell1, cell2], probe_case, context=context, max_time=float(CHEMISTRY_ROUNDS))

            motif_a = extract_motif_from_rules(
                [str(r) for r in best_a.declare_rules()],
                origin_lineage=best_a.lineage_id,
                origin_task="multiply_2cell",
                origin_signals=tuple(cell1.signals),
            )
            motif_b = extract_motif_from_rules(
                [str(r) for r in best_b.declare_rules()],
                origin_lineage=best_b.lineage_id,
                origin_task="multiply_2cell",
                origin_signals=tuple(cell2.signals),
            )

            store.save(motif_a, role="gate", task="multiply_2cell",
                       gate_err=gen_best_gate, generation=generation, experiment="epistasis_colony4")
            print(f"epistasis_colony4: motif_captured role=gate gen={generation} gate_err={gen_best_gate:.6f}")

            store.save(motif_b, role="echo", task="multiply_2cell",
                       gate_err=gen_best_gate, generation=generation, experiment="epistasis_colony4")
            print(f"epistasis_colony4: motif_captured role=echo gen={generation} gate_err={gen_best_gate:.6f}")

            motif_captured = True
```

**Note on `extract_motif_from_rules`:** This function from `genome.py` takes a `Sequence[str]` of declared rule names, so we must convert `declare_rules()` output (which returns `tuple[str | GateRule | ScaleRule, ...]`) to strings. The function already filters with `isinstance(rule, str)` internally, so passing stringified rules is safe — alternatively pass only the string elements directly:

```python
            motif_a = extract_motif_from_rules(
                [r for r in best_a.declare_rules() if isinstance(r, str)],
                origin_lineage=best_a.lineage_id,
                origin_task="multiply_2cell",
                origin_signals=tuple(cell1.signals),
            )
            motif_b = extract_motif_from_rules(
                [r for r in best_b.declare_rules() if isinstance(r, str)],
                origin_lineage=best_b.lineage_id,
                origin_task="multiply_2cell",
                origin_signals=tuple(cell2.signals),
            )
```

Use the `isinstance(r, str)` form — it is correct and matches `extract_motif_from_rules`'s own internal logic.

---

## Task 4 — Add `data/` to `.gitignore`

- [ ] Create `/home/mattthomson/workspace/AI-DNA/.gitignore` with content:

```
data/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
```

Note: `.gitignore` does not currently exist in the repo. The `__pycache__/` entry is already untracked (confirmed from git status) so including it here is correct.

---

## Task 5 — Run tests

- [ ] Run: `cd /home/mattthomson/workspace/AI-DNA && python -m pytest tests/test_motif_store.py -v`

All 6 tests must pass. If any fail, diagnose and fix before proceeding.

- [ ] Also run the full suite to confirm no regressions: `python -m pytest tests/ -v --ignore=tests/test_experiments.py -x`

(Ignoring `test_experiments.py` as it runs long experiment scripts.)

---

## Task 6 — Commit

- [ ] Stage and commit:

```bash
git add motif_store.py tests/test_motif_store.py experiments/33_epistasis_colony4.py .gitignore
git commit -m "$(cat <<'EOF'
Add MotifStore SQLite persistence and exp33 breakthrough capture

- New motif_store.py: thin SQLite wrapper with save() and query() for Motif objects
- tests/test_motif_store.py: 6 tests covering save, query, filter, roundtrip, top_k
- experiments/33_epistasis_colony4.py: capture gate+echo motifs on first gate_err < 0.05
- .gitignore: exclude data/ directory (runtime artifact)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
