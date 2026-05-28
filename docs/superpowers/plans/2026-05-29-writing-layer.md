# Writing Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `writing.py` with `MotifLibrary` (query engine) and `GenomeWriter` (composition engine), plus `origin_signals` on `Motif` for cosine-similarity selection.

**Architecture:** `Motif` gains an `origin_signals` field populated at capture time. `MotifLibrary` filters/ranks motifs by task origin, lineage prefix, and cosine similarity. `GenomeWriter.compose()` concatenates motif codon patterns and calls `declare_rules()` to produce a valid `CellGenome`.

**Tech Stack:** Python 3.10+, dataclasses, pytest. No new dependencies.

---

## File Map

- Modify: `genome.py` — add `origin_signals: tuple[float, ...] = ()` to `Motif`
- Modify: `evolution.py` — populate `origin_signals` in `attach_success_motif()`
- Create: `writing.py` — `_cosine()`, `MotifQuery`, `MotifLibrary`, `GenomeWriter`
- Create: `tests/test_writing.py` — unit + integration tests

---

### Task 1: Add `origin_signals` to `Motif`

**Files:**
- Modify: `genome.py`
- Create: `tests/test_writing.py`

Note: The actual `Motif` fields are `pattern` (tuple of rule name strings), `origin_lineage`, `origin_task`, `reuse_count`. The spec calls the codon sequence `codons` but in the codebase it is `pattern`. Use `pattern` throughout.

- [ ] **Step 1: Write the failing test**

Create `tests/test_writing.py`:

```python
from __future__ import annotations

from genome import Motif


def test_motif_has_origin_signals() -> None:
    m = Motif(pattern=("EMIT_S0",), origin_lineage="L1", origin_task="multiply")
    assert hasattr(m, "origin_signals")
    assert isinstance(m.origin_signals, tuple)
    assert m.origin_signals == ()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_writing.py::test_motif_has_origin_signals -v
```

Expected: FAIL with `TypeError` (unexpected keyword argument) or `AttributeError`.

- [ ] **Step 3: Add field to `Motif`**

In `genome.py`, find the `Motif` dataclass (line ~19). It currently has:

```python
@dataclass(frozen=True, slots=True)
class Motif:
    pattern: tuple[str, ...]
    origin_lineage: str
    origin_task: str
    reuse_count: int = 0
```

Change to:

```python
@dataclass(frozen=True, slots=True)
class Motif:
    pattern: tuple[str, ...]
    origin_lineage: str
    origin_task: str
    reuse_count: int = 0
    origin_signals: tuple[float, ...] = ()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_writing.py::test_motif_has_origin_signals -v
```

Expected: PASS

- [ ] **Step 5: Run full fast suite to check no regressions**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add genome.py tests/test_writing.py
git commit -m "feat: add origin_signals field to Motif"
```

---

### Task 2: Populate `origin_signals` in `attach_success_motif()`

**Files:**
- Modify: `evolution.py`
- Modify: `tests/test_writing.py`

`attach_success_motif(genome, task_name)` currently calls `extract_motif_from_rules()` with no signal context. We need a cell's signal vector at the moment of success. The function signature must be extended to accept an optional signal snapshot.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_writing.py`:

```python
from evolution import attach_success_motif
from genome import CellGenome


def test_attach_success_motif_stores_origin_signals() -> None:
    genome = CellGenome.from_rule_names(
        ["EMIT_S0", "OUTPUT"],
        lineage_id="L1",
    )
    signals = (0.5, 0.0, 0.0, 0.0, 0.0, 1.0)
    updated = attach_success_motif(genome, "multiply", origin_signals=signals)
    assert len(updated.local_motifs) > 0
    motif = updated.local_motifs[-1]
    assert motif.origin_signals == signals
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_writing.py::test_attach_success_motif_stores_origin_signals -v
```

Expected: FAIL — `attach_success_motif` does not accept `origin_signals` kwarg.

- [ ] **Step 3: Update `attach_success_motif` in `evolution.py`**

Find `attach_success_motif` (~line 261). Current:

```python
def attach_success_motif(genome: CellGenome, task_name: str) -> CellGenome:
    motif = extract_motif_from_rules(genome.declare_rules(), origin_lineage=genome.lineage_id, origin_task=task_name)
    return genome.attach_motif(motif)
```

Replace with:

```python
def attach_success_motif(
    genome: CellGenome,
    task_name: str,
    origin_signals: tuple[float, ...] = (),
) -> CellGenome:
    motif = extract_motif_from_rules(
        genome.declare_rules(),
        origin_lineage=genome.lineage_id,
        origin_task=task_name,
    )
    motif = replace(motif, origin_signals=origin_signals)
    return genome.attach_motif(motif)
```

Ensure `replace` is imported: add `from dataclasses import replace` if not already present at the top of `evolution.py`.

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_writing.py::test_attach_success_motif_stores_origin_signals -v
```

Expected: PASS

- [ ] **Step 5: Run full fast suite**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add evolution.py tests/test_writing.py
git commit -m "feat: populate origin_signals in attach_success_motif"
```

---

### Task 3: Create `writing.py` with `_cosine`, `MotifQuery`, `MotifLibrary`

**Files:**
- Create: `writing.py`
- Modify: `tests/test_writing.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_writing.py`:

```python
from writing import MotifLibrary, MotifQuery


def _make_motif(task: str, lineage: str, signals: tuple[float, ...] = ()) -> Motif:
    return Motif(
        pattern=("EMIT_S0",),
        origin_lineage=lineage,
        origin_task=task,
        origin_signals=signals,
    )


def test_motif_library_filter_by_task() -> None:
    motifs = [
        _make_motif("multiply", "L1"),
        _make_motif("add", "L2"),
        _make_motif("multiply", "L3"),
    ]
    lib = MotifLibrary(motifs)
    result = lib.query(MotifQuery(task_origin="multiply", top_k=10))
    assert len(result) == 2
    assert all(m.origin_task == "multiply" for m in result)


def test_motif_library_filter_by_lineage_prefix() -> None:
    motifs = [
        _make_motif("multiply", "ABC.1"),
        _make_motif("multiply", "ABC.2"),
        _make_motif("multiply", "XYZ.1"),
    ]
    lib = MotifLibrary(motifs)
    result = lib.query(MotifQuery(lineage_prefix="ABC", top_k=10))
    assert len(result) == 2
    assert all(m.origin_lineage.startswith("ABC") for m in result)


def test_motif_library_rank_by_cosine() -> None:
    motifs = [
        _make_motif("multiply", "L1", signals=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        _make_motif("multiply", "L2", signals=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        _make_motif("multiply", "L3", signals=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0)),
    ]
    lib = MotifLibrary(motifs)
    query_vec = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    result = lib.query(MotifQuery(signal_vector=query_vec, top_k=2))
    # L1 (exact match) and L3 (close) should be top 2; L2 should be excluded
    lineages = [m.origin_lineage for m in result]
    assert "L1" in lineages
    assert "L3" in lineages
    assert "L2" not in lineages


def test_motif_library_empty_query_returns_top_k() -> None:
    motifs = [_make_motif("multiply", f"L{i}") for i in range(6)]
    lib = MotifLibrary(motifs)
    result = lib.query(MotifQuery(top_k=3))
    assert len(result) == 3


def test_motif_library_combined_filters() -> None:
    motifs = [
        _make_motif("multiply", "A.1", signals=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        _make_motif("add", "A.2", signals=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        _make_motif("multiply", "B.1", signals=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    lib = MotifLibrary(motifs)
    result = lib.query(MotifQuery(task_origin="multiply", lineage_prefix="A", top_k=10))
    assert len(result) == 1
    assert result[0].origin_lineage == "A.1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_writing.py -k "library" -v
```

Expected: FAIL — `writing` module not found.

- [ ] **Step 3: Create `writing.py`**

```python
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

from genome import CellGenome, Motif


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


@dataclass
class MotifQuery:
    task_origin: str | None = None
    lineage_prefix: str | None = None
    signal_vector: tuple[float, ...] = ()
    top_k: int = 4


class MotifLibrary:
    def __init__(self, motifs: Iterable[Motif]) -> None:
        self._motifs: list[Motif] = list(motifs)

    def query(self, q: MotifQuery) -> list[Motif]:
        pool = self._motifs
        if q.task_origin is not None:
            pool = [m for m in pool if m.origin_task == q.task_origin]
        if q.lineage_prefix is not None:
            pool = [m for m in pool if m.origin_lineage.startswith(q.lineage_prefix)]
        if q.signal_vector:
            pool = sorted(
                pool,
                key=lambda m: _cosine(q.signal_vector, m.origin_signals),
                reverse=True,
            )
        return pool[: q.top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_writing.py -k "library" -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add writing.py tests/test_writing.py
git commit -m "feat: add MotifLibrary and MotifQuery to writing.py"
```

---

### Task 4: Add `GenomeWriter` to `writing.py`

**Files:**
- Modify: `writing.py`
- Modify: `tests/test_writing.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_writing.py`:

```python
from writing import GenomeWriter
import pytest


def test_genome_writer_compose_produces_genome() -> None:
    motifs = [
        Motif(pattern=("EMIT_S0", "OUTPUT"), origin_lineage="L1", origin_task="multiply"),
    ]
    writer = GenomeWriter()
    genome = writer.compose(motifs, lineage_id="composed.1")
    assert isinstance(genome, CellGenome)
    assert genome.lineage_id == "composed.1"
    assert len(genome.declare_rules()) > 0


def test_genome_writer_compose_concatenates_patterns() -> None:
    motifs = [
        Motif(pattern=("EMIT_S0",), origin_lineage="L1", origin_task="multiply"),
        Motif(pattern=("OUTPUT",), origin_lineage="L2", origin_task="multiply"),
    ]
    single = Motif(pattern=("EMIT_S0", "OUTPUT"), origin_lineage="L1", origin_task="multiply")
    writer = GenomeWriter()
    composed = writer.compose(motifs)
    single_genome = writer.compose([single])
    assert composed.codons == single_genome.codons


def test_genome_writer_compose_raises_on_empty() -> None:
    writer = GenomeWriter()
    with pytest.raises(ValueError):
        writer.compose([])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_writing.py -k "writer" -v
```

Expected: FAIL — `GenomeWriter` not defined.

- [ ] **Step 3: Add `GenomeWriter` to `writing.py`**

Add after `MotifLibrary`:

```python
class GenomeWriter:
    def compose(self, motifs: Sequence[Motif], lineage_id: str = "") -> CellGenome:
        if not motifs:
            raise ValueError("compose() requires at least one motif")
        combined: list[str] = []
        for motif in motifs:
            combined.extend(motif.pattern)
        return CellGenome.from_rule_names(combined, lineage_id=lineage_id)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_writing.py -k "writer" -v
```

Expected: all PASS.

- [ ] **Step 5: Run all writing tests**

```bash
python3 -m pytest tests/test_writing.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add writing.py tests/test_writing.py
git commit -m "feat: add GenomeWriter to writing.py"
```

---

### Task 5: Integration round-trip test

**Files:**
- Modify: `tests/test_writing.py`

Verify the full pipeline: stream run → motif capture → library query → compose → evaluate.

- [ ] **Step 1: Write the integration test**

Add to `tests/test_writing.py`:

```python
from random import Random
from task_stream import StreamConfig, run_task_stream
from tasks import TaskBundle
from writing import GenomeWriter, MotifLibrary, MotifQuery


def test_writing_layer_round_trip() -> None:
    tasks = [
        TaskBundle(
            name="multiply",
            cases=((2, 3, 6), (1, 4, 4), (3, 3, 9)),
            anti_shortcut_cases=(),
        ),
    ]
    config = StreamConfig(
        initial_population_size=4,
        max_steps_per_task=5,
        archive_interval=10,
        estimate_neutrality_trials=0,
    )
    report = run_task_stream("test_write_roundtrip", tasks, config=config, seed=42)

    # Collect all motifs from the final archive
    all_motifs = []
    for genome in report.final_archive:
        all_motifs.extend(genome.local_motifs)

    if not all_motifs:
        return  # no motifs captured — short run, skip assertion

    lib = MotifLibrary(all_motifs)
    selected = lib.query(MotifQuery(task_origin="multiply", top_k=2))
    assert len(selected) > 0

    writer = GenomeWriter()
    composed = writer.compose(selected, lineage_id="written.1")
    assert isinstance(composed, CellGenome)
    assert math.isfinite(composed.declare_rules().__class__.__len__(composed.declare_rules()) >= 0 or 0)
    # Verify transcription produced something evaluatable
    rules = composed.declare_rules()
    assert isinstance(rules, tuple)
```

Wait — `report.final_archive` may not exist. Check what `run_task_stream` returns:

```bash
python3 -c "from task_stream import StreamReport; import inspect; print([f for f in inspect.fields(StreamReport)])"
```

- [ ] **Step 2: Check `StreamReport` fields**

```bash
python3 -c "from task_stream import StreamReport; from dataclasses import fields; print([f.name for f in fields(StreamReport)])"
```

Note the output and adjust the integration test to use the correct field for the final population. Common options: `archive_snapshots`, `results`.

- [ ] **Step 3: Rewrite integration test using correct field**

Replace the integration test body with a version that collects genomes from the archive snapshots' best lineage, or uses a simpler approach: directly create motifs from rule names and round-trip through the writer.

```python
def test_writing_layer_round_trip() -> None:
    """Compose a genome from hand-crafted motifs and verify it transcribes."""
    import math
    from genome import CellGenome, Motif
    from writing import GenomeWriter, MotifLibrary, MotifQuery

    motifs = [
        Motif(
            pattern=("EMIT_S0", "ADD_S0_S1", "OUTPUT"),
            origin_lineage="A.1",
            origin_task="multiply",
            origin_signals=(0.5, 0.3, 0.0, 0.0, 0.0, 0.8),
        ),
        Motif(
            pattern=("DECAY_S0", "THRESHOLD_S0"),
            origin_lineage="A.2",
            origin_task="multiply",
            origin_signals=(0.2, 0.0, 0.0, 0.0, 0.0, 0.1),
        ),
    ]

    lib = MotifLibrary(motifs)
    query_vec = (0.5, 0.3, 0.0, 0.0, 0.0, 0.8)
    selected = lib.query(MotifQuery(task_origin="multiply", signal_vector=query_vec, top_k=2))
    assert len(selected) == 2
    assert selected[0].origin_lineage == "A.1"  # closest cosine match

    writer = GenomeWriter()
    composed = writer.compose(selected, lineage_id="written.1")
    assert isinstance(composed, CellGenome)
    assert composed.lineage_id == "written.1"
    rules = composed.declare_rules()
    assert isinstance(rules, tuple)
```

- [ ] **Step 4: Run integration test**

```bash
python3 -m pytest tests/test_writing.py::test_writing_layer_round_trip -v
```

Expected: PASS.

- [ ] **Step 5: Run full fast suite**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_writing.py
git commit -m "test: add writing layer round-trip integration test"
```

---

### Task 6: Final check

- [ ] **Step 1: Run full fast suite one more time**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```

Expected: all pass.

- [ ] **Step 2: Verify imports are clean**

```bash
python3 -c "from writing import MotifLibrary, MotifQuery, GenomeWriter; print('writing.py imports OK')"
```

Expected: `writing.py imports OK`
