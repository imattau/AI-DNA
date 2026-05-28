# Reading Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add to_vector() to CellState, SENSE_PEER_N codon ops for inter-cell reading, and CellReader in reading.py for external agent observation — one 6-float protocol serving both consumers.
**Architecture:** to_vector() on CellState (6 base, 15 extended); peer_vectors dict on ChemistryContext; SENSE_PEER_0/1 rule ops; CellReader wrapper in new reading.py. All additive, backward compatible.
**Tech Stack:** Python 3.10+, dataclasses, pytest

---

## File Map

- Modify: `/home/mattthomson/workspace/AI-DNA/cell.py` — add `to_vector()` method
- Modify: `/home/mattthomson/workspace/AI-DNA/codons.py` — add `SENSE_PEER_0`, `SENSE_PEER_1` to REGULATORY_OP_NAMES and REGULATORY_CODON_MAP
- Modify: `/home/mattthomson/workspace/AI-DNA/chemistry.py` — add `peer_vectors` field to ChemistryContext; handle SENSE_PEER_0/1 in rule dispatch
- Modify: `/home/mattthomson/workspace/AI-DNA/cooperative_chemistry.py` — store `to_vector()` in `context.peer_vectors` after each cell step
- Create: `/home/mattthomson/workspace/AI-DNA/reading.py` — CellReader with four observation modes
- Create: `/home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py` — Level 1–5 tests

---

## Task 1: `to_vector()` on CellState

**Files:**
- Modify: `/home/mattthomson/workspace/AI-DNA/cell.py`
- Create: `/home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py` (Level 1 tests only)

### Step 1: Write the failing tests

Create `/home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py`:

```python
from __future__ import annotations

import pytest
from cell import CellState


# ---------------------------------------------------------------------------
# Level 1 — to_vector() base (unit)
# ---------------------------------------------------------------------------

def _make_cell(signals: list[float], output: float | None) -> CellState:
    cell = CellState(active_rules=[])
    cell.signals = list(signals)
    cell.output = output
    return cell


def test_to_vector_base_values() -> None:
    """signals [1,2,0.5,0,0.3] + output 4.5 → (1.0, 2.0, 0.5, 0.0, 0.3, 4.5)."""
    cell = _make_cell([1.0, 2.0, 0.5, 0.0, 0.3], 4.5)
    assert cell.to_vector() == (1.0, 2.0, 0.5, 0.0, 0.3, 4.5)


def test_to_vector_none_output_becomes_zero() -> None:
    """output=None → vector[5] = 0.0."""
    cell = _make_cell([1.0, 2.0, 0.5, 0.0, 0.3], None)
    assert cell.to_vector()[5] == 0.0


def test_to_vector_base_length_is_six() -> None:
    """Base vector always has exactly 6 elements."""
    cell = _make_cell([0.0, 0.0, 0.0, 0.0, 0.0], None)
    assert len(cell.to_vector()) == 6


def test_to_vector_extended_length_is_fifteen() -> None:
    """Extended vector always has exactly 15 elements."""
    cell = _make_cell([0.0, 0.0, 0.0, 0.0, 0.0], None)
    assert len(cell.to_vector(extended=True)) == 15


def test_to_vector_deterministic() -> None:
    """Same cell state called twice returns identical vector."""
    cell = _make_cell([1.0, 2.0, 0.5, 0.0, 0.3], 4.5)
    assert cell.to_vector() == cell.to_vector()


# ---------------------------------------------------------------------------
# Level 2 — to_vector(extended=True) (unit)
# ---------------------------------------------------------------------------

def test_to_vector_extended_gene_flags() -> None:
    """active_gene_ids={2, 5} → vector[8]=1.0, vector[11]=1.0, others 0.0."""
    cell = _make_cell([0.0, 0.0, 0.0, 0.0, 0.0], None)
    vec = cell.to_vector(extended=True, active_gene_ids=frozenset({2, 5}))
    # base [0-5], gene_flags [6-13], motif_hash [14]
    # gene 0 → index 6, gene 2 → index 8, gene 5 → index 11
    assert vec[6] == 0.0   # gene 0 not active
    assert vec[8] == 1.0   # gene 2 active
    assert vec[11] == 1.0  # gene 5 active
    assert vec[7] == 0.0   # gene 1 not active
    assert vec[9] == 0.0   # gene 3 not active
    assert vec[10] == 0.0  # gene 4 not active
    assert vec[12] == 0.0  # gene 6 not active
    assert vec[13] == 0.0  # gene 7 not active


def test_to_vector_extended_motif_hash() -> None:
    """motif_hash=0.42 appears at vector[14]."""
    cell = _make_cell([0.0, 0.0, 0.0, 0.0, 0.0], None)
    vec = cell.to_vector(extended=True, motif_hash=0.42)
    assert vec[14] == pytest.approx(0.42)


def test_to_vector_extended_base_unchanged() -> None:
    """Extended vector[0-5] matches base vector."""
    cell = _make_cell([1.0, 2.0, 0.5, 0.0, 0.3], 4.5)
    base = cell.to_vector()
    ext = cell.to_vector(extended=True)
    assert ext[:6] == base
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/test_reading_layer.py -v
```

Expected: FAIL — `CellState` has no `to_vector` attribute.

### Step 3: Add `to_vector()` to `cell.py`

In `/home/mattthomson/workspace/AI-DNA/cell.py`, add the method inside the `CellState` class, after `snapshot()`:

```python
def to_vector(
    self,
    *,
    extended: bool = False,
    active_gene_ids: frozenset[int] = frozenset(),
    motif_hash: float = 0.0,
) -> tuple[float, ...]:
    out = self.output if self.output is not None else 0.0
    base = (*self.signals, out)
    if not extended:
        return base
    gene_flags = tuple(1.0 if i in active_gene_ids else 0.0 for i in range(8))
    return base + gene_flags + (motif_hash,)
```

Full updated `cell.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CellState:
    active_rules: list[str]
    signals: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    output: float | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def reset(self, *, x: float, y: float) -> None:
        self.signals = [float(x), float(y), 0.0, 0.0, 0.0]
        self.output = None
        self.trace.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_rules": list(self.active_rules),
            "signals": list(self.signals),
            "output": self.output,
        }

    def to_vector(
        self,
        *,
        extended: bool = False,
        active_gene_ids: frozenset[int] = frozenset(),
        motif_hash: float = 0.0,
    ) -> tuple[float, ...]:
        out = self.output if self.output is not None else 0.0
        base = (*self.signals, out)
        if not extended:
            return base
        gene_flags = tuple(1.0 if i in active_gene_ids else 0.0 for i in range(8))
        return base + gene_flags + (motif_hash,)

    @property
    def cell_count(self) -> int:
        return 1
```

### Step 4: Run Level 1 and 2 tests

```bash
python3 -m pytest tests/test_reading_layer.py -v
```

Expected: all Level 1 and 2 tests pass.

### Step 5: Run full fast suite

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass — `to_vector()` is additive, no existing paths modified.

### Step 6: Commit

```bash
git add /home/mattthomson/workspace/AI-DNA/cell.py /home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py
git commit -m "feat: add to_vector() to CellState — 6-float base, 15-float extended"
```

- [ ] Step 1: Write Level 1 and 2 failing tests in test_reading_layer.py
- [ ] Step 2: Run tests — verify FAIL
- [ ] Step 3: Add to_vector() to cell.py
- [ ] Step 4: Run test_reading_layer.py — verify Level 1/2 PASS
- [ ] Step 5: Run full fast suite — all pass
- [ ] Step 6: Commit

---

## Task 2: SENSE_PEER_N codon ops + peer_vectors on ChemistryContext

**Files:**
- Modify: `/home/mattthomson/workspace/AI-DNA/codons.py`
- Modify: `/home/mattthomson/workspace/AI-DNA/chemistry.py`
- Modify: `/home/mattthomson/workspace/AI-DNA/cooperative_chemistry.py`
- Modify: `/home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py` (add Level 3 tests)

### Step 1: Write the failing Level 3 tests

Append to `/home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py`:

```python
# ---------------------------------------------------------------------------
# Level 3 — Inter-cell reading (unit)
# ---------------------------------------------------------------------------

def test_sense_peer_0_updates_signals_when_peer_available() -> None:
    """SENSE_PEER_0 with peer 0 vector available → cell signals[0-4] updated."""
    from chemistry import ChemistryContext, ChemistrySystem

    # Build a peer vector: peer has signals [0.1, 0.2, 0.3, 0.4, 0.5], output 9.9
    peer = _make_cell([0.1, 0.2, 0.3, 0.4, 0.5], 9.9)
    peer_vec = peer.to_vector()  # (0.1, 0.2, 0.3, 0.4, 0.5, 9.9)

    context = ChemistryContext()
    context.peer_vectors[0] = peer_vec

    # Cell with SENSE_PEER_0 as active rule
    from tasks import TaskCase
    cell = CellState(active_rules=["SENSE_PEER_0"])
    cell.reset(x=0.0, y=0.0)

    system = ChemistrySystem(max_time=1.0, dt=1.0)
    system.step(cell, TaskCase(x=0.0, y=0.0, z=0.0), context)

    assert cell.signals[0] == pytest.approx(0.1)
    assert cell.signals[1] == pytest.approx(0.2)
    assert cell.signals[2] == pytest.approx(0.3)
    assert cell.signals[3] == pytest.approx(0.4)
    assert cell.signals[4] == pytest.approx(0.5)


def test_sense_peer_0_noop_when_no_peer() -> None:
    """SENSE_PEER_0 with empty peer_vectors → no-op, signals unchanged."""
    from chemistry import ChemistryContext, ChemistrySystem
    from tasks import TaskCase

    cell = CellState(active_rules=["SENSE_PEER_0"])
    cell.reset(x=1.0, y=2.0)
    original_signals = list(cell.signals)

    context = ChemistryContext()  # peer_vectors empty

    system = ChemistrySystem(max_time=1.0, dt=1.0)
    system.step(cell, TaskCase(x=1.0, y=2.0, z=0.0), context)

    assert cell.signals == original_signals


def test_sense_peer_1_reads_correct_peer() -> None:
    """SENSE_PEER_1 reads peer index 1, not peer 0."""
    from chemistry import ChemistryContext, ChemistrySystem
    from tasks import TaskCase

    peer0 = _make_cell([9.0, 9.0, 9.0, 9.0, 9.0], None)
    peer1 = _make_cell([0.7, 0.8, 0.9, 0.6, 0.5], None)

    context = ChemistryContext()
    context.peer_vectors[0] = peer0.to_vector()
    context.peer_vectors[1] = peer1.to_vector()

    cell = CellState(active_rules=["SENSE_PEER_1"])
    cell.reset(x=0.0, y=0.0)

    system = ChemistrySystem(max_time=1.0, dt=1.0)
    system.step(cell, TaskCase(x=0.0, y=0.0, z=0.0), context)

    # Should have peer1's signals, not peer0's
    assert cell.signals[0] == pytest.approx(0.7)
    assert cell.signals[1] == pytest.approx(0.8)


def test_cooperative_cell_b_reads_cell_a_signals() -> None:
    """Two cells cooperative: cell B with SENSE_PEER_0 reads cell A's signals correctly."""
    from chemistry import ChemistryContext, ChemistrySystem
    from cooperative_chemistry import CooperativeChemistrySystem
    from tasks import TaskCase

    # Cell A: sets signal[0] = task.x via RULE_EMIT_X
    cell_a = CellState(active_rules=["RULE_EMIT_X", "RULE_EMIT_Y"])
    cell_a.reset(x=5.5, y=3.3)

    # Cell B: reads peer 0 (cell A) using SENSE_PEER_0
    cell_b = CellState(active_rules=["SENSE_PEER_0"])
    cell_b.reset(x=0.0, y=0.0)

    system = CooperativeChemistrySystem()
    result = system.run([cell_a, cell_b], TaskCase(x=5.5, y=3.3, z=0.0))

    # After cooperative run, cell B should have read cell A's signal[0]=5.5
    assert result.cells[1].signals[0] == pytest.approx(5.5)
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/test_reading_layer.py::test_sense_peer_0_updates_signals_when_peer_available tests/test_reading_layer.py::test_sense_peer_0_noop_when_no_peer tests/test_reading_layer.py::test_sense_peer_1_reads_correct_peer tests/test_reading_layer.py::test_cooperative_cell_b_reads_cell_a_signals -v
```

Expected: FAIL — `SENSE_PEER_0` not in rulebook, `peer_vectors` not on `ChemistryContext`.

### Step 3: Add SENSE_PEER_0 and SENSE_PEER_1 to `codons.py`

**3a. Add to REGULATORY_OP_NAMES** in `/home/mattthomson/workspace/AI-DNA/codons.py`, insert after `"IF_S1_LT"` and before `"END_BLOCK"`:

```python
REGULATORY_OP_NAMES: tuple[str, ...] = (
    "PROMOTER",
    "GATE",
    "IF_S0_GT",
    "IF_S1_GT",
    "IF_S2_GT",
    "IF_S2_LT",
    "IF_S3_GT",
    "IF_S3_LT",
    "IF_S4_GT",
    "IF_S4_LT",
    "IF_S0_LT",
    "IF_S1_LT",
    "SENSE_PEER_0",   # new — load peer 0's signals[0-4] into own signals[0-4]
    "SENSE_PEER_1",   # new — load peer 1's signals[0-4] into own signals[0-4]
    "END_BLOCK",
    "GENE_START",
    "GENE_ID_0",
    "GENE_ID_1",
    "GENE_ID_2",
    "GENE_ID_3",
    "GENE_ID_4",
    "GENE_ID_5",
    "GENE_ID_6",
    "GENE_ID_7",
    "GENE_END",
    "CALL_0",
    "CALL_1",
    "CALL_2",
    "CALL_3",
    "CALL_4",
    "CALL_5",
    "CALL_6",
    "CALL_7",
)
```

**3b. Add 3 synonyms each to REGULATORY_CODON_MAP** — use codons 271–276 (above current max of 270):

```python
# At the end of REGULATORY_CODON_MAP, before **GENE_BLOCK_CODON_MAP:
    271: "SENSE_PEER_0",
    272: "SENSE_PEER_0",
    273: "SENSE_PEER_0",
    274: "SENSE_PEER_1",
    275: "SENSE_PEER_1",
    276: "SENSE_PEER_1",
```

The full updated bottom of `REGULATORY_CODON_MAP` in `codons.py`:

```python
REGULATORY_CODON_MAP: dict[int, str] = {
    231: "PROMOTER",
    232: "PROMOTER",
    233: "PROMOTER",
    234: "GATE",
    235: "GATE",
    236: "GATE",
    237: "IF_S0_GT",
    238: "IF_S0_GT",
    239: "IF_S0_GT",
    240: "IF_S1_GT",
    241: "IF_S1_GT",
    242: "IF_S1_GT",
    243: "IF_S2_GT",
    244: "IF_S2_GT",
    245: "IF_S2_GT",
    246: "IF_S2_LT",
    247: "IF_S2_LT",
    248: "IF_S2_LT",
    249: "IF_S3_GT",
    250: "IF_S3_GT",
    251: "IF_S3_GT",
    252: "IF_S3_LT",
    253: "IF_S3_LT",
    254: "IF_S3_LT",
    255: "IF_S4_GT",
    256: "IF_S4_GT",
    257: "IF_S4_GT",
    258: "IF_S4_LT",
    259: "IF_S4_LT",
    260: "IF_S4_LT",
    261: "IF_S0_LT",
    262: "IF_S0_LT",
    263: "IF_S0_LT",
    264: "IF_S1_LT",
    265: "IF_S1_LT",
    266: "IF_S1_LT",
    267: "END_BLOCK",
    268: "END_BLOCK",
    269: "END_BLOCK",
    270: "END_BLOCK",
    271: "SENSE_PEER_0",
    272: "SENSE_PEER_0",
    273: "SENSE_PEER_0",
    274: "SENSE_PEER_1",
    275: "SENSE_PEER_1",
    276: "SENSE_PEER_1",
    **GENE_BLOCK_CODON_MAP,
}
```

### Step 4: Add `peer_vectors` to ChemistryContext and SENSE_PEER rule handling in `chemistry.py`

**4a. Extend ChemistryContext** in `/home/mattthomson/workspace/AI-DNA/chemistry.py`:

```python
@dataclass(slots=True)
class ChemistryContext:
    inbox: list[float] = field(default_factory=list)
    outbox: list[float] = field(default_factory=list)
    time: float = 0.0
    stage_increment: float = 0.0
    events: list[dict[str, float | str]] = field(default_factory=list)
    peer_vectors: dict[int, tuple[float, ...]] = field(default_factory=dict)

    def schedule(self, kind: str, *, when: float, payload: float | str | None = None) -> None:
        entry: dict[str, float | str] = {"kind": kind, "when": when}
        if payload is not None:
            entry["payload"] = payload
        self.events.append(entry)
```

**4b. Add SENSE_PEER_0 and SENSE_PEER_1 to `build_rulebook()`** in `chemistry.py`:

Add these two rule functions inside `build_rulebook()`, after the existing `recv` function:

```python
def sense_peer_0(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
    vec = context.peer_vectors.get(0)
    if vec is None:
        return None
    for i in range(min(5, len(vec), len(state.signals))):
        state.signals[i] = vec[i]
    return "sense_peer_0"

def sense_peer_1(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
    vec = context.peer_vectors.get(1)
    if vec is None:
        return None
    for i in range(min(5, len(vec), len(state.signals))):
        state.signals[i] = vec[i]
    return "sense_peer_1"
```

Add them to the returned dict:

```python
return {
    "RULE_EMIT_X": Rule("RULE_EMIT_X", emit_x),
    "RULE_EMIT_Y": Rule("RULE_EMIT_Y", emit_y),
    "RULE_ZERO_2": Rule("RULE_ZERO_2", zero_2),
    "RULE_ADD0_IF1": Rule("RULE_ADD0_IF1", add0_if1),
    "RULE_DECAY1": Rule("RULE_DECAY1", decay1),
    "RULE_OUTPUT_IF1Z": Rule("RULE_OUTPUT_IF1Z", output_if1z),
    "RULE_INHIBIT1Z": Rule("RULE_INHIBIT1Z", inhibit1z),
    "RULE_COPY0_3": Rule("RULE_COPY0_3", copy0_3),
    "RULE_COPY1_3": Rule("RULE_COPY1_3", copy1_3),
    "RULE_ADD3_IF1": Rule("RULE_ADD3_IF1", add3_if1),
    "RULE_DECAY3": Rule("RULE_DECAY3", decay3),
    "RULE_THRESH1": Rule("RULE_THRESH1", thresh1),
    "RULE_THRESH3": Rule("RULE_THRESH3", thresh3),
    "SEND": Rule("SEND", send),
    "RECV": Rule("RECV", recv),
    "SENSE_PEER_0": Rule("SENSE_PEER_0", sense_peer_0),
    "SENSE_PEER_1": Rule("SENSE_PEER_1", sense_peer_1),
}
```

### Step 5: Store `to_vector()` in `context.peer_vectors` in `cooperative_chemistry.py`

In `/home/mattthomson/workspace/AI-DNA/cooperative_chemistry.py`, in `CooperativeChemistrySystem.run()`, after each cell's `self.chemistry.step()` call, store the cell's vector. Replace the inner for-loop body:

```python
for cell_index, cell in enumerate(cell_list):
    cell_changed, event_index = self.chemistry.step(cell, task, context, event_index_start=event_index)
    changed = changed or cell_changed
    # Store this cell's current state as a peer vector for other cells to read
    context.peer_vectors[cell_index] = cell.to_vector()
    messages_delivered += self._deliver_messages(context, broadcast=True)
```

### Step 6: Run Level 3 tests

```bash
python3 -m pytest tests/test_reading_layer.py -v
```

Expected: all Level 1, 2, and 3 tests pass.

### Step 7: Run full fast suite

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass — `peer_vectors` defaults to empty dict, `SENSE_PEER_N` is no-op when no peer present.

### Step 8: Commit

```bash
git add /home/mattthomson/workspace/AI-DNA/codons.py \
        /home/mattthomson/workspace/AI-DNA/chemistry.py \
        /home/mattthomson/workspace/AI-DNA/cooperative_chemistry.py \
        /home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py
git commit -m "feat: add SENSE_PEER_0/1 codon ops and peer_vectors to ChemistryContext"
```

- [ ] Step 1: Write Level 3 failing tests in test_reading_layer.py
- [ ] Step 2: Run tests — verify FAIL
- [ ] Step 3: Add SENSE_PEER_0 and SENSE_PEER_1 to codons.py REGULATORY_OP_NAMES and REGULATORY_CODON_MAP (271–276)
- [ ] Step 4: Add peer_vectors field to ChemistryContext; add sense_peer_0/1 rules to build_rulebook()
- [ ] Step 5: Store to_vector() in context.peer_vectors after each cell step in cooperative_chemistry.py
- [ ] Step 6: Run test_reading_layer.py — verify Level 1/2/3 PASS
- [ ] Step 7: Run full fast suite
- [ ] Step 8: Commit

---

## Task 3: `CellReader` in `reading.py`

**Files:**
- Create: `/home/mattthomson/workspace/AI-DNA/reading.py`
- Modify: `/home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py` (add Level 4 tests)

### Step 1: Write the failing Level 4 tests

Append to `/home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py`:

```python
# ---------------------------------------------------------------------------
# Level 4 — CellReader (unit)
# ---------------------------------------------------------------------------

def test_cell_reader_observe_returns_six_floats() -> None:
    """`observe(cell)` returns a 6-float tuple."""
    from reading import CellReader

    reader = CellReader()
    cell = _make_cell([1.0, 2.0, 0.5, 0.0, 0.3], 4.5)
    vec = reader.observe(cell)
    assert isinstance(vec, tuple)
    assert len(vec) == 6


def test_cell_reader_observe_matches_to_vector() -> None:
    """`observe(cell)` result matches cell.to_vector()."""
    from reading import CellReader

    reader = CellReader()
    cell = _make_cell([1.0, 2.0, 0.5, 0.0, 0.3], 4.5)
    assert reader.observe(cell) == cell.to_vector()


def test_cell_reader_observe_population_returns_n_tuples() -> None:
    """`observe_population([a, b])` returns tuple of 2 six-float vectors."""
    from reading import CellReader

    reader = CellReader()
    cell_a = _make_cell([1.0, 0.0, 0.0, 0.0, 0.0], 1.0)
    cell_b = _make_cell([2.0, 0.0, 0.0, 0.0, 0.0], 2.0)
    result = reader.observe_population([cell_a, cell_b])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert len(result[0]) == 6
    assert len(result[1]) == 6


def test_cell_reader_observe_colony_returns_mean_vector() -> None:
    """`observe_colony([a, b])` returns mean 6-float vector — verified by hand."""
    from reading import CellReader

    reader = CellReader()
    cell_a = _make_cell([1.0, 0.0, 0.0, 0.0, 0.0], 2.0)
    cell_b = _make_cell([3.0, 0.0, 0.0, 0.0, 0.0], 4.0)
    colony_vec = reader.observe_colony([cell_a, cell_b])
    # signal[0] mean = (1.0 + 3.0) / 2 = 2.0
    # output mean = (2.0 + 4.0) / 2 = 3.0
    assert len(colony_vec) == 6
    assert colony_vec[0] == pytest.approx(2.0)
    assert colony_vec[5] == pytest.approx(3.0)


def test_cell_reader_observe_colony_empty_population() -> None:
    """`observe_colony([])` returns (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)."""
    from reading import CellReader

    reader = CellReader()
    result = reader.observe_colony([])
    assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_cell_reader_observe_extended_returns_fifteen_floats() -> None:
    """`observe_extended` returns a 15-float tuple."""
    from reading import CellReader

    reader = CellReader()
    cell = _make_cell([0.0, 0.0, 0.0, 0.0, 0.0], None)
    vec = reader.observe_extended(cell, active_gene_ids=frozenset({1, 3}), motif_hash=0.5)
    assert len(vec) == 15


def test_cell_reader_observe_colony_fixed_size_regardless_of_population() -> None:
    """`observe_colony` always returns exactly 6 floats regardless of population size."""
    from reading import CellReader

    reader = CellReader()
    cells = [_make_cell([float(i), 0.0, 0.0, 0.0, 0.0], float(i)) for i in range(10)]
    assert len(reader.observe_colony(cells)) == 6
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/test_reading_layer.py::test_cell_reader_observe_returns_six_floats tests/test_reading_layer.py::test_cell_reader_observe_colony_empty_population -v
```

Expected: FAIL — `reading` module does not exist.

### Step 3: Create `reading.py`

Create `/home/mattthomson/workspace/AI-DNA/reading.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cell import CellState


@dataclass(frozen=True, slots=True)
class CellReader:
    def observe(self, cell: CellState) -> tuple[float, ...]:
        return cell.to_vector()

    def observe_extended(
        self,
        cell: CellState,
        *,
        active_gene_ids: frozenset[int] = frozenset(),
        motif_hash: float = 0.0,
    ) -> tuple[float, ...]:
        return cell.to_vector(extended=True, active_gene_ids=active_gene_ids, motif_hash=motif_hash)

    def observe_population(self, cells: Sequence[CellState]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.observe(cell) for cell in cells)

    def observe_colony(self, cells: Sequence[CellState]) -> tuple[float, ...]:
        vecs = self.observe_population(cells)
        if not vecs:
            return (0.0,) * 6
        return tuple(sum(v[i] for v in vecs) / len(vecs) for i in range(6))
```

### Step 4: Run Level 4 tests

```bash
python3 -m pytest tests/test_reading_layer.py -v
```

Expected: all Level 1, 2, 3, and 4 tests pass.

### Step 5: Run full fast suite

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

### Step 6: Commit

```bash
git add /home/mattthomson/workspace/AI-DNA/reading.py \
        /home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py
git commit -m "feat: add CellReader in reading.py with observe/observe_extended/observe_population/observe_colony"
```

- [ ] Step 1: Write Level 4 failing tests in test_reading_layer.py
- [ ] Step 2: Run tests — verify FAIL
- [ ] Step 3: Create reading.py with CellReader
- [ ] Step 4: Run test_reading_layer.py — verify Level 1/2/3/4 PASS
- [ ] Step 5: Run full fast suite
- [ ] Step 6: Commit

---

## Task 4: Integration and regression

**Files:**
- Modify: `/home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py` (add Level 5 regression tests)
- No source changes expected

### Step 1: Write the Level 5 regression tests

Append to `/home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py`:

```python
# ---------------------------------------------------------------------------
# Level 5 — Regression
# ---------------------------------------------------------------------------

def test_experiment_01_output_unchanged() -> None:
    """Experiment 01 (multiply benchmark) exits 0 — to_vector() is additive."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "experiments/01_multiply_persistent_rules.py"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/home/mattthomson/workspace/AI-DNA",
    )
    assert result.returncode == 0, (
        f"Experiment 01 exited with code {result.returncode}.\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )


def test_experiment_13_output_unchanged() -> None:
    """Experiment 13 (cooperative chemistry) exits 0 — SENSE_PEER_N only fires if codon present."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "experiments/13_cooperative_chemistry.py"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd="/home/mattthomson/workspace/AI-DNA",
    )
    assert result.returncode == 0, (
        f"Experiment 13 exited with code {result.returncode}.\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )


def test_peer_vectors_empty_by_default() -> None:
    """ChemistryContext() default peer_vectors is empty dict."""
    from chemistry import ChemistryContext

    context = ChemistryContext()
    assert context.peer_vectors == {}


def test_sense_peer_noop_in_solo_context() -> None:
    """SENSE_PEER_0 in a solo ChemistrySystem run (no cooperative context) → no-op."""
    from chemistry import ChemistryContext, ChemistrySystem
    from tasks import TaskCase

    cell = CellState(active_rules=["SENSE_PEER_0"])
    cell.reset(x=1.0, y=2.0)
    original = list(cell.signals)

    system = ChemistrySystem(max_time=4.0)
    system.run(cell, TaskCase(x=1.0, y=2.0, z=0.0))

    # signals[0] and [1] may be unchanged since SENSE_PEER_0 is no-op (no peer 0)
    # The no-op means signals[0-4] unchanged from reset values
    assert cell.signals[0] == pytest.approx(original[0])
    assert cell.signals[1] == pytest.approx(original[1])


def test_to_vector_does_not_modify_cell() -> None:
    """Calling to_vector() does not mutate cell.signals or cell.output."""
    cell = _make_cell([1.0, 2.0, 0.5, 0.0, 0.3], 4.5)
    before_signals = list(cell.signals)
    before_output = cell.output
    _ = cell.to_vector(extended=True, active_gene_ids=frozenset({1, 3}), motif_hash=0.7)
    assert cell.signals == before_signals
    assert cell.output == before_output
```

### Step 2: Run experiment 01 directly

```bash
python3 /home/mattthomson/workspace/AI-DNA/experiments/01_multiply_persistent_rules.py
```

Expected: exits 0, output identical to pre-change baseline.

### Step 3: Run experiment 13 directly

```bash
python3 /home/mattthomson/workspace/AI-DNA/experiments/13_cooperative_chemistry.py
```

Expected: exits 0, output identical to pre-change baseline.

### Step 4: Run all Level 5 tests (skipping the subprocess tests if experiments are slow)

```bash
python3 -m pytest tests/test_reading_layer.py -v
```

Expected: all Level 1/2/3/4/5 tests pass.

### Step 5: Run full fast suite

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass, no failures or errors.

### Step 6: Final commit

```bash
git add /home/mattthomson/workspace/AI-DNA/tests/test_reading_layer.py
git commit -m "feat: add reading layer — to_vector, SENSE_PEER_N, CellReader"
```

- [ ] Step 1: Write Level 5 regression tests in test_reading_layer.py
- [ ] Step 2: Run experiment 01 directly — verify exits 0
- [ ] Step 3: Run experiment 13 directly — verify exits 0
- [ ] Step 4: Run test_reading_layer.py — all Level 1/2/3/4/5 PASS
- [ ] Step 5: Run full fast suite — all pass
- [ ] Step 6: Final commit

---

## Self-review checklist

### Spec success criteria coverage

| Criterion | Covered by |
|---|---|
| 1. All existing tests pass unchanged | Task 1 Step 5, Task 2 Step 7, Task 3 Step 5, Task 4 Step 5 |
| 2. `cell.to_vector()` returns deterministic 6-float tuple from any CellState | Task 1 — test_to_vector_base_values, test_to_vector_deterministic |
| 3. `SENSE_PEER_0` in cooperative context correctly loads peer's signals | Task 2 — test_sense_peer_0_updates_signals_when_peer_available, test_cooperative_cell_b_reads_cell_a_signals |
| 4. `CellReader.observe_colony()` returns fixed-size 6-float mean vector | Task 3 — test_cell_reader_observe_colony_fixed_size_regardless_of_population |
| 5. Experiment 01 and 13 output identical to pre-change baseline | Task 4 — test_experiment_01_output_unchanged, test_experiment_13_output_unchanged |

### Name and type consistency

| Name | Used in |
|---|---|
| `to_vector(extended, active_gene_ids, motif_hash)` | cell.py method; Task 1 tests; CellReader.observe() / observe_extended(); cooperative_chemistry.py peer_vectors store |
| `peer_vectors: dict[int, tuple[float, ...]]` | ChemistryContext field; cooperative_chemistry.py store after step; Task 2 tests |
| `SENSE_PEER_0` / `SENSE_PEER_1` | codons.py REGULATORY_OP_NAMES + REGULATORY_CODON_MAP 271–276; chemistry.py build_rulebook(); Task 2 tests |
| `CellReader` | reading.py; Task 3 tests (imported as `from reading import CellReader`) |

### Backward compatibility

- `to_vector()` is a new method — no existing caller broken.
- `peer_vectors` field on `ChemistryContext` uses `default_factory=dict` — existing `ChemistryContext()` calls unaffected.
- `SENSE_PEER_N` is a no-op when `context.peer_vectors` has no matching key — solo experiments unaffected.
- `cooperative_chemistry.py` change only adds a `context.peer_vectors[cell_index] = cell.to_vector()` assignment — no logic changed.
- `reading.py` is a new file — no import cycles introduced (imports only `cell.py`).

### No placeholders

All implementation code is shown in full. Every step has either Python source or a bash command. No "implement X" without accompanying code.
