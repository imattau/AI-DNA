# Temporal Unfolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add signal[4] as a dedicated stage accumulator incremented each chemistry round, enabling genomes to encode developmental programs with distinct early/late gene expression phases.
**Architecture:** Extend CellState.signals to 5 slots; add stage_increment to ChemistryContext; increment signal[4] each round before rules fire; add IF_S3_GT/LT/IF_S4_GT/LT to codon table and _condition_passes(). Backward compatible — stage_increment=0.0 disables feature.
**Tech Stack:** Python 3.10+, dataclasses, pytest

---

## File Map

- Modify: `cell.py` — extend signals to 5 slots; update reset() and default_factory
- Modify: `codons.py` — add IF_S3_GT, IF_S3_LT, IF_S4_GT, IF_S4_LT to REGULATORY_CODON_MAP and REGULATORY_OP_NAMES
- Modify: `chemistry.py` — add stage_increment to ChemistryContext; increment signal[4] in run(); add 4 new conditions to _condition_passes(); update _record() type hint
- Create: `tests/test_temporal_unfolding.py` — all Level 1/2/3/4 tests

---

## Task 1: Signal slot extension and new codon ops

**Files:**
- Modify: `cell.py`
- Modify: `codons.py`
- Modify: `tests/test_codons.py`

### Step 1: Write the failing tests

Add to `tests/test_codons.py`:

```python
def test_temporal_op_names_present_in_table() -> None:
    """IF_S3_GT, IF_S3_LT, IF_S4_GT, IF_S4_LT appear in the default codon table."""
    from codons import default_codon_table

    table = default_codon_table()
    temporal_ops = {"IF_S3_GT", "IF_S3_LT", "IF_S4_GT", "IF_S4_LT"}
    present = set(table.codon_to_op.values())
    for op in temporal_ops:
        assert op in present, f"{op} missing from codon table"


def test_temporal_ops_have_multiple_synonyms() -> None:
    """Each new temporal op-name has at least 3 synonym codon values."""
    from codons import default_codon_table

    table = default_codon_table()
    for op in ("IF_S3_GT", "IF_S3_LT", "IF_S4_GT", "IF_S4_LT"):
        count = len(table.op_to_codons.get(op, []))
        assert count >= 3, f"{op} has only {count} synonyms, expected >= 3"


def test_temporal_round_trip_encode_decode() -> None:
    """Encoding then decoding IF_S4_GT and IF_S4_LT returns same op-names."""
    from codons import default_codon_table

    table = default_codon_table()
    ops = ["PROMOTER", "IF_S4_GT", "RULE_EMIT_X", "END_BLOCK",
           "GATE", "IF_S3_LT", "RULE_EMIT_Y", "END_BLOCK"]
    encoded = table.encode_ops(ops)
    decoded = table.decode(encoded)
    assert decoded == ops


def test_cell_state_has_five_signal_slots() -> None:
    """CellState() default has 5 signal slots, all 0.0."""
    from cell import CellState

    cell = CellState(active_rules=[])
    assert len(cell.signals) == 5, f"Expected 5 slots, got {len(cell.signals)}"
    assert all(s == 0.0 for s in cell.signals)


def test_cell_reset_initialises_five_slots() -> None:
    """reset(x=2, y=3) gives signals=[2, 3, 0, 0, 0]."""
    from cell import CellState

    cell = CellState(active_rules=[])
    cell.reset(x=2.0, y=3.0)
    assert cell.signals == [2.0, 3.0, 0.0, 0.0, 0.0], f"Got {cell.signals}"
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/test_codons.py::test_temporal_op_names_present_in_table tests/test_codons.py::test_temporal_ops_have_multiple_synonyms tests/test_codons.py::test_temporal_round_trip_encode_decode tests/test_codons.py::test_cell_state_has_five_signal_slots tests/test_codons.py::test_cell_reset_initialises_five_slots -v
```

Expected: FAIL — new ops not in table, signals has 4 slots.

### Step 3: Extend CellState.signals to 5 slots in `cell.py`

In `/home/mattthomson/workspace/AI-DNA/cell.py`, replace lines 10 and 15:

```python
# Before line 10:
signals: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

# After:
signals: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
```

```python
# Before line 15:
self.signals = [float(x), float(y), 0.0, 0.0]

# After:
self.signals = [float(x), float(y), 0.0, 0.0, 0.0]
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

    @property
    def cell_count(self) -> int:
        return 1
```

### Step 4: Add 4 new op-names to `codons.py`

In `/home/mattthomson/workspace/AI-DNA/codons.py`:

**4a. Add to REGULATORY_OP_NAMES tuple** (after `"IF_S1_LT"`, before `"END_BLOCK"`):

```python
REGULATORY_OP_NAMES: tuple[str, ...] = (
    "PROMOTER",
    "GATE",
    "IF_S0_GT",
    "IF_S1_GT",
    "IF_S2_GT",
    "IF_S0_LT",
    "IF_S1_LT",
    "IF_S2_LT",   # add if missing — keep alphabetical signal order
    "IF_S3_GT",   # new
    "IF_S3_LT",   # new
    "IF_S4_GT",   # new — stage gate
    "IF_S4_LT",   # new — early-stage gate
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

**4b. Add 4 new entries to REGULATORY_CODON_MAP** (extending from codon 256 onward, 3 synonyms each):

The existing map ends at 255. Add entries at 256–267 for 4 ops × 3 synonyms:

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
    246: "IF_S0_LT",
    247: "IF_S0_LT",
    248: "IF_S0_LT",
    249: "IF_S1_LT",
    250: "IF_S1_LT",
    251: "IF_S1_LT",
    252: "END_BLOCK",
    253: "END_BLOCK",
    254: "END_BLOCK",
    255: "END_BLOCK",
    # Temporal unfolding — signal[3] and signal[4] conditions
    256: "IF_S3_GT",
    257: "IF_S3_GT",
    258: "IF_S3_GT",
    259: "IF_S3_LT",
    260: "IF_S3_LT",
    261: "IF_S3_LT",
    262: "IF_S4_GT",
    263: "IF_S4_GT",
    264: "IF_S4_GT",
    265: "IF_S4_LT",
    266: "IF_S4_LT",
    267: "IF_S4_LT",
    **GENE_BLOCK_CODON_MAP,
}
```

Note: codon values 256–267 are above the 256-modulus boundary. The `default_codon_table()` function uses `codon_to_op.update(REGULATORY_CODON_MAP)` — these keys are stored as-is and looked up directly. The `encode_ops` / `random_codon` path returns one of these values; callers that use `codon % 256` to index the table will fold them back. To keep lookup clean, verify `default_codon_table()` returns a dict keyed by the raw integer. Since `table.encode_ops(ops)` calls `random_codon(op)` which returns the raw value from `op_to_codons`, and `table.op(codon)` looks up `codon_to_op[codon]` directly, values above 255 work correctly in the dict.

If the codebase ever applies `codon % 256` before lookup (check `genome.py` `declare_rules`), the new codons must be within 0–255. In that case, reuse codon values 200–211 (currently NOOP-filled by `build_codon_table` with `table_size=231`, so 200–230 are NOOP). Safer: place the 4 new ops in the 200–211 range inside REGULATORY_CODON_MAP:

```python
    # Temporal unfolding — placed in 200-211 (previously NOOP-filled)
    200: "IF_S3_GT",
    201: "IF_S3_GT",
    202: "IF_S3_GT",
    203: "IF_S3_LT",
    204: "IF_S3_LT",
    205: "IF_S3_LT",
    206: "IF_S4_GT",
    207: "IF_S4_GT",
    208: "IF_S4_GT",
    209: "IF_S4_LT",
    210: "IF_S4_LT",
    211: "IF_S4_LT",
```

**Use the 200–211 range** — it is safe because `build_codon_table(allocation, table_size=231)` fills 0–230 with rule ops and NOOPs, and `REGULATORY_CODON_MAP.update()` overwrites whichever positions it specifies. Codons 200–211 were previously NOOP; overwriting is safe.

### Step 5: Run tests

```bash
python3 -m pytest tests/test_codons.py -v
```

Expected: all pass including the 5 new tests.

### Step 6: Run full fast suite to check no regressions

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass. If any existing test asserts `len(cell.signals) == 4` or `cell.signals == [x, y, 0.0, 0.0]`, update it to 5 slots now.

### Step 7: Commit

```bash
git add cell.py codons.py tests/test_codons.py
git commit -m "feat: extend signals to 5 slots and add IF_S3_GT/LT IF_S4_GT/LT to codon table"
```

- [ ] Step 1: Write failing tests in test_codons.py
- [ ] Step 2: Run tests — verify FAIL
- [ ] Step 3: Update cell.py signals to 5 slots
- [ ] Step 4: Add 4 new op-names to codons.py REGULATORY_CODON_MAP and REGULATORY_OP_NAMES
- [ ] Step 5: Run test_codons.py — verify PASS
- [ ] Step 6: Run full fast suite — fix any 4-slot references
- [ ] Step 7: Commit

---

## Task 2: ChemistryContext stage_increment + VM increment

**Files:**
- Modify: `chemistry.py`

### Step 1: Write the failing tests

Create `/home/mattthomson/workspace/AI-DNA/tests/test_temporal_unfolding.py`:

```python
from __future__ import annotations

import pytest
from cell import CellState
from chemistry import ChemistryContext, ChemistrySystem
from tasks import TaskCase


# ---------------------------------------------------------------------------
# Level 1 — Signal slot extension (unit)
# ---------------------------------------------------------------------------

def test_cellstate_default_has_five_slots() -> None:
    """CellState() default signals has 5 slots, all 0.0."""
    cell = CellState(active_rules=[])
    assert len(cell.signals) == 5
    assert cell.signals[4] == 0.0


def test_cellstate_reset_five_slots() -> None:
    """reset(x=2, y=3) gives signals=[2, 3, 0, 0, 0]."""
    cell = CellState(active_rules=[])
    cell.reset(x=2.0, y=3.0)
    assert cell.signals == [2.0, 3.0, 0.0, 0.0, 0.0]


def test_cellstate_signal4_starts_zero_after_reset() -> None:
    """signal[4] always starts at 0.0 after reset regardless of prior value."""
    cell = CellState(active_rules=[])
    cell.signals[4] = 0.9
    cell.reset(x=1.0, y=2.0)
    assert cell.signals[4] == 0.0


# ---------------------------------------------------------------------------
# Level 2 — VM stage increment (unit)
# ---------------------------------------------------------------------------

def test_stage_increment_zero_leaves_signal4_unchanged() -> None:
    """stage_increment=0.0 keeps signal[4] at 0.0 throughout all rounds."""
    cell = CellState(active_rules=[])
    cell.reset(x=0.0, y=0.0)
    task = TaskCase(x=0.0, y=0.0, z=0.0)
    context = ChemistryContext(stage_increment=0.0)
    system = ChemistrySystem(max_time=10.0)
    system.run(cell, task, context=context)
    assert cell.signals[4] == 0.0, f"Expected 0.0, got {cell.signals[4]}"


def test_stage_increment_point1_after_one_round() -> None:
    """With stage_increment=0.1, signal[4]=0.1 after exactly 1 round."""
    cell = CellState(active_rules=[])
    cell.reset(x=0.0, y=0.0)
    task = TaskCase(x=0.0, y=0.0, z=0.0)
    context = ChemistryContext(stage_increment=0.1)
    system = ChemistrySystem(max_time=1.0, dt=1.0)
    # Run exactly 1 step
    system.step(cell, task, context)
    assert abs(cell.signals[4] - 0.1) < 1e-9, f"Expected 0.1 after 1 round, got {cell.signals[4]}"


def test_stage_increment_point1_after_five_rounds() -> None:
    """With stage_increment=0.1, signal[4]=0.5 after exactly 5 rounds."""
    cell = CellState(active_rules=[])
    cell.reset(x=0.0, y=0.0)
    task = TaskCase(x=0.0, y=0.0, z=0.0)
    context = ChemistryContext(stage_increment=0.1)
    system = ChemistrySystem(max_time=32.0, dt=1.0)
    for _ in range(5):
        system.step(cell, task, context)
        context.time += system.dt
    assert abs(cell.signals[4] - 0.5) < 1e-9, f"Expected 0.5 after 5 rounds, got {cell.signals[4]}"


def test_stage_increment_clamps_at_one() -> None:
    """signal[4] saturates at 1.0 after 10 rounds with stage_increment=0.1 (not 1.1)."""
    cell = CellState(active_rules=[])
    cell.reset(x=0.0, y=0.0)
    task = TaskCase(x=0.0, y=0.0, z=0.0)
    context = ChemistryContext(stage_increment=0.1)
    system = ChemistrySystem(max_time=32.0, dt=1.0)
    for _ in range(12):  # 12 rounds — should clamp, not overflow
        system.step(cell, task, context)
        context.time += system.dt
    assert cell.signals[4] == 1.0, f"Expected 1.0 (clamped), got {cell.signals[4]}"


def test_episode_reset_clears_signal4() -> None:
    """cell.reset() sets signal[4]=0.0 even after it has been incremented."""
    cell = CellState(active_rules=[])
    cell.reset(x=1.0, y=2.0)
    cell.signals[4] = 0.7  # simulate mid-episode accumulation
    cell.reset(x=3.0, y=4.0)
    assert cell.signals[4] == 0.0


def test_if_s4_gt_true_when_signal4_high() -> None:
    """_condition_passes('IF_S4_GT', cell) returns True when signal[4]=0.8."""
    from chemistry import _condition_passes

    cell = CellState(active_rules=[])
    cell.signals[4] = 0.8
    assert _condition_passes("IF_S4_GT", cell) is True


def test_if_s4_gt_false_when_signal4_low() -> None:
    """_condition_passes('IF_S4_GT', cell) returns False when signal[4]=0.2."""
    from chemistry import _condition_passes

    cell = CellState(active_rules=[])
    cell.signals[4] = 0.2
    assert _condition_passes("IF_S4_GT", cell) is False


def test_if_s4_lt_true_when_signal4_low() -> None:
    """_condition_passes('IF_S4_LT', cell) returns True when signal[4]=0.2."""
    from chemistry import _condition_passes

    cell = CellState(active_rules=[])
    cell.signals[4] = 0.2
    assert _condition_passes("IF_S4_LT", cell) is True


def test_if_s4_lt_false_when_signal4_high() -> None:
    """_condition_passes('IF_S4_LT', cell) returns False when signal[4]=0.8."""
    from chemistry import _condition_passes

    cell = CellState(active_rules=[])
    cell.signals[4] = 0.8
    assert _condition_passes("IF_S4_LT", cell) is False


def test_if_s3_gt_and_lt_work() -> None:
    """_condition_passes handles IF_S3_GT and IF_S3_LT correctly."""
    from chemistry import _condition_passes

    cell = CellState(active_rules=[])
    cell.signals[3] = 0.8
    assert _condition_passes("IF_S3_GT", cell) is True
    assert _condition_passes("IF_S3_LT", cell) is False

    cell.signals[3] = 0.2
    assert _condition_passes("IF_S3_GT", cell) is False
    assert _condition_passes("IF_S3_LT", cell) is True
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/test_temporal_unfolding.py -v
```

Expected: FAIL — `ChemistryContext` has no `stage_increment`, `_condition_passes` missing new conditions.

### Step 3: Add stage_increment to ChemistryContext in `chemistry.py`

In `/home/mattthomson/workspace/AI-DNA/chemistry.py`, extend the `ChemistryContext` dataclass:

```python
@dataclass(slots=True)
class ChemistryContext:
    inbox: list[float] = field(default_factory=list)
    outbox: list[float] = field(default_factory=list)
    time: float = 0.0
    events: list[dict[str, float | str]] = field(default_factory=list)
    stage_increment: float = 0.0  # 0.0 = disabled (backward compatible); 0.1 = default rate

    def schedule(self, kind: str, *, when: float, payload: float | str | None = None) -> None:
        entry: dict[str, float | str] = {"kind": kind, "when": when}
        if payload is not None:
            entry["payload"] = payload
        self.events.append(entry)
```

### Step 4: Add 4 new conditions to `_condition_passes()` in `chemistry.py`

Replace the existing `_condition_passes` function (lines 174–185):

```python
def _condition_passes(condition: str, cell: CellState) -> bool:
    if condition == "IF_S0_GT":
        return cell.signals[0] > 0.5
    if condition == "IF_S1_GT":
        return cell.signals[1] > 0.5
    if condition == "IF_S2_GT":
        return cell.signals[2] > 0.5
    if condition == "IF_S0_LT":
        return cell.signals[0] < 0.5
    if condition == "IF_S1_LT":
        return cell.signals[1] < 0.5
    if condition == "IF_S3_GT":
        return cell.signals[3] > 0.5
    if condition == "IF_S3_LT":
        return cell.signals[3] < 0.5
    if condition == "IF_S4_GT":
        return cell.signals[4] > 0.5
    if condition == "IF_S4_LT":
        return cell.signals[4] < 0.5
    return True
```

### Step 5: Increment signal[4] at start of each round in `ChemistrySystem.step()`

In `/home/mattthomson/workspace/AI-DNA/chemistry.py`, update the `step` method to increment signal[4] before any rules fire. Add the increment as the very first line of the step body (before the `execute` helper definition):

```python
def step(
    self,
    cell: CellState,
    task: TaskCase,
    context: ChemistryContext,
    *,
    event_index_start: int = 0,
) -> tuple[bool, int]:
    # Increment stage accumulator before rules fire (clamped at 1.0)
    if context.stage_increment > 0.0:
        cell.signals[4] = min(cell.signals[4] + context.stage_increment, 1.0)

    changed = False
    event_index = event_index_start

    def execute(rule_entry: str | GateRule) -> None:
        nonlocal changed, event_index
        if isinstance(rule_entry, GateRule):
            before = tuple(cell.signals)
            gate_name = f"GATE[{rule_entry.condition}]"
            if _condition_passes(rule_entry.condition, cell):
                _record(cell, event_index, gate_name, before, "open")
                event_index += 1
                for nested in rule_entry.rules:
                    execute(nested)
            else:
                _record(cell, event_index, gate_name, before, "closed")
                event_index += 1
            return

        rule = self.rulebook.get(rule_entry)
        if rule is None:
            return
        before = tuple(cell.signals)
        note = rule.apply(cell, task, context)
        after = tuple(cell.signals)
        if after != before or note:
            changed = True
            _record(cell, event_index, rule_entry, before, note or "")
            event_index += 1

    for rule_name in cell.active_rules:
        execute(rule_name)
    return changed, event_index
```

### Step 6: Update `_record()` type hint in `chemistry.py`

Replace the `before` parameter type hint from 4-tuple to 5-tuple:

```python
def _record(
    state: CellState,
    event_index: int,
    rule_name: str,
    before: tuple[float, float, float, float, float],
    note: str = "",
) -> None:
    state.trace.append(
        {
            "round": event_index,
            "rule": rule_name,
            "before": before,
            "after": tuple(state.signals),
            "note": note,
        }
    )
```

### Step 7: Run Level 2 tests

```bash
python3 -m pytest tests/test_temporal_unfolding.py -v
```

Expected: all Level 1 and Level 2 tests pass.

### Step 8: Run full fast suite

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

### Step 9: Commit

```bash
git add chemistry.py
git commit -m "feat: add stage_increment to ChemistryContext and increment signal[4] each round"
```

- [ ] Step 1: Write failing tests in test_temporal_unfolding.py (Level 1 + 2)
- [ ] Step 2: Run tests — verify FAIL
- [ ] Step 3: Add stage_increment to ChemistryContext
- [ ] Step 4: Add IF_S3_GT/LT, IF_S4_GT/LT to _condition_passes()
- [ ] Step 5: Add signal[4] increment at start of step()
- [ ] Step 6: Update _record() type hint
- [ ] Step 7: Run test_temporal_unfolding.py — verify PASS
- [ ] Step 8: Run full fast suite
- [ ] Step 9: Commit

---

## Task 3: Stage gating integration

**Files:**
- Modify: `tests/test_temporal_unfolding.py` — add Level 3 tests

### Step 1: Write the failing tests

Append to `/home/mattthomson/workspace/AI-DNA/tests/test_temporal_unfolding.py`:

```python
# ---------------------------------------------------------------------------
# Level 3 — Stage gating integration
# ---------------------------------------------------------------------------

def _genome_from_ops(ops: list[str]) -> "CellGenome":
    """Build a CellGenome whose codons encode the given op sequence."""
    from codons import default_codon_table
    from genome import CellGenome

    table = default_codon_table()
    codons = tuple(table.encode_ops(ops))
    return CellGenome(codons=codons, local_motifs=(), lineage_id="test")


def test_promoter_if_s4_gt_excludes_rule_when_signal4_low() -> None:
    """PROMOTER + IF_S4_GT: RULE_EMIT_X absent in active rules when signal[4]=0.2."""
    genome = _genome_from_ops([
        "PROMOTER", "IF_S4_GT",
        "RULE_EMIT_X",
        "END_BLOCK",
    ])
    signals = (0.0, 0.0, 0.0, 0.0, 0.2)  # signal[4]=0.2, below threshold
    rules = genome.declare_rules(signals=signals)
    rule_names = [r for r in rules if isinstance(r, str)]
    assert "RULE_EMIT_X" not in rule_names, (
        f"RULE_EMIT_X should be absent when signal[4]=0.2, got {rule_names}"
    )


def test_promoter_if_s4_gt_includes_rule_when_signal4_high() -> None:
    """PROMOTER + IF_S4_GT: RULE_EMIT_X present in active rules when signal[4]=0.8."""
    genome = _genome_from_ops([
        "PROMOTER", "IF_S4_GT",
        "RULE_EMIT_X",
        "END_BLOCK",
    ])
    signals = (0.0, 0.0, 0.0, 0.0, 0.8)  # signal[4]=0.8, above threshold
    rules = genome.declare_rules(signals=signals)
    rule_names = [r for r in rules if isinstance(r, str)]
    assert "RULE_EMIT_X" in rule_names, (
        f"RULE_EMIT_X should be present when signal[4]=0.8, got {rule_names}"
    )


def test_gate_if_s4_gt_does_not_fire_in_early_rounds() -> None:
    """GateRule(IF_S4_GT) — RULE_EMIT_X does not fire in rounds 1-4 (signal[4]<0.5)."""
    from genome import GateRule

    task = TaskCase(x=5.0, y=0.0, z=5.0)
    gate = GateRule(condition="IF_S4_GT", rules=("RULE_EMIT_X",))

    cell = CellState(active_rules=[gate])
    cell.reset(x=0.0, y=0.0)

    context = ChemistryContext(stage_increment=0.1)
    system = ChemistrySystem(max_time=32.0, dt=1.0)

    # Run exactly 4 rounds manually — signal[4] will be 0.4 after round 4
    for _ in range(4):
        system.step(cell, task, context)
        context.time += system.dt

    # signal[4]=0.4, gate should not have opened
    assert cell.signals[4] <= 0.4 + 1e-9
    assert cell.signals[0] == 0.0, (
        f"RULE_EMIT_X should not have fired (signal[4]<0.5), but signal[0]={cell.signals[0]}"
    )


def test_gate_if_s4_gt_fires_from_round_five() -> None:
    """GateRule(IF_S4_GT) — RULE_EMIT_X fires from round 5 onward (signal[4]>=0.5)."""
    from genome import GateRule

    task = TaskCase(x=5.0, y=0.0, z=5.0)
    gate = GateRule(condition="IF_S4_GT", rules=("RULE_EMIT_X",))

    cell = CellState(active_rules=[gate])
    cell.reset(x=0.0, y=0.0)

    context = ChemistryContext(stage_increment=0.1)
    system = ChemistrySystem(max_time=32.0, dt=1.0)

    # Run 5 rounds — signal[4] will be 0.5 after round 5
    for _ in range(5):
        system.step(cell, task, context)
        context.time += system.dt

    # signal[4]=0.5, gate should have opened on round 5
    assert cell.signals[4] >= 0.5 - 1e-9
    assert cell.signals[0] == task.x, (
        f"RULE_EMIT_X should have fired by round 5, but signal[0]={cell.signals[0]}"
    )


def test_two_phase_genome_different_active_rules_per_phase() -> None:
    """Two-phase genome: gene 0 always active, gene 1 gated by IF_S4_GT.

    Active rules differ between round 4 (signal[4]=0.4) and round 6 (signal[4]=0.6).
    """
    from genome import CellGenome, GateRule

    # Build genome manually (direct ops sequence using declare_rules with different signals)
    genome = _genome_from_ops([
        # Gene 0: unconditional — emit x input
        "RULE_EMIT_X",
        # Gene 1: late-stage only — PROMOTER gated by IF_S4_GT
        "PROMOTER", "IF_S4_GT",
        "RULE_EMIT_Y",
        "END_BLOCK",
    ])

    # Early phase: signal[4] = 0.4 (rounds 1-4)
    signals_early = (0.0, 0.0, 0.0, 0.0, 0.4)
    rules_early = genome.declare_rules(signals=signals_early)
    rule_names_early = [r for r in rules_early if isinstance(r, str)]

    # Late phase: signal[4] = 0.6 (rounds 5+)
    signals_late = (0.0, 0.0, 0.0, 0.0, 0.6)
    rules_late = genome.declare_rules(signals=signals_late)
    rule_names_late = [r for r in rules_late if isinstance(r, str)]

    # RULE_EMIT_X always present
    assert "RULE_EMIT_X" in rule_names_early, "RULE_EMIT_X should be in early rules"
    assert "RULE_EMIT_X" in rule_names_late, "RULE_EMIT_X should be in late rules"

    # RULE_EMIT_Y only in late phase
    assert "RULE_EMIT_Y" not in rule_names_early, (
        f"RULE_EMIT_Y should be absent in early phase, got {rule_names_early}"
    )
    assert "RULE_EMIT_Y" in rule_names_late, (
        f"RULE_EMIT_Y should be present in late phase, got {rule_names_late}"
    )
```

Note: `genome.declare_rules(signals=...)` currently accepts a 4-tuple. After Task 1 extends signals to 5 slots, update `genome.py`'s `_evaluate_condition` helper to handle signal index 4. See Step 3.

### Step 2: Update `genome.py` _evaluate_condition to handle signal[4]

In `/home/mattthomson/workspace/AI-DNA/genome.py`, extend `_evaluate_condition` and the `_CONDITION_OPS` frozenset to include the 4 new ops:

```python
_CONDITION_OPS: frozenset[str] = frozenset({
    "IF_S0_GT", "IF_S1_GT", "IF_S2_GT",
    "IF_S0_LT", "IF_S1_LT",
    "IF_S3_GT", "IF_S3_LT",   # new
    "IF_S4_GT", "IF_S4_LT",   # new
})


def _evaluate_condition(condition: str, signals: tuple[float, ...]) -> bool:
    """Return True if the condition op-name is satisfied by the current signals.

    Unknown condition op-names default to True (constitutive expression).
    Threshold is fixed at 0.5.
    """
    if condition == "IF_S0_GT":
        return signals[0] > 0.5 if len(signals) > 0 else True
    if condition == "IF_S1_GT":
        return signals[1] > 0.5 if len(signals) > 1 else True
    if condition == "IF_S2_GT":
        return signals[2] > 0.5 if len(signals) > 2 else True
    if condition == "IF_S0_LT":
        return signals[0] < 0.5 if len(signals) > 0 else True
    if condition == "IF_S1_LT":
        return signals[1] < 0.5 if len(signals) > 1 else True
    if condition == "IF_S3_GT":
        return signals[3] > 0.5 if len(signals) > 3 else True
    if condition == "IF_S3_LT":
        return signals[3] < 0.5 if len(signals) > 3 else True
    if condition == "IF_S4_GT":
        return signals[4] > 0.5 if len(signals) > 4 else True
    if condition == "IF_S4_LT":
        return signals[4] < 0.5 if len(signals) > 4 else True
    # Unknown condition → constitutive expression (always true)
    return True
```

Also check the `declare_rules` signature — if `signals` is typed as `tuple[float, float, float, float] | None`, update to `tuple[float, ...] | None`.

### Step 3: Run Level 3 tests

```bash
python3 -m pytest tests/test_temporal_unfolding.py -v
```

Expected: all Level 1, 2, 3 tests pass.

### Step 4: Run full fast suite

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

### Step 5: Commit

```bash
git add tests/test_temporal_unfolding.py genome.py
git commit -m "feat: add stage gating integration tests and extend genome _evaluate_condition for signal[4]"
```

- [ ] Step 1: Write Level 3 tests in test_temporal_unfolding.py
- [ ] Step 2: Update genome.py _evaluate_condition and _CONDITION_OPS
- [ ] Step 3: Run test_temporal_unfolding.py — verify all Level 1/2/3 PASS
- [ ] Step 4: Run full fast suite
- [ ] Step 5: Commit

---

## Task 4: Regression and integration

**Files:**
- Modify: `tests/test_temporal_unfolding.py` — add Level 4 regression test
- No source changes expected

### Step 1: Write the regression test

Append to `/home/mattthomson/workspace/AI-DNA/tests/test_temporal_unfolding.py`:

```python
# ---------------------------------------------------------------------------
# Level 4 — Regression and integration
# ---------------------------------------------------------------------------

def test_experiment_01_runs_unchanged_with_default_stage_increment() -> None:
    """Experiment 01 (multiply benchmark) exits 0 with stage_increment=0.0 default.

    Verifies backward compatibility: existing genomes without temporal codons
    are unaffected because ChemistryContext.stage_increment defaults to 0.0.
    """
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


def test_stage_increment_zero_is_default_on_chemistry_context() -> None:
    """ChemistryContext() default stage_increment is 0.0 (backward compatible)."""
    context = ChemistryContext()
    assert context.stage_increment == 0.0, (
        f"Expected default stage_increment=0.0, got {context.stage_increment}"
    )


def test_existing_4slot_callers_work_with_5slot_signals() -> None:
    """Rules that only touch signals[0-3] are unaffected by the 5th slot."""
    task = TaskCase(x=3.0, y=2.0, z=6.0)
    cell = CellState(active_rules=["RULE_EMIT_X", "RULE_EMIT_Y"])
    cell.reset(x=task.x, y=task.y)

    context = ChemistryContext(stage_increment=0.0)
    system = ChemistrySystem(max_time=10.0)
    system.run(cell, task, context=context)

    assert cell.signals[0] == task.x
    assert cell.signals[1] == task.y
    assert cell.signals[4] == 0.0  # untouched


def test_signal4_does_not_affect_existing_rules() -> None:
    """Running multiply rules with stage_increment=0.1 still produces correct output."""
    task = TaskCase(x=3.0, y=4.0, z=12.0)
    cell = CellState(active_rules=[
        "RULE_EMIT_X", "RULE_EMIT_Y",
        "RULE_ZERO_2", "RULE_COPY1_3",
        "RULE_ADD0_IF1", "RULE_DECAY1",
        "RULE_OUTPUT_IF1Z",
    ])
    cell.reset(x=task.x, y=task.y)

    context = ChemistryContext(stage_increment=0.1)
    system = ChemistrySystem(max_time=32.0)
    system.run(cell, task, context=context)

    # Multiply produces x*y = 12.0 regardless of stage_increment
    assert cell.output == pytest.approx(task.z, abs=0.1), (
        f"Expected output≈12.0, got {cell.output}"
    )
```

### Step 2: Run experiment 01 directly

```bash
python3 /home/mattthomson/workspace/AI-DNA/experiments/01_multiply_persistent_rules.py
```

Expected: exits 0, output identical to pre-change baseline.

### Step 3: Run all Level 4 tests

```bash
python3 -m pytest tests/test_temporal_unfolding.py -v
```

Expected: all Level 1/2/3/4 tests pass.

### Step 4: Run full fast suite

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass, no failures or errors.

### Step 5: Check for any 4-slot signal references in existing tests

```bash
grep -rn "0\.0, 0\.0, 0\.0, 0\.0" /home/mattthomson/workspace/AI-DNA/tests/
grep -rn "signals\[3\]" /home/mattthomson/workspace/AI-DNA/tests/
grep -rn "len(.*signals.*) == 4" /home/mattthomson/workspace/AI-DNA/tests/
```

For each match: update 4-slot tuple/list literals to 5-slot and re-run the suite.

### Step 6: Final commit

```bash
git add tests/test_temporal_unfolding.py
git commit -m "feat: add temporal unfolding — signal[4] stage accumulator"
```

- [ ] Step 1: Write Level 4 regression tests
- [ ] Step 2: Run experiment 01 directly — verify exits 0
- [ ] Step 3: Run test_temporal_unfolding.py — all Level 1/2/3/4 PASS
- [ ] Step 4: Run full fast suite — all pass
- [ ] Step 5: Check for 4-slot references in existing tests and update
- [ ] Step 6: Final commit

---

## Self-review checklist

### Spec success criteria coverage

| Criterion | Covered by |
|---|---|
| 1. All existing tests pass unchanged (stage_increment=0.0 default) | Task 4 Step 4 full suite; Task 2 Step 8 |
| 2. signal[4] increments correctly each round and clamps at 1.0 | Task 2 — test_stage_increment_clamps_at_one |
| 3. IF_S4_GT correctly gates gene expression from round 5 onward | Task 3 — test_gate_if_s4_gt_fires_from_round_five |
| 4. Two-phase genome produces different active rule sets in rounds 1-4 vs 5+ | Task 3 — test_two_phase_genome_different_active_rules_per_phase |
| 5. Experiment 01 output identical to pre-change baseline | Task 4 — test_experiment_01_runs_unchanged_with_default_stage_increment |

### Name consistency

| Name | Used in |
|---|---|
| `signal[4]` | cell.py, chemistry.py step(), all tests |
| `stage_increment` | ChemistryContext field, test names, all references |
| `IF_S4_GT` / `IF_S4_LT` | codons.py REGULATORY_CODON_MAP, chemistry.py _condition_passes(), genome.py _CONDITION_OPS and _evaluate_condition, all tests |
| `IF_S3_GT` / `IF_S3_LT` | same files — added for completeness per spec |

### Backward compatibility

- `ChemistryContext.stage_increment` defaults to `0.0` — existing code that creates `ChemistryContext()` without keyword args is unaffected.
- `CellState.signals` changes from 4 to 5 slots — existing rules only index [0]-[3], so adding [4] does not affect their behaviour.
- `_record()` type hint update is cosmetic only — runtime behaviour unchanged.
- `declare_rules(signals=None)` — existing callers that pass no signals continue to work; new 5-tuple signals are handled by `_evaluate_condition`'s `len(signals) > 4` guard.

### No placeholders

All implementation code shown in full. No "implement X" without accompanying Python.

### Circular import guard

- `cell.py` imports nothing from project modules.
- `codons.py` imports nothing from project modules.
- `genome.py` imports from `codons.py` only.
- `chemistry.py` imports from `cell.py`, `genome.py`, `tasks.py`, `tracing.py` — no circular dependency introduced.
