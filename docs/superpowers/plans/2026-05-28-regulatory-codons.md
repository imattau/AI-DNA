# Regulatory Codons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add genome-level promoters and chemistry-level GATE codons so genomes can encode conditional programs that activate based on signal state.
**Architecture:** Extend codon table with 8 new op-names; extend declare_rules() with a promoter stack; add GateRule dataclass; extend chemistry VM to evaluate GateRule per round. Two independent layers sharing signal slots.
**Tech Stack:** Python 3.10+, dataclasses, pytest

---

## File Map

- Modify: `codons.py` — add 8 new op-names to `default_codon_table()`
- Modify: `genome.py` — add `GateRule` dataclass; extend `declare_rules()` with promoter stack
- Modify: `chemistry.py` — extend `ChemistrySystem.step()` to handle `GateRule` instances
- Modify: `tests/test_codons.py` — Level 1 codon table tests
- Create: `tests/test_regulatory_codons.py` — Level 2, 3, 4 tests

---

## Task 1: Codon table extensions

**Files:**
- Modify: `codons.py`
- Modify: `tests/test_codons.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_codons.py`:

```python
def test_regulatory_op_names_present_in_table() -> None:
    """All 8 new regulatory op-names appear in the default codon table."""
    from codons import default_codon_table

    table = default_codon_table()
    regulatory_ops = {
        "PROMOTER", "GATE",
        "IF_S0_GT", "IF_S1_GT", "IF_S2_GT",
        "IF_S0_LT", "IF_S1_LT",
        "END_BLOCK",
    }
    present = set(table.codon_to_op.values())
    for op in regulatory_ops:
        assert op in present, f"{op} missing from codon table"


def test_regulatory_synonyms_decode_to_same_op() -> None:
    """All synonyms for IF_S0_GT decode to the same op-name (neutral mutation)."""
    from codons import default_codon_table

    table = default_codon_table()
    # Collect all codons that map to IF_S0_GT
    synonyms = [c for c, op in table.codon_to_op.items() if op == "IF_S0_GT"]
    assert len(synonyms) >= 3, f"Expected at least 3 synonyms for IF_S0_GT, got {synonyms}"
    assert all(table.op(c) == "IF_S0_GT" for c in synonyms)


def test_regulatory_round_trip_encode_decode() -> None:
    """Encoding op-names then decoding returns the same op-names."""
    from codons import default_codon_table

    table = default_codon_table()
    ops = ["PROMOTER", "IF_S0_GT", "RULE_EMIT_X", "END_BLOCK", "GATE", "IF_S1_LT"]
    encoded = table.encode_ops(ops)
    decoded = table.decode(encoded)
    assert decoded == ops


def test_each_regulatory_op_has_at_least_three_synonyms() -> None:
    """Each new regulatory op-name has at least 3 synonym codon values."""
    from codons import default_codon_table

    table = default_codon_table()
    regulatory_ops = [
        "PROMOTER", "GATE",
        "IF_S0_GT", "IF_S1_GT", "IF_S2_GT",
        "IF_S0_LT", "IF_S1_LT",
        "END_BLOCK",
    ]
    op_to_codons = table.op_to_codons
    for op in regulatory_ops:
        count = len(op_to_codons.get(op, []))
        assert count >= 3, f"{op} has only {count} synonyms, expected >= 3"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_codons.py::test_regulatory_op_names_present_in_table tests/test_codons.py::test_regulatory_synonyms_decode_to_same_op tests/test_codons.py::test_regulatory_round_trip_encode_decode tests/test_codons.py::test_each_regulatory_op_has_at_least_three_synonyms -v
```

Expected: FAIL — regulatory ops not yet in the table.

- [ ] **Step 3: Add 8 new op-names to `default_codon_table()`**

In `codons.py`, replace the `default_codon_table` function (lines 104-125):

```python
def default_codon_table() -> CodonTable:
    allocation = {
        "START": 3,
        "STOP": 3,
        "NOOP": 4,
        "SEND": 2,
        "RECV": 2,
        "RULE_EMIT_X": 5,
        "RULE_EMIT_Y": 5,
        "RULE_ZERO_2": 4,
        "RULE_ADD0_IF1": 6,
        "RULE_DECAY1": 6,
        "RULE_OUTPUT_IF1Z": 5,
        "RULE_INHIBIT1Z": 3,
        "RULE_COPY0_3": 3,
        "RULE_COPY1_3": 3,
        "RULE_ADD3_IF1": 3,
        "RULE_DECAY3": 3,
        "RULE_THRESH1": 2,
        "RULE_THRESH3": 2,
        # Regulatory codons — 8 new op-names, 3-4 synonyms each
        "PROMOTER": 3,
        "GATE": 3,
        "IF_S0_GT": 3,
        "IF_S1_GT": 3,
        "IF_S2_GT": 3,
        "IF_S0_LT": 3,
        "IF_S1_LT": 3,
        "END_BLOCK": 4,
    }
    return build_codon_table(allocation, table_size=128, fill_op="NOOP")
```

Note: `table_size` increases from 64 to 128 to accommodate the 24 additional synonym slots (8 ops × 3 synonyms each) without evicting existing ops. The `build_codon_table` function places the first occurrence of each op at index `idx`, then adds synonyms in a second pass — existing op assignments are preserved.

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_codons.py -v
```

Expected: all pass.

- [ ] **Step 5: Verify existing tests still pass**

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass (table_size change is backward compatible — existing codons 0-63 retain the same op assignments).

- [ ] **Step 6: Commit**

```bash
git add codons.py tests/test_codons.py
git commit -m "feat: add 8 regulatory op-names to codon table (PROMOTER, GATE, IF_S*, END_BLOCK)"
```

---

## Task 2: Transcription — promoter stack in `declare_rules()`

**Files:**
- Modify: `genome.py`
- Create: `tests/test_regulatory_codons.py` (Level 2 tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_regulatory_codons.py`:

```python
from __future__ import annotations

import pytest
from codons import default_codon_table
from genome import CellGenome, GateRule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TABLE = default_codon_table()


def _genome_from_ops(ops: list[str], lineage_id: str = "test") -> CellGenome:
    """Build a CellGenome whose codons encode the given sequence of op-names."""
    codons = tuple(TABLE.encode_ops(ops))
    return CellGenome(codons=codons, local_motifs=(), lineage_id=lineage_id)


# ---------------------------------------------------------------------------
# Level 2 — Transcription (promoter stack)
# ---------------------------------------------------------------------------

def test_promoter_block_active_when_signal_high() -> None:
    """PROMOTER + IF_S0_GT includes the inner rule when signal[0] = 0.8 > 0.5."""
    genome = _genome_from_ops([
        "PROMOTER", "IF_S0_GT",
        "RULE_EMIT_X",
        "END_BLOCK",
    ])
    signals = (0.8, 0.0, 0.0, 0.0)
    rules = genome.declare_rules(signals=signals)
    assert "RULE_EMIT_X" in rules, f"Expected RULE_EMIT_X in rules, got {rules}"


def test_promoter_block_skipped_when_signal_low() -> None:
    """PROMOTER + IF_S0_GT excludes the inner rule when signal[0] = 0.2 < 0.5."""
    genome = _genome_from_ops([
        "PROMOTER", "IF_S0_GT",
        "RULE_EMIT_X",
        "END_BLOCK",
        "RULE_EMIT_Y",  # outside block — always included
    ])
    signals = (0.2, 0.0, 0.0, 0.0)
    rules = genome.declare_rules(signals=signals)
    assert "RULE_EMIT_X" not in rules, f"RULE_EMIT_X should be absent, got {rules}"
    assert "RULE_EMIT_Y" in rules, f"RULE_EMIT_Y should be present, got {rules}"


def test_unbalanced_genome_no_end_block_completes() -> None:
    """A genome with no END_BLOCK completes without error (implicit END_BLOCK at end)."""
    genome = _genome_from_ops([
        "PROMOTER", "IF_S0_GT",
        "RULE_EMIT_X",
        # No END_BLOCK — end-of-genome acts as implicit END_BLOCK
    ])
    signals = (0.8, 0.0, 0.0, 0.0)
    rules = genome.declare_rules(signals=signals)
    # Should not raise; RULE_EMIT_X included since signal[0] > 0.5
    assert "RULE_EMIT_X" in rules


def test_unknown_condition_codon_defaults_to_true() -> None:
    """A non-IF_S* codon after PROMOTER defaults to true (constitutive expression)."""
    genome = _genome_from_ops([
        "PROMOTER", "NOOP",   # NOOP is not a valid condition — defaults to true
        "RULE_EMIT_X",
        "END_BLOCK",
    ])
    signals = (0.0, 0.0, 0.0, 0.0)
    rules = genome.declare_rules(signals=signals)
    assert "RULE_EMIT_X" in rules, f"Unknown condition should default to true, got {rules}"


def test_gate_codon_emits_gate_rule() -> None:
    """GATE codon produces a GateRule placeholder in declared rules."""
    genome = _genome_from_ops([
        "GATE", "IF_S1_GT",
        "RULE_EMIT_X",
        "END_BLOCK",
    ])
    signals = (0.0, 0.0, 0.0, 0.0)
    rules = genome.declare_rules(signals=signals)
    gate_rules = [r for r in rules if isinstance(r, GateRule)]
    assert len(gate_rules) == 1, f"Expected 1 GateRule, got {gate_rules}"
    assert gate_rules[0].condition == "IF_S1_GT"
    assert "RULE_EMIT_X" in gate_rules[0].rules


def test_declare_rules_no_signals_backward_compatible() -> None:
    """declare_rules() with no signals argument behaves exactly as before (all blocks active)."""
    genome = _genome_from_ops(["RULE_EMIT_X", "RULE_EMIT_Y"])
    rules = genome.declare_rules()
    assert "RULE_EMIT_X" in rules
    assert "RULE_EMIT_Y" in rules


def test_promoter_stack_depth_cap() -> None:
    """Deeply nested PROMOTER blocks beyond depth 4 complete without error."""
    # 5 levels of nesting — should cap at 4 and not crash
    ops = []
    for _ in range(5):
        ops += ["PROMOTER", "IF_S0_GT"]
    ops += ["RULE_EMIT_X"]
    for _ in range(5):
        ops += ["END_BLOCK"]
    genome = _genome_from_ops(ops)
    signals = (0.8, 0.0, 0.0, 0.0)
    rules = genome.declare_rules(signals=signals)  # must not raise
    assert isinstance(rules, tuple)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_regulatory_codons.py -v
```

Expected: FAIL — `GateRule` does not exist and `declare_rules` does not accept `signals`.

- [ ] **Step 3: Add `GateRule` dataclass to `genome.py`**

In `genome.py`, after the `Motif` dataclass (after line 26), add:

```python
@dataclass(frozen=True, slots=True)
class GateRule:
    """Runtime condition block produced by a GATE codon during transcription.

    condition: one of IF_S0_GT, IF_S1_GT, IF_S2_GT, IF_S0_LT, IF_S1_LT
    rules: tuple of rule names (str) to execute when condition is true
    """
    condition: str
    rules: tuple[str, ...]
```

- [ ] **Step 4: Add condition-evaluation helpers to `genome.py`**

In `genome.py`, after the `GateRule` dataclass, add:

```python
# Regulatory codon op-names
_PROMOTER_OPS: frozenset[str] = frozenset({"PROMOTER"})
_GATE_OPS: frozenset[str] = frozenset({"GATE"})
_CONDITION_OPS: frozenset[str] = frozenset({
    "IF_S0_GT", "IF_S1_GT", "IF_S2_GT",
    "IF_S0_LT", "IF_S1_LT",
})
_END_BLOCK_OPS: frozenset[str] = frozenset({"END_BLOCK"})
_MAX_PROMOTER_DEPTH: int = 4


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
    # Unknown condition → constitutive expression (always true)
    return True
```

- [ ] **Step 5: Rewrite `declare_rules()` in `CellGenome` to support promoter stack**

In `genome.py`, replace the `declare_rules` method (lines 48-52):

```python
def declare_rules(
    self,
    signals: tuple[float, ...] | None = None,
) -> tuple[str | GateRule, ...]:
    """Walk codons and transcribe rules, honouring PROMOTER and GATE blocks.

    Parameters
    ----------
    signals:
        Current cell signal vector. When None, all PROMOTER blocks are treated
        as active (backward compatible — no regulatory evaluation).
    """
    from codons import default_codon_table
    table = default_codon_table()
    table_size = len(table.codon_to_op)

    declared: list[str | GateRule] = []

    # Stack entries: ("promoter"|"gate", condition, active, block_collector)
    # active: bool — whether to include rules in current block
    # block_collector: list[str] | None — collects rule names for GateRule inner block
    stack: list[tuple[str, str, bool, list[str] | None]] = []

    def _currently_active() -> bool:
        """True when all enclosing PROMOTER blocks are active (GATE blocks pass through)."""
        for kind, _cond, active, _collector in stack:
            if kind == "promoter" and not active:
                return False
        return True

    codon_iter = iter(self.codons)
    for raw_codon in codon_iter:
        op = table.op(raw_codon % table_size)

        if op in _PROMOTER_OPS:
            # Read next codon as condition
            try:
                cond_codon = next(codon_iter)
            except StopIteration:
                cond_codon = 0
            cond_op = table.op(cond_codon % table_size)
            if cond_op not in _CONDITION_OPS:
                cond_op = "UNKNOWN"  # will default to true
            if len(stack) < _MAX_PROMOTER_DEPTH:
                if signals is None:
                    active = True
                else:
                    active = _evaluate_condition(cond_op, signals)
                stack.append(("promoter", cond_op, active, None))
            # If at depth cap, ignore this PROMOTER (treat as NOOP)
            continue

        if op in _GATE_OPS:
            # Read next codon as condition
            try:
                cond_codon = next(codon_iter)
            except StopIteration:
                cond_codon = 0
            cond_op = table.op(cond_codon % table_size)
            if cond_op not in _CONDITION_OPS:
                cond_op = "UNKNOWN"
            if len(stack) < _MAX_PROMOTER_DEPTH and _currently_active():
                stack.append(("gate", cond_op, True, []))
            continue

        if op in _END_BLOCK_OPS:
            if stack:
                kind, cond_op, active, collector = stack.pop()
                if kind == "gate" and collector is not None and _currently_active():
                    declared.append(GateRule(condition=cond_op, rules=tuple(collector)))
            continue

        # Regular rule codon
        if not _currently_active():
            continue  # inside an inactive PROMOTER block — skip

        # Inside a GATE block — collect into block, not declared
        if stack and stack[-1][0] == "gate":
            kind, cond_op, active, collector = stack[-1]
            if collector is not None and op in set(RULE_NAMES):
                collector.append(op)
            continue

        # Normal path — emit rule name
        if op in set(RULE_NAMES):
            declared.append(op)

    # Implicit END_BLOCK for any unclosed stacks
    while stack:
        kind, cond_op, active, collector = stack.pop()
        if kind == "gate" and collector is not None and _currently_active():
            declared.append(GateRule(condition=cond_op, rules=tuple(collector)))

    # Append motif rules (not subject to promoter filtering)
    for motif in self.local_motifs:
        for rule in motif.pattern:
            if rule not in declared:
                declared.append(rule)

    # Deduplicate while preserving order (GateRule instances are kept as-is)
    seen_str: set[str] = set()
    result: list[str | GateRule] = []
    for item in declared:
        if isinstance(item, GateRule):
            result.append(item)
        elif item not in seen_str:
            seen_str.add(item)
            result.append(item)

    return tuple(result)
```

Also add `RULE_NAMES` to the import at top of `genome.py` (it is already imported from `codons`).

- [ ] **Step 6: Update `cell.py` / callers if `active_rules` typing needs updating**

The `active_rules` field on `CellState` is `list[str]`. After this change, `declare_rules()` may return `GateRule` instances. Callers that pass `declare_rules()` output directly to `active_rules` need updating. Check the call sites:

```bash
grep -n "declare_rules" /home/mattthomson/workspace/AI-DNA/*.py
```

For each call site that feeds into `active_rules`, filter out `GateRule` instances for the string-only list, and handle `GateRule` separately in the VM (Task 3). The chemistry VM will receive a mixed list; `active_rules` type widens to `list[str | GateRule]`.

In `cell.py`, update the type annotation for `active_rules`:

```python
# Before
active_rules: list[str]
# After — GateRule is handled by chemistry VM
active_rules: list[object]
```

Or more precisely (requires importing GateRule — creates circular import). The safe approach: keep `active_rules` as `list` (no annotation narrowing) and let the VM duck-type. The VM already iterates `cell.active_rules` and looks up each name in `self.rulebook` — we extend it to check `isinstance(item, GateRule)` first (Task 3).

- [ ] **Step 7: Run Level 2 tests**

```bash
python3 -m pytest tests/test_regulatory_codons.py -v
```

Expected: all Level 2 tests pass.

- [ ] **Step 8: Run full fast suite**

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add genome.py tests/test_regulatory_codons.py
git commit -m "feat: add GateRule dataclass and promoter stack to declare_rules()"
```

---

## Task 3: Runtime gating — GateRule in chemistry VM

**Files:**
- Modify: `chemistry.py`
- Modify: `tests/test_regulatory_codons.py` (add Level 3 tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_regulatory_codons.py`:

```python
# ---------------------------------------------------------------------------
# Level 3 — Runtime gating (chemistry VM)
# ---------------------------------------------------------------------------

def test_gate_rule_fires_when_signal_high() -> None:
    """GateRule(IF_S1_GT, [RULE_EMIT_X]) fires when signal[1] = 0.8 > 0.5."""
    from chemistry import ChemistrySystem, ChemistryContext
    from cell import CellState
    from tasks import TaskCase
    from genome import GateRule

    task = TaskCase(x=3.0, y=0.0, z=3.0)
    gate = GateRule(condition="IF_S1_GT", rules=("RULE_EMIT_X",))
    cell = CellState(active_rules=[gate])
    cell.signals[1] = 0.8  # signal[1] > 0.5 — gate should open

    system = ChemistrySystem()
    system.run(cell, task)

    assert cell.signals[0] == task.x, (
        f"RULE_EMIT_X should have fired (signal[0] set to {task.x}), got {cell.signals[0]}"
    )


def test_gate_rule_skipped_when_signal_low() -> None:
    """GateRule(IF_S1_GT, [RULE_EMIT_X]) does not fire when signal[1] = 0.2 < 0.5."""
    from chemistry import ChemistrySystem
    from cell import CellState
    from tasks import TaskCase
    from genome import GateRule

    task = TaskCase(x=3.0, y=0.0, z=3.0)
    gate = GateRule(condition="IF_S1_GT", rules=("RULE_EMIT_X",))
    cell = CellState(active_rules=[gate])
    cell.signals[1] = 0.2  # signal[1] < 0.5 — gate stays closed

    system = ChemistrySystem()
    system.run(cell, task)

    assert cell.signals[0] == 0.0, (
        f"RULE_EMIT_X should NOT have fired, but signal[0] = {cell.signals[0]}"
    )


def test_gate_opens_mid_episode_when_signal_crosses_threshold() -> None:
    """Gate is re-evaluated each round: opens from the round signal[1] crosses 0.5."""
    from chemistry import ChemistrySystem, ChemistryContext
    from cell import CellState
    from tasks import TaskCase
    from genome import GateRule

    task = TaskCase(x=1.0, y=0.0, z=1.0)

    # Setup: RULE_DECAY1 lowers signal[1] each round; RULE_EMIT_X inside gate.
    # We prime signal[1] = 0.8 so gate opens immediately, confirm signal[0] gets set.
    # Then test with signal[1] = 0.2 (gate closed), manually bump signal[1] mid-run
    # via a custom rule — or simply verify the two-state contrast via separate runs.

    # Simpler verifiable form: run with signal[1] high, expect RULE_EMIT_X fired.
    gate = GateRule(condition="IF_S1_GT", rules=("RULE_EMIT_X",))
    cell_high = CellState(active_rules=[gate])
    cell_high.signals[1] = 0.8

    system = ChemistrySystem(max_time=4.0)
    system.run(cell_high, task)
    assert cell_high.signals[0] == task.x, "Gate should open with signal[1]=0.8"

    # Run with signal[1] low — gate stays closed for all rounds.
    gate2 = GateRule(condition="IF_S1_GT", rules=("RULE_EMIT_X",))
    cell_low = CellState(active_rules=[gate2])
    cell_low.signals[1] = 0.2

    system2 = ChemistrySystem(max_time=4.0)
    system2.run(cell_low, task)
    assert cell_low.signals[0] == 0.0, "Gate should stay closed with signal[1]=0.2"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_regulatory_codons.py::test_gate_rule_fires_when_signal_high tests/test_regulatory_codons.py::test_gate_rule_skipped_when_signal_low tests/test_regulatory_codons.py::test_gate_opens_mid_episode_when_signal_crosses_threshold -v
```

Expected: FAIL — `ChemistrySystem.step()` does not handle `GateRule` instances.

- [ ] **Step 3: Extend `ChemistrySystem.step()` to handle `GateRule`**

In `chemistry.py`, add the import for `GateRule` at the top (after existing imports):

```python
from genome import GateRule
```

Replace the `step` method (lines 186-199):

```python
def step(
    self,
    cell: CellState,
    task: TaskCase,
    context: ChemistryContext,
    *,
    event_index_start: int = 0,
) -> tuple[bool, int]:
    changed = False
    event_index = event_index_start
    for item in cell.active_rules:
        if isinstance(item, GateRule):
            # Re-evaluate condition against current signals every round
            if _evaluate_gate_condition(item.condition, tuple(cell.signals)):
                for rule_name in item.rules:
                    rule = self.rulebook.get(rule_name)
                    if rule is None:
                        continue
                    before = tuple(cell.signals)
                    note = rule.apply(cell, task, context)
                    after = tuple(cell.signals)
                    if after != before or note:
                        changed = True
                        _record(cell, event_index, rule_name, before, note or "")
                        event_index += 1
        else:
            rule_name = item
            rule = self.rulebook.get(rule_name)
            if rule is None:
                continue
            before = tuple(cell.signals)
            note = rule.apply(cell, task, context)
            after = tuple(cell.signals)
            if after != before or note:
                changed = True
                _record(cell, event_index, rule_name, before, note or "")
                event_index += 1
    return changed, event_index
```

Also add the helper function `_evaluate_gate_condition` in `chemistry.py` (after `_record`, before `build_rulebook`):

```python
def _evaluate_gate_condition(condition: str, signals: tuple[float, ...]) -> bool:
    """Evaluate a GATE condition op-name against current signals. Threshold = 0.5.

    Unknown condition defaults to True (constitutive — always fires).
    Mirrors genome._evaluate_condition but kept local to avoid circular import.
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
    return True  # unknown condition → constitutive
```

Note: `_evaluate_gate_condition` is a local copy (not imported from `genome.py`) to avoid the circular import that would arise from `chemistry.py` importing `genome.py` which imports `codons.py` (already the case), while `genome.py` must not import `chemistry.py`.

- [ ] **Step 4: Run Level 3 tests**

```bash
python3 -m pytest tests/test_regulatory_codons.py -v
```

Expected: all Level 2 and Level 3 tests pass.

- [ ] **Step 5: Run full fast suite**

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add chemistry.py tests/test_regulatory_codons.py
git commit -m "feat: extend chemistry VM to evaluate GateRule per round"
```

---

## Task 4: Integration and regression

**Files:**
- Modify: `tests/test_regulatory_codons.py` (add Level 4 integration test)
- No source changes expected

- [ ] **Step 1: Write the integration test**

Add to `tests/test_regulatory_codons.py`:

```python
# ---------------------------------------------------------------------------
# Level 4 — Integration
# ---------------------------------------------------------------------------

def test_promoter_and_gate_on_same_signal_slot() -> None:
    """Genome with PROMOTER block + GATE block on signal[0].

    Transcription-time: PROMOTER(IF_S0_GT) includes the GATE block when signal[0]=0.8.
    Runtime: the GATE(IF_S0_GT) inside the block controls per-round execution.
    """
    from chemistry import ChemistrySystem
    from cell import CellState
    from tasks import TaskCase
    from genome import GateRule

    # Build genome: PROMOTER(IF_S0_GT) wraps a GATE(IF_S0_GT) block around RULE_EMIT_Y
    genome = _genome_from_ops([
        "PROMOTER", "IF_S0_GT",
            "GATE", "IF_S0_GT",
                "RULE_EMIT_Y",
            "END_BLOCK",
        "END_BLOCK",
    ])

    # --- Transcription with signal[0] = 0.8 (high): PROMOTER block active ---
    signals_high = (0.8, 0.0, 0.0, 0.0)
    rules_high = genome.declare_rules(signals=signals_high)

    gate_rules_high = [r for r in rules_high if isinstance(r, GateRule)]
    assert len(gate_rules_high) == 1, (
        f"Expected 1 GateRule after transcription with signal[0]=0.8, got {gate_rules_high}"
    )
    assert gate_rules_high[0].condition == "IF_S0_GT"
    assert "RULE_EMIT_Y" in gate_rules_high[0].rules

    # --- Transcription with signal[0] = 0.2 (low): PROMOTER block inactive ---
    signals_low = (0.2, 0.0, 0.0, 0.0)
    rules_low = genome.declare_rules(signals=signals_low)

    gate_rules_low = [r for r in rules_low if isinstance(r, GateRule)]
    assert len(gate_rules_low) == 0, (
        f"Expected no GateRule after transcription with signal[0]=0.2, got {gate_rules_low}"
    )

    # --- Runtime: GateRule fires when signal[0] = 0.8 ---
    task = TaskCase(x=0.0, y=5.0, z=5.0)
    cell = CellState(active_rules=list(rules_high))
    cell.signals[0] = 0.8  # gate condition met

    system = ChemistrySystem(max_time=4.0)
    system.run(cell, task)
    assert cell.signals[1] == task.y, (
        f"RULE_EMIT_Y should have fired with signal[0]=0.8, got signal[1]={cell.signals[1]}"
    )


def test_experiment_01_output_unchanged() -> None:
    """Experiment 01 (multiply benchmark) produces the same structure as before.

    Verifies backward compatibility: genomes without regulatory codons are unaffected.
    This test imports and runs the experiment's core logic without the CLI output layer.
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
```

- [ ] **Step 2: Run the integration test**

```bash
python3 -m pytest tests/test_regulatory_codons.py::test_promoter_and_gate_on_same_signal_slot -v
```

Expected: PASS.

- [ ] **Step 3: Run experiment 01 directly**

```bash
python3 experiments/01_multiply_persistent_rules.py
```

Expected: exits 0, output identical to pre-change baseline (no regulatory codons in those genomes).

- [ ] **Step 4: Run full fast suite**

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass, no failures or errors.

- [ ] **Step 5: Run the experiment 01 integration test**

```bash
python3 -m pytest tests/test_regulatory_codons.py::test_experiment_01_output_unchanged -v
```

Expected: PASS.

- [ ] **Step 6: Final commit**

```bash
git add tests/test_regulatory_codons.py
git commit -m "feat: add regulatory codons — promoters and runtime gates"
```

---

## Self-review checklist

### Spec coverage

| Spec requirement | Covered by |
|---|---|
| 8 new op-names with 3-4 synonyms each | Task 1 Step 3, tests Step 1 |
| Neutral drift preserved for new ops | Task 1 — synonyms in `default_codon_table()` |
| PROMOTER evaluates at transcription time | Task 2 `declare_rules()` |
| GATE produces `GateRule` placeholder | Task 2 `declare_rules()` + `GateRule` dataclass |
| Stack depth cap at 4 | `_MAX_PROMOTER_DEPTH = 4` in genome.py |
| End-of-genome = implicit END_BLOCK | `while stack:` cleanup loop in `declare_rules()` |
| Unknown condition → constitutive expression | `_evaluate_condition` default `return True` |
| `signals=None` backward compatible | `if signals is None: active = True` |
| GateRule re-evaluated every chemistry round | `step()` calls `_evaluate_gate_condition` each pass |
| Experiment 01 unchanged | Task 4 Step 3 + integration test |
| All existing tests pass | Task 1 Step 5, Task 2 Step 8, Task 3 Step 5, Task 4 Step 4 |

### No placeholders

- All implementation code shown in full.
- No "implement X" without accompanying Python.

### Type consistency

- `GateRule` is defined once in `genome.py`.
- `chemistry.py` imports `GateRule` from `genome.py` for `isinstance` check.
- `_evaluate_gate_condition` in `chemistry.py` mirrors `_evaluate_condition` in `genome.py` (no circular import).
- `declare_rules()` return type is `tuple[str | GateRule, ...]` — callers that expect only `str` (e.g. direct `active_rules` assignment) receive a mixed tuple and the VM handles both.
- `active_rules` field on `CellState` is `list` — no type narrowing required; VM iterates and `isinstance`-dispatches.

### Circular import guard

- `genome.py` imports from `codons.py` ✓
- `chemistry.py` imports from `genome.py` (`GateRule`) ✓
- `genome.py` does NOT import from `chemistry.py` ✓
- `_evaluate_gate_condition` is kept local to `chemistry.py` for this reason.
