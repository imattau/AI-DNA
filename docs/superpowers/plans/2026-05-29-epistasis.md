# Epistasis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SCALE_BY_Sn` codons that multiply all signal deltas of the next rule by `signal[n]`, enabling analog epistatic modulation of gene output.

**Architecture:** Four new modifier codons (SCALE_BY_S0–S3) set a pending slot during transcription; the next rule is wrapped in a new `ScaleRule` dataclass. The chemistry VM dispatches `ScaleRule` by snapshotting signals before/after inner rule execution and scaling each delta by the modulator value.

**Tech Stack:** Python 3.10+, dataclasses, pytest.

---

## File Map

- Modify: `codons.py` — add SCALE_BY_S0–S3 to ALL_CODONS and byte table (bytes 271–282)
- Modify: `genome.py` — add `ScaleRule` dataclass; add pending-modifier logic to `_transcribe_rules()`
- Modify: `chemistry.py` — add `ScaleRule` import and dispatch in `execute()`
- Create: `tests/test_epistasis.py` — unit + integration tests

---

### Task 1: Add `SCALE_BY_S0–S3` to `codons.py`

**Files:**
- Modify: `codons.py`
- Create: `tests/test_epistasis.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_epistasis.py`:

```python
from __future__ import annotations

from codons import ALL_CODONS, CODON_TABLE


def test_scale_by_codons_in_all_codons() -> None:
    for name in ("SCALE_BY_S0", "SCALE_BY_S1", "SCALE_BY_S2", "SCALE_BY_S3"):
        assert name in ALL_CODONS, f"{name} missing from ALL_CODONS"


def test_scale_by_codons_in_byte_table() -> None:
    names_in_table = set(CODON_TABLE.values())
    for name in ("SCALE_BY_S0", "SCALE_BY_S1", "SCALE_BY_S2", "SCALE_BY_S3"):
        assert name in names_in_table, f"{name} missing from CODON_TABLE"
    # Each should have exactly 3 synonyms
    for name in ("SCALE_BY_S0", "SCALE_BY_S1", "SCALE_BY_S2", "SCALE_BY_S3"):
        count = sum(1 for v in CODON_TABLE.values() if v == name)
        assert count == 3, f"{name} has {count} synonyms, expected 3"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_epistasis.py -v
```

Expected: FAIL — SCALE_BY_S0 not in ALL_CODONS.

- [ ] **Step 3: Add codons to `codons.py`**

In `codons.py`, find `ALL_CODONS` (the list of all codon name strings). Add after the last entry:

```python
"SCALE_BY_S0",
"SCALE_BY_S1",
"SCALE_BY_S2",
"SCALE_BY_S3",
```

In `codons.py`, find the byte→name dict (the one ending at byte 270). Add after byte 270:

```python
271: "SCALE_BY_S0",
272: "SCALE_BY_S0",
273: "SCALE_BY_S0",
274: "SCALE_BY_S1",
275: "SCALE_BY_S1",
276: "SCALE_BY_S1",
277: "SCALE_BY_S2",
278: "SCALE_BY_S2",
279: "SCALE_BY_S2",
280: "SCALE_BY_S3",
281: "SCALE_BY_S3",
282: "SCALE_BY_S3",
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_epistasis.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codons.py tests/test_epistasis.py
git commit -m "feat: add SCALE_BY_S0-S3 codons"
```

---

### Task 2: Add `ScaleRule` dataclass to `genome.py`

**Files:**
- Modify: `genome.py`
- Modify: `tests/test_epistasis.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_epistasis.py`:

```python
from genome import ScaleRule, GateRule


def test_scale_rule_is_dataclass() -> None:
    inner = "EMIT_S0"
    sr = ScaleRule(signal_slot=2, inner=inner)
    assert sr.signal_slot == 2
    assert sr.inner == "EMIT_S0"


def test_scale_rule_accepts_gate_rule_inner() -> None:
    gate = GateRule(condition="IF_S0_GT", rules=("EMIT_S0",))
    sr = ScaleRule(signal_slot=0, inner=gate)
    assert isinstance(sr.inner, GateRule)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_epistasis.py::test_scale_rule_is_dataclass -v
```

Expected: FAIL — cannot import `ScaleRule`.

- [ ] **Step 3: Add `ScaleRule` to `genome.py`**

In `genome.py`, find the `GateRule` dataclass (around line 31):

```python
@dataclass(frozen=True, slots=True)
class GateRule:
    condition: str
    rules: tuple[str | "GateRule", ...]
```

Add immediately after it:

```python
@dataclass(frozen=True, slots=True)
class ScaleRule:
    signal_slot: int
    inner: str | GateRule
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_epistasis.py::test_scale_rule_is_dataclass tests/test_epistasis.py::test_scale_rule_accepts_gate_rule_inner -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genome.py tests/test_epistasis.py
git commit -m "feat: add ScaleRule dataclass to genome.py"
```

---

### Task 3: Wire `SCALE_BY_Sn` into `_transcribe_rules()`

**Files:**
- Modify: `genome.py`
- Modify: `tests/test_epistasis.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_epistasis.py`:

```python
from genome import CellGenome, ScaleRule


def test_transcribe_scale_by_wraps_next_rule() -> None:
    genome = CellGenome.from_rule_names(
        ["SCALE_BY_S2", "EMIT_S0", "OUTPUT"],
        lineage_id="L1",
    )
    rules = genome.declare_rules()
    assert len(rules) == 2
    assert isinstance(rules[0], ScaleRule)
    assert rules[0].signal_slot == 2
    assert rules[0].inner == "EMIT_S0"
    assert rules[1] == "OUTPUT"


def test_transcribe_dangling_scale_by_is_dropped() -> None:
    genome = CellGenome.from_rule_names(
        ["EMIT_S0", "SCALE_BY_S1"],
        lineage_id="L1",
    )
    rules = genome.declare_rules()
    assert len(rules) == 1
    assert rules[0] == "EMIT_S0"


def test_transcribe_consecutive_scale_by_last_wins() -> None:
    genome = CellGenome.from_rule_names(
        ["SCALE_BY_S0", "SCALE_BY_S3", "EMIT_S0"],
        lineage_id="L1",
    )
    rules = genome.declare_rules()
    assert len(rules) == 1
    assert isinstance(rules[0], ScaleRule)
    assert rules[0].signal_slot == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_epistasis.py -k "transcribe" -v
```

Expected: FAIL — SCALE_BY_S2 treated as unknown rule string, not a modifier.

- [ ] **Step 3: Update `_transcribe_rules()` in `genome.py`**

Find `_transcribe_rules()` (~line 228). Inside the `parse()` inner function, find the section that handles known opcodes (PROMOTER, GATE, CALL_, etc.) and the final `active.append(op)` fallthrough.

Add handling for SCALE_BY before the fallthrough `active.append(op)` line:

```python
            if op.startswith("SCALE_BY_S"):
                try:
                    pending_scale = int(op[-1])
                except (ValueError, IndexError):
                    pending_scale = None
                index += 1
                continue
```

Then change the fallthrough `active.append(op)` to apply any pending scale:

```python
            if pending_scale is not None:
                active.append(ScaleRule(signal_slot=pending_scale, inner=op))
                pending_scale = None
            else:
                active.append(op)
            index += 1
```

`pending_scale` must be initialised to `None` at the top of `parse()`:

```python
    def parse(index: int, depth: int) -> tuple[list[str | GateRule], int]:
        active: list[str | GateRule] = []
        pending_scale: int | None = None
        ...
```

Also add `ScaleRule` to the import at the top of the file where `GateRule` is used (it's defined in the same file, so no import needed — just ensure `ScaleRule` is referenced after its definition).

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_epistasis.py -k "transcribe" -v
```

Expected: all PASS.

- [ ] **Step 5: Run full fast suite**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add genome.py tests/test_epistasis.py
git commit -m "feat: wire SCALE_BY_Sn into _transcribe_rules"
```

---

### Task 4: Dispatch `ScaleRule` in the chemistry VM

**Files:**
- Modify: `chemistry.py`
- Modify: `tests/test_epistasis.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_epistasis.py`:

```python
from cell import CellState
from chemistry import ChemistryContext, RuleBook, build_default_rulebook
from genome import CellGenome, ScaleRule


def _run_genome(genome: CellGenome, signals: list[float]) -> CellState:
    """Run one chemistry round and return the resulting cell state."""
    from chemistry import CellChemistry
    rulebook = build_default_rulebook()
    cell = CellState()
    cell.signals = signals[:]
    rules = genome.declare_rules()
    context = ChemistryContext()
    chem = CellChemistry(rules=rules, rulebook=rulebook)
    chem.step(cell, task=None, context=context)
    return cell


def test_scale_rule_halves_delta() -> None:
    # EMIT_S0 writes 1.0 to signal[0]; with SCALE_BY_S1 and signal[1]=0.5, delta should be halved
    genome_scaled = CellGenome.from_rule_names(
        ["SCALE_BY_S1", "EMIT_S0"],
        lineage_id="L1",
    )
    genome_unscaled = CellGenome.from_rule_names(
        ["EMIT_S0"],
        lineage_id="L2",
    )
    signals_base = [0.0, 0.5, 0.0, 0.0, 0.0]
    cell_scaled = _run_genome(genome_scaled, signals_base)
    cell_unscaled = _run_genome(genome_unscaled, signals_base)

    delta_scaled = cell_scaled.signals[0]
    delta_unscaled = cell_unscaled.signals[0]
    assert delta_scaled == pytest.approx(delta_unscaled * 0.5, abs=1e-6)


def test_scale_rule_zero_modulator_suppresses_delta() -> None:
    genome = CellGenome.from_rule_names(
        ["SCALE_BY_S2", "EMIT_S0"],
        lineage_id="L1",
    )
    # signal[2] = 0.0 → delta should be 0
    cell = _run_genome(genome, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert cell.signals[0] == pytest.approx(0.0, abs=1e-9)


def test_scale_rule_full_modulator_unchanged() -> None:
    genome_scaled = CellGenome.from_rule_names(
        ["SCALE_BY_S1", "EMIT_S0"],
        lineage_id="L1",
    )
    genome_unscaled = CellGenome.from_rule_names(
        ["EMIT_S0"],
        lineage_id="L2",
    )
    signals = [0.0, 1.0, 0.0, 0.0, 0.0]
    cell_scaled = _run_genome(genome_scaled, signals)
    cell_unscaled = _run_genome(genome_unscaled, signals)
    assert cell_scaled.signals[0] == pytest.approx(cell_unscaled.signals[0], abs=1e-6)
```

Also add `import pytest` at the top of `tests/test_epistasis.py` if not already present.

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_epistasis.py -k "scale_rule" -v
```

Expected: FAIL — chemistry VM does not handle `ScaleRule`, likely passes it to rulebook lookup which returns None.

- [ ] **Step 3: Add `ScaleRule` import to `chemistry.py`**

At the top of `chemistry.py`, find the genome import line:

```python
from genome import GateRule
```

Change to:

```python
from genome import GateRule, ScaleRule
```

- [ ] **Step 4: Add `ScaleRule` dispatch in `execute()` in `chemistry.py`**

Find the `execute()` inner function (~line 236). It currently starts:

```python
        def execute(rule_entry: str | GateRule) -> None:
            nonlocal changed, event_index
            if isinstance(rule_entry, GateRule):
                ...
                return

            rule = self.rulebook.get(rule_entry)
```

Add a `ScaleRule` branch immediately after the `GateRule` branch:

```python
            if isinstance(rule_entry, ScaleRule):
                modulator = cell.signals[rule_entry.signal_slot] if rule_entry.signal_slot < len(cell.signals) else 0.0
                before_scale = tuple(cell.signals)
                execute(rule_entry.inner)
                after_scale = tuple(cell.signals)
                for i in range(len(cell.signals)):
                    delta = after_scale[i] - before_scale[i]
                    if delta != 0.0:
                        cell.signals[i] = max(0.0, min(1.0, before_scale[i] + delta * modulator))
                return
```

Update the type annotation of `execute` to accept `ScaleRule`:

```python
        def execute(rule_entry: str | GateRule | ScaleRule) -> None:
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_epistasis.py -k "scale_rule" -v
```

Expected: all PASS.

- [ ] **Step 6: Run full fast suite**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add chemistry.py tests/test_epistasis.py
git commit -m "feat: dispatch ScaleRule in chemistry VM"
```

---

### Task 5: Integration smoke test

**Files:**
- Modify: `tests/test_epistasis.py`

- [ ] **Step 1: Write the integration test**

Add to `tests/test_epistasis.py`:

```python
from task_stream import StreamConfig, run_task_stream
from tasks import TaskBundle


def test_epistasis_genome_evaluates_without_error() -> None:
    """A genome with SCALE_BY codons should not crash task stream evaluation."""
    tasks = [
        TaskBundle(
            name="multiply",
            cases=((2, 3, 6), (1, 4, 4), (3, 3, 9)),
            anti_shortcut_cases=(),
        ),
    ]
    config = StreamConfig(
        initial_population_size=4,
        max_steps_per_task=3,
        archive_interval=10,
        estimate_neutrality_trials=0,
    )
    report = run_task_stream("test_epistasis", tasks, config=config, seed=99)
    assert len(report.results) > 0
    for r in report.results:
        assert r.mean_error >= 0.0
```

- [ ] **Step 2: Run the integration test**

```bash
python3 -m pytest tests/test_epistasis.py::test_epistasis_genome_evaluates_without_error -v
```

Expected: PASS (genomes with SCALE_BY codons may appear via random mutation; stream should not crash).

- [ ] **Step 3: Run all epistasis tests**

```bash
python3 -m pytest tests/test_epistasis.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_epistasis.py
git commit -m "test: add epistasis integration smoke test"
```

---

### Task 6: Final check

- [ ] **Step 1: Run full fast suite**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```

Expected: all pass.

- [ ] **Step 2: Verify ScaleRule appears in transcribed rules when seed forces it**

```bash
python3 -c "
from genome import CellGenome, ScaleRule
g = CellGenome.from_rule_names(['SCALE_BY_S0', 'EMIT_S0', 'SCALE_BY_S2', 'OUTPUT'], lineage_id='test')
rules = g.declare_rules()
print(rules)
assert isinstance(rules[0], ScaleRule)
assert rules[0].signal_slot == 0
assert isinstance(rules[1], ScaleRule)
assert rules[1].signal_slot == 2
print('ScaleRule transcription OK')
"
```

Expected: prints rule tuple and `ScaleRule transcription OK`.
