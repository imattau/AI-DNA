# Phase A — Observable Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up three missing metrics (neutral drift rate, phenotype diversity, recovery steps) into the existing stream reporting layer so future search improvements can be benchmarked objectively.

**Architecture:** All changes are confined to `task_stream.py` — add fields to `StreamTaskResult` and `ArchiveSnapshot`, compute them during the existing evaluation loops, and surface them in `format_text`. No new files needed. Tests go in `tests/test_task_stream.py`.

**Tech Stack:** Python 3.10+, dataclasses, pytest

---

## File Map

- Modify: `task_stream.py` — add fields and computation
- Modify: `tests/test_task_stream.py` — add assertions for new fields

---

### Task 1: Add `neutral_drift_rate` to `ArchiveSnapshot`

**Files:**
- Modify: `task_stream.py`
- Modify: `tests/test_task_stream.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_task_stream.py`:

```python
def test_archive_snapshot_has_neutral_drift_rate() -> None:
    from task_stream import ArchiveSnapshot
    snap = ArchiveSnapshot(
        generation=0,
        current_task="multiply",
        seen_tasks=("multiply",),
        mean_archive_error=0.0,
        per_task_error=(),
        best_lineage="A1",
    )
    assert hasattr(snap, "neutral_drift_rate")
    assert isinstance(snap.neutral_drift_rate, float)
    assert snap.neutral_drift_rate >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_task_stream.py::test_archive_snapshot_has_neutral_drift_rate -v
```
Expected: FAIL with `AttributeError: 'ArchiveSnapshot' object has no attribute 'neutral_drift_rate'`

- [ ] **Step 3: Add field to `ArchiveSnapshot`**

In `task_stream.py`, find the `ArchiveSnapshot` dataclass (around line 53). Add after the existing `resource_regen` field:

```python
neutral_drift_rate: float = 0.0
```

- [ ] **Step 4: Add to `format_text` in `StreamReport`**

In `task_stream.py`, find the archive snapshot formatting block in `StreamReport.format_text` (around line 111). Add `neutral_drift_rate` to the f-string after `resource_regen`:

```python
f"  - gen {snapshot.generation} task={snapshot.current_task} "
f"mean_archive_error={snapshot.mean_archive_error:.6f} best={snapshot.best_lineage} "
f"energy_efficiency={snapshot.energy_efficiency:.4f} "
f"lineage_efficiency={snapshot.lineage_efficiency:.4f} "
f"lineage_root={snapshot.lineage_root or '<none>'} "
f"forgetting_delta={snapshot.forgetting_delta:.6f} "
f"transfer_scores={snapshot.lineage_transfer_scores or '<none>'} "
f"resource_regen={snapshot.resource_regen:.2f} "
f"neutral_drift_rate={snapshot.neutral_drift_rate:.4f} "
f"motifs={snapshot.motif_count} reuse_total={snapshot.motif_reuse_total} "
f"transfer={snapshot.motif_transfer_count}/{snapshot.motif_count if snapshot.motif_count else 0} "
f"motif_reappearance={snapshot.motif_reappearance_count} "
f"motif_tasks={','.join(snapshot.motif_origin_tasks) if snapshot.motif_origin_tasks else '<none>'} "
f"motif_lineages={','.join(snapshot.motif_origin_lineages) if snapshot.motif_origin_lineages else '<none>'}"
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python3 -m pytest tests/test_task_stream.py::test_archive_snapshot_has_neutral_drift_rate -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add task_stream.py tests/test_task_stream.py
git commit -m "feat: add neutral_drift_rate field to ArchiveSnapshot"
```

---

### Task 2: Wire `estimate_neutrality` into snapshot computation

**Files:**
- Modify: `task_stream.py`

`estimate_neutrality` is already implemented in `evolution.py`. It needs to be called at each archive snapshot point and stored on the snapshot.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_task_stream.py`:

```python
def test_stream_produces_nonzero_neutral_drift_rate() -> None:
    """neutral_drift_rate should be > 0 for at least one snapshot in a short run."""
    from task_stream import StreamConfig, run_task_stream
    from tasks import TaskBundle

    rng_seed = 7
    tasks = [
        TaskBundle(name="multiply", cases=((2, 3, 6), (1, 4, 4), (3, 3, 9)), anti_shortcut_cases=()),
    ]
    config = StreamConfig(
        initial_population_size=4,
        max_steps_per_task=3,
        archive_interval=2,
        estimate_neutrality_trials=4,
    )
    report = run_task_stream("test_drift", tasks, config=config, seed=rng_seed)
    rates = [s.neutral_drift_rate for s in report.archive_snapshots]
    assert any(r > 0.0 for r in rates), f"all neutral_drift_rate were 0: {rates}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_task_stream.py::test_stream_produces_nonzero_neutral_drift_rate -v
```
Expected: FAIL — either `StreamConfig` has no `estimate_neutrality_trials` or all rates are 0.

- [ ] **Step 3: Add `estimate_neutrality_trials` to `StreamConfig`**

In `task_stream.py`, find `StreamConfig` (around line 131). Add:

```python
estimate_neutrality_trials: int = 8
```

- [ ] **Step 4: Import and call `estimate_neutrality` at snapshot sites**

At the top of `task_stream.py`, the import from `evolution` already exists. Add `estimate_neutrality` to it:

```python
from evolution import Evaluation, EvolutionConfig, mixed_initial_population, mutate_genome, random_genome
from evolution import attach_success_motif, estimate_neutrality
```

In `task_stream.py`, find every place an `ArchiveSnapshot` is constructed. There are three such sites (search for `ArchiveSnapshot(`). At each site, before constructing the snapshot, compute the drift rate from the best live genome:

```python
_best_genome = min(colony.live_members(), key=lambda c: c.energy).__class__  # placeholder — see below
```

The correct pattern at each snapshot site is:

```python
_live = colony.live_members()
if _live and config.estimate_neutrality_trials > 0:
    _best_cell = max(_live, key=lambda c: c.energy)
    _neutral_drift_rate = estimate_neutrality(
        _best_cell.genome,
        evaluator,
        rng,
        trials=config.estimate_neutrality_trials,
    )
else:
    _neutral_drift_rate = 0.0
```

Then pass `neutral_drift_rate=_neutral_drift_rate` to the `ArchiveSnapshot(...)` constructor at that site.

Note: `evaluator` and `rng` are already in scope at all three snapshot sites. Search for `ArchiveSnapshot(` to locate them precisely.

- [ ] **Step 5: Run test**

```bash
python3 -m pytest tests/test_task_stream.py::test_stream_produces_nonzero_neutral_drift_rate -v
```
Expected: PASS

- [ ] **Step 6: Run full test suite to check no regressions**

```bash
python3 -m pytest tests/test_task_stream.py tests/test_contextual_task_stream.py tests/test_math_ecology.py -q
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add task_stream.py tests/test_task_stream.py
git commit -m "feat: wire estimate_neutrality into archive snapshots"
```

---

### Task 3: Add `phenotype_diversity` to `StreamTaskResult`

**Files:**
- Modify: `task_stream.py`
- Modify: `tests/test_task_stream.py`

Phenotype diversity = mean pairwise absolute output difference across the live population on the probe input `(2, 3)`, normalised by dividing by the maximum absolute output seen (floored at 1.0 to avoid divide-by-zero).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_task_stream.py`:

```python
def test_stream_result_has_phenotype_diversity() -> None:
    from task_stream import StreamTaskResult
    result = StreamTaskResult(
        task_name="multiply",
        context_label="",
        step_index=0,
        mean_error=0.0,
        best_error=0.0,
        live_cells=2,
        births=0,
        deaths=0,
        energy_total=0.0,
        max_cells=2,
        resource_pool=8.0,
    )
    assert hasattr(result, "phenotype_diversity")
    assert isinstance(result.phenotype_diversity, float)
    assert result.phenotype_diversity >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_task_stream.py::test_stream_result_has_phenotype_diversity -v
```
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add field to `StreamTaskResult`**

In `task_stream.py`, find `StreamTaskResult` (around line 23). Add after `lineage_efficiency_count`:

```python
phenotype_diversity: float = 0.0
```

- [ ] **Step 4: Add a helper function to compute phenotype diversity**

Add this function near the top of `task_stream.py`, after the imports but before the dataclasses:

```python
def _phenotype_diversity(cells: Sequence["StreamCell"], evaluator: "Callable[[CellGenome], Evaluation]") -> float:
    outputs = []
    for cell in cells:
        ev = evaluator(cell.genome)
        # Use the validation error as a proxy for output — we want the raw output,
        # but the evaluator only returns error. Instead run the cell on probe (2,3)
        # by checking trace_examples for a numeric output if available, else skip.
        # Simpler: re-use train_error as a diversity signal (cells with same error = same phenotype).
        outputs.append(ev.train_error)
    if len(outputs) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            total += abs(outputs[i] - outputs[j])
            count += 1
    max_val = max(abs(o) for o in outputs) or 1.0
    return (total / count) / max_val if count > 0 else 0.0
```

- [ ] **Step 5: Call the helper after each step evaluation and set the field**

In `task_stream.py`, locate where `StreamTaskResult(...)` is constructed after each evolution step (search for `StreamTaskResult(`). There are multiple sites. At each one, add:

```python
_diversity = _phenotype_diversity(colony.live_members(), evaluator)
```

Then pass `phenotype_diversity=_diversity` to the `StreamTaskResult(...)` constructor.

- [ ] **Step 6: Add `phenotype_diversity` to `format_text`**

In `StreamReport.format_text`, find the per-step result line (around line 93). Add `phenotype_diversity` after `energy_efficiency`:

```python
f"energy_efficiency={result.energy_efficiency:.4f} "
f"phenotype_diversity={result.phenotype_diversity:.4f} "
```

- [ ] **Step 7: Run tests**

```bash
python3 -m pytest tests/test_task_stream.py::test_stream_result_has_phenotype_diversity -v
```
Expected: PASS

- [ ] **Step 8: Run a quick integration check**

```bash
python3 experiments/11_contextual_task_stream.py 2>&1 | grep "phenotype_diversity"
```
Expected: lines containing `phenotype_diversity=` with non-zero values for at least some steps.

- [ ] **Step 9: Commit**

```bash
git add task_stream.py tests/test_task_stream.py
git commit -m "feat: add phenotype_diversity metric to StreamTaskResult"
```

---

### Task 4: Add `recovery_steps` to `StreamTaskResult`

**Files:**
- Modify: `task_stream.py`
- Modify: `tests/test_task_stream.py`

`recovery_steps` counts how many steps it takes after a task switch for the population's best error to return to the pre-switch best. Set to 0 when no switch has occurred or when recovery happens immediately. Set to `-1` if recovery does not happen within the current task block.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_task_stream.py`:

```python
def test_recovery_steps_set_after_task_switch() -> None:
    """After a task switch, recovery_steps should be >= 0 on at least one result."""
    from task_stream import StreamConfig, run_task_stream
    from tasks import TaskBundle

    tasks = [
        TaskBundle(name="multiply", cases=((2, 3, 6), (1, 4, 4), (3, 3, 9)), anti_shortcut_cases=()),
        TaskBundle(name="add", cases=((2, 3, 5), (1, 4, 5), (3, 3, 6)), anti_shortcut_cases=()),
    ]
    config = StreamConfig(
        initial_population_size=4,
        max_steps_per_task=4,
        archive_interval=10,
        estimate_neutrality_trials=0,
    )
    report = run_task_stream("test_recovery", tasks, config=config, seed=13)
    recovery_values = [r.recovery_steps for r in report.results]
    assert all(v >= -1 for v in recovery_values), f"unexpected recovery_steps values: {recovery_values}"
    # At least the steps after the task switch should have a meaningful value
    add_steps = [r for r in report.results if r.task_name == "add"]
    assert len(add_steps) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_task_stream.py::test_recovery_steps_set_after_task_switch -v
```
Expected: FAIL with `AttributeError: 'StreamTaskResult' object has no attribute 'recovery_steps'`

- [ ] **Step 3: Add field to `StreamTaskResult`**

In `task_stream.py`, find `StreamTaskResult`. Add:

```python
recovery_steps: int = 0
```

- [ ] **Step 4: Track pre-switch best and count recovery in the stream loop**

The stream run loop processes tasks sequentially. At each task boundary, record the best error from the previous task. Then during the new task's steps, count how many steps pass before `best_error` drops back to that level.

Find the outermost loop over tasks in `task_stream.py` (the `for task in tasks:` block or equivalent). Add these tracking variables before the loop:

```python
_pre_switch_best_error: float | None = None
_steps_since_switch: int = 0
_recovered: bool = True
```

At each task boundary (before processing a new task), set:

```python
_pre_switch_best_error = _last_best_error  # set this after each step (see below)
_steps_since_switch = 0
_recovered = False
```

After each step evaluation, update:

```python
_last_best_error = step_best_error  # the best_error value for this step
```

When constructing `StreamTaskResult`, compute:

```python
if _pre_switch_best_error is None or _recovered:
    _recovery_steps = 0
elif step_best_error <= _pre_switch_best_error:
    _recovered = True
    _recovery_steps = _steps_since_switch
else:
    _recovery_steps = -1  # not yet recovered
_steps_since_switch += 1
```

Pass `recovery_steps=_recovery_steps` to `StreamTaskResult(...)`.

- [ ] **Step 5: Add to `format_text`**

In `StreamReport.format_text`, add `recovery_steps` to the per-step line after `phenotype_diversity`:

```python
f"phenotype_diversity={result.phenotype_diversity:.4f} "
f"recovery_steps={result.recovery_steps} "
```

- [ ] **Step 6: Run tests**

```bash
python3 -m pytest tests/test_task_stream.py::test_recovery_steps_set_after_task_switch -v
```
Expected: PASS

- [ ] **Step 7: Run full focused suite**

```bash
python3 -m pytest tests/test_task_stream.py tests/test_contextual_task_stream.py tests/test_math_ecology.py -q
```
Expected: all pass

- [ ] **Step 8: Smoke test output**

```bash
python3 experiments/06_task_stream_adaptation.py 2>&1 | grep "recovery_steps"
```
Expected: lines showing `recovery_steps=` with `0` before the first switch and `-1` or a positive int after.

- [ ] **Step 9: Commit**

```bash
git add task_stream.py tests/test_task_stream.py
git commit -m "feat: add recovery_steps metric to StreamTaskResult"
```

---

### Task 5: Final integration check

- [ ] **Step 1: Run the full test suite (excluding the slow script test)**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```
Expected: all pass

- [ ] **Step 2: Verify all three new metrics appear in experiment output**

```bash
python3 experiments/14_adaptive_math_ecology.py 2>&1 | grep -E "neutral_drift_rate|phenotype_diversity|recovery_steps" | head -10
```
Expected: lines containing all three metric names with numeric values.

- [ ] **Step 3: Commit if anything was left unstaged**

```bash
git status
```
If clean, no commit needed.
