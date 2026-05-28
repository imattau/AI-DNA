# Phase A — Observable Metrics Design

Date: 2026-05-28

## Summary

Add four missing metrics to the AI-DNA experiment reporting layer so future search-quality work (Phase B) can be benchmarked objectively. The data is largely already being produced; this phase wires it up and surfaces it.

## What Already Exists (do not re-implement)

- `forgetting_delta` — computed in `task_stream.py` as the drop in best transfer score after a task switch. Already in `StreamTaskResult` and `ArchiveSnapshot`.
- `lineage_transfer_scores` — per-lineage transfer already computed and printed.
- `estimate_neutrality()` — exists in `evolution.py` but is **never called** from `task_stream.py`. Just needs wiring.
- `neutrality_estimate` field on `Evaluation` — exists but always 0.0 in practice.

## What Is Missing

### 1. Neutral drift rate in stream output

`estimate_neutrality()` is implemented but not called during stream runs. It should be sampled on the best genome at each archive snapshot interval and reported as `neutral_drift_rate` in `ArchiveSnapshot`.

**Where:** `task_stream.py` — at the point archive snapshots are taken, call `estimate_neutrality` on the best-scoring live genome. Use the existing `evaluator` in scope. Add `neutral_drift_rate: float = 0.0` to `ArchiveSnapshot` and surface it in `format_text`.

**Cost:** `trials=8` per snapshot — cheap. Gate it behind an `estimate_neutrality_trials: int = 8` field on `StreamConfig` (0 to disable).

### 2. Population diversity per generation

No diversity metric exists anywhere. Add a per-step phenotype diversity measure to `StreamTaskResult`.

**Definition:** phenotype diversity = mean pairwise absolute output difference across the live population on a fixed probe input (e.g., inputs `(2, 3)` for multiply). Normalise to [0, 1] by dividing by the max observed output magnitude.

**Where:** compute in `task_stream.py` after each step evaluation loop. Add `phenotype_diversity: float = 0.0` to `StreamTaskResult` and print it in `format_text`.

**Why pairwise output difference, not genotype Hamming:** outputs are what selection sees. Genotype diversity can be high while phenotype diversity is zero (synonymous codons). We want to know if the population has collapsed to one behavioural attractor.

### 3. Recovery steps after task switch

`forgetting_delta` is a one-step delta. It tells you how much performance dropped but not how long recovery takes. Add `recovery_steps: int = 0` to `StreamTaskResult` — the number of steps after a task switch until the population re-reaches its pre-switch best error.

**Where:** in `task_stream.py`, track `pre_switch_best_error` at each task boundary. After a switch, count steps until `best_error <= pre_switch_best_error`. Set `recovery_steps` on the first result after recovery (or on the last result of that task block if recovery doesn't happen).

**Note:** `forgetting_delta` stays as-is; `recovery_steps` is additive.

### 4. Fix test_experiments.py assertion (already done)

The assert on line 43 now accepts `"spatial_development:"` and `"spatial3d_development:"` prefixes. No further work needed here.

## Files Changed

| File | Change |
|---|---|
| `task_stream.py` | Wire `estimate_neutrality`, add `neutral_drift_rate` to `ArchiveSnapshot`; add `phenotype_diversity` and `recovery_steps` to `StreamTaskResult`; update `format_text` for both |
| `evolution.py` | No change needed — `estimate_neutrality` already correct |
| `tracing.py` | No change needed |
| `tests/test_task_stream.py` | Add assertions that new fields are non-negative floats; add a task-switch scenario and assert `recovery_steps >= 0` |
| `tests/test_experiments.py` | Already fixed |

## Out of Scope for Phase A

- Motif origin tracking improvements (checklist item 2 refinements) — already partially implemented, leave as-is
- Per-lineage forgetting (each lineage tracked separately) — Phase B
- Diversity histogram (full distribution) — Phase B tooling

## Success Criteria

1. All existing tests pass.
2. Running `experiments/11_contextual_task_stream.py` or `experiments/14_adaptive_math_ecology.py` prints non-zero `neutral_drift_rate`, `phenotype_diversity`, and `recovery_steps` for at least some steps.
3. New test coverage in `test_task_stream.py` passes.
