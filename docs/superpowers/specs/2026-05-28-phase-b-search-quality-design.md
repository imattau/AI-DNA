# Phase B — Search Quality Design

Date: 2026-05-28

## Summary

Add four independently-toggleable search improvements to `evolution.py`, `colony.py`, and `task_stream.py`. The goal is to make the evolver smarter on hard targets (exp 09 `matrix_multiplication_search`, exp 14 `adaptive_math_ecology`) without seeding full solutions. All changes are backward-compatible: new parameters default to current behaviour.

## Reference Experiments

- **Exp 09** `experiments/09_matrix_multiplication_search.py` — hard arithmetic target, long convergence plateau.
- **Exp 14** `experiments/14_adaptive_math_ecology.py` — task-switching ecology, population collapses after switches.

Phase A metrics (`neutral_drift_rate`, `phenotype_diversity`, `recovery_steps`) provide the quantitative signal for benchmarking.

## What Already Exists (do not re-implement)

- `select_top_diverse` in `evolution.py` — deduplicates by genome signature but applies no crowding penalty.
- `crossover_genomes` in `evolution.py` — delegates to `crossover_codons`, which performs a random splice with no boundary awareness.
- `mutate_genome` / `spawn_sibling_variants` — accept a fixed `mutation_rate`; no per-lineage adaptation.
- `estimate_neutrality` in `evolution.py` — implemented and correct; used by Phase A to compute `neutral_drift_rate`.
- `EvolutionConfig` dataclass — holds scalar mutation/crossover rates; no adaptive flags.
- `local_motifs: tuple[Motif, ...]` on `CellGenome` — motifs carry `origin_lineage` and `origin_task`; `declare_rules()` returns rule names.
- `TaskStreamColony` and `StreamConfig` in `task_stream.py` — manage stream-mode evolution; hold an energy-ranked archive.

## Four Deliverables

---

### 1. Fitness Sharing in `select_top_diverse`

**Problem.** The current selection deduplicates only by exact genome signature. Two candidates with distinct signatures but nearly identical error scores (same behavioural niche) both survive, crowding out candidates from different niches.

**Design.** Add a `sharing_radius: float = 0.0` parameter to `select_top_diverse`. When non-zero, apply a niche penalty before ranking:

```
effective_score(c) = raw_score(c) + sum over already-selected s:
    max(0, sharing_strength * (1 - |score_c - score_s| / sharing_radius))
```

where `sharing_strength: float = 1.0` is a second new parameter. Candidates whose score is within `sharing_radius` of an already-selected survivor receive a penalty proportional to closeness. The loop iterates ranked candidates in order of raw score and applies the penalty before deciding whether to accept each one.

**Signature change.**

```python
def select_top_diverse(
    evaluations: Sequence[Evaluation],
    *,
    k: int,
    sharing_radius: float = 0.0,
    sharing_strength: float = 1.0,
) -> tuple[Evaluation, ...]:
```

Defaults of `sharing_radius=0.0` reproduce current behaviour exactly (no penalty applied).

**Where:** `evolution.py` only. No changes to callers (`colony.py`, `task_stream.py`) needed for backward compatibility — they call without keyword args.

**Config surface.** Add to `EvolutionConfig`:
```python
sharing_radius: float = 0.0
sharing_strength: float = 1.0
```

`Colony.advance` passes these through when calling `select_top_diverse`.

---

### 2. Motif-Aware Crossover in `crossover_genomes`

**Problem.** `crossover_codons` picks a random splice point, which frequently cuts through the middle of a functional motif, producing broken offspring and increasing the cost of crossover.

**Design.** When both parents have `local_motifs`, compute motif boundary positions (the codon index immediately after each motif's last codon) and bias the crossover point to land at one of those positions.

Algorithm:
1. Collect boundary indices from `left.local_motifs` and `right.local_motifs` by walking `left.codons` and `right.codons` and matching rule names to codon spans.
2. For `left`, candidate splice points are its motif boundaries plus position 0 and `len(left.codons)`.
3. With probability `motif_crossover_bias` (new parameter, default `0.0`), choose a splice point uniformly from the left-parent boundary set; otherwise fall back to `crossover_codons` (random splice).
4. At the chosen splice point on `left`, take codons `left.codons[:split]`; pair with a randomly-chosen suffix from `right.codons[r_split:]` where `r_split` is the nearest motif boundary in `right` to the proportional position `split / len(left.codons)`.
5. Combine: `child_codons = left.codons[:split] + right.codons[r_split:]`.
6. Inherit motifs: `left.local_motifs + right.local_motifs` (unchanged from current).

**Signature change.**

```python
def crossover_genomes(
    left: CellGenome,
    right: CellGenome,
    rng: Random,
    *,
    lineage_id: str,
    motif_crossover_bias: float = 0.0,
) -> CellGenome:
```

Default `motif_crossover_bias=0.0` falls through to `crossover_codons`, preserving current behaviour.

**Config surface.** Add to `EvolutionConfig`:
```python
motif_crossover_bias: float = 0.0
```

`Colony.advance` passes this to `crossover_genomes` calls (lines 81–86 of current `colony.py`).

**Fallback rule.** If either parent has no `local_motifs`, or if boundary extraction yields fewer than 2 positions, fall back to `crossover_codons` regardless of `motif_crossover_bias`.

---

### 3. Adaptive Mutation Rate per Lineage in `task_stream.py`

**Problem.** Every lineage uses the same `config.mutation_rate`. Lineages with high `neutral_drift_rate` are in a flat fitness landscape — they benefit from a higher rate to escape. High-performing lineages near a fitness peak need a lower rate to avoid disrupting good solutions.

**Design.** Add `adaptive_mutation: bool = False` to `StreamConfig`. When enabled, before spawning offspring for each live cell, compute a per-cell mutation rate multiplier:

```
multiplier = clamp(base * (1 + alpha * neutral_drift_rate - beta * fitness_rank_fraction), lo, hi)
```

where:
- `base` = `config.mutation_rate` (unchanged default path)
- `neutral_drift_rate` = Phase A metric, sampled at snapshot intervals and stored per-lineage in `StreamTaskResult`
- `fitness_rank_fraction` = rank of this cell in the current generation / population size (0 = best, 1 = worst)
- `alpha: float = 2.0`, `beta: float = 0.5` — tuning constants, exposed as `StreamConfig` fields
- `lo: float = 0.01`, `hi: float = 0.25` — clamp bounds, exposed as `StreamConfig` fields

Add to `StreamConfig`:
```python
adaptive_mutation: bool = False
adaptive_mutation_alpha: float = 2.0
adaptive_mutation_beta: float = 0.5
adaptive_mutation_lo: float = 0.01
adaptive_mutation_hi: float = 0.25
```

**Where the multiplier is applied.** In `task_stream.py`, in the offspring-spawning loop that calls `mutate_genome` or `spawn_sibling_variants`. Compute the per-cell rate, construct a local `EvolutionConfig` copy with `mutation_rate` overridden, pass it to the spawn call. Do not mutate `StreamConfig` in place.

**Neutral drift source.** Use the `neutral_drift_rate` already computed by Phase A (`ArchiveSnapshot.neutral_drift_rate`). Cache the last snapshot value per lineage in a `dict[str, float]` local to the advance loop. If no snapshot value exists yet for a lineage, use `0.0` (falls back to `base` rate).

**Backward compatibility.** `adaptive_mutation=False` (default) leaves all mutation rates identical to current behaviour.

---

### 4. Quality-Diversity (MAP-Elites) Archive in `task_stream.py`

**Problem.** The current archive is a flat energy-ranked list. It keeps only the globally best genomes, discarding solutions that are good in a different behavioural regime. This causes convergence to a single attractor on multi-modal problems like exp 09.

**Design.** Maintain a behavioural grid alongside the existing archive. Enable via `qd_archive: bool = False` on `StreamConfig`.

**Grid axes (2D):**
- Axis 0 — `error_bucket`: discretised `best_error`, `n_error_buckets` evenly-spaced bins over `[0, error_axis_max]`.
- Axis 1 — `diversity_bucket`: discretised `phenotype_diversity` (Phase A metric), `n_diversity_buckets` bins over `[0, 1]`.

Add to `StreamConfig`:
```python
qd_archive: bool = False
n_error_buckets: int = 10
n_diversity_buckets: int = 5
error_axis_max: float = 100.0
```

**Data structure.** `dict[tuple[int, int], Evaluation]` — maps grid cell to the best `Evaluation` seen for that cell (best = lowest `.score`). Store as `_qd_grid` on the `TaskStreamColony` instance (not exposed in `StreamConfig`).

**Update rule.** After each generation's evaluation loop, for each `Evaluation` in the current population:
1. Compute `(eb, db)` from `error` and current `phenotype_diversity`.
2. If `(eb, db)` not in `_qd_grid`, insert.
3. If `(eb, db)` in `_qd_grid` and new `.score < existing.score`, replace.

**Selection from grid.** When `qd_archive=True`, replace the energy-ranked archive draw with a two-phase draw:
1. With probability `qd_exploit_fraction` (new `StreamConfig` field, default `0.7`), draw from the standard archive (current behaviour) — exploit known good solutions.
2. With probability `1 - qd_exploit_fraction`, draw uniformly from a random occupied QD cell — explore novel behavioural niches.

Add to `StreamConfig`:
```python
qd_exploit_fraction: float = 0.7
```

**Reporting.** Add `qd_cells_occupied: int = 0` to `ArchiveSnapshot`. When `qd_archive=True`, set it to `len(_qd_grid)`. Print in `format_text` alongside other snapshot fields.

**Backward compatibility.** `qd_archive=False` (default) leaves archive behaviour identical to current.

---

## Files Changed

| File | Changes |
|---|---|
| `evolution.py` | `select_top_diverse`: add `sharing_radius`, `sharing_strength` params + penalty logic. `crossover_genomes`: add `motif_crossover_bias` param + boundary-aware splice. `EvolutionConfig`: add `sharing_radius`, `sharing_strength`, `motif_crossover_bias` fields. |
| `colony.py` | `Colony.advance`: pass `sharing_radius`, `sharing_strength` to `select_top_diverse`; pass `motif_crossover_bias` to `crossover_genomes`. Read from `config` fields. |
| `task_stream.py` | `StreamConfig`: add `adaptive_mutation*` and `qd_archive*` fields. `TaskStreamColony`: add `_qd_grid` dict, per-lineage neutral-drift cache, adaptive rate logic, QD update + draw logic. `ArchiveSnapshot`: add `qd_cells_occupied`. Update `format_text`. |

No new files required.

## Out of Scope for Phase B

- Per-lineage forgetting tracked separately in the QD grid — Phase C.
- Island-model parallelism across multiple colonies — Phase C.
- Crossover between motifs from different tasks (cross-task recombination) — Phase C.
- Diversity histogram output — Phase C tooling.

## Success Criteria

1. All existing tests pass with default config (no behaviour change).
2. New unit tests in `tests/test_evolution.py`:
   - `test_fitness_sharing_penalises_crowded_niche`: two candidates with close scores — with `sharing_radius > 0`, the second is displaced by a more diverse third candidate.
   - `test_motif_crossover_respects_boundaries`: crossover with `motif_crossover_bias=1.0` always splices at a motif boundary when boundaries exist.
   - `test_adaptive_mutation_scales_with_neutrality`: high `neutral_drift_rate` lineage receives mutation rate above `base`; low-rank lineage receives rate below `base`.
3. New unit tests in `tests/test_task_stream.py`:
   - `test_qd_archive_populates_grid`: after one generation, `_qd_grid` has at least one entry.
   - `test_qd_archive_replaces_on_improvement`: inserting a better evaluation into an occupied cell replaces it.
   - `test_qd_cells_occupied_reported`: `ArchiveSnapshot.qd_cells_occupied > 0` when `qd_archive=True`.
4. Benchmark (manual, not automated): running exp 09 with `sharing_radius=0.5`, `motif_crossover_bias=0.5`, `qd_archive=True` converges to error < 1.0 in fewer generations than the baseline. Running exp 14 with `adaptive_mutation=True` shows lower `recovery_steps` than baseline. Record results in README under "Measured Results".
