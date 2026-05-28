# Phase C — Spatial-Ecology Integration Design

Date: 2026-05-28

## Summary

Phase C unifies the two independent halves of the AI-DNA codebase: the energy-based ecology in `task_stream.py` / `colony.py` and the spatial development engine in `spatial.py` / `spatial_routing.py`. Spatial cells currently develop structure across experiments 15–24 but never pay maintenance costs, reproduce under energy pressure, or respond to changing task environments. This phase adds energy bookkeeping to spatial cells, drives spatial arenas through a changing sequence of target body plans, revisits matrix-fabric convergence with sharper evaluation, and culminates in a single cross-layer experiment that exercises all four systems simultaneously.

## What Already Exists (do not re-implement)

- `SpatialArena`, `SpatialCell`, `MorphogenField`, `run_spatial_development` — full spatial execution engine in `spatial.py`.
- `SPATIAL_OPS` vocabulary including `DIVIDE_*`, `KILL`, `HALT`, `REPAIR_*`, `ADHERE`, `WANDER`, `EMIT_0`, `SENSE_*`.
- `SpatialArena.spawn_child`, `move_cell`, `_adhere`, `_wander` — positional reproduction and movement already implemented.
- `StreamCell`, `StreamConfig`, `StreamReport`, `StreamTaskResult`, `ArchiveSnapshot`, `TaskStreamColony` — full energy/ecology/stream infrastructure in `task_stream.py`.
- `Colony.advance` with `select_top_diverse`, `crossover_genomes`, `mutate_genome` — genome-level evolution in `colony.py` / `evolution.py`.
- `MatrixFabricTarget`, `MatrixFabricEvaluation`, `spatial_matrix_fabric.py` — spatial routing and matrix convergence scaffold.
- `SpatialRoutingTarget`, `SpatialRoutingSearchConfig`, `run_spatial_routing_search` — routing search in `spatial_routing.py`.
- Phase A metrics (`neutral_drift_rate`, `phenotype_diversity`, `recovery_steps`) and Phase B search improvements (`sharing_radius`, `motif_crossover_bias`, `adaptive_mutation`, `qd_archive`) — assumed complete before Phase C implementation begins.

## What Is Missing

### 1. Spatial cells in ecological competition

**Gap.** `SpatialCell` has no `energy` field, no maintenance cost, no reward for fitness, no reproduction threshold, and no death-by-starvation. `SpatialArena.step` develops cells but never applies selection pressure. The genome-level evolver in `colony.py` does not know about spatial positions or morphogen state.

**Design.** Add `energy: float = 6.0` and `alive` (already present) semantics to `SpatialCell`. After each arena development round, run a survival step:

1. Each live cell pays `maintenance_cost` from its energy.
2. Each cell gains `reward_scale * fitness` energy, where `fitness` is a callable that maps a `SpatialCell` to a float in [0, 1] — supplied by the caller, not hardcoded.
3. Any cell whose energy falls to or below `death_threshold` is marked `alive = False` and pruned.
4. Any cell whose energy exceeds `reproduction_threshold` attempts `spawn_child` in a random free cardinal direction, paying `spawn_cost`, and the child inherits `initial_energy`.

**New class: `SpatialColony`** (new file `spatial_colony.py`).

```python
@dataclass(slots=True)
class SpatialColonyConfig:
    maintenance_cost: float = 0.5
    spawn_cost: float = 2.0
    reproduction_threshold: float = 8.0
    initial_energy: float = 6.0
    reward_scale: float = 4.0
    death_threshold: float = 0.0
    development_steps_per_round: int = 8
    mutation_rate: float = 0.08
    max_cells: int = 64

@dataclass(slots=True)
class SpatialColony:
    arena: SpatialArena
    config: SpatialColonyConfig
    rng: Random
    generation: int = 0
    energies: dict[tuple[int, int], float] = field(default_factory=dict)

    def advance(
        self,
        fitness_fn: Callable[[SpatialCell, SpatialArena], float],
    ) -> SpatialColonyReport: ...
```

`SpatialColony.advance`:
1. Runs `arena.step()` for `development_steps_per_round` ticks.
2. Applies the survival step described above.
3. For cells that reproduced, the child genome is mutated via `mutate_genome` (existing) using `config.mutation_rate`.
4. Returns a `SpatialColonyReport` with live cell count, mean/best fitness, births, deaths, energy totals, and generation number.

**`SpatialColonyReport`** (in `spatial_colony.py`):

```python
@dataclass(slots=True)
class SpatialColonyReport:
    generation: int
    live_cells: int
    births: int
    deaths: int
    mean_fitness: float
    best_fitness: float
    energy_total: float
    morphogen_peak: float

    def format_text(self) -> str: ...
```

**Backward compatibility.** `spatial.py` is unchanged. `SpatialCell` gains `energy: float = 6.0` as a new optional field with a default so existing construction sites (experiments 15–24, `run_spatial_development`) continue to work. The survival step is only invoked when using `SpatialColony`, not during plain `SpatialArena.run`.

**File:** `spatial_colony.py` (new). `spatial.py` gains only the `energy` field on `SpatialCell`.

---

### 2. Changing spatial task streams

**Gap.** All spatial experiments run a single fixed target body plan for a fixed number of steps. There is no equivalent of experiments 11–14 where the environment changes and cells must adapt. Specifically, there is no `StreamConfig`-driven infrastructure for spatial arenas, no task-switch event, and no reporting of `forgetting_delta` or `recovery_steps` in spatial context.

**Design.** Create `SpatialStreamConfig` (in `spatial_colony.py`) that extends `SpatialColonyConfig` with stream parameters:

```python
@dataclass(frozen=True, slots=True)
class SpatialStreamConfig:
    colony: SpatialColonyConfig = field(default_factory=SpatialColonyConfig)
    steps_per_task: int = 12
    snapshot_interval: int = 4
```

Define at least three spatial tasks as named body-plan targets, implemented as fitness functions:

| Task name | Body-plan description | Fitness signal |
|---|---|---|
| `"stripe_h"` | Cells form a horizontal band across the arena centre (y in [H//2-1, H//2+1]) | Fraction of live cells in target band |
| `"stripe_v"` | Cells form a vertical band across arena centre (x in [W//2-1, W//2+1]) | Fraction of live cells in target band |
| `"cluster_nw"` | Cells cluster in the NW quadrant (x < W//2, y < H//2) | Fraction of live cells in quadrant |

These tasks form a meaningful adaptation sequence: stripe_h → stripe_v (requires positional reorientation) → cluster_nw (requires both axes to change).

**New experiment `experiments/25_spatial_task_stream.py`:**
- Creates a `SpatialColony` with a single seed genome.
- Runs through the three tasks in sequence using `SpatialStreamConfig.steps_per_task` rounds each.
- At each task boundary, records `forgetting_delta` (best fitness before switch minus best fitness one round after switch) and counts `recovery_steps` (rounds until best fitness recovers to pre-switch level).
- Prints a `SpatialStreamReport` (new dataclass in `spatial_colony.py`) mirroring `StreamReport.format_text` format, prefixed `"spatial_stream:"`.

**`SpatialStreamReport`** fields: `sequence_name`, `results: tuple[SpatialStreamTaskResult, ...]`, `config`.
**`SpatialStreamTaskResult`** fields: `task_name`, `round_index`, `mean_fitness`, `best_fitness`, `live_cells`, `births`, `deaths`, `forgetting_delta`, `recovery_steps`.

**File:** `spatial_colony.py` (additions). `experiments/25_spatial_task_stream.py` (new).

---

### 3. Spatial matrix fabric convergence

**Gap.** The matrix-fabric benchmark in `spatial_matrix_fabric.py` is described as "still partial" — `MatrixFabricEvaluation.exact_match` is never `True` in practice. Three specific deficiencies cause this:

1. **Population too small.** Current search uses a default population that is too small to maintain diversity across the large spatial search space (4×4 fabric on a 16×16 grid). Exact convergence requires at least 32 candidates per generation.
2. **Evaluation is too coarse.** `layout_error`, `input_error`, and `output_error` are weighted but `matrix_error` (the actual computation correctness) dominates only at the end. Early search has no gradient toward correct routing because partial coverage scores the same as random coverage once layout is satisfied. A curriculum is needed: first reward cell placement, then reward routing connectivity, then reward correct output values.
3. **No mutation diversity pressure.** The search mutates from a single parent without crossover between distinct surviving fabric configurations. Different candidate fabrics solving different sub-problems (row routing vs. column routing) are never recombined.

**Spec changes to `spatial_matrix_fabric.py`:**

Add `MatrixFabricSearchConfig` field:
```python
population_size: int = 32          # was implicit ~8
curriculum_phase: int = 0          # 0=layout, 1=routing, 2=full
phase_thresholds: tuple[float, float] = (0.8, 0.9)
```

**Curriculum scoring** — `score_fabric_evaluation` returns a weighted composite that shifts weight by `curriculum_phase`:
- Phase 0 (layout): `score = layout_error` only. Advance to phase 1 when mean `layout_error < phase_thresholds[0]`.
- Phase 1 (routing): `score = 0.4 * layout_error + 0.6 * (input_error + role_error)`. Advance to phase 2 when mean routing score `< phase_thresholds[1]`.
- Phase 2 (full): `score = 0.15 * layout_error + 0.25 * role_error + 0.6 * matrix_error`.

`MatrixFabricSearchConfig` also adds:
```python
crossover_rate: float = 0.3        # crossover between top-2 fabric genomes
sharing_radius: float = 0.15       # Phase B fitness sharing for diversity
```

The search loop in `run_matrix_fabric_search` calls `select_top_diverse` with `sharing_radius` (Phase B feature), and performs crossover between the top two survivors when `rng.random() < crossover_rate`.

**Success definition**: `exact_match = True` is set when `matrix_error < 0.01` across all test cases in `MatrixFabricTarget.test_cases`. This threshold must be explicit in the evaluation function (currently it is never checked against a real tolerance).

**File:** `spatial_matrix_fabric.py` (additions to `MatrixFabricSearchConfig`, `score_fabric_evaluation`, and the main search loop). No new files.

---

### 4. Cross-layer integration experiment

**Gap.** No single experiment exercises chemistry signal rules + energy-based ecology + spatial development + changing task streams together. This is the north-star milestone: a multicellular spatial organism that evolves under a changing mathematical environment.

**Design.** New experiment `experiments/26_spatial_ecology_stream.py`:

```
Layer stack (bottom to top):
  chemistry     — MorphogenField diffusion/decay (already in SpatialArena)
  spatial cells — SpatialCell with energy fields (Phase C step 1)
  ecology       — SpatialColony survival/reproduction loop (Phase C step 1)
  task stream   — three-task sequence from Phase C step 2 (stripe_h → stripe_v → cluster_nw)
```

The experiment:
1. Initialises a `SpatialColony` (8×8 arena, `SpatialColonyConfig` defaults) seeded with 3 cells, each carrying the same base genome.
2. Runs through the three spatial task stream targets defined in step 2 using `SpatialStreamConfig`.
3. At each round, the morphogen field provides spatial context: cells sense the gradient and use it to bias division direction (existing `SENSE_GRAD_X_0` / `SENSE_GRAD_Y_0` ops).
4. After every task switch, logs `forgetting_delta` and `recovery_steps` to the report.
5. Genome mutation uses Phase B `adaptive_mutation` when `SpatialColonyConfig.adaptive_mutation: bool = False` (opt-in). To enable it the experiment sets `adaptive_mutation=True` and `mutation_rate=0.08`.
6. Prints a `SpatialStreamReport` (from step 2) with prefix `"spatial_ecology_stream:"` and also prints final arena layout as ASCII art (type_id 0 = `.`, type_id 1 = `#`, empty = ` `).

**ASCII art layout format** (new method `SpatialArena.format_ascii`) — one character per cell, rows separated by newlines, 1-char border of spaces.

**File:** `experiments/26_spatial_ecology_stream.py` (new). `spatial.py` gains `SpatialArena.format_ascii` method.

---

## Files Changed

| File | Change |
|---|---|
| `spatial.py` | Add `energy: float = 6.0` field to `SpatialCell`; add `SpatialArena.format_ascii() -> str` method |
| `spatial_colony.py` | New file: `SpatialColonyConfig`, `SpatialColony`, `SpatialColonyReport`, `SpatialStreamConfig`, `SpatialStreamTaskResult`, `SpatialStreamReport` |
| `spatial_matrix_fabric.py` | Add `population_size`, `curriculum_phase`, `phase_thresholds`, `crossover_rate`, `sharing_radius` to `MatrixFabricSearchConfig`; add curriculum scoring logic; fix `exact_match` threshold; wire `select_top_diverse` with `sharing_radius`; add crossover between top survivors |
| `experiments/25_spatial_task_stream.py` | New experiment: three-task spatial stream with forgetting/recovery reporting |
| `experiments/26_spatial_ecology_stream.py` | New experiment: full cross-layer integration (chemistry + ecology + spatial + stream) |

## Out of Scope for Phase C

- Island-model parallelism across multiple independent spatial arenas.
- 3D spatial ecology (`spatial3d.py` is not modified).
- Per-lineage QD grid for spatial body plans (QD grid remains in `task_stream.py` only).
- Cross-task motif recombination between spatial and arithmetic genomes.
- Automated benchmark regression in CI (manual benchmark only, results recorded in README).
- Visualisation beyond ASCII art (no matplotlib, no animation).
- Spatial routing search (`spatial_routing.py`) changes — routing targets are a separate concern from body-plan task streams.

## Success Criteria

1. All existing tests pass. `experiments/15` through `experiments/24` produce identical output to pre-Phase-C baseline (backward compatibility of `SpatialCell.energy` default confirmed).
2. `experiments/25_spatial_task_stream.py` runs to completion, prints `"spatial_stream:"` output with three named task blocks, and reports non-zero `forgetting_delta` on at least one task switch.
3. `experiments/26_spatial_ecology_stream.py` runs to completion, prints `"spatial_ecology_stream:"` output, and the ASCII art shows at least two distinct cell types in the final arena.
4. Running `spatial_matrix_fabric.py` search with `curriculum_phase=0` auto-advancing to phase 2 achieves `exact_match=True` on at least one candidate within 200 generations (manual benchmark, not automated).
5. New unit tests in `tests/test_spatial_colony.py`:
   - `test_spatial_colony_energy_maintenance`: after one round with no fitness reward, all cells have energy reduced by `maintenance_cost`.
   - `test_spatial_colony_death_at_zero_energy`: a cell starting at `energy=0.0` is pruned after one survival step.
   - `test_spatial_colony_reproduction_above_threshold`: a cell starting above `reproduction_threshold` spawns a child and pays `spawn_cost`.
   - `test_spatial_stream_forgetting_delta_nonnegative`: running two-task stream always yields `forgetting_delta >= 0`.
6. `SpatialArena.format_ascii` produces a string with exactly `height` newline-separated rows each of length `width`.
