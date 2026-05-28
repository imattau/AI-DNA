# Phase C — Spatial-Ecology Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring spatial organisms into the full energy/ecology loop so they evolve under changing tasks, not just develop structure.
**Architecture:** New `spatial_colony.py` adds energy bookkeeping to `SpatialArena`; `spatial_matrix_fabric.py` gets curriculum scoring and larger population; two new experiments wire everything together.
**Tech Stack:** Python 3.10+, dataclasses, pytest

---

## File Map

- Modify: `spatial.py` — add `energy: float = 6.0` to `SpatialCell`; add `SpatialArena.format_ascii() -> str`
- Create: `spatial_colony.py` — `SpatialColonyConfig`, `SpatialColony`, `SpatialColonyReport`, `SpatialStreamConfig`, `SpatialStreamTaskResult`, `SpatialStreamReport`
- Modify: `spatial_matrix_fabric.py` — `MatrixFabricSearchConfig` new fields; curriculum scoring; crossover between top survivors
- Create: `experiments/25_spatial_task_stream.py` — three-task spatial stream experiment
- Create: `experiments/26_spatial_ecology_stream.py` — cross-layer integration experiment
- Create: `tests/test_spatial_colony.py` — unit tests for Tasks 1 and 2

---

## Task 1: `SpatialCell` energy fields + `SpatialColony` survival loop

**Files:**
- Modify: `spatial.py`
- Create: `spatial_colony.py`
- Create: `tests/test_spatial_colony.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spatial_colony.py`:

```python
from __future__ import annotations

import pytest
from spatial import SpatialArena, SpatialCell, build_spatial_genome
from spatial_colony import SpatialColony, SpatialColonyConfig


def _make_arena_with_cells(n: int, energy: float) -> SpatialArena:
    arena = SpatialArena(width=8, height=8)
    genome = build_spatial_genome(("HALT",), lineage_id="T0")
    for i in range(n):
        cell = SpatialCell(genome=genome, x=i, y=0, energy=energy)
        arena.place(cell)
    return arena


def test_spatial_colony_energy_maintenance() -> None:
    """After one advance with zero fitness reward, all cells lose exactly maintenance_cost energy."""
    from random import Random
    config = SpatialColonyConfig(
        maintenance_cost=0.5,
        reward_scale=0.0,
        death_threshold=-999.0,
        reproduction_threshold=999.0,
        initial_energy=6.0,
        development_steps_per_round=1,
    )
    arena = _make_arena_with_cells(2, energy=6.0)
    colony = SpatialColony(arena=arena, config=config, rng=Random(1))
    report = colony.advance(fitness_fn=lambda cell, arena: 0.0)
    assert report.live_cells == 2
    for cell in arena.cells.values():
        assert abs(cell.energy - 5.5) < 1e-9, f"Expected 5.5, got {cell.energy}"


def test_spatial_colony_death_at_zero_energy() -> None:
    """A cell starting at energy=0.0 is pruned after one survival step."""
    from random import Random
    config = SpatialColonyConfig(
        maintenance_cost=0.5,
        reward_scale=0.0,
        death_threshold=0.0,
        reproduction_threshold=999.0,
        initial_energy=6.0,
        development_steps_per_round=1,
    )
    arena = _make_arena_with_cells(1, energy=0.0)
    colony = SpatialColony(arena=arena, config=config, rng=Random(2))
    report = colony.advance(fitness_fn=lambda cell, arena: 0.0)
    assert report.deaths >= 1
    assert report.live_cells == 0


def test_spatial_colony_reproduction_above_threshold() -> None:
    """A cell starting above reproduction_threshold spawns a child and pays spawn_cost."""
    from random import Random
    config = SpatialColonyConfig(
        maintenance_cost=0.0,
        reward_scale=0.0,
        death_threshold=-999.0,
        reproduction_threshold=5.0,
        spawn_cost=2.0,
        initial_energy=6.0,
        development_steps_per_round=1,
        max_cells=64,
    )
    arena = SpatialArena(width=8, height=8)
    genome = build_spatial_genome(("HALT",), lineage_id="T1")
    cell = SpatialCell(genome=genome, x=4, y=4, energy=8.0)
    arena.place(cell)
    colony = SpatialColony(arena=arena, config=config, rng=Random(3))
    report = colony.advance(fitness_fn=lambda cell, arena: 1.0)
    assert report.births >= 1
    parent = next(
        c for c in arena.cells.values() if c.lineage_id.startswith("T1") and "." not in c.lineage_id.replace("T1", "")
    )
    assert parent.energy < 8.0  # paid spawn_cost


def test_spatial_colony_high_error_cell_loses_energy_faster() -> None:
    """Cell with fitness=0 (error=1.0) loses energy faster than cell with fitness=1 (error=0)."""
    from random import Random
    config = SpatialColonyConfig(
        maintenance_cost=0.5,
        reward_scale=4.0,
        death_threshold=-999.0,
        reproduction_threshold=999.0,
        initial_energy=6.0,
        development_steps_per_round=1,
    )
    arena = SpatialArena(width=8, height=8)
    genome_a = build_spatial_genome(("HALT",), lineage_id="A")
    genome_b = build_spatial_genome(("HALT",), lineage_id="B")
    cell_a = SpatialCell(genome=genome_a, x=0, y=0, energy=6.0)
    cell_b = SpatialCell(genome=genome_b, x=1, y=0, energy=6.0)
    arena.place(cell_a)
    arena.place(cell_b)

    def scorer(cell: SpatialCell, arena: SpatialArena) -> float:
        return 1.0 if cell.lineage_id == "A" else 0.0

    colony = SpatialColony(arena=arena, config=config, rng=Random(4))
    colony.advance(fitness_fn=scorer)

    energies = {c.lineage_id: c.energy for c in arena.cells.values() if c.lineage_id in ("A", "B")}
    assert energies["A"] > energies["B"], f"A={energies['A']}, B={energies['B']}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_spatial_colony.py -v 2>&1 | head -30
```

Expected: FAIL — `SpatialCell` has no `energy` field; `spatial_colony` module does not exist.

- [ ] **Step 3: Add `energy` field to `SpatialCell` in `spatial.py`**

In `spatial.py`, find `SpatialCell` (line 118). Add `energy: float = 6.0` after `alive: bool = True`:

```python
@dataclass(slots=True)
class SpatialCell:
    genome: CellGenome
    x: int
    y: int
    pc: int = 0
    registers: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    type_id: int = 0
    output: float | None = None
    alive: bool = True
    energy: float = 6.0
    halted: bool = False
    age: int = 0
    trace: list[dict[str, object]] = field(default_factory=list)
```

- [ ] **Step 4: Add `SpatialArena.format_ascii` to `spatial.py`**

In `spatial.py`, add this method to `SpatialArena` after the `report` method (after line 358):

```python
    def format_ascii(self) -> str:
        """Return a height×width ASCII grid: 'O' for occupied cell, '.' for empty, newline per row."""
        rows: list[str] = []
        for y in range(self.height):
            row_chars: list[str] = []
            for x in range(self.width):
                row_chars.append("O" if (x, y) in self.cells else ".")
            rows.append("".join(row_chars))
        return "\n".join(rows)
```

- [ ] **Step 5: Create `spatial_colony.py`**

Create `/home/mattthomson/workspace/AI-DNA/spatial_colony.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Callable

from evolution import mutate_genome
from spatial import SpatialArena, SpatialCell


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
class SpatialColonyReport:
    generation: int
    live_cells: int
    births: int
    deaths: int
    mean_fitness: float
    best_fitness: float
    energy_total: float
    morphogen_peak: float

    def format_text(self) -> str:
        return (
            f"spatial_colony: generation={self.generation} live={self.live_cells} "
            f"births={self.births} deaths={self.deaths} "
            f"mean_fitness={self.mean_fitness:.4f} best_fitness={self.best_fitness:.4f} "
            f"energy_total={self.energy_total:.2f} morphogen_peak={self.morphogen_peak:.4f}"
        )


@dataclass(slots=True)
class SpatialStreamTaskResult:
    task_name: str
    round_index: int
    mean_fitness: float
    best_fitness: float
    live_cells: int
    births: int
    deaths: int
    forgetting_delta: float = 0.0
    recovery_steps: int = 0


@dataclass(slots=True)
class SpatialStreamConfig:
    colony: SpatialColonyConfig = field(default_factory=SpatialColonyConfig)
    steps_per_task: int = 12
    snapshot_interval: int = 4


@dataclass(slots=True)
class SpatialStreamReport:
    sequence_name: str
    results: tuple[SpatialStreamTaskResult, ...]
    config: SpatialStreamConfig

    def format_text(self) -> str:
        lines: list[str] = [f"spatial_stream: sequence={self.sequence_name} tasks={len(self.results)}"]
        for r in self.results:
            lines.append(
                f"spatial_stream: task={r.task_name} round={r.round_index} "
                f"mean_fitness={r.mean_fitness:.4f} best_fitness={r.best_fitness:.4f} "
                f"live={r.live_cells} births={r.births} deaths={r.deaths} "
                f"forgetting_delta={r.forgetting_delta:.4f} recovery_steps={r.recovery_steps}"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class SpatialColony:
    arena: SpatialArena
    config: SpatialColonyConfig
    rng: Random
    generation: int = 0

    def advance(
        self,
        fitness_fn: Callable[[SpatialCell, SpatialArena], float],
    ) -> SpatialColonyReport:
        """Run one generation: develop arena, apply survival, reproduce, prune dead."""
        # 1. Development ticks
        for _ in range(self.config.development_steps_per_round):
            self.arena.step()

        # 2. Evaluate fitness for each live cell
        live_cells = [c for c in self.arena.cells.values() if c.alive]
        fitnesses: dict[tuple[int, int], float] = {}
        for cell in live_cells:
            fitnesses[(cell.x, cell.y)] = fitness_fn(cell, self.arena)

        # 3. Award energy, deduct maintenance
        for cell in live_cells:
            cell.energy += self.config.reward_scale * fitnesses.get((cell.x, cell.y), 0.0)
            cell.energy -= self.config.maintenance_cost

        # 4. Count deaths before pruning
        deaths = sum(
            1 for c in self.arena.cells.values()
            if c.alive and c.energy <= self.config.death_threshold
        )
        for cell in list(self.arena.cells.values()):
            if cell.alive and cell.energy <= self.config.death_threshold:
                cell.alive = False

        # Prune dead cells from arena
        dead_positions = [pos for pos, c in self.arena.cells.items() if not c.alive]
        for pos in dead_positions:
            self.arena.cells.pop(pos, None)

        # 5. Reproduction: cells above threshold attempt to spawn in a free cardinal direction
        births = 0
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
        reproducers = [
            c for c in list(self.arena.cells.values())
            if c.energy > self.config.reproduction_threshold
            and len(self.arena.cells) < self.config.max_cells
        ]
        for cell in reproducers:
            self.rng.shuffle(list(directions))
            for dx, dy in self.rng.sample(directions, len(directions)):
                child_x = cell.x + dx
                child_y = cell.y + dy
                if self.arena._wrap(child_x, child_y) is None:
                    continue
                if (child_x, child_y) in self.arena.cells:
                    continue
                if len(self.arena.cells) >= self.config.max_cells:
                    break
                # Mutate genome for child
                child_genome = mutate_genome(
                    cell.genome,
                    self.rng,
                    lineage_id=f"{cell.lineage_id}.c{self.generation}",
                    mutation_rate=self.config.mutation_rate,
                    synonym_rate=0.7,
                    insertion_rate=0.05,
                    deletion_rate=0.03,
                    motif_mutation_rate=0.0,
                )
                from spatial import SpatialCell as _SC
                child = _SC(
                    genome=child_genome,
                    x=child_x,
                    y=child_y,
                    energy=self.config.initial_energy,
                )
                self.arena.cells[(child_x, child_y)] = child
                cell.energy -= self.config.spawn_cost
                births += 1
                break

        # 6. Build report
        remaining = list(self.arena.cells.values())
        fitness_vals = [fitness_fn(c, self.arena) for c in remaining] if remaining else [0.0]
        mean_fitness = sum(fitness_vals) / len(fitness_vals) if fitness_vals else 0.0
        best_fitness = max(fitness_vals) if fitness_vals else 0.0
        energy_total = sum(c.energy for c in remaining)

        self.generation += 1
        return SpatialColonyReport(
            generation=self.generation,
            live_cells=len(remaining),
            births=births,
            deaths=deaths,
            mean_fitness=mean_fitness,
            best_fitness=best_fitness,
            energy_total=energy_total,
            morphogen_peak=self.arena.morphogen_field.peak(),
        )
```

- [ ] **Step 6: Run tests**

```bash
python3 -m pytest tests/test_spatial_colony.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 7: Run regression suite**

```bash
python3 -m pytest tests/test_spatial.py tests/test_spatial_body_plan.py -q
```

Expected: all pass (backward compatibility confirmed — `SpatialCell.energy` defaults to `6.0`).

- [ ] **Step 8: Commit**

```bash
git add spatial.py spatial_colony.py tests/test_spatial_colony.py
git commit -m "feat: add energy field to SpatialCell and SpatialColony survival loop"
```

---

## Task 2: Changing spatial task stream (experiment 25)

**Files:**
- Create: `spatial_task_stream.py`
- Create: `experiments/25_spatial_task_stream.py`
- Modify: `tests/test_spatial_colony.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_spatial_colony.py`:

```python
def test_spatial_stream_runs_three_tasks_without_error() -> None:
    """Stream runs through 3 tasks and report contains at least 3 result entries."""
    from spatial_task_stream import run_spatial_task_stream, SpatialStreamConfig
    from spatial_colony import SpatialColonyConfig

    config = SpatialStreamConfig(
        colony=SpatialColonyConfig(
            maintenance_cost=0.1,
            reward_scale=1.0,
            death_threshold=-999.0,
            reproduction_threshold=999.0,
            initial_energy=6.0,
            development_steps_per_round=2,
            max_cells=16,
        ),
        steps_per_task=3,
        snapshot_interval=1,
    )
    report = run_spatial_task_stream("test_stream", config=config, seed=42)
    assert len(report.results) >= 3
    assert report.sequence_name == "test_stream"


def test_spatial_stream_forgetting_delta_nonnegative() -> None:
    """Running a two-task spatial stream always yields forgetting_delta >= 0 on the second task."""
    from spatial_task_stream import run_spatial_task_stream, SpatialStreamConfig
    from spatial_colony import SpatialColonyConfig

    config = SpatialStreamConfig(
        colony=SpatialColonyConfig(
            maintenance_cost=0.0,
            reward_scale=1.0,
            death_threshold=-999.0,
            reproduction_threshold=999.0,
            initial_energy=6.0,
            development_steps_per_round=2,
            max_cells=16,
        ),
        steps_per_task=4,
        snapshot_interval=2,
    )
    report = run_spatial_task_stream("test_forgetting", config=config, seed=7)
    # forgetting_delta should be >= 0 for all task-switch entries
    switch_results = [r for r in report.results if r.forgetting_delta != 0.0]
    for r in switch_results:
        assert r.forgetting_delta >= 0.0, f"Negative forgetting_delta={r.forgetting_delta} for task {r.task_name}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_spatial_colony.py::test_spatial_stream_runs_three_tasks_without_error tests/test_spatial_colony.py::test_spatial_stream_forgetting_delta_nonnegative -v
```

Expected: FAIL — `spatial_task_stream` module does not exist.

- [ ] **Step 3: Create `spatial_task_stream.py`**

Create `/home/mattthomson/workspace/AI-DNA/spatial_task_stream.py`:

```python
from __future__ import annotations

from random import Random
from typing import Callable

from spatial import SpatialArena, SpatialCell, build_spatial_genome
from spatial_colony import (
    SpatialColony,
    SpatialColonyConfig,
    SpatialStreamConfig,
    SpatialStreamReport,
    SpatialStreamTaskResult,
)


# ---------------------------------------------------------------------------
# Body-plan fitness functions
# ---------------------------------------------------------------------------

def _stripe_h_fitness(cell: SpatialCell, arena: SpatialArena) -> float:
    """Fraction of live cells in the horizontal centre band (y in [H//2-1, H//2+1])."""
    h = arena.height
    live = [c for c in arena.cells.values() if c.alive]
    if not live:
        return 0.0
    band_lo = h // 2 - 1
    band_hi = h // 2 + 1
    in_band = sum(1 for c in live if band_lo <= c.y <= band_hi)
    return in_band / len(live)


def _stripe_v_fitness(cell: SpatialCell, arena: SpatialArena) -> float:
    """Fraction of live cells in the vertical centre band (x in [W//2-1, W//2+1])."""
    w = arena.width
    live = [c for c in arena.cells.values() if c.alive]
    if not live:
        return 0.0
    band_lo = w // 2 - 1
    band_hi = w // 2 + 1
    in_band = sum(1 for c in live if band_lo <= c.x <= band_hi)
    return in_band / len(live)


def _cluster_nw_fitness(cell: SpatialCell, arena: SpatialArena) -> float:
    """Fraction of live cells in the NW quadrant (x < W//2, y < H//2)."""
    w = arena.width
    h = arena.height
    live = [c for c in arena.cells.values() if c.alive]
    if not live:
        return 0.0
    in_quad = sum(1 for c in live if c.x < w // 2 and c.y < h // 2)
    return in_quad / len(live)


SPATIAL_TASKS: dict[str, Callable[[SpatialCell, SpatialArena], float]] = {
    "stripe_h": _stripe_h_fitness,
    "stripe_v": _stripe_v_fitness,
    "cluster_nw": _cluster_nw_fitness,
}

TASK_SEQUENCE: tuple[str, ...] = ("stripe_h", "stripe_v", "cluster_nw")


def run_spatial_task_stream(
    name: str,
    *,
    config: SpatialStreamConfig | None = None,
    seed: int = 0,
) -> SpatialStreamReport:
    """Run a colony through the three spatial task targets in sequence.

    Records forgetting_delta and recovery_steps at each task boundary.
    Returns a SpatialStreamReport with one SpatialStreamTaskResult per round.
    """
    if config is None:
        config = SpatialStreamConfig()

    rng = Random(seed)
    arena = SpatialArena(width=8, height=8)
    base_genome = build_spatial_genome(
        ("GET_X", "GET_Y", "EMIT_0", "SENSE_0", "DIVIDE_EAST", "DIVIDE_SOUTH", "HALT"),
        lineage_id=f"S{seed}",
    )
    seed_cell = SpatialCell(genome=base_genome, x=4, y=4, energy=config.colony.initial_energy)
    arena.place(seed_cell)

    colony = SpatialColony(arena=arena, config=config.colony, rng=rng)
    results: list[SpatialStreamTaskResult] = []

    round_index = 0
    # best_fitness_before_switch[task_name] -> best fitness at end of previous task block
    pre_switch_best: float | None = None

    for task_name in TASK_SEQUENCE:
        fitness_fn = SPATIAL_TASKS[task_name]
        task_best_fitness_history: list[float] = []
        task_births = 0
        task_deaths = 0

        for step in range(config.steps_per_task):
            report = colony.advance(fitness_fn=fitness_fn)
            task_best_fitness_history.append(report.best_fitness)
            task_births += report.births
            task_deaths += report.deaths

            if step % config.snapshot_interval == 0 or step == config.steps_per_task - 1:
                # Compute forgetting_delta on first round of each new task
                forgetting_delta = 0.0
                if pre_switch_best is not None and step == 0:
                    forgetting_delta = max(0.0, pre_switch_best - report.best_fitness)

                # Compute recovery_steps: rounds until best_fitness >= pre_switch_best
                recovery_steps = 0
                if pre_switch_best is not None and step == config.steps_per_task - 1:
                    for i, bf in enumerate(task_best_fitness_history):
                        if bf >= pre_switch_best:
                            recovery_steps = i
                            break
                    else:
                        recovery_steps = len(task_best_fitness_history)

                results.append(
                    SpatialStreamTaskResult(
                        task_name=task_name,
                        round_index=round_index,
                        mean_fitness=report.mean_fitness,
                        best_fitness=report.best_fitness,
                        live_cells=report.live_cells,
                        births=task_births,
                        deaths=task_deaths,
                        forgetting_delta=forgetting_delta,
                        recovery_steps=recovery_steps,
                    )
                )
                round_index += 1

        pre_switch_best = max(task_best_fitness_history) if task_best_fitness_history else 0.0

    return SpatialStreamReport(
        sequence_name=name,
        results=tuple(results),
        config=config,
    )
```

- [ ] **Step 4: Create `experiments/25_spatial_task_stream.py`**

Create `/home/mattthomson/workspace/AI-DNA/experiments/25_spatial_task_stream.py`:

```python
#!/usr/bin/env python3
"""Experiment 25 — Spatial task stream: three body-plan targets in sequence."""
from __future__ import annotations

from spatial_colony import SpatialColonyConfig, SpatialStreamConfig
from spatial_task_stream import run_spatial_task_stream


def main() -> None:
    config = SpatialStreamConfig(
        colony=SpatialColonyConfig(
            maintenance_cost=0.3,
            reward_scale=3.0,
            death_threshold=0.0,
            reproduction_threshold=10.0,
            spawn_cost=2.0,
            initial_energy=6.0,
            development_steps_per_round=6,
            mutation_rate=0.08,
            max_cells=48,
        ),
        steps_per_task=12,
        snapshot_interval=4,
    )
    report = run_spatial_task_stream("exp25_spatial_stream", config=config, seed=25)
    print(report.format_text())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_spatial_colony.py::test_spatial_stream_runs_three_tasks_without_error tests/test_spatial_colony.py::test_spatial_stream_forgetting_delta_nonnegative -v
```

Expected: both PASS.

- [ ] **Step 6: Run experiment 25**

```bash
python3 experiments/25_spatial_task_stream.py
```

Expected: exits 0 and prints multiple lines beginning with `spatial_stream:`.

- [ ] **Step 7: Run regression suite**

```bash
python3 -m pytest tests/test_spatial.py tests/test_spatial_body_plan.py tests/test_spatial_colony.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add spatial_task_stream.py experiments/25_spatial_task_stream.py tests/test_spatial_colony.py
git commit -m "feat: add spatial task stream with forgetting/recovery reporting (exp 25)"
```

---

## Task 3: Matrix fabric convergence improvements

**Files:**
- Modify: `spatial_matrix_fabric.py`
- Modify: `tests/test_spatial_body_plan.py` (or create `tests/test_spatial_matrix_fabric.py` if not present)

- [ ] **Step 1: Write the failing test**

Create `tests/test_spatial_matrix_fabric.py`:

```python
from __future__ import annotations


def test_matrix_fabric_curriculum_phase_field_exists() -> None:
    """MatrixFabricSearchConfig accepts curriculum_phase, crossover_rate, sharing_radius."""
    from spatial_matrix_fabric import MatrixFabricSearchConfig
    cfg = MatrixFabricSearchConfig(
        population_size=32,
        curriculum_phase=1,
        crossover_rate=0.3,
        sharing_radius=0.15,
    )
    assert cfg.population_size == 32
    assert cfg.curriculum_phase == 1
    assert cfg.crossover_rate == 0.3
    assert cfg.sharing_radius == 0.15


def test_score_fabric_evaluation_phase0_uses_layout_only() -> None:
    """Phase 0: score equals layout_error only."""
    from spatial_matrix_fabric import MatrixFabricEvaluation, score_fabric_evaluation
    from genome import CellGenome

    genome = CellGenome(codons=(0,), local_motifs=(), lineage_id="T")
    ev = MatrixFabricEvaluation(
        genome=genome,
        score=0.0,
        exact_match=False,
        layout_error=0.4,
        input_error=0.5,
        output_error=0.5,
        role_error=0.6,
        matrix_error=0.8,
        correct_output_count=0,
        missing_cells=2,
        extra_cells=1,
        dropout_robustness=0.0,
        occupied_positions=(),
        outputs=(),
        trace_examples=(),
    )
    score = score_fabric_evaluation(ev, curriculum_phase=0)
    assert abs(score - 0.4) < 1e-9, f"Expected 0.4, got {score}"


def test_score_fabric_evaluation_phase1_uses_routing_blend() -> None:
    """Phase 1: score = 0.4 * layout_error + 0.6 * (input_error + role_error)."""
    from spatial_matrix_fabric import MatrixFabricEvaluation, score_fabric_evaluation
    from genome import CellGenome

    genome = CellGenome(codons=(0,), local_motifs=(), lineage_id="T")
    ev = MatrixFabricEvaluation(
        genome=genome,
        score=0.0,
        exact_match=False,
        layout_error=0.4,
        input_error=0.2,
        output_error=0.5,
        role_error=0.3,
        matrix_error=0.8,
        correct_output_count=0,
        missing_cells=0,
        extra_cells=0,
        dropout_robustness=0.0,
        occupied_positions=(),
        outputs=(),
        trace_examples=(),
    )
    expected = 0.4 * 0.4 + 0.6 * (0.2 + 0.3)
    score = score_fabric_evaluation(ev, curriculum_phase=1)
    assert abs(score - expected) < 1e-9, f"Expected {expected}, got {score}"


def test_score_fabric_evaluation_phase2_full() -> None:
    """Phase 2: score = 0.15 * layout_error + 0.25 * role_error + 0.6 * matrix_error."""
    from spatial_matrix_fabric import MatrixFabricEvaluation, score_fabric_evaluation
    from genome import CellGenome

    genome = CellGenome(codons=(0,), local_motifs=(), lineage_id="T")
    ev = MatrixFabricEvaluation(
        genome=genome,
        score=0.0,
        exact_match=False,
        layout_error=0.4,
        input_error=0.2,
        output_error=0.5,
        role_error=0.3,
        matrix_error=0.7,
        correct_output_count=0,
        missing_cells=0,
        extra_cells=0,
        dropout_robustness=0.0,
        occupied_positions=(),
        outputs=(),
        trace_examples=(),
    )
    expected = 0.15 * 0.4 + 0.25 * 0.3 + 0.6 * 0.7
    score = score_fabric_evaluation(ev, curriculum_phase=2)
    assert abs(score - expected) < 1e-9, f"Expected {expected}, got {score}"


def test_matrix_fabric_search_with_curriculum_and_crossover() -> None:
    """Search with population_size=32 and crossover_survivors=True runs 2 generations without error."""
    from spatial_matrix_fabric import MatrixFabricSearchConfig, MatrixFabricTarget, run_matrix_fabric_search

    config = MatrixFabricSearchConfig(
        population_size=32,
        restarts=1,
        generations=2,
        curriculum_phase=1,
        crossover_rate=0.3,
        sharing_radius=0.15,
        survivor_count=4,
        siblings_per_survivor=2,
    )
    target = MatrixFabricTarget()
    report = run_matrix_fabric_search("test_curriculum", target, config=config, seed=99)
    assert report is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_spatial_matrix_fabric.py -v 2>&1 | head -30
```

Expected: FAIL — `MatrixFabricSearchConfig` lacks `curriculum_phase`; `score_fabric_evaluation` does not exist.

- [ ] **Step 3: Read `spatial_matrix_fabric.py` lines 100-200 to find existing config and search loop**

```bash
python3 -c "
import ast, sys
with open('spatial_matrix_fabric.py') as f:
    src = f.read()
print(src[3000:6000])
"
```

- [ ] **Step 4: Add new fields to `MatrixFabricSearchConfig`**

In `spatial_matrix_fabric.py`, find `MatrixFabricSearchConfig` (line 89). After `crossover_rate: float = 0.35`, add:

```python
    population_size: int = 32
    curriculum_phase: int = 0
    phase_thresholds: tuple[float, float] = (0.8, 0.9)
    sharing_radius: float = 0.15
```

Note: `population_size` already exists at line 90 with value 18 — change its default to 32 and remove any duplicate:

```python
    population_size: int = 32          # increased from 18
    restarts: int = 5
    generations: int = 16
    curriculum_generations: int = 6
    survivor_count: int = 5
    siblings_per_survivor: int = 3
    mutation_rate: float = 0.1
    synonym_rate: float = 0.7
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    crossover_rate: float = 0.35
    curriculum_phase: int = 0
    phase_thresholds: tuple[float, float] = (0.8, 0.9)
    sharing_radius: float = 0.15
```

- [ ] **Step 5: Add `score_fabric_evaluation` function**

In `spatial_matrix_fabric.py`, add this function after the `MatrixFabricEvaluation` dataclass:

```python
def score_fabric_evaluation(
    ev: MatrixFabricEvaluation,
    *,
    curriculum_phase: int = 0,
) -> float:
    """Return a composite score based on curriculum phase.

    Phase 0 (layout):  score = layout_error
    Phase 1 (routing): score = 0.4 * layout_error + 0.6 * (input_error + role_error)
    Phase 2 (full):    score = 0.15 * layout_error + 0.25 * role_error + 0.6 * matrix_error
    Lower is better (error scale).
    """
    if curriculum_phase == 0:
        return ev.layout_error
    elif curriculum_phase == 1:
        return 0.4 * ev.layout_error + 0.6 * (ev.input_error + ev.role_error)
    else:
        return 0.15 * ev.layout_error + 0.25 * ev.role_error + 0.6 * ev.matrix_error
```

- [ ] **Step 6: Wire curriculum phase auto-advance and crossover into `run_matrix_fabric_search`**

In `spatial_matrix_fabric.py`, find the main search loop in `run_matrix_fabric_search`. After evaluating the population each generation, add curriculum phase advancement logic and crossover between top-2 survivors.

Locate the block that evaluates genomes and produces a ranked list. After computing `evaluations`, add:

```python
        # Curriculum phase auto-advance
        current_phase = config.curriculum_phase
        mean_score = sum(score_fabric_evaluation(e, curriculum_phase=current_phase) for e in evaluations) / max(len(evaluations), 1)
        if current_phase == 0 and mean_score < config.phase_thresholds[0]:
            current_phase = 1
        elif current_phase == 1 and mean_score < config.phase_thresholds[1]:
            current_phase = 2

        # Re-score evaluations using updated phase for selection
        scored = sorted(evaluations, key=lambda e: score_fabric_evaluation(e, curriculum_phase=current_phase))
```

After deriving `survivors` (top-k by score), add crossover between top-2 when `crossover_rate > 0`:

```python
        # Crossover between top-2 fabric genomes
        if len(survivors) >= 2 and rng.random() < config.crossover_rate:
            from codons import crossover_codons
            top_a = survivors[0].genome
            top_b = survivors[1].genome
            child_codons = crossover_codons(top_a.codons, top_b.codons, rng)
            from genome import CellGenome
            crossover_child = CellGenome(
                codons=child_codons,
                local_motifs=top_a.local_motifs + top_b.local_motifs,
                lineage_id=f"{top_a.lineage_id}.cx{generation}",
            )
            next_population.append(crossover_child)
```

Also update `select_top_diverse` call to pass `sharing_radius`:

```python
        survivors = select_top_diverse(
            evaluations,
            k=config.survivor_count,
            sharing_radius=config.sharing_radius,
        )
```

And ensure `exact_match` is set when `matrix_error < 0.01` in the evaluation function. Find where `MatrixFabricEvaluation` is constructed in the evaluation helper and update:

```python
        exact_match=matrix_error < 0.01,
```

- [ ] **Step 7: Run tests**

```bash
python3 -m pytest tests/test_spatial_matrix_fabric.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 8: Run regression suite**

```bash
python3 -m pytest tests/test_spatial.py tests/test_spatial_body_plan.py -q
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add spatial_matrix_fabric.py tests/test_spatial_matrix_fabric.py
git commit -m "feat: add curriculum scoring, crossover, and population_size=32 to matrix fabric search"
```

---

## Task 4: Cross-layer integration experiment (experiment 26)

**Files:**
- Modify: `spatial.py` (verify `format_ascii` already added in Task 1)
- Create: `experiments/26_spatial_ecology_stream.py`
- Modify: `tests/test_spatial_colony.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_spatial_colony.py`:

```python
def test_experiment_26_exits_zero_and_prints_prefix() -> None:
    """Experiment 26 exits 0 and stdout contains 'spatial_ecology_stream:'."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "experiments/26_spatial_ecology_stream.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Non-zero exit: {result.stderr}"
    assert "spatial_ecology_stream:" in result.stdout, (
        f"Missing prefix in stdout:\n{result.stdout[:500]}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_spatial_colony.py::test_experiment_26_exits_zero_and_prints_prefix -v
```

Expected: FAIL — `experiments/26_spatial_ecology_stream.py` does not exist.

- [ ] **Step 3: Verify `format_ascii` already present in `spatial.py`**

```bash
python3 -c "from spatial import SpatialArena; a = SpatialArena(4,4); print(a.format_ascii())"
```

Expected: prints a 4×4 grid of dots. If method is missing, confirm Task 1 Step 4 was applied.

- [ ] **Step 4: Create `experiments/26_spatial_ecology_stream.py`**

Create `/home/mattthomson/workspace/AI-DNA/experiments/26_spatial_ecology_stream.py`:

```python
#!/usr/bin/env python3
"""Experiment 26 — Cross-layer integration: chemistry + ecology + spatial + task stream."""
from __future__ import annotations

from random import Random

from spatial import SpatialArena, SpatialCell, build_spatial_genome
from spatial_colony import (
    SpatialColony,
    SpatialColonyConfig,
    SpatialStreamConfig,
    SpatialStreamReport,
    SpatialStreamTaskResult,
)
from spatial_task_stream import SPATIAL_TASKS, TASK_SEQUENCE, run_spatial_task_stream


def main() -> None:
    config = SpatialStreamConfig(
        colony=SpatialColonyConfig(
            maintenance_cost=0.3,
            reward_scale=3.0,
            death_threshold=0.0,
            reproduction_threshold=9.0,
            spawn_cost=2.0,
            initial_energy=6.0,
            development_steps_per_round=6,
            mutation_rate=0.08,
            max_cells=48,
        ),
        steps_per_task=10,
        snapshot_interval=4,
    )

    # Run the three-task spatial stream (stripe_h -> stripe_v -> cluster_nw)
    report = run_spatial_task_stream("exp26_ecology_stream", config=config, seed=26)

    # Print stream results with ecology prefix
    for line in report.format_text().splitlines():
        print(f"spatial_ecology_stream: {line}")

    # Reconstruct final arena state for ASCII display
    rng = Random(26)
    arena = SpatialArena(width=8, height=8)
    base_genome = build_spatial_genome(
        ("GET_X", "GET_Y", "EMIT_0", "SENSE_GRAD_X_0", "SENSE_GRAD_Y_0",
         "DIVIDE_EAST", "DIVIDE_SOUTH", "HALT"),
        lineage_id="E26",
    )
    # Seed 3 cells in a small cluster
    for i, (cx, cy) in enumerate([(3, 3), (4, 3), (3, 4)]):
        cell = SpatialCell(
            genome=base_genome.with_lineage(f"E26.{i}"),
            x=cx,
            y=cy,
            energy=config.colony.initial_energy,
        )
        arena.place(cell)

    colony = SpatialColony(arena=arena, config=config.colony, rng=rng)
    for task_name in TASK_SEQUENCE:
        fitness_fn = SPATIAL_TASKS[task_name]
        for _ in range(config.steps_per_task):
            colony.advance(fitness_fn=fitness_fn)

    print("spatial_ecology_stream: final_arena_layout:")
    for row in arena.format_ascii().splitlines():
        print(f"spatial_ecology_stream:   {row}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run experiment 26 manually**

```bash
python3 experiments/26_spatial_ecology_stream.py
```

Expected: exits 0; stdout contains multiple lines beginning with `spatial_ecology_stream:` and an ASCII layout.

- [ ] **Step 6: Run the failing test**

```bash
python3 -m pytest tests/test_spatial_colony.py::test_experiment_26_exits_zero_and_prints_prefix -v
```

Expected: PASS.

- [ ] **Step 7: Run full test suite**

```bash
python3 -m pytest tests/test_spatial.py tests/test_spatial_body_plan.py tests/test_spatial_colony.py tests/test_spatial_matrix_fabric.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add experiments/26_spatial_ecology_stream.py tests/test_spatial_colony.py
git commit -m "feat: add cross-layer ecology stream experiment (exp 26)"
```

---

## Task 5: Final integration

**Files:** no new changes — verification and final commit only

- [ ] **Step 1: Run full spatial test suite**

```bash
python3 -m pytest tests/test_spatial.py tests/test_spatial_body_plan.py tests/test_spatial_colony.py tests/test_spatial_matrix_fabric.py -q
```

Expected: all pass with no failures or errors.

- [ ] **Step 2: Run experiment 25**

```bash
python3 experiments/25_spatial_task_stream.py
```

Expected: exits 0; prints multiple lines beginning with `spatial_stream:` including three named task blocks (`task=stripe_h`, `task=stripe_v`, `task=cluster_nw`).

- [ ] **Step 3: Run experiment 26**

```bash
python3 experiments/26_spatial_ecology_stream.py
```

Expected: exits 0; prints `spatial_ecology_stream:` lines; prints ASCII layout showing at least one `O` character.

- [ ] **Step 4: Verify backward compatibility of experiments 15–24**

```bash
for exp in experiments/1[5-9]_*.py experiments/2[0-4]_*.py; do
    python3 "$exp" > /dev/null 2>&1 && echo "OK: $exp" || echo "FAIL: $exp"
done
```

Expected: all print `OK`. If any fail, check whether the `SpatialCell.energy = 6.0` default broke construction — the default must not be a positional argument.

- [ ] **Step 5: Verify `format_ascii` output dimensions**

```bash
python3 -c "
from spatial import SpatialArena, SpatialCell, build_spatial_demo_genome
arena = SpatialArena(width=6, height=4)
arena.place(SpatialCell(genome=build_spatial_demo_genome(), x=3, y=2))
ascii_art = arena.format_ascii()
rows = ascii_art.splitlines()
assert len(rows) == 4, f'Expected 4 rows, got {len(rows)}'
assert all(len(r) == 6 for r in rows), f'Row lengths: {[len(r) for r in rows]}'
print('format_ascii dimensions: OK')
print(ascii_art)
"
```

Expected: prints `format_ascii dimensions: OK` and a 4-row × 6-column grid.

- [ ] **Step 6: Final commit**

```bash
git add spatial.py spatial_colony.py spatial_task_stream.py spatial_matrix_fabric.py \
    experiments/25_spatial_task_stream.py experiments/26_spatial_ecology_stream.py \
    tests/test_spatial_colony.py tests/test_spatial_matrix_fabric.py
git commit -m "feat: Phase C spatial-ecology integration"
```

---

## Self-review: spec coverage check

| Spec requirement | Task | Status |
|---|---|---|
| `SpatialCell.energy: float = 6.0` | Task 1 Step 3 | Covered |
| `SpatialColonyConfig` dataclass with all 9 fields | Task 1 Step 5 | Covered |
| `SpatialColony.advance(fitness_fn)` — develop, reward, maintain, kill, divide | Task 1 Step 5 | Covered |
| `SpatialColonyReport` with `format_text` | Task 1 Step 5 | Covered |
| `test_spatial_colony_energy_maintenance` | Task 1 Step 1 | Covered |
| `test_spatial_colony_death_at_zero_energy` | Task 1 Step 1 | Covered |
| `test_spatial_colony_reproduction_above_threshold` | Task 1 Step 1 | Covered |
| Three body-plan fitness functions (`stripe_h`, `stripe_v`, `cluster_nw`) | Task 2 Step 3 | Covered |
| `SpatialStreamConfig`, `SpatialStreamReport`, `SpatialStreamTaskResult` | Task 1 Step 5 / Task 2 Step 3 | Covered |
| `forgetting_delta` and `recovery_steps` tracking | Task 2 Step 3 | Covered |
| `test_spatial_stream_forgetting_delta_nonnegative` | Task 2 Step 1 | Covered |
| `experiments/25_spatial_task_stream.py` | Task 2 Step 4 | Covered |
| `MatrixFabricSearchConfig.population_size = 32` | Task 3 Step 4 | Covered |
| `curriculum_phase`, `phase_thresholds`, `sharing_radius` fields | Task 3 Step 4 | Covered |
| `score_fabric_evaluation` with 3-phase logic | Task 3 Step 5 | Covered |
| Crossover between top-2 survivors | Task 3 Step 6 | Covered |
| `exact_match = matrix_error < 0.01` | Task 3 Step 6 | Covered |
| `SpatialArena.format_ascii()` | Task 1 Step 4 | Covered |
| `experiments/26_spatial_ecology_stream.py` with `spatial_ecology_stream:` prefix | Task 4 Step 4 | Covered |
| ASCII art in experiment 26 output | Task 4 Step 4 | Covered |
| Backward compat of experiments 15–24 | Task 5 Step 4 | Covered |
