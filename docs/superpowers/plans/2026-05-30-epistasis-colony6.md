# Epistasis Colony 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `experiments/35_epistasis_colony6.py` — a motif-seeded three-cell colony experiment with a two-stage blended curriculum, testing whether stored gate/echo circuits from exp33 accelerate discovery of three-cell multiply.

**Architecture:** Hybrid-seed pop_a/pop_b from stored gate motifs and pop_c from echo motifs via `MotifStore` + `GenomeWriter`. Run a two-stage blended-α curriculum: Stage 1 (gens 0–149) trains cell3 to echo c/3 and cell2 to gate b×c/9 (two-cell sub-task) while cell1 echoes a/3; Stage 2 (gens 150–399) blends cell1 from echo_a toward full three-cell gate a×b×c/27.

**Tech Stack:** Python stdlib, existing `CooperativeChemistrySystem`, `MotifStore`, `GenomeWriter`, `mutate_genome`, `CellGenome`, `CellState`, `ChemistryContext`, `TaskCase`.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `experiments/35_epistasis_colony6.py` | Full experiment entrypoint |
| Modify | `tests/test_experiments.py` | Add script + prefix to regression test |

---

## Task 1 — Create `experiments/35_epistasis_colony6.py`

**Files:**
- Create: `experiments/35_epistasis_colony6.py`

- [ ] **Step 1: Write the experiment file**

```python
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from random import Random
from cell import CellState
from chemistry import ChemistryContext
from cooperative_chemistry import CooperativeChemistrySystem
from evolution import mutate_genome
from genome import CellGenome
from codons import random_codons
from tasks import TaskCase
from motif_store import MotifStore
from writing import GenomeWriter


SPLIT_CASES_3: tuple[tuple[int, int, int], ...] = tuple(
    (a, b, c) for a in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3)
)
SPLIT_CASES_BC: tuple[tuple[int, int], ...] = tuple(
    (b, c) for b in (1, 2, 3) for c in (1, 2, 3)
)
CHEMISTRY_ROUNDS = 6
POPULATION_SIZE = 8
SURVIVORS = 3
PARTNER_SAMPLES = 3
TOTAL_GENS = 400
STAGE1_GENS = 150


def _make_case(x: float, y: float, target: float) -> TaskCase:
    return TaskCase(x=x, y=y, target=target, task_name="multiply")


def _random_genome(rng: Random, lineage_id: str) -> CellGenome:
    length = rng.randint(6, 20)
    return CellGenome(
        codons=tuple(random_codons(rng, length)),
        local_motifs=(),
        lineage_id=lineage_id,
    )


def _seed_population(
    motifs: list,
    n_seeded: int,
    n_total: int,
    rng: Random,
    prefix: str,
) -> list[CellGenome]:
    """Build a population with up to n_seeded genomes composed from motifs, rest random."""
    writer = GenomeWriter()
    pop: list[CellGenome] = []
    for i in range(min(n_seeded, len(motifs))):
        motif = motifs[i % len(motifs)]
        genome = writer.compose([motif], lineage_id=f"{prefix}S{i + 1}")
        pop.append(genome)
    while len(pop) < n_total:
        j = len(pop)
        pop.append(_random_genome(rng, f"{prefix}{j + 1}"))
    return pop


# ── Fitness helpers ───────────────────────────────────────────────────────────

def _echo_err(genome: CellGenome, signal_slot: int, system: CooperativeChemistrySystem) -> float:
    """Mean squared error of a cell echoing its own input from signal_slot (solo run)."""
    err = 0.0
    for v in (1, 2, 3):
        cell = CellState(active_rules=list(genome.declare_rules()))
        cell.signals = [0.0, 0.0, 0.0, 0.0, 0.0]
        cell.signals[signal_slot] = v / 3.0
        case = _make_case(float(v), 0.0, v / 3.0)
        context = ChemistryContext()
        system.run([cell], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        out = cell.output if cell.output is not None else cell.signals[2]
        err += (out - v / 3.0) ** 2
    return err / 3


def _gate_bc_err(
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> float:
    """Two-cell sub-task: mean((c2*c2_s3 - b*c/9)²) over all (b,c) pairs."""
    err = 0.0
    for b, c in SPLIT_CASES_BC:
        inter = (b * c) / 9.0
        case = _make_case(float(b), float(c), inter)
        cell2 = CellState(active_rules=list(genome_b.declare_rules()))
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        cell3 = CellState(active_rules=list(genome_c.declare_rules()))
        cell3.signals = [0.0, 0.0, c / 3.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell2, cell3], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c2 = cell2.output if cell2.output is not None else cell2.signals[2]
        c2_s3 = max(0.0, min(1.0, cell2.signals[3]))
        err += (c2 * c2_s3 - inter) ** 2
    return err / len(SPLIT_CASES_BC)


def _gate_abc_err(
    genome_a: CellGenome,
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> float:
    """Full three-cell: mean((c1*c1_s3 - a*b*c/27)²) over all 27 cases."""
    err = 0.0
    for a, b, c in SPLIT_CASES_3:
        target = (a * b * c) / 27.0
        case = _make_case(float(a), float(b), target)
        cell1 = CellState(active_rules=list(genome_a.declare_rules()))
        cell1.signals = [a / 3.0, 0.0, 0.0, 0.0, 0.0]
        cell2 = CellState(active_rules=list(genome_b.declare_rules()))
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        cell3 = CellState(active_rules=list(genome_c.declare_rules()))
        cell3.signals = [0.0, 0.0, c / 3.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell1, cell2, cell3], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c1 = cell1.output if cell1.output is not None else cell1.signals[2]
        c1_s3 = max(0.0, min(1.0, cell1.signals[3]))
        err += (c1 * c1_s3 - target) ** 2
    return err / len(SPLIT_CASES_3)


# ── Population scoring ────────────────────────────────────────────────────────

def _score_a(
    genome: CellGenome,
    pop_b: list[CellGenome],
    pop_c: list[CellGenome],
    system: CooperativeChemistrySystem,
    rng: Random,
    generation: int,
) -> float:
    echo_a = _echo_err(genome, 0, system)
    if generation < STAGE1_GENS:
        return echo_a
    alpha2 = (generation - STAGE1_GENS) / (TOTAL_GENS - STAGE1_GENS - 1)
    gate_abc = sum(
        _gate_abc_err(genome, rng.choice(pop_b), rng.choice(pop_c), system)
        for _ in range(PARTNER_SAMPLES)
    ) / PARTNER_SAMPLES
    return (1.0 - alpha2) * echo_a + alpha2 * gate_abc


def _score_b(
    genome: CellGenome,
    pop_c: list[CellGenome],
    system: CooperativeChemistrySystem,
    rng: Random,
    generation: int,
) -> float:
    echo_b = _echo_err(genome, 1, system)
    gate_bc = sum(
        _gate_bc_err(genome, rng.choice(pop_c), system)
        for _ in range(PARTNER_SAMPLES)
    ) / PARTNER_SAMPLES
    if generation < STAGE1_GENS:
        alpha1 = generation / (STAGE1_GENS - 1)
        return (1.0 - alpha1) * echo_b + alpha1 * gate_bc
    return gate_bc


def _score_c(
    genome: CellGenome,
    system: CooperativeChemistrySystem,
) -> float:
    return _echo_err(genome, 2, system)


# ── Trio evaluation (for reporting) ──────────────────────────────────────────

def _run_trio(
    genome_a: CellGenome,
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[float, list[float]]:
    """Return (colony_err, probe) over all 27 cases. Probe = [c1_out, c1_s3, c2_s3, colony_out] at a=b=c=2."""
    colony_err = 0.0
    probe = [0.0, 0.0, 0.0, 0.0]
    for a, b, c in SPLIT_CASES_3:
        target = (a * b * c) / 27.0
        case = _make_case(float(a), float(b), target)
        cell1 = CellState(active_rules=list(genome_a.declare_rules()))
        cell1.signals = [a / 3.0, 0.0, 0.0, 0.0, 0.0]
        cell2 = CellState(active_rules=list(genome_b.declare_rules()))
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        cell3 = CellState(active_rules=list(genome_c.declare_rules()))
        cell3.signals = [0.0, 0.0, c / 3.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell1, cell2, cell3], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c1 = cell1.output if cell1.output is not None else cell1.signals[2]
        c1_s3 = max(0.0, min(1.0, cell1.signals[3]))
        c2_s3 = max(0.0, min(1.0, cell2.signals[3]))
        colony_out = c1 * c1_s3
        colony_err += (colony_out - target) ** 2
        if a == 2 and b == 2 and c == 2:
            probe = [c1, c1_s3, c2_s3, colony_out]
    return colony_err / len(SPLIT_CASES_3), probe


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    system = CooperativeChemistrySystem()
    rng = Random(35)

    # Load stored motifs
    try:
        store = MotifStore()
        gate_motifs = store.query(role="gate", task="multiply_2cell", top_k=3)
        echo_motifs = store.query(role="echo", task="multiply_2cell", top_k=3)
        store.close()
        print(
            f"epistasis_colony6: loaded {len(gate_motifs)} gate motifs, "
            f"{len(echo_motifs)} echo motifs from store"
        )
    except Exception:
        gate_motifs = []
        echo_motifs = []
        print("epistasis_colony6: no motif store found, starting fully random")

    pop_a = _seed_population(gate_motifs, 3, POPULATION_SIZE, rng, "A")
    pop_b = _seed_population(gate_motifs, 3, POPULATION_SIZE, rng, "B")
    pop_c = _seed_population(echo_motifs, 3, POPULATION_SIZE, rng, "C")

    n_a = min(3, len(gate_motifs))
    n_b = min(3, len(gate_motifs))
    n_c = min(3, len(echo_motifs))
    print(f"epistasis_colony6: seeded pop_a={n_a} pop_b={n_b} pop_c={n_c} from motifs")

    best_a_lineage = pop_a[0].lineage_id
    best_b_lineage = pop_b[0].lineage_id
    best_c_lineage = pop_c[0].lineage_id

    for generation in range(TOTAL_GENS):
        scores_a = sorted(
            [(_score_a(g, pop_b, pop_c, system, rng, generation), g) for g in pop_a],
            key=lambda x: x[0],
        )
        scores_b = sorted(
            [(_score_b(g, pop_c, system, rng, generation), g) for g in pop_b],
            key=lambda x: x[0],
        )
        scores_c = sorted(
            [(_score_c(g, system), g) for g in pop_c],
            key=lambda x: x[0],
        )

        gen_best_colony = float("inf")
        probe_best = [0.0, 0.0, 0.0, 0.0]
        for _, ga in scores_a[:3]:
            for _, gb in scores_b[:3]:
                for _, gc in scores_c[:3]:
                    ce, probe = _run_trio(ga, gb, gc, system)
                    if ce < gen_best_colony:
                        gen_best_colony = ce
                        probe_best = probe
                        best_a_lineage = ga.lineage_id
                        best_b_lineage = gb.lineage_id
                        best_c_lineage = gc.lineage_id

        c1_out, c1_s3, c2_s3, colony_out = probe_best
        print(
            f"epistasis_colony6: gen={generation} colony_err={gen_best_colony:.6f} "
            f"colony_out={colony_out:.4f} probe_target=0.2963 "
            f"cell1_out={c1_out:.4f} cell1_s3={c1_s3:.4f} cell2_s3={c2_s3:.4f} "
            f"lineage_a={best_a_lineage} lineage_b={best_b_lineage} lineage_c={best_c_lineage}"
        )

        if generation % 20 == 0:
            best_a = scores_a[0][1]
            best_b = scores_b[0][1]
            best_c = scores_c[0][1]
            probe_case = _make_case(2.0, 2.0, 8 / 27.0)
            cell1 = CellState(active_rules=list(best_a.declare_rules()))
            cell1.signals = [2 / 3.0, 0.0, 0.0, 0.0, 0.0]
            cell2 = CellState(active_rules=list(best_b.declare_rules()))
            cell2.signals = [0.0, 2 / 3.0, 0.0, 0.0, 0.0]
            cell3 = CellState(active_rules=list(best_c.declare_rules()))
            cell3.signals = [0.0, 0.0, 2 / 3.0, 0.0, 0.0]
            context = ChemistryContext()
            system.run([cell1, cell2, cell3], probe_case, context=context, max_time=float(CHEMISTRY_ROUNDS))
            print(
                f"epistasis_colony6: snapshot gen={generation} "
                f"cell1_signals={[round(s, 3) for s in cell1.signals]} "
                f"cell2_signals={[round(s, 3) for s in cell2.signals]} "
                f"cell3_signals={[round(s, 3) for s in cell3.signals]}"
            )

        survivors_a = [g for _, g in scores_a[:SURVIVORS]]
        next_a: list[CellGenome] = list(survivors_a)
        while len(next_a) < POPULATION_SIZE:
            parent = rng.choice(survivors_a)
            next_a.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation + 1}.{len(next_a)}"))
        pop_a = next_a

        survivors_b = [g for _, g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            next_b.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation + 1}.{len(next_b)}"))
        pop_b = next_b

        survivors_c = [g for _, g in scores_c[:SURVIVORS]]
        next_c: list[CellGenome] = list(survivors_c)
        while len(next_c) < POPULATION_SIZE:
            parent = rng.choice(survivors_c)
            next_c.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation + 1}.{len(next_c)}"))
        pop_c = next_c


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file exists**

Run: `ls experiments/35_epistasis_colony6.py`
Expected: file listed.

- [ ] **Step 3: Commit**

```bash
git add experiments/35_epistasis_colony6.py
git commit -m "feat: add epistasis colony6 motif-seeded three-cell experiment (35)"
```

---

## Task 2 — Register in `tests/test_experiments.py`

**Files:**
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Add the script name to the scripts list**

In `tests/test_experiments.py`, find the `scripts` list and add `"35_epistasis_colony6.py"` after `"34_epistasis_colony5.py"`:

```python
        "34_epistasis_colony5.py",
        "35_epistasis_colony6.py",
```

- [ ] **Step 2: Add the output prefix**

In the same file, find the `prefix in` tuple and add `"epistasis_colony6:"` after `"epistasis_colony5:"`:

```python
                "epistasis_colony5:",
                "epistasis_colony6:",
```

- [ ] **Step 3: Run the smoke test for just the new script**

```bash
python3 experiments/35_epistasis_colony6.py 2>&1 | head -20
```

Expected: First few lines show `epistasis_colony6: loaded ... motifs` (or `no motif store found`) then `epistasis_colony6: seeded ...` then `epistasis_colony6: gen=0 colony_err=...`. No Python errors.

- [ ] **Step 4: Run non-experiment tests**

```bash
python3 -m pytest tests/ -v --ignore=tests/test_experiments.py -x -q 2>&1 | tail -5
```

Expected: All existing tests pass (the pre-existing `test_task_stream_runs_multiple_tasks_and_grows` failure is unrelated to this work — skip or ignore it).

- [ ] **Step 5: Commit**

```bash
git add tests/test_experiments.py
git commit -m "test: register epistasis_colony6 experiment in test suite"
```

---

## Task 3 — Push

- [ ] **Step 1: Push to remote**

```bash
git push
```

Expected: `master -> master` confirmation.
