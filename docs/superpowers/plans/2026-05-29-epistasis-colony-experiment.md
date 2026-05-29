# Epistasis Colony Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `experiments/30_epistasis_colony.py` — a 200-generation two-cell colony experiment demonstrating emergent specialisation on split-input multiplication with an epistasis mixing gate.

**Architecture:** A standalone experiment script with a custom `score_split_pair()` function. Cell 1 gets `signal[0]=a/3`, cell 2 gets `signal[1]=b/3`. Colony output = `cell1.output × cell1.signals[3] + cell2.output × (1 − cell1.signals[3])`. Evolution loop uses `mutate_genome()` and tournament selection. Output prefix `epistasis_colony:`.

**Tech Stack:** Python 3.10+, existing `CooperativeChemistrySystem`, `mutate_genome`, `select_top_diverse`, pytest.

---

## File Map

- Create: `experiments/30_epistasis_colony.py` — full experiment script
- Modify: `tests/test_experiments.py` — add script to list, add `"epistasis_colony:"` to recognised prefixes

---

### Task 1: Scaffold the experiment with a smoke-test run

**Files:**
- Create: `experiments/30_epistasis_colony.py`

- [ ] **Step 1: Create the experiment file**

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
from evolution import mutate_genome, select_top_diverse, EvolutionConfig
from genome import CellGenome
from codons import random_codons
from tasks import TaskCase


SPLIT_CASES: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in (1, 2, 3) for b in (1, 2, 3)
)


def _make_case(a: int, b: int) -> TaskCase:
    return TaskCase(inputs=(float(a), float(b)), target=(a * b) / 9.0)


def score_split_pair(
    genome: CellGenome,
    system: CooperativeChemistrySystem,
    chemistry_rounds: int = 6,
) -> tuple[float, float, float, float]:
    """Return (mean_error, mixing_weight, cell1_out, cell2_out) for probe case (2,3)."""
    total_error = 0.0
    probe_mixing = 0.0
    probe_c1 = 0.0
    probe_c2 = 0.0

    for a, b in SPLIT_CASES:
        case = _make_case(a, b)
        cell1 = CellState()
        cell1.signals = [a / 3.0, 0.0, 0.0, 0.0, 0.0]
        cell2 = CellState()
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]

        context = ChemistryContext()
        system.run([cell1, cell2], case, context=context, max_time=float(chemistry_rounds))

        c1_out = cell1.output if cell1.output is not None else cell1.signals[2]
        c2_out = cell2.output if cell2.output is not None else cell2.signals[2]
        w = max(0.0, min(1.0, cell1.signals[3]))
        colony_out = c1_out * w + c2_out * (1.0 - w)
        total_error += (colony_out - case.target) ** 2

        if a == 2 and b == 3:
            probe_mixing = w
            probe_c1 = c1_out
            probe_c2 = c2_out

    return total_error / len(SPLIT_CASES), probe_mixing, probe_c1, probe_c2


def _random_genome(rng: Random, lineage_id: str) -> CellGenome:
    length = rng.randint(6, 20)
    return CellGenome(
        codons=tuple(random_codons(rng, length)),
        local_motifs=(),
        lineage_id=lineage_id,
    )


def main() -> None:
    rng = Random(30)
    system = CooperativeChemistrySystem()
    population = [_random_genome(rng, f"C{i+1}") for i in range(8)]

    for generation in range(200):
        scored = []
        for genome in population:
            err, w, c1, c2 = score_split_pair(genome, system)
            scored.append((err, w, c1, c2, genome))

        scored.sort(key=lambda x: x[0])
        best_err, best_w, best_c1, best_c2, _ = scored[0]
        mean_err = sum(s[0] for s in scored) / len(scored)

        print(
            f"epistasis_colony: gen={generation} "
            f"mean_error={mean_err:.6f} best_error={best_err:.6f} "
            f"mixing_weight={best_w:.4f} cell1_out={best_c1:.4f} cell2_out={best_c2:.4f}"
        )

        if generation % 20 == 0:
            best_genome = scored[0][4]
            probe_case = _make_case(2, 3)
            cell1 = CellState()
            cell1.signals = [2 / 3.0, 0.0, 0.0, 0.0, 0.0]
            cell2 = CellState()
            cell2.signals = [0.0, 3 / 3.0, 0.0, 0.0, 0.0]
            context = ChemistryContext()
            system.run([cell1, cell2], probe_case, context=context, max_time=6.0)
            print(
                f"epistasis_colony: snapshot gen={generation} "
                f"cell1_signals={[round(s, 3) for s in cell1.signals]} "
                f"cell2_signals={[round(s, 3) for s in cell2.signals]}"
            )

        # Selection + reproduction
        survivors = [s[4] for s in scored[:3]]
        next_gen: list[CellGenome] = list(survivors)
        evo_config = EvolutionConfig()
        while len(next_gen) < 8:
            parent = rng.choice(survivors)
            child_id = f"{parent.lineage_id}.{generation+1}.{len(next_gen)}"
            child = mutate_genome(parent, child_id, rng, config=evo_config)
            next_gen.append(child)
        population = next_gen


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment to verify it produces output**

```bash
python3 experiments/30_epistasis_colony.py 2>&1 | head -5
```

Expected: lines starting with `epistasis_colony: gen=0 ...`

- [ ] **Step 3: Commit**

```bash
git add experiments/30_epistasis_colony.py
git commit -m "feat: add epistasis colony experiment (30)"
```

---

### Task 2: Register in `test_experiments.py`

**Files:**
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Add script to the list**

In `tests/test_experiments.py`, find the `scripts` list. Add after the last entry:

```python
"30_epistasis_colony.py",
```

- [ ] **Step 2: Add recognised output prefix**

In `tests/test_experiments.py`, find the `assert any(prefix in completed.stdout ...)` block. Add `"epistasis_colony:"` to the tuple of recognised prefixes:

```python
        assert any(
            prefix in completed.stdout
            for prefix in (
                "experiment:",
                "stream:",
                "cooperative_stream:",
                "spatial_development:",
                "spatial3d_development:",
                "spatial_stream:",
                "spatial_ecology_stream:",
                "temporal_unfolding:",
                "reading_layer:",
                "epistasis_colony:",
            )
        ), f"{script_name} produced no recognised output prefix:\n{completed.stdout}"
```

- [ ] **Step 3: Run the targeted test**

```bash
python3 -m pytest tests/test_experiments.py -k "test_remaining" -v --no-header 2>&1 | grep -E "PASS|FAIL|ERROR|30_epistasis"
```

Note: `test_remaining_experiments` runs all scripts sequentially and is slow. Run only if you have time; otherwise skip to the fast suite.

- [ ] **Step 4: Run the fast suite**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_experiments.py
git commit -m "test: register epistasis colony experiment in test_experiments"
```

---

### Task 3: Verify specialisation signals emerge

**Files:**
- No code changes — observation only

- [ ] **Step 1: Run the full 200-generation experiment**

```bash
python3 experiments/30_epistasis_colony.py 2>&1 | tee /tmp/epistasis_colony_run.txt
```

- [ ] **Step 2: Check for error reduction**

```bash
grep "epistasis_colony: gen=" /tmp/epistasis_colony_run.txt | awk '{print $3, $4}' | head -5
grep "epistasis_colony: gen=" /tmp/epistasis_colony_run.txt | awk '{print $3, $4}' | tail -5
```

Expected: `best_error` decreases over generations (not necessarily monotonically).

- [ ] **Step 3: Check mixing weight movement**

```bash
grep "epistasis_colony: gen=" /tmp/epistasis_colony_run.txt | grep "mixing_weight" | awk -F'mixing_weight=' '{print $2}' | awk '{print $1}' | head -20
```

Expected: mixing weight moves away from initial value over generations. Any stable value != 0.5 is interesting.

- [ ] **Step 4: Check cell output divergence at gen 100+**

```bash
grep "epistasis_colony: gen=1[0-9][0-9]" /tmp/epistasis_colony_run.txt | head -5
```

Expected: `cell1_out` and `cell2_out` differ by > 0.05 on the probe case in later generations.
