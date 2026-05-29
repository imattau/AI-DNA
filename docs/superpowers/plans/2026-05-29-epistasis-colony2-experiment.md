# Epistasis Colony 2 Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `experiments/31_epistasis_colony2.py` — two independent co-evolving populations (A=cell1, B=cell2) with forced equal mixing, diagnosing whether genuine role specialisation emerges when the degenerate `w=0` attractor is removed.

**Architecture:** Two populations of 8 genomes each. Individual fitness = mean colony MSE across 3 random partner samples from the opposing population. Colony output = (cell1.output + cell2.output) / 2. 200 generations, output prefix `epistasis_colony2:`.

**Tech Stack:** Python 3.10+, existing `CooperativeChemistrySystem`, `mutate_genome`, pytest.

---

## File Map

- Create: `experiments/31_epistasis_colony2.py`
- Modify: `tests/test_experiments.py` — add script + prefix

---

### Task 1: Create the experiment script

**Files:**
- Create: `experiments/31_epistasis_colony2.py`

- [ ] **Step 1: Create the file**

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
from evolution import mutate_genome, EvolutionConfig
from genome import CellGenome
from codons import random_codons
from tasks import TaskCase


SPLIT_CASES: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in (1, 2, 3) for b in (1, 2, 3)
)
CHEMISTRY_ROUNDS = 6
POPULATION_SIZE = 8
SURVIVORS = 3
PARTNER_SAMPLES = 3
GENERATIONS = 200


def _make_case(a: int, b: int) -> TaskCase:
    return TaskCase(inputs=(float(a), float(b)), target=(a * b) / 9.0)


def _run_pair(
    genome_a: CellGenome,
    genome_b: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[float, float, float, float, float]:
    """Return (mean_mse, probe_c1_out, probe_c2_out, probe_c1_s3, probe_c2_s3)."""
    total_error = 0.0
    probe_c1 = 0.0
    probe_c2 = 0.0
    probe_c1_s3 = 0.0
    probe_c2_s3 = 0.0

    for a, b in SPLIT_CASES:
        case = _make_case(a, b)
        cell1 = CellState()
        cell1.signals = [a / 3.0, 0.0, 0.0, 0.0, 0.0]
        cell2 = CellState()
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell1, cell2], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c1 = cell1.output if cell1.output is not None else cell1.signals[2]
        c2 = cell2.output if cell2.output is not None else cell2.signals[2]
        colony = (c1 + c2) / 2.0
        total_error += (colony - case.target) ** 2
        if a == 2 and b == 3:
            probe_c1 = c1
            probe_c2 = c2
            probe_c1_s3 = cell1.signals[3]
            probe_c2_s3 = cell2.signals[3]

    return total_error / len(SPLIT_CASES), probe_c1, probe_c2, probe_c1_s3, probe_c2_s3


def _score_genome(
    genome: CellGenome,
    role: str,
    partners: list[CellGenome],
    system: CooperativeChemistrySystem,
    rng: Random,
) -> float:
    """Mean colony MSE across PARTNER_SAMPLES random partners."""
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    errors = []
    for partner in sampled:
        if role == "A":
            err, *_ = _run_pair(genome, partner, system)
        else:
            err, *_ = _run_pair(partner, genome, system)
        errors.append(err)
    return sum(errors) / len(errors)


def _random_genome(rng: Random, lineage_id: str) -> CellGenome:
    length = rng.randint(6, 20)
    return CellGenome(
        codons=tuple(random_codons(rng, length)),
        local_motifs=(),
        lineage_id=lineage_id,
    )


def main() -> None:
    rng = Random(31)
    system = CooperativeChemistrySystem()
    evo_config = EvolutionConfig()

    pop_a = [_random_genome(rng, f"A{i+1}") for i in range(POPULATION_SIZE)]
    pop_b = [_random_genome(rng, f"B{i+1}") for i in range(POPULATION_SIZE)]

    best_pair_error = float("inf")
    best_a_lineage = pop_a[0].lineage_id
    best_b_lineage = pop_b[0].lineage_id

    for generation in range(GENERATIONS):
        # Score population A
        scores_a = [
            (_score_genome(g, "A", pop_b, system, rng), g)
            for g in pop_a
        ]
        scores_a.sort(key=lambda x: x[0])

        # Score population B
        scores_b = [
            (_score_genome(g, "B", pop_a, system, rng), g)
            for g in pop_b
        ]
        scores_b.sort(key=lambda x: x[0])

        # Find best pair this generation
        gen_best_error = float("inf")
        gen_best_c1 = 0.0
        gen_best_c2 = 0.0
        for err_a, ga in scores_a[:3]:
            for err_b, gb in scores_b[:3]:
                pair_err, c1, c2, _, _ = _run_pair(ga, gb, system)
                if pair_err < gen_best_error:
                    gen_best_error = pair_err
                    gen_best_c1 = c1
                    gen_best_c2 = c2
                    best_a_lineage = ga.lineage_id
                    best_b_lineage = gb.lineage_id

        if gen_best_error < best_pair_error:
            best_pair_error = gen_best_error

        mean_error = (
            sum(s[0] for s in scores_a) + sum(s[0] for s in scores_b)
        ) / (len(scores_a) + len(scores_b))

        print(
            f"epistasis_colony2: gen={generation} "
            f"mean_error={mean_error:.6f} best_error={gen_best_error:.6f} "
            f"cell1_out={gen_best_c1:.4f} cell2_out={gen_best_c2:.4f} "
            f"lineage_a={best_a_lineage} lineage_b={best_b_lineage}"
        )

        if generation % 20 == 0:
            best_a = scores_a[0][1]
            best_b = scores_b[0][1]
            probe_case = _make_case(2, 3)
            cell1 = CellState()
            cell1.signals = [2 / 3.0, 0.0, 0.0, 0.0, 0.0]
            cell2 = CellState()
            cell2.signals = [0.0, 1.0, 0.0, 0.0, 0.0]
            context = ChemistryContext()
            system.run([cell1, cell2], probe_case, context=context, max_time=float(CHEMISTRY_ROUNDS))
            print(
                f"epistasis_colony2: snapshot gen={generation} "
                f"cell1_signals={[round(s, 3) for s in cell1.signals]} "
                f"cell2_signals={[round(s, 3) for s in cell2.signals]}"
            )

        # Reproduce population A
        survivors_a = [g for _, g in scores_a[:SURVIVORS]]
        next_a: list[CellGenome] = list(survivors_a)
        while len(next_a) < POPULATION_SIZE:
            parent = rng.choice(survivors_a)
            child_id = f"{parent.lineage_id}.{generation+1}.{len(next_a)}"
            next_a.append(mutate_genome(parent, child_id, rng, config=evo_config))
        pop_a = next_a

        # Reproduce population B
        survivors_b = [g for _, g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            child_id = f"{parent.lineage_id}.{generation+1}.{len(next_b)}"
            next_b.append(mutate_genome(parent, child_id, rng, config=evo_config))
        pop_b = next_b


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a smoke test (first 3 generations only)**

```bash
python3 -c "
import sys
sys.argv = []
from pathlib import Path
import subprocess, sys
result = subprocess.run(
    [sys.executable, 'experiments/31_epistasis_colony2.py'],
    capture_output=True, text=True, timeout=60,
    cwd='$(pwd)'
)
print(result.stdout[:500])
print(result.stderr[:200] if result.stderr else '')
" 2>&1 | head -20
```

Simpler — just run and grab first few lines:

```bash
timeout 60 python3 experiments/31_epistasis_colony2.py 2>&1 | head -5
```

Expected: lines starting with `epistasis_colony2: gen=0 ...`

- [ ] **Step 3: Commit**

```bash
git add experiments/31_epistasis_colony2.py
git commit -m "feat: add epistasis colony 2 experiment (31)"
```

---

### Task 2: Register in `test_experiments.py`

**Files:**
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Add to script list**

In `tests/test_experiments.py`, find the `scripts` list. Add after `"30_epistasis_colony.py"`:

```python
"31_epistasis_colony2.py",
```

- [ ] **Step 2: Add recognised prefix**

Find the `assert any(prefix in completed.stdout ...)` block. Add `"epistasis_colony2:"` to the tuple:

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
                "epistasis_colony2:",
            )
        ), f"{script_name} produced no recognised output prefix:\n{completed.stdout}"
```

- [ ] **Step 3: Run fast suite**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_experiments.py
git commit -m "test: register epistasis colony 2 in test_experiments"
```

---

### Task 3: Run full experiment and observe

**Files:** none

- [ ] **Step 1: Run the full 200-generation experiment**

```bash
python3 experiments/31_epistasis_colony2.py 2>&1 | tee /tmp/colony2_run.txt
```

- [ ] **Step 2: Check error trend**

```bash
grep "epistasis_colony2: gen=" /tmp/colony2_run.txt | awk -F'best_error=' '{print NR, $2}' | awk '{print $1, $2}' | head -20
grep "epistasis_colony2: gen=" /tmp/colony2_run.txt | awk -F'best_error=' '{print NR, $2}' | awk '{print $1, $2}' | tail -10
```

Expected: `best_error` decreasing over generations.

- [ ] **Step 3: Check cell output divergence**

```bash
grep "epistasis_colony2: gen=1[5-9][0-9]" /tmp/colony2_run.txt | head -10
```

Expected: `cell1_out` and `cell2_out` differ by > 0.05 in later generations.

- [ ] **Step 4: Check snapshots for signal divergence**

```bash
grep "snapshot" /tmp/colony2_run.txt
```

Expected: `cell1_signals` and `cell2_signals` develop different patterns as generations progress.
