# Epistasis Colony 5 Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `experiments/34_epistasis_colony5.py` — three-cell symmetric colony with split inputs (a/3, b/3, c/3), warm-started from exp 33's proven two-cell genomes, evolving to compute `a*b*c/27` via emergent self-organisation.

**Architecture:** Three independent populations (pop_a, pop_b, pop_c). Warm-start: re-run exp 32 curriculum (seed=32, 200 gens) + exp 33 gate phase (seed=33, 400 gens) to recover pop_a and pop_b; pop_c initialised randomly (seed=34). Colony fitness = best-of-three cell outputs vs target. Individual fitness = own cell's output error vs target, averaged over best-of-3 partner trios. Gate phase: 400 generations. Cases: all 27 combinations of a,b,c ∈ {1,2,3}.

**Tech Stack:** Python 3.10+, existing `CooperativeChemistrySystem`, `mutate_genome`, pytest.

---

## File Map

- Create: `experiments/34_epistasis_colony5.py`
- Modify: `tests/test_experiments.py` — add script + prefix

---

### Task 1: Create the experiment script

**Files:**
- Create: `experiments/34_epistasis_colony5.py`

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
from evolution import mutate_genome
from genome import CellGenome
from codons import random_codons
from tasks import TaskCase


SPLIT_CASES_3: tuple[tuple[int, int, int], ...] = tuple(
    (a, b, c) for a in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3)
)
SPLIT_CASES_2: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in (1, 2, 3) for b in (1, 2, 3)
)
CHEMISTRY_ROUNDS = 6
POPULATION_SIZE = 8
SURVIVORS = 3
PARTNER_SAMPLES = 3
WARMUP_GENS = 200
GATE_GENS_2 = 400
GATE_GENS_3 = 400


def _make_case_2(a: int, b: int) -> TaskCase:
    return TaskCase(x=float(a), y=float(b), target=(a * b) / 9.0, task_name="multiply")


def _make_case_3(a: int, b: int, c: int) -> TaskCase:
    return TaskCase(x=float(a), y=float(b), target=(a * b * c) / 27.0, task_name="multiply")


def _random_genome(rng: Random, lineage_id: str) -> CellGenome:
    length = rng.randint(6, 20)
    return CellGenome(
        codons=tuple(random_codons(rng, length)),
        local_motifs=(),
        lineage_id=lineage_id,
    )


# ── Exp 32 curriculum warm-up ────────────────────────────────────────────────

def _warmup_alpha(generation: int) -> tuple[float, str, str]:
    if generation < 100:
        return generation / 99.0, "s1", "s2"
    return (generation - 100) / 99.0, "s2", "s3"


def _warmup_run_pair(
    genome_a: CellGenome,
    genome_b: CellGenome,
    system: CooperativeChemistrySystem,
) -> dict[str, float]:
    s1_a = s1_b = s2_a = s2_b = s3_err = 0.0
    for a, b in SPLIT_CASES_2:
        case = _make_case_2(a, b)
        cell1 = CellState(active_rules=list(genome_a.declare_rules()))
        cell1.signals = [a / 3.0, 0.0, 0.0, 0.0, 0.0]
        cell2 = CellState(active_rules=list(genome_b.declare_rules()))
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell1, cell2], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c1 = cell1.output if cell1.output is not None else cell1.signals[2]
        c2 = cell2.output if cell2.output is not None else cell2.signals[2]
        s1_a += (c1 - a / 3.0) ** 2
        s1_b += (c2 - b / 3.0) ** 2
        s2_a += (c1 - b / 3.0) ** 2
        s2_b += (c2 - a / 3.0) ** 2
        s3_err += ((c1 + c2) / 2.0 - (a * b) / 9.0) ** 2
    n = len(SPLIT_CASES_2)
    return {"s1_a": s1_a/n, "s1_b": s1_b/n, "s2_a": s2_a/n, "s2_b": s2_b/n, "s3": s3_err/n}


def _warmup_blended(metrics: dict[str, float], role: str, alpha: float, sf: str, st: str) -> float:
    def e(s: str) -> float:
        return metrics.get(f"{s}_{role}" if s != "s3" else "s3", 0.0)
    return (1.0 - alpha) * e(sf) + alpha * e(st)


def _warmup_score(genome: CellGenome, role: str, partners: list[CellGenome],
                  system: CooperativeChemistrySystem, rng: Random,
                  alpha: float, sf: str, st: str) -> float:
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    errors = []
    for partner in sampled:
        m = _warmup_run_pair(genome, partner, system) if role == "A" else _warmup_run_pair(partner, genome, system)
        errors.append(_warmup_blended(m, role.lower(), alpha, sf, st))
    return sum(errors) / len(errors)


def _run_exp32_warmup(system: CooperativeChemistrySystem) -> tuple[list[CellGenome], list[CellGenome]]:
    rng = Random(32)
    pop_a = [_random_genome(rng, f"A{i+1}") for i in range(POPULATION_SIZE)]
    pop_b = [_random_genome(rng, f"B{i+1}") for i in range(POPULATION_SIZE)]
    for generation in range(WARMUP_GENS):
        alpha, sf, st = _warmup_alpha(generation)
        scores_a = sorted(
            [(_warmup_score(g, "A", pop_b, system, rng, alpha, sf, st), g) for g in pop_a],
            key=lambda x: x[0]
        )
        scores_b = sorted(
            [(_warmup_score(g, "B", pop_a, system, rng, alpha, sf, st), g) for g in pop_b],
            key=lambda x: x[0]
        )
        survivors_a = [g for _, g in scores_a[:SURVIVORS]]
        next_a: list[CellGenome] = list(survivors_a)
        while len(next_a) < POPULATION_SIZE:
            parent = rng.choice(survivors_a)
            next_a.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.{generation+1}.{len(next_a)}"))
        pop_a = next_a
        survivors_b = [g for _, g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            next_b.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.{generation+1}.{len(next_b)}"))
        pop_b = next_b
    return pop_a, pop_b


# ── Exp 33 gate phase ─────────────────────────────────────────────────────────

def _gate_errors_2(
    genome_a: CellGenome,
    genome_b: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[float, float]:
    gate_err = echo_err = 0.0
    for a, b in SPLIT_CASES_2:
        case = _make_case_2(a, b)
        cell1 = CellState(active_rules=list(genome_a.declare_rules()))
        cell1.signals = [a / 3.0, 0.0, 0.0, 0.0, 0.0]
        cell2 = CellState(active_rules=list(genome_b.declare_rules()))
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell1, cell2], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c1 = cell1.output if cell1.output is not None else cell1.signals[2]
        c2 = cell2.output if cell2.output is not None else cell2.signals[2]
        s3 = max(0.0, min(1.0, cell1.signals[3]))
        gate_err += (c1 * s3 - (a * b) / 9.0) ** 2
        echo_err += (c2 - b / 3.0) ** 2
    n = len(SPLIT_CASES_2)
    return gate_err / n, echo_err / n


def _score_a_2(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    return sum(_gate_errors_2(genome, p, system)[0] for p in sampled) / PARTNER_SAMPLES


def _score_b_2(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    return sum(_gate_errors_2(p, genome, system)[1] for p in sampled) / PARTNER_SAMPLES


def _run_exp33_gate(
    pop_a: list[CellGenome],
    pop_b: list[CellGenome],
    system: CooperativeChemistrySystem,
) -> tuple[list[CellGenome], list[CellGenome]]:
    rng = Random(33)
    for generation in range(GATE_GENS_2):
        scores_a = sorted([(_score_a_2(g, pop_b, system, rng), g) for g in pop_a], key=lambda x: x[0])
        scores_b = sorted([(_score_b_2(g, pop_a, system, rng), g) for g in pop_b], key=lambda x: x[0])
        survivors_a = [g for _, g in scores_a[:SURVIVORS]]
        next_a: list[CellGenome] = list(survivors_a)
        while len(next_a) < POPULATION_SIZE:
            parent = rng.choice(survivors_a)
            next_a.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation+1}.{len(next_a)}"))
        pop_a = next_a
        survivors_b = [g for _, g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            next_b.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation+1}.{len(next_b)}"))
        pop_b = next_b
    return pop_a, pop_b


def _run_warmup(system: CooperativeChemistrySystem) -> tuple[list[CellGenome], list[CellGenome]]:
    print("epistasis_colony5: warmup exp32 curriculum (200 gens)...")
    pop_a, pop_b = _run_exp32_warmup(system)
    print("epistasis_colony5: warmup exp33 gate (400 gens)...")
    pop_a, pop_b = _run_exp33_gate(pop_a, pop_b, system)
    print("epistasis_colony5: warmup complete")
    return pop_a, pop_b


# ── Three-cell gate phase ─────────────────────────────────────────────────────

def _run_trio(
    genome_a: CellGenome,
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[float, float, float, float, list[float]]:
    """Return (colony_err, score_a, score_b, score_c, probe) over all 27 cases."""
    colony_err = score_a = score_b = score_c = 0.0
    probe = [0.0, 0.0, 0.0, 0.0]  # c1, c2, c3, best_out

    for a, b, c in SPLIT_CASES_3:
        case = _make_case_3(a, b, c)
        target = (a * b * c) / 27.0
        cell1 = CellState(active_rules=list(genome_a.declare_rules()))
        cell1.signals = [a / 3.0, 0.0, 0.0, 0.0, 0.0]
        cell2 = CellState(active_rules=list(genome_b.declare_rules()))
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        cell3 = CellState(active_rules=list(genome_c.declare_rules()))
        cell3.signals = [0.0, 0.0, c / 3.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell1, cell2, cell3], case, context=context, max_time=float(CHEMISTRY_ROUNDS))

        c1 = cell1.output if cell1.output is not None else cell1.signals[2]
        c2 = cell2.output if cell2.output is not None else cell2.signals[2]
        c3 = cell3.output if cell3.output is not None else cell3.signals[2]

        e1 = (c1 - target) ** 2
        e2 = (c2 - target) ** 2
        e3 = (c3 - target) ** 2
        colony_err += min(e1, e2, e3)
        score_a += e1
        score_b += e2
        score_c += e3

        if a == 2 and b == 2 and c == 2:
            best_out = min([(e1, c1), (e2, c2), (e3, c3)])[1]
            probe = [c1, c2, c3, best_out]

    n = len(SPLIT_CASES_3)
    return colony_err / n, score_a / n, score_b / n, score_c / n, probe


def _score_genome_a(genome: CellGenome, pop_b: list[CellGenome], pop_c: list[CellGenome],
                    system: CooperativeChemistrySystem, rng: Random) -> float:
    total = 0.0
    for _ in range(PARTNER_SAMPLES):
        gb = rng.choice(pop_b)
        gc = rng.choice(pop_c)
        _, sa, _, _, _ = _run_trio(genome, gb, gc, system)
        total += sa
    return total / PARTNER_SAMPLES


def _score_genome_b(genome: CellGenome, pop_a: list[CellGenome], pop_c: list[CellGenome],
                    system: CooperativeChemistrySystem, rng: Random) -> float:
    total = 0.0
    for _ in range(PARTNER_SAMPLES):
        ga = rng.choice(pop_a)
        gc = rng.choice(pop_c)
        _, _, sb, _, _ = _run_trio(ga, genome, gc, system)
        total += sb
    return total / PARTNER_SAMPLES


def _score_genome_c(genome: CellGenome, pop_a: list[CellGenome], pop_b: list[CellGenome],
                    system: CooperativeChemistrySystem, rng: Random) -> float:
    total = 0.0
    for _ in range(PARTNER_SAMPLES):
        ga = rng.choice(pop_a)
        gb = rng.choice(pop_b)
        _, _, _, sc, _ = _run_trio(ga, gb, genome, system)
        total += sc
    return total / PARTNER_SAMPLES


def main() -> None:
    system = CooperativeChemistrySystem()
    pop_a, pop_b = _run_warmup(system)

    rng = Random(34)
    pop_c = [_random_genome(rng, f"C{i+1}") for i in range(POPULATION_SIZE)]

    best_a_lineage = pop_a[0].lineage_id
    best_b_lineage = pop_b[0].lineage_id
    best_c_lineage = pop_c[0].lineage_id

    for generation in range(GATE_GENS_3):
        scores_a = sorted([(_score_genome_a(g, pop_b, pop_c, system, rng), g) for g in pop_a], key=lambda x: x[0])
        scores_b = sorted([(_score_genome_b(g, pop_a, pop_c, system, rng), g) for g in pop_b], key=lambda x: x[0])
        scores_c = sorted([(_score_genome_c(g, pop_a, pop_b, system, rng), g) for g in pop_c], key=lambda x: x[0])

        gen_best_colony = float("inf")
        probe_best = [0.0, 0.0, 0.0, 0.0]
        best_cell_idx = 0
        for _, ga in scores_a[:3]:
            for _, gb in scores_b[:3]:
                for _, gc in scores_c[:3]:
                    ce, _, _, _, probe = _run_trio(ga, gb, gc, system)
                    if ce < gen_best_colony:
                        gen_best_colony = ce
                        probe_best = probe
                        best_a_lineage = ga.lineage_id
                        best_b_lineage = gb.lineage_id
                        best_c_lineage = gc.lineage_id
                        c1, c2, c3, _ = probe
                        target_probe = (2 * 2 * 2) / 27.0
                        errs = [(abs(c1 - target_probe), 0), (abs(c2 - target_probe), 1), (abs(c3 - target_probe), 2)]
                        best_cell_idx = min(errs)[1]

        c1, c2, c3, best_out = probe_best
        print(
            f"epistasis_colony5: gen={generation} colony_err={gen_best_colony:.6f} "
            f"best_cell={best_cell_idx} probe_out={best_out:.4f} probe_target=0.2963 "
            f"cell1_out={c1:.4f} cell2_out={c2:.4f} cell3_out={c3:.4f} "
            f"lineage_a={best_a_lineage} lineage_b={best_b_lineage} lineage_c={best_c_lineage}"
        )

        if generation % 20 == 0:
            best_a = scores_a[0][1]
            best_b = scores_b[0][1]
            best_c = scores_c[0][1]
            probe_case = _make_case_3(2, 2, 2)
            cell1 = CellState(active_rules=list(best_a.declare_rules()))
            cell1.signals = [2 / 3.0, 0.0, 0.0, 0.0, 0.0]
            cell2 = CellState(active_rules=list(best_b.declare_rules()))
            cell2.signals = [0.0, 2 / 3.0, 0.0, 0.0, 0.0]
            cell3 = CellState(active_rules=list(best_c.declare_rules()))
            cell3.signals = [0.0, 0.0, 2 / 3.0, 0.0, 0.0]
            context = ChemistryContext()
            system.run([cell1, cell2, cell3], probe_case, context=context, max_time=float(CHEMISTRY_ROUNDS))
            print(
                f"epistasis_colony5: snapshot gen={generation} "
                f"cell1_signals={[round(s, 3) for s in cell1.signals]} "
                f"cell2_signals={[round(s, 3) for s in cell2.signals]} "
                f"cell3_signals={[round(s, 3) for s in cell3.signals]}"
            )

        survivors_a = [g for _, g in scores_a[:SURVIVORS]]
        next_a: list[CellGenome] = list(survivors_a)
        while len(next_a) < POPULATION_SIZE:
            parent = rng.choice(survivors_a)
            next_a.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation+1}.{len(next_a)}"))
        pop_a = next_a

        survivors_b = [g for _, g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            next_b.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation+1}.{len(next_b)}"))
        pop_b = next_b

        survivors_c = [g for _, g in scores_c[:SURVIVORS]]
        next_c: list[CellGenome] = list(survivors_c)
        while len(next_c) < POPULATION_SIZE:
            parent = rng.choice(survivors_c)
            next_c.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation+1}.{len(next_c)}"))
        pop_c = next_c


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```bash
timeout 300 python3 experiments/34_epistasis_colony5.py 2>&1 | head -6
```

Expected: warmup messages, then lines starting with `epistasis_colony5: gen=0 ...`

- [ ] **Step 3: Commit**

```bash
git add experiments/34_epistasis_colony5.py
git commit -m "feat: add epistasis colony 5 three-cell symmetric experiment (34)"
```

---

### Task 2: Register in `test_experiments.py`

**Files:**
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Add to script list**

Find `"33_epistasis_colony4.py",` and add after:

```python
"34_epistasis_colony5.py",
```

- [ ] **Step 2: Add recognised prefix**

Find `"epistasis_colony4:",` and add after:

```python
"epistasis_colony5:",
```

- [ ] **Step 3: Run fast suite**

```bash
python3 -m pytest -q -k "not test_remaining_experiments"
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_experiments.py
git commit -m "test: register epistasis colony 5 in test_experiments"
```

---

### Task 3: Run and observe

- [ ] **Step 1: Run full experiment**

```bash
python3 experiments/34_epistasis_colony5.py 2>&1 | tee /tmp/colony5_run.txt
```

- [ ] **Step 2: Check colony error trend**

```bash
grep "epistasis_colony5: gen=" /tmp/colony5_run.txt | grep -oP 'gen=\K\d+|colony_err=\K[0-9.]+' | paste - - | awk 'NR%20==0 || NR==1'
```

Expected: `colony_err` declining from initial value.

- [ ] **Step 3: Check best_cell stability**

```bash
grep "epistasis_colony5: gen=" /tmp/colony5_run.txt | grep -oP 'best_cell=\K\d' | sort | uniq -c
```

Expected: one cell index dominates, indicating emergent role specialisation.

- [ ] **Step 4: Check probe_out approaching target**

```bash
grep "epistasis_colony5: gen=" /tmp/colony5_run.txt | grep -oP 'probe_out=\K[0-9.]+' | tail -20
```

Expected: `probe_out` approaching 0.2963 (= 8/27) in later generations.
