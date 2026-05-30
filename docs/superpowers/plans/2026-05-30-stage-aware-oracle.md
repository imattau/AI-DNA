# Stage-Aware Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stage detection and stage-constrained template menus to `CodonOracle`, then wire them into experiment 38 so tinyllama proposes COPY codons when cell2 can't echo, SENSE_PEER codons when it can't gate, and SCALE codons when the gate product needs refinement.

**Architecture:** Two new methods on `CodonOracle` (`detect_stage`, `build_stage_prompt`) and an extended `try_inject` signature. Experiment 38 computes echo_err and cell2_s3_mean each generation alongside gate_err, passes them to the oracle, and logs the detected stage.

**Tech Stack:** Python stdlib only; tinyllama via ollama HTTP; pytest for tests.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `codon_oracle.py` | Modify | Add `detect_stage`, `build_stage_prompt`; extend `try_inject` |
| `experiments/38_stage_oracle.py` | Create | bc gate loop + separate echo_err + stage oracle wiring |
| `tests/test_codon_oracle.py` | Modify | Add 6 stage-detection tests |
| `tests/test_experiments.py` | Modify | Register exp38 |

---

### Task 1: Add stage detection tests and implementation

**Files:**
- Modify: `tests/test_codon_oracle.py`
- Modify: `codon_oracle.py`

- [ ] **Step 1: Add failing tests to tests/test_codon_oracle.py**

Append these tests to the existing file:

```python
def test_detect_stage_echo():
    oracle = CodonOracle()
    assert oracle.detect_stage(echo_err=0.10, gate_err=0.27, cell2_s3_mean=0.0) == "echo"


def test_detect_stage_sense():
    oracle = CodonOracle()
    assert oracle.detect_stage(echo_err=0.02, gate_err=0.27, cell2_s3_mean=0.05) == "sense"


def test_detect_stage_scale():
    oracle = CodonOracle()
    assert oracle.detect_stage(echo_err=0.02, gate_err=0.10, cell2_s3_mean=0.50) == "scale"


def test_build_stage_prompt_echo_contains_copy():
    oracle = CodonOracle()
    prompt = oracle.build_stage_prompt("echo", ["SENSE_PEER_0"], [0.27, 0.27], ["RULE_COPY1_2"])
    assert "COPY_S_TO_D" in prompt
    assert "SENSE_PEER" not in prompt


def test_build_stage_prompt_sense_contains_sense_peer():
    oracle = CodonOracle()
    prompt = oracle.build_stage_prompt("sense", ["SENSE_PEER_0"], [0.27, 0.27], ["RULE_COPY1_2"])
    assert "SENSE_PEER_S_TO_D" in prompt
    assert "COPY_S_TO_D" not in prompt


def test_build_stage_prompt_scale_contains_scale_by():
    oracle = CodonOracle()
    prompt = oracle.build_stage_prompt("scale", ["SENSE_PEER_0"], [0.10, 0.10], ["RULE_COPY1_2"])
    assert "SCALE_BY_SN" in prompt
    assert "COPY_S_TO_D" not in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_codon_oracle.py::test_detect_stage_echo -v`
Expected: `FAILED` with `AttributeError: 'CodonOracle' object has no attribute 'detect_stage'`

- [ ] **Step 3: Add `detect_stage` and `build_stage_prompt` to codon_oracle.py**

Add these methods to the `CodonOracle` class (after `build_prompt`):

```python
def detect_stage(self, echo_err: float, gate_err: float, cell2_s3_mean: float) -> str:
    if echo_err > 0.05:
        return "echo"
    if cell2_s3_mean < 0.1:
        return "sense"
    return "scale"

def build_stage_prompt(
    self,
    stage: str,
    codon_names: list[str],
    fitness_history: list[float],
    best_motif: list[str],
) -> str:
    history_str = ", ".join(f"{v:.4f}" for v in fitness_history[-10:])
    motif_str = " → ".join(best_motif) if best_motif else "none"
    names_str = ", ".join(codon_names[:20])

    if stage == "echo":
        menu = "COPY_S_TO_D (params: s, d in 0-4, s≠d)"
        framing = "cell2 cannot yet output proportional values — scaffold signal routing within the cell"
        example = '{"template": "COPY_S_TO_D", "s": 1, "d": 2}'
    elif stage == "sense":
        menu = "SENSE_PEER_S_TO_D (params: s, d in 0-4, s≠d)"
        framing = "cell2 outputs correctly but cannot read from its peer — scaffold cross-cell sensing"
        example = '{"template": "SENSE_PEER_S_TO_D", "s": 2, "d": 3}'
    else:  # scale
        menu = "SCALE_BY_SN (param: n in 0-4), IF_SN_GT (param: n in 0-4), IF_SN_LT (param: n in 0-4)"
        framing = "cell2 reads peer but gate product needs refinement — scaffold conditional scaling"
        example = '{"template": "SCALE_BY_SN", "n": 3}'

    return (
        f"You are a genome expander for an evolutionary system. "
        f"Current codons: {names_str}. "
        f"Recent fitness (lower=better): [{history_str}]. "
        f"Best circuit: {motif_str}. "
        f"Stage: {stage} — {framing}. "
        f"Propose ONE new codon using exactly this template: {menu}. "
        f"Reply with ONLY a JSON object, e.g.: {example}"
    )
```

- [ ] **Step 4: Run all stage tests**

Run: `python3 -m pytest tests/test_codon_oracle.py -v -k "stage"`
Expected: 6 tests pass

- [ ] **Step 5: Extend `try_inject` to accept stage params**

Replace the existing `try_inject` signature in `codon_oracle.py`:

```python
def try_inject(
    self,
    generation: int,
    gate_err_history: list[float],
    codon_names: list[str],
    best_motif: list[str],
    rulebook: dict[str, Rule],
    codon_map: dict[int, str],
    op_names: list[str],
    echo_err: float = 0.5,
    gate_err: float = 0.5,
    cell2_s3_mean: float = 0.0,
) -> None:
    if not self.is_stagnating(gate_err_history):
        return
    stage = self.detect_stage(echo_err, gate_err, cell2_s3_mean)
    prompt = self.build_stage_prompt(stage, codon_names, gate_err_history, best_motif)
    proposal = self.query(prompt)
    if proposal is None:
        print(f"llm_oracle: gen={generation} stage={stage} ollama_unreachable_or_bad_json")
        return
    name = self.inject(proposal, rulebook, codon_map, op_names)
    print(
        f"llm_oracle: gen={generation} stage={stage} proposed={proposal} "
        f"injected={'True' if name else 'False'} codon={name}"
    )
```

Note: default values preserve backward compatibility with exp37's existing `try_inject` calls.

- [ ] **Step 6: Run full oracle test suite**

Run: `python3 -m pytest tests/test_codon_oracle.py -v`
Expected: all 13 tests pass

- [ ] **Step 7: Commit**

```bash
git add codon_oracle.py tests/test_codon_oracle.py
git commit -m "Add stage detection to CodonOracle (detect_stage, build_stage_prompt)"
```

---

### Task 2: Create experiment 38

**Files:**
- Create: `experiments/38_stage_oracle.py`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write experiments/38_stage_oracle.py**

```python
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from random import Random
from cell import CellState
from chemistry import ChemistryContext, build_rulebook
from codons import REGULATORY_CODON_MAP, REGULATORY_OP_NAMES
from cooperative_chemistry import CooperativeChemistrySystem
from codon_oracle import CodonOracle
from evolution import mutate_genome
from genome import CellGenome, extract_motif_from_rules
from codons import random_codons
from tasks import TaskCase
from motif_store import MotifStore

SPLIT_CASES: tuple[tuple[int, int], ...] = tuple(
    (b, c) for b in (1, 2, 3) for c in (1, 2, 3)
)
CHEMISTRY_ROUNDS = 6
POPULATION_SIZE = 8
SURVIVORS = 3
PARTNER_SAMPLES = 3
GATE_GENS = 600
ORACLE_WINDOW = 50


def _make_case(b: int, c: int) -> TaskCase:
    return TaskCase(x=float(b), y=float(c), target=(b * c) / 9.0, task_name="multiply")


def _run_pair(
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[float, float, float, list[float]]:
    """Return (gate_err, echo_err_b, cell2_s3_mean, probe) over all 9 cases."""
    gate_errs, echo_errs, s3s = [], [], []
    probe = [0.0, 0.0, 0.0, 0.0]
    for b, c in SPLIT_CASES:
        case = _make_case(b, c)
        cell2 = CellState(active_rules=list(genome_b.declare_rules()))
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        cell3 = CellState(active_rules=list(genome_c.declare_rules()))
        cell3.signals = [0.0, 0.0, c / 3.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell2, cell3], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c2 = cell2.output if cell2.output is not None else cell2.signals[2]
        c3 = cell3.output if cell3.output is not None else cell3.signals[2]
        s3 = max(0.0, min(1.0, cell2.signals[3]))
        gate_errs.append((c2 * s3 - (b * c) / 9.0) ** 2)
        echo_errs.append((c2 - b / 3.0) ** 2)
        s3s.append(s3)
        if b == 2 and c == 3:
            probe = [c2, c3, s3, c2 * s3]
    n = len(SPLIT_CASES)
    return sum(gate_errs) / n, sum(echo_errs) / n, sum(s3s) / n, probe


def _score_b(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    return sum(_run_pair(genome, rng.choice(partners), system)[0] for _ in range(PARTNER_SAMPLES)) / PARTNER_SAMPLES


def _score_c(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    errors = []
    for _ in range(PARTNER_SAMPLES):
        partner = rng.choice(partners)
        case = _make_case(2, 3)
        cell2 = CellState(active_rules=list(partner.declare_rules()))
        cell2.signals = [0.0, 2 / 3.0, 0.0, 0.0, 0.0]
        cell3 = CellState(active_rules=list(genome.declare_rules()))
        cell3.signals = [0.0, 0.0, 1.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell2, cell3], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c3 = cell3.output if cell3.output is not None else cell3.signals[2]
        errors.append((c3 - 1.0) ** 2)
    return sum(errors) / len(errors)


def _random_genome(rng: Random, lineage_id: str) -> CellGenome:
    return CellGenome(codons=tuple(random_codons(rng, rng.randint(6, 20))), local_motifs=(), lineage_id=lineage_id)


def main() -> None:
    system = CooperativeChemistrySystem()
    rng = Random(38)
    oracle = CodonOracle()

    live_rulebook = system.rulebook if hasattr(system, 'rulebook') else build_rulebook()
    live_codon_map = dict(REGULATORY_CODON_MAP)
    live_op_names = list(REGULATORY_OP_NAMES)

    pop_b = [_random_genome(rng, f"B{i+1}") for i in range(POPULATION_SIZE)]
    pop_c = [_random_genome(rng, f"C{i+1}") for i in range(POPULATION_SIZE)]

    store = MotifStore()
    motif_captured = False
    gate_err_history: list[float] = []

    for generation in range(GATE_GENS):
        scores_b = sorted([(_score_b(g, pop_c, system, rng), g) for g in pop_b], key=lambda x: x[0])
        scores_c = sorted([(_score_c(g, pop_b, system, rng), g) for g in pop_c], key=lambda x: x[0])

        gen_best_gate = float("inf")
        gen_best_echo = 0.5
        gen_best_s3 = 0.0
        probe_best = [0.0, 0.0, 0.0, 0.0]
        for _, gb in scores_b[:3]:
            for _, gc in scores_c[:3]:
                ge, ee, s3m, probe = _run_pair(gb, gc, system)
                if ge < gen_best_gate:
                    gen_best_gate = ge
                    gen_best_echo = ee
                    gen_best_s3 = s3m
                    probe_best = probe

        gate_err_history.append(gen_best_gate)
        c2, c3, s3, colony = probe_best
        stage = oracle.detect_stage(gen_best_echo, gen_best_gate, gen_best_s3)
        print(
            f"stage_oracle: gen={generation} stage={stage} gate_err={gen_best_gate:.6f} "
            f"echo_err={gen_best_echo:.6f} cell2_s3={gen_best_s3:.4f} "
            f"cell2_out={c2:.4f} colony_out={colony:.4f}"
        )

        best_motif = [str(r) for r in scores_b[0][1].declare_rules() if isinstance(r, str)]
        oracle.try_inject(
            generation, gate_err_history,
            live_op_names, best_motif,
            live_rulebook, live_codon_map, live_op_names,
            echo_err=gen_best_echo,
            gate_err=gen_best_gate,
            cell2_s3_mean=gen_best_s3,
        )

        if not motif_captured and gen_best_gate < 0.05:
            store.save(
                extract_motif_from_rules(
                    [r for r in scores_b[0][1].declare_rules() if isinstance(r, str)],
                    origin_lineage=scores_b[0][1].lineage_id,
                    origin_task="multiply_3cell_bc",
                    origin_signals=(0.0, 2/3.0, 0.0, 0.0, 0.0),
                ),
                role="gate_bc", task="multiply_3cell_bc",
                gate_err=gen_best_gate, generation=generation, experiment="stage_oracle",
            )
            print(f"stage_oracle: motif_captured gen={generation} gate_err={gen_best_gate:.6f}")
            motif_captured = True

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

Run: `timeout 20 python3 -u experiments/38_stage_oracle.py 2>&1 | head -5`
Expected: lines starting with `stage_oracle: gen=0 stage=echo gate_err=...`

- [ ] **Step 3: Register in test suite**

In `tests/test_experiments.py`:
- Add `"38_stage_oracle.py"` to the `scripts` list
- Add `"stage_oracle:"` to the recognised prefixes list

- [ ] **Step 4: Commit**

```bash
git add experiments/38_stage_oracle.py tests/test_experiments.py
git commit -m "Add experiment 38: stage-aware LLM codon oracle"
```

---

## Self-Review

**Spec coverage:**
- ✅ `detect_stage(echo_err, gate_err, cell2_s3_mean)` → stage string
- ✅ `build_stage_prompt` with constrained menu per stage
- ✅ `try_inject` extended with echo_err, gate_err, cell2_s3_mean params (backward compatible defaults)
- ✅ echo_err and cell2_s3_mean computed per generation in exp38 via `_run_pair`
- ✅ Stage logged in output: `stage_oracle: gen=N stage=echo ...`
- ✅ 6 new stage tests
- ✅ exp38 registered in test suite with `stage_oracle:` prefix

**No placeholders found.**

**Type consistency:** `detect_stage` returns `str`, `build_stage_prompt` takes `str` stage — consistent across Task 1 and Task 2.
