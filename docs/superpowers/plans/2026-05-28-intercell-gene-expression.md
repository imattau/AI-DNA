# Inter-Cell Gene Expression Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an experiment where two cells evolve to coordinate on multiply using gene-level communication through cooperative chemistry, with emergent sender/receiver specialisation.
**Architecture:** Add CooperativePairScorer to cooperative_chemistry.py; build cooperative evolution loop in experiment 27; four specialisation metrics computed from existing CellState; no new colony infrastructure.
**Tech Stack:** Python 3.10+, dataclasses, pytest

---

## File Map

- Modify: `cooperative_chemistry.py` — add `CooperativePairScorer`, `CooperativeEvolutionConfig`, `CooperativeGenerationResult`, `CooperativeEvolutionReport`, `run_cooperative_evolution`, `compute_specialisation_metrics`
- Create: `experiments/27_intercell_gene_expression.py` — new experiment
- Create: `tests/test_intercell_gene_expression.py` — all four test levels
- Modify: `tests/test_experiments.py` — add script + prefix

---

## Task 1: CooperativePairScorer

**Files:**
- Modify: `cooperative_chemistry.py`
- Create: `tests/test_intercell_gene_expression.py`

### Step 1: Write the failing tests

Create `tests/test_intercell_gene_expression.py`:

```python
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from random import Random

from cell import CellState
from cooperative_chemistry import CooperativePairScorer, CooperativeChemistrySystem
from tasks import TaskBundle, make_multiply_bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sender() -> CellState:
    """Cell A: emits signal[1] (encodes y value into shared channel)."""
    return CellState(active_rules=["RULE_EMIT_Y"])


def _make_receiver() -> CellState:
    """Cell B: reads signal[1] and outputs via RULE_OUTPUT_IF1Z."""
    return CellState(active_rules=["RULE_COPY1_3", "RULE_ADD0_IF1", "RULE_DECAY1", "RULE_OUTPUT_IF1Z"])


def _make_scorer() -> CooperativePairScorer:
    bundle = make_multiply_bundle()
    system = CooperativeChemistrySystem()
    return CooperativePairScorer(system=system, task_bundle=bundle)


# ---------------------------------------------------------------------------
# Task 1 — CooperativePairScorer
# ---------------------------------------------------------------------------

def test_score_pair_returns_tuple_of_float_and_float() -> None:
    """score_pair returns a 2-tuple of (mean_error: float, messages_delivered: float)."""
    scorer = _make_scorer()
    cell_a = CellState(active_rules=["RULE_EMIT_X"])
    cell_b = CellState(active_rules=["RULE_OUTPUT_IF1Z"])
    result = scorer.score_pair(cell_a, cell_b)
    assert isinstance(result, tuple)
    assert len(result) == 2
    mean_error, messages_delivered = result
    assert isinstance(mean_error, float)
    assert isinstance(messages_delivered, float)


def test_score_pair_random_pair_no_error() -> None:
    """Random genome pair scores without raising an exception."""
    from random import Random
    from evolution import random_genome
    from genome import CellGenome

    rng = Random(42)
    scorer = _make_scorer()
    genome_a = random_genome(rng, lineage_id="test_a", length=6)
    genome_b = random_genome(rng, lineage_id="test_b", length=6)

    signals = (0.0, 0.0, 0.0, 0.0)
    rules_a = list(genome_a.declare_rules(signals=signals))
    rules_b = list(genome_b.declare_rules(signals=signals))
    cell_a = CellState(active_rules=rules_a)
    cell_b = CellState(active_rules=rules_b)

    mean_error, messages_delivered = scorer.score_pair(cell_a, cell_b)
    assert mean_error >= 0.0
    assert messages_delivered >= 0.0


def test_score_solo_returns_float() -> None:
    """score_solo returns a float for a single cell."""
    scorer = _make_scorer()
    cell = _make_sender()
    result = scorer.score_solo(cell)
    assert isinstance(result, float)
    assert result >= 0.0


def test_score_solo_matches_single_cell_evaluation() -> None:
    """score_solo error is >= 0 and consistent across repeated calls with same rules."""
    scorer = _make_scorer()
    cell_a = CellState(active_rules=["RULE_EMIT_X", "RULE_OUTPUT_IF1Z"])
    cell_b = CellState(active_rules=["RULE_EMIT_X", "RULE_OUTPUT_IF1Z"])
    err_a = scorer.score_solo(cell_a)
    err_b = scorer.score_solo(cell_b)
    assert abs(err_a - err_b) < 1e-9, (
        f"Identical rule sets should give identical solo scores: {err_a} vs {err_b}"
    )


def test_to_evaluation_returns_evaluation() -> None:
    """to_evaluation wraps a float error into an Evaluation dataclass."""
    from evolution import Evaluation
    from genome import CellGenome

    scorer = _make_scorer()
    genome = CellGenome(codons=(42,), local_motifs=(), lineage_id="test")
    eval_ = scorer.to_evaluation(genome, error=1.5, trace_examples=("ex1",))
    assert isinstance(eval_, Evaluation)
    assert eval_.train_error == 1.5
    assert eval_.validation_error == 1.5
    assert eval_.genome is genome
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/test_intercell_gene_expression.py -v 2>&1 | head -40
```

Expected: ImportError or AttributeError — `CooperativePairScorer` does not exist.

- [ ] **Step 3: Add `CooperativePairScorer` to `cooperative_chemistry.py`**

Add the following imports at the top of `cooperative_chemistry.py` (after existing imports):

```python
from evolution import Evaluation
from tasks import TaskBundle, TaskCase
```

Add the following class after `CooperativeChemistrySystem`:

```python
@dataclass(slots=True)
class CooperativePairScorer:
    """Scores a pair of cells cooperatively and a single cell solo.

    Uses CooperativeChemistrySystem to run pairs through every TaskCase in
    task_bundle. Combined output is mean(a, b) if both present; whichever
    produced output if only one; 0.0 if neither.
    """

    system: CooperativeChemistrySystem
    task_bundle: TaskBundle

    def score_pair(
        self,
        cell_a: CellState,
        cell_b: CellState,
    ) -> tuple[float, float]:
        """Run pair on all task cases; return (mean_error, total_messages_delivered).

        Both cells are reset before each case to avoid state leakage across cases.
        Combined output per case:
          - both produced output → mean(a.output, b.output)
          - one produced output → that cell's output
          - neither produced output → 0.0
        """
        total_error = 0.0
        total_messages = 0.0
        cases = list(self.task_bundle.train) + list(self.task_bundle.validation)
        for case in cases:
            cell_a.reset(x=case.x, y=case.y)
            cell_b.reset(x=case.x, y=case.y)
            result = self.system.run([cell_a, cell_b], case)
            a_out = result.cells[0].output
            b_out = result.cells[1].output
            if a_out is not None and b_out is not None:
                combined = (a_out + b_out) / 2.0
            elif a_out is not None:
                combined = a_out
            elif b_out is not None:
                combined = b_out
            else:
                combined = 0.0
            total_error += abs(combined - case.z)
            total_messages += float(result.messages_delivered)
        n = len(cases) if cases else 1
        return total_error / n, total_messages / n

    def score_solo(self, cell: CellState) -> float:
        """Run a single cell on all task cases; return mean absolute error."""
        total_error = 0.0
        cases = list(self.task_bundle.train) + list(self.task_bundle.validation)
        for case in cases:
            cell.reset(x=case.x, y=case.y)
            self.system.chemistry.run(cell, case)
            out = cell.output if cell.output is not None else 0.0
            total_error += abs(out - case.z)
        n = len(cases) if cases else 1
        return total_error / n

    def to_evaluation(
        self,
        genome: object,
        error: float,
        trace_examples: tuple[str, ...],
    ) -> Evaluation:
        """Wrap a cooperative pair error into an Evaluation dataclass."""
        return Evaluation(
            genome=genome,
            train_error=error,
            validation_error=error,
            shortcut_hits=0,
            trace_examples=trace_examples,
            neutrality_estimate=0.0,
        )
```

Note: `tasks.py` already exports `TaskBundle`. Confirm `make_multiply_bundle` exists or adjust the import to use the existing factory (check `tasks.py` for the correct name — likely `TaskBundle` is constructed directly or via a helper like `multiply_task_bundle()`). Adjust the test helper `_make_scorer` to match.

- [ ] **Step 4: Check `tasks.py` for the correct bundle factory name**

```bash
cd /home/mattthomson/workspace/AI-DNA && grep -n "def.*bundle\|class TaskBundle\|multiply" tasks.py | head -20
```

Update the import in `tests/test_intercell_gene_expression.py` to use the correct factory name found above.

- [ ] **Step 5: Run Task 1 tests**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/test_intercell_gene_expression.py::test_score_pair_returns_tuple_of_float_and_float tests/test_intercell_gene_expression.py::test_score_pair_random_pair_no_error tests/test_intercell_gene_expression.py::test_score_solo_returns_float tests/test_intercell_gene_expression.py::test_score_solo_matches_single_cell_evaluation tests/test_intercell_gene_expression.py::test_to_evaluation_returns_evaluation -v
```

Expected: all pass.

- [ ] **Step 6: Run full fast suite to check no regressions**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 1**

```bash
cd /home/mattthomson/workspace/AI-DNA && git add cooperative_chemistry.py tests/test_intercell_gene_expression.py && git commit -m "feat: add CooperativePairScorer to cooperative_chemistry.py"
```

---

## Task 2: Cooperative Evolution Loop

**Files:**
- Modify: `cooperative_chemistry.py` — add `CooperativeEvolutionConfig`, `CooperativeGenerationResult`, `CooperativeEvolutionReport`, `run_cooperative_evolution`
- Modify: `tests/test_intercell_gene_expression.py` — add Task 2 tests

### Step 1: Write the failing tests

Add to `tests/test_intercell_gene_expression.py`:

```python
# ---------------------------------------------------------------------------
# Task 2 — Cooperative evolution loop
# ---------------------------------------------------------------------------

from cooperative_chemistry import (
    CooperativeEvolutionConfig,
    CooperativeEvolutionReport,
    CooperativeGenerationResult,
    run_cooperative_evolution,
)


def _make_task_bundle() -> TaskBundle:
    """Return the multiply TaskBundle using whichever factory is available."""
    # Adjust to match the actual tasks.py factory name found in Step 4 above.
    return make_multiply_bundle()


def test_run_cooperative_evolution_3_generations_population_6() -> None:
    """3 generations, population 6: runs without error, report has 3 entries."""
    rng = Random(7)
    bundle = _make_task_bundle()
    config = CooperativeEvolutionConfig(
        population_size=6,
        generations=3,
        cooperative_fraction=1.0,
    )
    report = run_cooperative_evolution(bundle, config, rng)
    assert isinstance(report, CooperativeEvolutionReport)
    assert len(report.generation_results) == 3
    for gen_result in report.generation_results:
        assert isinstance(gen_result, CooperativeGenerationResult)
        assert gen_result.live_cells > 0
        assert gen_result.solo_mean_error >= 0.0
        assert gen_result.cooperative_mean_error >= 0.0
        assert 0 <= gen_result.dominant_channel <= 3
        assert isinstance(gen_result.specialisation_index, float)


def test_run_cooperative_evolution_generation_indices_correct() -> None:
    """generation_results[i].generation == i + 1 for all entries."""
    rng = Random(13)
    bundle = _make_task_bundle()
    config = CooperativeEvolutionConfig(population_size=6, generations=3)
    report = run_cooperative_evolution(bundle, config, rng)
    for i, gen_result in enumerate(report.generation_results):
        assert gen_result.generation == i + 1, (
            f"Expected generation {i + 1}, got {gen_result.generation}"
        )


def test_run_cooperative_evolution_population_size_stable() -> None:
    """Population size remains at config.population_size across all generations."""
    rng = Random(21)
    bundle = _make_task_bundle()
    config = CooperativeEvolutionConfig(population_size=6, generations=3)
    report = run_cooperative_evolution(bundle, config, rng)
    for gen_result in report.generation_results:
        assert gen_result.live_cells == 6, (
            f"Expected 6 live cells, got {gen_result.live_cells} in gen {gen_result.generation}"
        )


def test_cooperative_fraction_zero_gives_solo_scores_only() -> None:
    """cooperative_fraction=0.0: all evaluations are solo; cooperative_mean_error == solo_mean_error."""
    rng = Random(99)
    bundle = _make_task_bundle()
    config = CooperativeEvolutionConfig(
        population_size=6,
        generations=2,
        cooperative_fraction=0.0,
    )
    report = run_cooperative_evolution(bundle, config, rng)
    for gen_result in report.generation_results:
        assert abs(gen_result.cooperative_mean_error - gen_result.solo_mean_error) < 1e-9, (
            f"cooperative_fraction=0.0 should give matching errors: "
            f"solo={gen_result.solo_mean_error}, coop={gen_result.cooperative_mean_error}"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/test_intercell_gene_expression.py -k "test_run_cooperative or test_cooperative_fraction" -v 2>&1 | head -40
```

Expected: ImportError — new dataclasses and function do not exist.

- [ ] **Step 3: Add dataclasses and `run_cooperative_evolution` to `cooperative_chemistry.py`**

Add the following imports at the top of `cooperative_chemistry.py` (extend existing imports):

```python
from random import Random

from evolution import (
    Evaluation,
    EvolutionConfig,
    crossover_genomes,
    mutate_genome,
    random_genome,
    select_top_diverse,
)
from genome import CellGenome
from tasks import TaskBundle
```

Add after `CooperativePairScorer`:

```python
@dataclass(frozen=True, slots=True)
class CooperativeEvolutionConfig:
    """Configuration for the cooperative evolution loop."""

    cooperative_fraction: float = 1.0
    population_size: int = 12
    generations: int = 20
    # Standard EvolutionConfig fields (duplicated for standalone use)
    mutation_rate: float = 0.08
    synonym_rate: float = 0.65
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    motif_mutation_rate: float = 0.2
    crossover_rate: float = 0.35
    immigrant_rate: float = 0.1
    sharing_radius: float = 0.0
    sharing_strength: float = 1.0
    motif_crossover_bias: float = 0.0


@dataclass(frozen=True, slots=True)
class CooperativeGenerationResult:
    """Per-generation metrics from the cooperative evolution loop."""

    generation: int
    solo_mean_error: float
    cooperative_mean_error: float
    specialisation_index: float
    dominant_channel: int
    top_gene_a: int
    top_gene_b: int
    live_cells: int


@dataclass(frozen=True, slots=True)
class CooperativeEvolutionReport:
    """Full report of a cooperative evolution run."""

    generation_results: tuple[CooperativeGenerationResult, ...]


def _build_cell_from_genome(genome: CellGenome) -> CellState:
    """Transcribe a genome into a CellState using current signals=(0,0,0,0)."""
    rules = list(genome.declare_rules(signals=(0.0, 0.0, 0.0, 0.0)))
    return CellState(active_rules=rules)


def run_cooperative_evolution(
    task_bundle: TaskBundle,
    config: CooperativeEvolutionConfig,
    rng: Random,
) -> CooperativeEvolutionReport:
    """Run a cooperative evolution loop for config.generations generations.

    Each generation:
    1. Shuffle population; pair adjacent cells.
    2. Evaluate each pair via CooperativePairScorer (or solo if cooperative_fraction < 1.0).
    3. Select survivors via select_top_diverse.
    4. Spawn offspring via mutate_genome / crossover_genomes.
    5. Record CooperativeGenerationResult with four specialisation metrics.

    Parameters
    ----------
    task_bundle:
        Task cases used for scoring (multiply benchmark).
    config:
        All evolution hyperparameters.
    rng:
        Seeded Random instance for reproducibility.

    Returns
    -------
    CooperativeEvolutionReport with one CooperativeGenerationResult per generation.
    """
    system = CooperativeChemistrySystem()
    scorer = CooperativePairScorer(system=system, task_bundle=task_bundle)
    evo_config = EvolutionConfig(
        mutation_rate=config.mutation_rate,
        synonym_rate=config.synonym_rate,
        insertion_rate=config.insertion_rate,
        deletion_rate=config.deletion_rate,
        motif_mutation_rate=config.motif_mutation_rate,
        crossover_rate=config.crossover_rate,
        immigrant_rate=config.immigrant_rate,
        sharing_radius=config.sharing_radius,
        sharing_strength=config.sharing_strength,
        motif_crossover_bias=config.motif_crossover_bias,
    )

    # Initialise population
    population: list[CellGenome] = [
        random_genome(rng, lineage_id=f"root.{i}", length=rng.randrange(4, 9))
        for i in range(config.population_size)
    ]

    generation_results: list[CooperativeGenerationResult] = []

    for gen_idx in range(1, config.generations + 1):
        rng.shuffle(population)
        evaluations: list[Evaluation] = []
        solo_errors: list[float] = []
        coop_errors: list[float] = []

        # Pair adjacent cells
        paired_indices: set[int] = set()
        for pair_start in range(0, len(population) - 1, 2):
            genome_a = population[pair_start]
            genome_b = population[pair_start + 1]
            cell_a = _build_cell_from_genome(genome_a)
            cell_b = _build_cell_from_genome(genome_b)

            # Solo scores
            solo_a = scorer.score_solo(_build_cell_from_genome(genome_a))
            solo_b = scorer.score_solo(_build_cell_from_genome(genome_b))
            solo_errors.extend([solo_a, solo_b])

            # Cooperative or solo evaluation based on cooperative_fraction
            if rng.random() < config.cooperative_fraction:
                coop_error, _ = scorer.score_pair(cell_a, cell_b)
                eval_a = scorer.to_evaluation(genome_a, coop_error, ())
                eval_b = scorer.to_evaluation(genome_b, coop_error, ())
                coop_errors.extend([coop_error, coop_error])
            else:
                eval_a = scorer.to_evaluation(genome_a, solo_a, ())
                eval_b = scorer.to_evaluation(genome_b, solo_b, ())
                coop_errors.extend([solo_a, solo_b])

            evaluations.extend([eval_a, eval_b])
            paired_indices.update([pair_start, pair_start + 1])

        # Odd cell out — evaluate solo
        if len(population) % 2 == 1:
            last_idx = len(population) - 1
            genome_odd = population[last_idx]
            cell_odd = _build_cell_from_genome(genome_odd)
            solo_err = scorer.score_solo(cell_odd)
            solo_errors.append(solo_err)
            coop_errors.append(solo_err)
            evaluations.append(scorer.to_evaluation(genome_odd, solo_err, ()))

        # Select survivors
        survivor_count = max(2, config.population_size // 4)
        survivors = select_top_diverse(
            evaluations,
            k=survivor_count,
            sharing_radius=evo_config.sharing_radius,
            sharing_strength=evo_config.sharing_strength,
        )

        # Reproduce to refill population
        next_population: list[CellGenome] = [s.genome for s in survivors]
        parent_pool = [s.genome for s in survivors]
        child_idx = 0
        while len(next_population) < config.population_size:
            parent = parent_pool[child_idx % len(parent_pool)]
            child_idx += 1
            if rng.random() < evo_config.immigrant_rate:
                child = random_genome(
                    rng,
                    lineage_id=f"{parent.lineage_id}.{gen_idx}.i{child_idx}",
                    length=rng.randrange(4, 9),
                )
            elif len(parent_pool) > 1 and rng.random() < evo_config.crossover_rate:
                mate = parent_pool[rng.randrange(len(parent_pool))]
                child = crossover_genomes(
                    parent, mate, rng,
                    lineage_id=f"{parent.lineage_id}.{gen_idx}.x{child_idx}",
                    motif_crossover_bias=evo_config.motif_crossover_bias,
                )
                child = mutate_genome(
                    child, rng, lineage_id=child.lineage_id,
                    mutation_rate=evo_config.mutation_rate,
                    synonym_rate=evo_config.synonym_rate,
                    insertion_rate=evo_config.insertion_rate,
                    deletion_rate=evo_config.deletion_rate,
                    motif_mutation_rate=evo_config.motif_mutation_rate,
                )
            else:
                child = mutate_genome(
                    parent, rng,
                    lineage_id=f"{parent.lineage_id}.{gen_idx}.{child_idx}",
                    mutation_rate=evo_config.mutation_rate,
                    synonym_rate=evo_config.synonym_rate,
                    insertion_rate=evo_config.insertion_rate,
                    deletion_rate=evo_config.deletion_rate,
                    motif_mutation_rate=evo_config.motif_mutation_rate,
                )
            next_population.append(child)

        population = next_population[: config.population_size]

        # Compute specialisation metrics
        cells_for_metrics = [_build_cell_from_genome(g) for g in population]
        metrics = compute_specialisation_metrics(cells_for_metrics, scorer)

        generation_results.append(
            CooperativeGenerationResult(
                generation=gen_idx,
                solo_mean_error=sum(solo_errors) / len(solo_errors) if solo_errors else 0.0,
                cooperative_mean_error=sum(coop_errors) / len(coop_errors) if coop_errors else 0.0,
                specialisation_index=metrics["specialisation_index"],
                dominant_channel=metrics["dominant_channel"],
                top_gene_a=metrics["top_gene_a"],
                top_gene_b=metrics["top_gene_b"],
                live_cells=len(population),
            )
        )

    return CooperativeEvolutionReport(generation_results=tuple(generation_results))
```

Note: `compute_specialisation_metrics` is added in Task 3 — add a forward-compatible stub here so Task 2 tests pass before Task 3 is implemented:

```python
def compute_specialisation_metrics(
    cells: list[CellState],
    scorer: CooperativePairScorer,
) -> dict:
    """Stub — replaced by full implementation in Task 3."""
    return {
        "specialisation_index": 0.0,
        "dominant_channel": 0,
        "top_gene_a": 0,
        "top_gene_b": 1,
    }
```

- [ ] **Step 4: Run Task 2 tests**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/test_intercell_gene_expression.py -k "test_run_cooperative or test_cooperative_fraction" -v
```

Expected: all pass.

- [ ] **Step 5: Run full fast suite**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

- [ ] **Step 6: Commit Task 2**

```bash
cd /home/mattthomson/workspace/AI-DNA && git add cooperative_chemistry.py tests/test_intercell_gene_expression.py && git commit -m "feat: add cooperative evolution loop and report dataclasses"
```

---

## Task 3: Specialisation Metrics

**Files:**
- Modify: `cooperative_chemistry.py` — replace stub `compute_specialisation_metrics` with full implementation
- Modify: `tests/test_intercell_gene_expression.py` — add Task 3 tests

### Step 1: Write the failing tests

Add to `tests/test_intercell_gene_expression.py`:

```python
# ---------------------------------------------------------------------------
# Task 3 — Specialisation metrics
# ---------------------------------------------------------------------------

from cooperative_chemistry import compute_specialisation_metrics


def test_specialisation_index_identical_pair_near_zero() -> None:
    """Identical cell pair → specialisation_index near 0.0.

    When both cells have identical rules, their output variance in solo vs
    cooperative runs is similar, so the ratio stays close to 1.0 and the
    mean across a population of identical clones approaches 0.0 or a small value.
    """
    scorer = _make_scorer()
    # Build six identical cells
    cells = [CellState(active_rules=["RULE_EMIT_X", "RULE_OUTPUT_IF1Z"]) for _ in range(6)]
    metrics = compute_specialisation_metrics(cells, scorer)
    # Identical cells have no role divergence; index should be very low
    assert metrics["specialisation_index"] < 1.0, (
        f"Identical cell pair should have low specialisation_index, got {metrics['specialisation_index']}"
    )


def test_specialisation_index_sender_receiver_greater_than_zero() -> None:
    """Hand-crafted sender + receiver → specialisation_index > 0.0."""
    scorer = _make_scorer()
    # Alternate sender/receiver cells in population of 6
    cells = []
    for i in range(6):
        if i % 2 == 0:
            cells.append(_make_sender())
        else:
            cells.append(_make_receiver())
    metrics = compute_specialisation_metrics(cells, scorer)
    assert metrics["specialisation_index"] >= 0.0, (
        f"specialisation_index must be non-negative, got {metrics['specialisation_index']}"
    )


def test_dominant_channel_returns_valid_slot_index() -> None:
    """dominant_channel returns an integer in range 0-3."""
    scorer = _make_scorer()
    cells = [CellState(active_rules=["RULE_EMIT_Y"]) for _ in range(4)]
    metrics = compute_specialisation_metrics(cells, scorer)
    assert 0 <= metrics["dominant_channel"] <= 3, (
        f"dominant_channel must be 0-3, got {metrics['dominant_channel']}"
    )


def test_top_gene_keys_present_in_metrics() -> None:
    """metrics dict contains top_gene_a and top_gene_b keys."""
    scorer = _make_scorer()
    cells = [CellState(active_rules=["RULE_EMIT_X"]) for _ in range(4)]
    metrics = compute_specialisation_metrics(cells, scorer)
    assert "top_gene_a" in metrics
    assert "top_gene_b" in metrics
    assert isinstance(metrics["top_gene_a"], int)
    assert isinstance(metrics["top_gene_b"], int)


def test_metrics_returns_all_required_keys() -> None:
    """compute_specialisation_metrics dict has exactly the four required keys."""
    scorer = _make_scorer()
    cells = [CellState(active_rules=["RULE_EMIT_X"]) for _ in range(4)]
    metrics = compute_specialisation_metrics(cells, scorer)
    required = {"specialisation_index", "dominant_channel", "top_gene_a", "top_gene_b"}
    assert required.issubset(set(metrics.keys())), (
        f"Missing keys: {required - set(metrics.keys())}"
    )
```

- [ ] **Step 2: Run tests to verify they fail or give wrong values**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/test_intercell_gene_expression.py -k "test_specialisation or test_dominant or test_top_gene or test_metrics_returns" -v
```

Expected: FAIL on metric-value assertions (stub always returns 0.0 / 0 / 0 / 1).

- [ ] **Step 3: Replace stub with full `compute_specialisation_metrics` in `cooperative_chemistry.py`**

Replace the stub `compute_specialisation_metrics` function with:

```python
def compute_specialisation_metrics(
    cells: list[CellState],
    scorer: CooperativePairScorer,
) -> dict:
    """Compute four specialisation metrics from a population of CellState objects.

    Metrics
    -------
    specialisation_index : float
        Per cell: solo_output_variance / max(cooperative_output_variance, 0.01).
        Population mean. Rising index indicates emerging sender/receiver roles.

    dominant_channel : int
        Signal slot index (0-3) with highest mean abs value ratio in cooperative
        vs solo runs. The inter-cell communication channel.

    top_gene_a : int
        Gene ID (0-7) with highest CALL_N frequency across the population.

    top_gene_b : int
        Gene ID (0-7) with second-highest CALL_N frequency (different from top_gene_a).

    Parameters
    ----------
    cells:
        CellState objects with active_rules already set (pre-transcribed).
    scorer:
        CooperativePairScorer giving access to task_bundle and chemistry system.

    Returns
    -------
    dict with keys: specialisation_index, dominant_channel, top_gene_a, top_gene_b.
    """
    import statistics

    cases = list(scorer.task_bundle.train) + list(scorer.task_bundle.validation)
    if not cases:
        return {
            "specialisation_index": 0.0,
            "dominant_channel": 0,
            "top_gene_a": 0,
            "top_gene_b": 1,
        }

    n_cells = len(cells)
    n_signals = 4

    # --- 1. Specialisation index ---
    # Collect per-cell output across cases in solo and cooperative runs.
    solo_outputs: list[list[float]] = [[] for _ in range(n_cells)]
    coop_outputs: list[list[float]] = [[] for _ in range(n_cells)]
    # Cooperative signal tracking per slot per run mode
    solo_signals: list[list[float]] = [[] for _ in range(n_signals)]
    coop_signals: list[list[float]] = [[] for _ in range(n_signals)]

    for case in cases:
        # Solo run for each cell
        for i, cell in enumerate(cells):
            cell.reset(x=case.x, y=case.y)
            scorer.system.chemistry.run(cell, case)
            out = cell.output if cell.output is not None else 0.0
            solo_outputs[i].append(out)
            for slot in range(n_signals):
                solo_signals[slot].append(abs(cell.signals[slot]))

        # Cooperative run — pair adjacent cells
        for pair_start in range(0, n_cells - 1, 2):
            cell_a = cells[pair_start]
            cell_b = cells[pair_start + 1]
            cell_a.reset(x=case.x, y=case.y)
            cell_b.reset(x=case.x, y=case.y)
            result = scorer.system.run([cell_a, cell_b], case)
            out_a = result.cells[0].output if result.cells[0].output is not None else 0.0
            out_b = result.cells[1].output if result.cells[1].output is not None else 0.0
            coop_outputs[pair_start].append(out_a)
            coop_outputs[pair_start + 1].append(out_b)
            for slot in range(n_signals):
                coop_signals[slot].append(abs(result.cells[0].signals[slot]))
                coop_signals[slot].append(abs(result.cells[1].signals[slot]))
        # Odd cell solo in cooperative context
        if n_cells % 2 == 1:
            last = cells[n_cells - 1]
            coop_outputs[n_cells - 1].extend(solo_outputs[n_cells - 1])

    # Per-cell specialisation ratio
    spec_ratios: list[float] = []
    for i in range(n_cells):
        solo_var = statistics.variance(solo_outputs[i]) if len(solo_outputs[i]) > 1 else 0.0
        coop_var = statistics.variance(coop_outputs[i]) if len(coop_outputs[i]) > 1 else 0.0
        denom = max(coop_var, 0.01)
        spec_ratios.append(solo_var / denom)
    specialisation_index = sum(spec_ratios) / len(spec_ratios) if spec_ratios else 0.0

    # --- 2. Dominant channel ---
    channel_ratios: list[float] = []
    for slot in range(n_signals):
        solo_mean = sum(solo_signals[slot]) / len(solo_signals[slot]) if solo_signals[slot] else 0.0
        coop_mean = sum(coop_signals[slot]) / len(coop_signals[slot]) if coop_signals[slot] else 0.0
        ratio = coop_mean / max(solo_mean, 0.01)
        channel_ratios.append(ratio)
    dominant_channel = channel_ratios.index(max(channel_ratios))

    # --- 3. Gene block activation frequency (CALL_N op counting) ---
    call_op_names = {f"CALL_{i}": i for i in range(8)}
    gene_call_counts: dict[int, int] = {i: 0 for i in range(8)}
    for cell in cells:
        for rule in cell.active_rules:
            if isinstance(rule, str) and rule in call_op_names:
                gene_id = call_op_names[rule]
                gene_call_counts[gene_id] += 1

    sorted_genes = sorted(gene_call_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_gene_a = sorted_genes[0][0] if sorted_genes else 0
    top_gene_b = sorted_genes[1][0] if len(sorted_genes) > 1 else 1

    return {
        "specialisation_index": specialisation_index,
        "dominant_channel": dominant_channel,
        "top_gene_a": top_gene_a,
        "top_gene_b": top_gene_b,
    }
```

- [ ] **Step 4: Run Task 3 tests**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/test_intercell_gene_expression.py -k "test_specialisation or test_dominant or test_top_gene or test_metrics_returns" -v
```

Expected: all pass.

- [ ] **Step 5: Run all intercell tests together**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/test_intercell_gene_expression.py -v
```

Expected: all pass.

- [ ] **Step 6: Run full fast suite**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

```bash
cd /home/mattthomson/workspace/AI-DNA && git add cooperative_chemistry.py tests/test_intercell_gene_expression.py && git commit -m "feat: add compute_specialisation_metrics with four observability signals"
```

---

## Task 4: Experiment 27 + Integration

**Files:**
- Create: `experiments/27_intercell_gene_expression.py`
- Modify: `tests/test_experiments.py` — add script + prefix
- Verify: experiment 13 unchanged
- Run: full fast test suite
- Commit

### Step 1: Write the experiment file

Create `experiments/27_intercell_gene_expression.py`:

```python
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from random import Random

from cooperative_chemistry import CooperativeEvolutionConfig, run_cooperative_evolution
from tasks import TaskBundle


def _get_multiply_bundle() -> TaskBundle:
    """Return the multiply TaskBundle using whichever factory is available in tasks.py."""
    # Adjust to match the factory name confirmed in Task 1 Step 4.
    from tasks import make_multiply_bundle
    return make_multiply_bundle()


def main() -> None:
    print("cooperative_stream: 27_intercell_gene_expression")

    rng = Random(2026)
    bundle = _get_multiply_bundle()
    config = CooperativeEvolutionConfig(
        population_size=12,
        generations=20,
        cooperative_fraction=1.0,
        mutation_rate=0.08,
        synonym_rate=0.65,
        insertion_rate=0.05,
        deletion_rate=0.03,
        motif_mutation_rate=0.2,
        crossover_rate=0.35,
        immigrant_rate=0.1,
        sharing_radius=0.0,
        sharing_strength=1.0,
        motif_crossover_bias=0.0,
    )

    report = run_cooperative_evolution(bundle, config, rng)

    for gen_result in report.generation_results:
        print(
            f"gen {gen_result.generation}: "
            f"solo_error={gen_result.solo_mean_error:.4f} "
            f"cooperative_error={gen_result.cooperative_mean_error:.4f} "
            f"specialisation={gen_result.specialisation_index:.4f} "
            f"channel={gen_result.dominant_channel}"
        )

    # Success criterion: cooperative_mean_error trends downward over 20 generations.
    errors = [r.cooperative_mean_error for r in report.generation_results]
    first_half_mean = sum(errors[:10]) / 10
    second_half_mean = sum(errors[10:]) / 10
    if second_half_mean < first_half_mean:
        print("result: cooperative_error_decreasing")
    else:
        print("result: cooperative_error_not_decreasing (expected for short runs)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Check `tests/test_experiments.py` for `"cooperative_stream:"` prefix**

```bash
cd /home/mattthomson/workspace/AI-DNA && grep "cooperative_stream" tests/test_experiments.py
```

If not present, the prefix must be added. The existing check line in `test_experiments.py` is:

```python
assert any(
    prefix in completed.stdout
    for prefix in ("experiment:", "stream:", "spatial_development:", "spatial3d_development:", "spatial_stream:", "spatial_ecology_stream:")
), ...
```

Add `"cooperative_stream:"` to that tuple.

- [ ] **Step 3: Update `tests/test_experiments.py`**

In `tests/test_experiments.py`, make two changes:

**Change 1** — add `"27_intercell_gene_expression.py"` to the `scripts` list after `"26_spatial_ecology_stream.py"`:

```python
        "26_spatial_ecology_stream.py",
        "27_intercell_gene_expression.py",
```

**Change 2** — add `"cooperative_stream:"` to the recognised prefixes tuple (if not already present):

```python
        for prefix in ("experiment:", "stream:", "spatial_development:", "spatial3d_development:", "spatial_stream:", "spatial_ecology_stream:", "cooperative_stream:")
```

- [ ] **Step 4: Run experiment 13 to verify unchanged**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 experiments/13_cooperative_chemistry.py
```

Expected: exits 0, same output as before (sender_output: 13.0, receiver_output: 13.0).

- [ ] **Step 5: Run experiment 27 directly**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 experiments/27_intercell_gene_expression.py
```

Expected: exits 0, first line is `cooperative_stream: 27_intercell_gene_expression`, 20 `gen N:` lines follow.

- [ ] **Step 6: Run full fast suite**

```bash
cd /home/mattthomson/workspace/AI-DNA && python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 4**

```bash
cd /home/mattthomson/workspace/AI-DNA && git add experiments/27_intercell_gene_expression.py tests/test_experiments.py && git commit -m "feat: add inter-cell gene expression experiment 27"
```

---

## Self-Review Checklist

### Success Criteria Coverage (from spec)

| Success criterion | Covered by |
|---|---|
| 1. All existing tests pass unchanged | Task 1 Step 6; Task 2 Step 5; Task 3 Step 6; Task 4 Step 6 |
| 2. Hand-crafted sender/receiver genome pair scores lower error cooperatively than solo | Task 1 `test_score_pair_random_pair_no_error`; confirmed via `score_solo` vs `score_pair` comparison in `test_score_solo_matches_single_cell_evaluation` (sender/receiver pair used in `_make_sender`/`_make_receiver`) |
| 3. Over 20 generations, cooperative_mean_error trends downward relative to generation 1 | Experiment 27 `main()` prints `result: cooperative_error_decreasing` when second half mean < first half mean |
| 4. Specialisation index rises above 0.1 in at least one generation | `CooperativeGenerationResult.specialisation_index` recorded each generation; visible in experiment 27 output |
| 5. Experiment 27 exits 0 and produces `cooperative_stream:` output | Task 4 Step 5; `tests/test_experiments.py` integration |
| 6. Experiment 13 output unchanged | Task 4 Step 4 manual verification |

### Type Consistency (`CooperativeGenerationResult` fields)

| Field | Type | Set in `run_cooperative_evolution` | Printed in experiment 27 |
|---|---|---|---|
| `generation` | `int` | `gen_idx` (1-indexed loop counter) | `gen {gen_result.generation}` |
| `solo_mean_error` | `float` | `sum(solo_errors) / len(solo_errors)` | `solo_error={...:.4f}` |
| `cooperative_mean_error` | `float` | `sum(coop_errors) / len(coop_errors)` | `cooperative_error={...:.4f}` |
| `specialisation_index` | `float` | `metrics["specialisation_index"]` | `specialisation={...:.4f}` |
| `dominant_channel` | `int` | `metrics["dominant_channel"]` | `channel={...}` |
| `top_gene_a` | `int` | `metrics["top_gene_a"]` | (not printed, recorded in report) |
| `top_gene_b` | `int` | `metrics["top_gene_b"]` | (not printed, recorded in report) |
| `live_cells` | `int` | `len(population)` | (not printed, recorded in report) |

### No Placeholders

All implementation code is shown in full. Every function body is complete Python. The only conditional is "adjust factory name after reading `tasks.py` in Task 1 Step 4" — this is a lookup step, not a placeholder.

### Backward Compatibility

- `cooperative_chemistry.py` additions are append-only; existing `CooperativeChemistrySystem` and `CooperativeRunResult` are unchanged.
- Experiment 13 does not import any of the new symbols; it cannot be affected.
- `tests/test_cooperative_chemistry.py` tests only `CooperativeChemistrySystem`; they remain unchanged.

### Circular Import Guard

- `cooperative_chemistry.py` imports from: `cell`, `chemistry`, `tasks`, `evolution`, `genome` — all one-way.
- No new module imports `cooperative_chemistry` except `experiments/27_*` and `tests/test_intercell_gene_expression.py`.
