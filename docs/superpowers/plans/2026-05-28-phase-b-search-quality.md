# Phase B — Search Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four independently-toggleable search improvements (fitness sharing, motif-aware crossover, adaptive mutation rate, QD archive) to make the evolver converge faster on hard targets without breaking existing behaviour.

**Architecture:** Tasks 1-2 modify `evolution.py` and `colony.py` only; Tasks 3-4 modify `task_stream.py` only. All new parameters default to current behaviour (`sharing_radius=0.0`, `motif_crossover_bias=0.0`, `adaptive_mutation=False`, `qd_archive=False`). Tests are added to existing test files `tests/test_evolution.py` and `tests/test_task_stream.py`.

**Tech Stack:** Python 3.10+, dataclasses, pytest

---

## File Map

- Modify: `evolution.py` — `EvolutionConfig`, `select_top_diverse`, `crossover_genomes`
- Modify: `colony.py` — `Colony.advance` (pass new config fields to updated functions)
- Modify: `task_stream.py` — `StreamConfig`, `ArchiveSnapshot`, `TaskStreamColony` advance loop
- Modify: `tests/test_evolution.py` — new tests for Tasks 1 and 2
- Modify: `tests/test_task_stream.py` — new tests for Tasks 3 and 4

---

## Task 1: Fitness sharing in `select_top_diverse`

**Files:**
- Modify: `evolution.py`
- Modify: `colony.py`
- Modify: `tests/test_evolution.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evolution.py`:

```python
def test_fitness_sharing_penalises_crowded_niche() -> None:
    """With sharing_radius > 0, a diverse third candidate displaces a near-duplicate second."""
    from evolution import Evaluation, select_top_diverse
    from genome import CellGenome

    def _build_candidate(lineage: str, score: float) -> Evaluation:
        genome = CellGenome(codons=(hash(lineage) % 256,), local_motifs=(), lineage_id=lineage)
        return Evaluation(
            genome=genome,
            train_error=score,
            validation_error=0.0,
            shortcut_hits=0,
            trace_examples=(),
        )

    # A: score 1.0, B: score 1.05 (crowded near A), C: score 2.0 (different niche)
    a = _build_candidate("A", 1.0)
    b = _build_candidate("B", 1.05)
    c = _build_candidate("C", 2.0)

    # Without sharing: top-2 are A and B
    result_no_sharing = select_top_diverse([a, b, c], k=2, sharing_radius=0.0)
    assert set(e.genome.lineage_id for e in result_no_sharing) == {"A", "B"}

    # With sharing_radius=0.5: B is penalised because its score is within 0.5 of A
    # so C (different niche) should be selected instead of B
    result_sharing = select_top_diverse([a, b, c], k=2, sharing_radius=0.5, sharing_strength=1.0)
    ids = set(e.genome.lineage_id for e in result_sharing)
    assert "A" in ids
    assert "C" in ids, f"Expected C in result but got {ids}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_evolution.py::test_fitness_sharing_penalises_crowded_niche -v
```

Expected: FAIL — `select_top_diverse` does not accept `sharing_radius` keyword argument.

- [ ] **Step 3: Add `sharing_radius` and `sharing_strength` to `EvolutionConfig`**

In `evolution.py`, find the `EvolutionConfig` dataclass (line 36). Add two fields after `immigrant_rate`:

```python
sharing_radius: float = 0.0
sharing_strength: float = 1.0
```

- [ ] **Step 4: Update `select_top_diverse` signature and add penalty logic**

In `evolution.py`, replace the existing `select_top_diverse` function (lines 156-175):

```python
def select_top_diverse(
    evaluations: Sequence[Evaluation],
    *,
    k: int,
    sharing_radius: float = 0.0,
    sharing_strength: float = 1.0,
) -> tuple[Evaluation, ...]:
    ranked = sorted(evaluations, key=lambda evaluation: evaluation.score)
    diverse: list[Evaluation] = []
    seen: set[tuple[int, ...]] = set()
    selected_scores: list[float] = []

    for evaluation in ranked:
        signature = evaluation.genome.signature()
        if signature in seen:
            continue

        effective_score = evaluation.score
        if sharing_radius > 0.0:
            overlap_count = sum(
                max(0.0, sharing_strength * (1.0 - abs(evaluation.score - s) / sharing_radius))
                for s in selected_scores
                if abs(evaluation.score - s) < sharing_radius
            )
            effective_score = evaluation.score + overlap_count

        # Re-sort would be expensive; instead only accept if effective_score is not dominated.
        # Because ranked is sorted by raw score, a candidate with high penalty should be
        # skipped in favour of the next candidate. We insert into diverse only if
        # effective_score <= the raw score of the next unselected candidate, OR if we
        # have not yet filled k slots from the already-seen signature-unique set.
        # Simplified: accept if effective_score is within sharing_radius of best so far,
        # otherwise defer. For correctness with the test: add to seen but skip diverse append
        # when effective_score exceeds the raw score of all remaining candidates.
        # Practical implementation: always accept unique genomes in raw-score order but
        # apply the penalty to produce a second pass ranking.
        seen.add(signature)
        diverse.append((effective_score, evaluation))
        selected_scores.append(evaluation.score)

    # Re-rank by effective score and take top k
    diverse.sort(key=lambda pair: pair[0])
    result = tuple(pair[1] for pair in diverse[:k])

    if len(result) < k:
        extras = [e for e in ranked if e not in result]
        result = result + tuple(extras[: k - len(result)])
    return result
```

- [ ] **Step 5: Update `Colony.advance` to pass sharing params**

In `colony.py`, find line 41 where `select_top_diverse` is called:

```python
survivors = select_top_diverse(evaluations, k=survivor_count)
```

Replace with:

```python
survivors = select_top_diverse(
    evaluations,
    k=survivor_count,
    sharing_radius=config.sharing_radius,
    sharing_strength=config.sharing_strength,
)
```

- [ ] **Step 6: Run tests**

```bash
python3 -m pytest tests/test_evolution.py::test_fitness_sharing_penalises_crowded_niche -v
```

Expected: PASS

- [ ] **Step 7: Run regression suite**

```bash
python3 -m pytest tests/test_task_stream.py tests/test_chemistry_and_colony.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add evolution.py colony.py tests/test_evolution.py
git commit -m "feat: add fitness sharing to select_top_diverse"
```

---

## Task 2: Motif-aware crossover in `crossover_genomes`

**Files:**
- Modify: `evolution.py`
- Modify: `colony.py`
- Modify: `tests/test_evolution.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evolution.py`:

```python
def test_motif_crossover_respects_boundaries() -> None:
    """With motif_crossover_bias=1.0 and motifs present, crossover always splices at a motif boundary."""
    from random import Random
    from evolution import crossover_genomes
    from genome import CellGenome, Motif

    rng = Random(42)

    # Left genome: 6 codons, motif covers 3 codons (boundary at index 3)
    left_motif = Motif(
        pattern=("RULE_EMIT_X", "RULE_ADD0_IF1", "RULE_DECAY1"),
        origin_lineage="L",
        origin_task="t",
    )
    left = CellGenome(codons=(10, 20, 30, 40, 50, 60), local_motifs=(left_motif,), lineage_id="L1")

    # Right genome: 6 codons, motif covers 4 codons (boundary at index 4)
    right_motif = Motif(
        pattern=("RULE_EMIT_Y", "RULE_COPY1_3", "RULE_ADD0_IF1", "RULE_DECAY1"),
        origin_lineage="R",
        origin_task="t",
    )
    right = CellGenome(codons=(11, 21, 31, 41, 51, 61), local_motifs=(right_motif,), lineage_id="R1")

    # With bias=1.0, every crossover must split left at its boundary (index 3)
    split_points: set[int] = set()
    for _ in range(20):
        child = crossover_genomes(left, right, rng, lineage_id="C1", motif_crossover_bias=1.0)
        # Determine how many leading codons came from left
        left_prefix_len = 0
        for i, codon in enumerate(child.codons):
            if i < len(left.codons) and codon == left.codons[i]:
                left_prefix_len = i + 1
            else:
                break
        split_points.add(left_prefix_len)

    # All splits should be at boundary 3 (the only left motif boundary)
    assert split_points <= {3}, f"Expected only split at motif boundary 3, got {split_points}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_evolution.py::test_motif_crossover_respects_boundaries -v
```

Expected: FAIL — `crossover_genomes` does not accept `motif_crossover_bias`.

- [ ] **Step 3: Add `motif_crossover_bias` to `EvolutionConfig`**

In `evolution.py`, add to `EvolutionConfig` after `sharing_strength`:

```python
motif_crossover_bias: float = 0.0
```

- [ ] **Step 4: Add `_motif_boundaries` helper before `crossover_genomes`**

In `evolution.py`, insert this function before `crossover_genomes` (before line 87):

```python
def _motif_boundaries(genome: CellGenome) -> list[int]:
    """Return end-of-motif codon indices for each motif in genome.

    Walks motifs in order, accumulating codon spans from pattern lengths.
    The boundary for motif i is the index of the first codon after that motif.
    Clamped to len(genome.codons).
    """
    boundaries: list[int] = []
    pos = 0
    for motif in genome.local_motifs:
        pos = min(pos + len(motif.pattern), len(genome.codons))
        boundaries.append(pos)
    return boundaries
```

- [ ] **Step 5: Update `crossover_genomes` with boundary-aware splice**

In `evolution.py`, replace `crossover_genomes` (lines 87-92):

```python
def crossover_genomes(
    left: CellGenome,
    right: CellGenome,
    rng: Random,
    *,
    lineage_id: str,
    motif_crossover_bias: float = 0.0,
) -> CellGenome:
    left_boundaries = _motif_boundaries(left)
    right_boundaries = _motif_boundaries(right)

    use_boundary = (
        motif_crossover_bias > 0.0
        and len(left_boundaries) >= 1
        and len(right_boundaries) >= 1
        and rng.random() < motif_crossover_bias
    )

    if use_boundary:
        pivot_left = rng.choice(left_boundaries)
        prop = pivot_left / max(len(left.codons), 1)
        pivot_right = min(
            right_boundaries,
            key=lambda b: abs(b / max(len(right.codons), 1) - prop),
        )
        child_codons = tuple(left.codons[:pivot_left]) + tuple(right.codons[pivot_right:])
    else:
        child_codons = crossover_codons(left.codons, right.codons, rng)

    return CellGenome(
        codons=child_codons,
        local_motifs=left.local_motifs + right.local_motifs,
        lineage_id=lineage_id,
    )
```

- [ ] **Step 6: Update `Colony.advance` to pass `motif_crossover_bias`**

In `colony.py`, find the `crossover_genomes` call (around line 81):

```python
child = crossover_genomes(
    parent,
    mate,
    self.rng,
    lineage_id=f"{parent.lineage_id}.{self.generation + 1}.x{child_index}",
)
```

Replace with:

```python
child = crossover_genomes(
    parent,
    mate,
    self.rng,
    lineage_id=f"{parent.lineage_id}.{self.generation + 1}.x{child_index}",
    motif_crossover_bias=config.motif_crossover_bias,
)
```

- [ ] **Step 7: Run tests**

```bash
python3 -m pytest tests/test_evolution.py::test_motif_crossover_respects_boundaries -v
```

Expected: PASS

- [ ] **Step 8: Run regression suite**

```bash
python3 -m pytest tests/test_task_stream.py tests/test_chemistry_and_colony.py -q
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add evolution.py colony.py tests/test_evolution.py
git commit -m "feat: add motif-aware crossover bias to crossover_genomes"
```

---

## Task 3: Adaptive mutation rate in `task_stream.py`

**Files:**
- Modify: `task_stream.py`
- Modify: `tests/test_task_stream.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_task_stream.py`:

```python
def test_adaptive_mutation_scales_with_rank() -> None:
    """Best-ranked cell receives a lower effective mutation rate than worst-ranked."""
    from task_stream import StreamConfig, _compute_adaptive_rate

    config = StreamConfig(
        mutation_rate=0.08,
        adaptive_mutation=True,
        adaptive_mutation_alpha=2.0,
        adaptive_mutation_beta=0.5,
        adaptive_mutation_lo=0.01,
        adaptive_mutation_hi=0.25,
    )
    neutral_drift = 0.3

    best_rate = _compute_adaptive_rate(
        base_rate=config.mutation_rate,
        fitness_rank_fraction=0.0,
        neutral_drift_rate=neutral_drift,
        alpha=config.adaptive_mutation_alpha,
        beta=config.adaptive_mutation_beta,
        lo=config.adaptive_mutation_lo,
        hi=config.adaptive_mutation_hi,
    )
    worst_rate = _compute_adaptive_rate(
        base_rate=config.mutation_rate,
        fitness_rank_fraction=1.0,
        neutral_drift_rate=neutral_drift,
        alpha=config.adaptive_mutation_alpha,
        beta=config.adaptive_mutation_beta,
        lo=config.adaptive_mutation_lo,
        hi=config.adaptive_mutation_hi,
    )

    assert best_rate < worst_rate, f"Expected best_rate < worst_rate, got {best_rate} vs {worst_rate}"
    assert config.adaptive_mutation_lo <= best_rate <= config.adaptive_mutation_hi
    assert config.adaptive_mutation_lo <= worst_rate <= config.adaptive_mutation_hi
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_task_stream.py::test_adaptive_mutation_scales_with_rank -v
```

Expected: FAIL — `StreamConfig` has no `adaptive_mutation` field and `_compute_adaptive_rate` does not exist.

- [ ] **Step 3: Add fields to `StreamConfig`**

In `task_stream.py`, find `StreamConfig` (line 139). Add after `estimate_neutrality_trials`:

```python
adaptive_mutation: bool = False
adaptive_mutation_alpha: float = 2.0
adaptive_mutation_beta: float = 0.5
adaptive_mutation_lo: float = 0.01
adaptive_mutation_hi: float = 0.25
```

- [ ] **Step 4: Add `_compute_adaptive_rate` helper**

In `task_stream.py`, add this function after `_phenotype_diversity` (around line 200):

```python
def _compute_adaptive_rate(
    *,
    base_rate: float,
    fitness_rank_fraction: float,
    neutral_drift_rate: float,
    alpha: float,
    beta: float,
    lo: float,
    hi: float,
) -> float:
    """Per-cell mutation rate: base * (1 + alpha * drift - beta * rank_fraction), clamped to [lo, hi]."""
    rate = base_rate * (1.0 + alpha * neutral_drift_rate - beta * fitness_rank_fraction)
    return max(lo, min(hi, rate))
```

- [ ] **Step 5: Apply adaptive rate in the offspring-spawning loop**

In `task_stream.py`, find the loop inside `TaskStreamColony` (or the stream advance function) where `mutate_genome` is called for each live cell. Declare a per-lineage drift cache before the loop:

```python
_lineage_drift: dict[str, float] = {}
```

Update `_lineage_drift` after each `ArchiveSnapshot` is constructed (the snapshot's `neutral_drift_rate` applies to all live lineages at that point):

```python
for _cell in colony.live_members():
    _lineage_drift[_cell.lineage_id] = snapshot.neutral_drift_rate
```

Before each `mutate_genome` call for cell `c`, compute the effective rate:

```python
if config.adaptive_mutation:
    _sorted_live = sorted(colony.live_members(), key=lambda x: x.energy, reverse=True)
    _rank = next((i for i, x in enumerate(_sorted_live) if x is c), 0)
    _rank_fraction = _rank / max(len(_sorted_live) - 1, 1)
    _eff_mutation_rate = _compute_adaptive_rate(
        base_rate=config.mutation_rate,
        fitness_rank_fraction=_rank_fraction,
        neutral_drift_rate=_lineage_drift.get(c.lineage_id, 0.0),
        alpha=config.adaptive_mutation_alpha,
        beta=config.adaptive_mutation_beta,
        lo=config.adaptive_mutation_lo,
        hi=config.adaptive_mutation_hi,
    )
else:
    _eff_mutation_rate = config.mutation_rate
```

Then construct a local config for the spawn call (do not mutate `config` in place):

```python
_evo_config = EvolutionConfig(
    mutation_rate=_eff_mutation_rate,
    synonym_rate=config.synonym_rate,
    insertion_rate=config.insertion_rate,
    deletion_rate=config.deletion_rate,
    motif_mutation_rate=config.motif_mutation_rate,
    crossover_rate=config.crossover_rate,
    immigrant_rate=config.immigrant_rate,
)
```

Pass `_evo_config` to `spawn_sibling_variants` or use `mutation_rate=_eff_mutation_rate` in the `mutate_genome` call directly.

- [ ] **Step 6: Run tests**

```bash
python3 -m pytest tests/test_task_stream.py::test_adaptive_mutation_scales_with_rank -v
```

Expected: PASS

- [ ] **Step 7: Run regression suite**

```bash
python3 -m pytest tests/test_task_stream.py tests/test_chemistry_and_colony.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add task_stream.py tests/test_task_stream.py
git commit -m "feat: add adaptive mutation rate per lineage to task_stream"
```

---

## Task 4: QD (MAP-Elites) archive in `task_stream.py`

**Files:**
- Modify: `task_stream.py`
- Modify: `tests/test_task_stream.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_task_stream.py`:

```python
def test_qd_archive_populates_grid() -> None:
    """After inserting an evaluation, qd_grid has at least one entry."""
    from task_stream import QDArchive
    from evolution import Evaluation
    from genome import CellGenome

    archive = QDArchive(n_error_buckets=5, n_diversity_buckets=5, error_axis_max=10.0)
    genome = CellGenome(codons=(1, 2, 3), local_motifs=(), lineage_id="A1")
    ev = Evaluation(genome=genome, train_error=1.0, validation_error=1.0, shortcut_hits=0, trace_examples=())
    archive.insert(ev, error=2.0, diversity_score=0.5)
    assert len(archive.grid) > 0


def test_qd_archive_replaces_on_improvement() -> None:
    """Inserting a better evaluation into an occupied cell replaces the existing entry."""
    from task_stream import QDArchive
    from evolution import Evaluation
    from genome import CellGenome

    archive = QDArchive(n_error_buckets=5, n_diversity_buckets=5, error_axis_max=10.0)
    genome_a = CellGenome(codons=(1,), local_motifs=(), lineage_id="A1")
    genome_b = CellGenome(codons=(2,), local_motifs=(), lineage_id="B1")
    ev_a = Evaluation(genome=genome_a, train_error=3.0, validation_error=3.0, shortcut_hits=0, trace_examples=())
    ev_b = Evaluation(genome=genome_b, train_error=1.0, validation_error=1.0, shortcut_hits=0, trace_examples=())

    archive.insert(ev_a, error=2.0, diversity_score=0.5)
    key = list(archive.grid.keys())[0]
    archive.insert(ev_b, error=2.0, diversity_score=0.5)
    assert archive.grid[key].genome.lineage_id == "B1"


def test_qd_cells_occupied_reported() -> None:
    """ArchiveSnapshot.qd_cells_occupied > 0 after a short run with qd_archive=True."""
    from task_stream import StreamConfig, run_task_stream
    from tasks import TaskBundle

    tasks = [
        TaskBundle(name="multiply", cases=((2, 3, 6), (1, 4, 4), (3, 3, 9)), anti_shortcut_cases=()),
    ]
    config = StreamConfig(
        initial_population_size=4,
        max_steps_per_task=4,
        archive_interval=2,
        estimate_neutrality_trials=0,
        qd_archive=True,
        n_error_buckets=5,
        n_diversity_buckets=5,
        error_axis_max=100.0,
    )
    report = run_task_stream("test_qd", tasks, config=config, seed=99)
    occupied = [s.qd_cells_occupied for s in report.archive_snapshots]
    assert any(v > 0 for v in occupied), f"qd_cells_occupied was always 0: {occupied}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_task_stream.py::test_qd_archive_populates_grid tests/test_task_stream.py::test_qd_archive_replaces_on_improvement tests/test_task_stream.py::test_qd_cells_occupied_reported -v
```

Expected: FAIL — `QDArchive` does not exist and `StreamConfig` lacks `qd_archive`.

- [ ] **Step 3: Add fields to `StreamConfig`**

In `task_stream.py`, add to `StreamConfig` after `adaptive_mutation_hi`:

```python
qd_archive: bool = False
n_error_buckets: int = 10
n_diversity_buckets: int = 5
error_axis_max: float = 100.0
qd_exploit_fraction: float = 0.7
```

- [ ] **Step 4: Add `qd_cells_occupied` to `ArchiveSnapshot`**

In `task_stream.py`, find `ArchiveSnapshot` (line 57). Add after `neutral_drift_rate`:

```python
qd_cells_occupied: int = 0
```

Update `format_text` in `StreamReport` to include the new field. In the archive snapshot formatting block (around line 127), add `qd_cells_occupied` after `neutral_drift_rate`:

```python
f"neutral_drift_rate={snapshot.neutral_drift_rate:.4f} "
f"qd_cells_occupied={snapshot.qd_cells_occupied} "
```

- [ ] **Step 5: Add `QDArchive` dataclass**

In `task_stream.py`, add this dataclass after `ArchiveSnapshot` and before `StreamReport`:

```python
@dataclass(slots=True)
class QDArchive:
    n_error_buckets: int
    n_diversity_buckets: int
    error_axis_max: float
    grid: dict[tuple[int, int], Evaluation] = field(default_factory=dict)

    def insert(self, evaluation: Evaluation, *, error: float, diversity_score: float) -> None:
        """Insert evaluation, replacing existing entry only when new score is lower."""
        eb = min(
            int(error / self.error_axis_max * self.n_error_buckets),
            self.n_error_buckets - 1,
        )
        db = min(
            int(diversity_score * self.n_diversity_buckets),
            self.n_diversity_buckets - 1,
        )
        key = (eb, db)
        existing = self.grid.get(key)
        if existing is None or evaluation.score < existing.score:
            self.grid[key] = evaluation
```

- [ ] **Step 6: Maintain `QDArchive` in the stream advance loop**

In `task_stream.py`, at the start of the stream run function, initialise the archive:

```python
_qd_archive: QDArchive | None = (
    QDArchive(
        n_error_buckets=config.n_error_buckets,
        n_diversity_buckets=config.n_diversity_buckets,
        error_axis_max=config.error_axis_max,
    )
    if config.qd_archive
    else None
)
```

After each generation's evaluation loop, insert live evaluations into the QD archive:

```python
if _qd_archive is not None:
    for cell, evaluation, _ in step_evaluations:
        if cell.alive:
            _qd_archive.insert(
                evaluation,
                error=evaluation.score,
                diversity_score=_current_phenotype_diversity,
            )
```

Where `_current_phenotype_diversity` is the `phenotype_diversity` value already computed for the current step (from Phase A).

When drawing parents for offspring in the selection phase, apply two-phase draw:

```python
if _qd_archive is not None and _qd_archive.grid and rng.random() > config.qd_exploit_fraction:
    # Explore: draw uniformly from a random occupied QD cell
    parent_genome = rng.choice(list(_qd_archive.grid.values())).genome
else:
    # Exploit: use standard energy-ranked archive draw (unchanged path)
    parent_genome = _standard_archive_draw(...)
```

When constructing each `ArchiveSnapshot`, pass:

```python
qd_cells_occupied=len(_qd_archive.grid) if _qd_archive is not None else 0,
```

- [ ] **Step 7: Run tests**

```bash
python3 -m pytest tests/test_task_stream.py::test_qd_archive_populates_grid tests/test_task_stream.py::test_qd_archive_replaces_on_improvement tests/test_task_stream.py::test_qd_cells_occupied_reported -v
```

Expected: all PASS

- [ ] **Step 8: Run regression suite**

```bash
python3 -m pytest tests/test_task_stream.py tests/test_chemistry_and_colony.py -q
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add task_stream.py tests/test_task_stream.py
git commit -m "feat: add QD MAP-Elites archive to task_stream"
```

---

## Task 5: Integration check

**Files:** no changes — verification only

- [ ] **Step 1: Run the focused test suites**

```bash
python3 -m pytest tests/test_task_stream.py tests/test_chemistry_and_colony.py -q
```

Expected: all pass with no failures or errors.

- [ ] **Step 2: Enable all four features and run exp 14**

```bash
python3 experiments/14_adaptive_math_ecology.py
```

Expected: exits 0. If it raises an exception, diagnose using the traceback — do not suppress errors.

- [ ] **Step 3: Verify feature flags appear in output**

```bash
python3 experiments/14_adaptive_math_ecology.py 2>&1 | grep -E "qd_cells_occupied|neutral_drift_rate|phenotype_diversity" | head -10
```

Expected: lines containing `qd_cells_occupied=` and `neutral_drift_rate=` with numeric values.

- [ ] **Step 4: Commit all Phase B work**

```bash
git add evolution.py colony.py task_stream.py tests/test_evolution.py tests/test_task_stream.py
git commit -m "feat: Phase B search quality improvements"
```
