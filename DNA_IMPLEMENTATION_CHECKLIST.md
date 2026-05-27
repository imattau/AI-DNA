# DNA-Inspired Implementation Checklist

Ordered by impact on evolvability, robustness, and long-running adaptation.

## 1. Codon Degeneracy and Neutrality

Goal:
- Keep multiple codons mapping to the same operation so most mutations are neutral or near-neutral.

Why this matters:
- This is the main mechanism that creates searchable neutral networks instead of brittle one-step solutions.

Current repo seam:
- `codons.py`
- `matrix_experiment.py`
- `replication.py`

Current status:
- Implemented.

Next refinements:
- Increase neutral neighborhoods for the most useful operations.
- Track neutral drift rate in experiment reports.
- Measure how often mutations preserve phenotype.

## 2. Modular Motifs and Local Inheritance

Goal:
- Keep short reusable motifs lineage-local, then let successful motifs propagate through offspring only.

Why this matters:
- It creates reusable building blocks without introducing a global motif memory that short-circuits evolution.

Current repo seam:
- `genome.py`
- `evolution.py`
- `task_stream.py`
- `matrix_experiment.py`
- `replication.py`

Current status:
- Implemented.

Next refinements:
- Track motif origin, reuse count, and task of emergence more explicitly in reports.
- Compare motif reuse across lineages and tasks.

## 3. Energy-Based Survival and Reproduction

Goal:
- Make reproduction cost energy and make death arise from maintenance pressure.

Why this matters:
- This is the closest match to ecological selection pressure in the current model.

Current repo seam:
- `task_stream.py`
- `colony.py`
- `evolution.py`
- `replication.py`

Current status:
- Implemented.

Next refinements:
- Add richer resource dynamics such as spatial scarcity or delayed regeneration.
- Track energy efficiency as a first-class metric.

## 4. Contextual and Changing Environments

Goal:
- Evaluate the same genomes under changing tasks, noisy inputs, shifted input ranges, and changing resource regimes.

Why this matters:
- This is what turns a single-task solver into a continual-adaptation testbed.

Current repo seam:
- `task_stream.py`
- `tasks.py`
- `math_ecology.py`
- `experiments/11_contextual_task_stream.py`
- `experiments/12_math_task_ecology.py`

Current status:
- Implemented.

Next refinements:
- Add more structured context families.
- Add explicit retention metrics between contexts.
- Add harder transitions and randomized task orderings.

## 5. Archive, Lineage, and Transfer Tracking

Goal:
- Keep lineage history, archive successful survivors, and measure transfer from old tasks to new ones.

Why this matters:
- The system should remember how it got better, not just the latest best score.

Current repo seam:
- `tracing.py`
- `task_stream.py`
- `experiments/runner.py`

Current status:
- Implemented.

Next refinements:
- Add per-lineage transfer scores.
- Track which motifs and lineages reappear after task switches.
- Report forgetting explicitly as a metric.

## 6. Symbolic or Exact Verification

Goal:
- Use exact verification when the search reaches a candidate that is close enough to justify proof-level checking.

Why this matters:
- This separates empirical success from mathematically valid success.

Current repo seam:
- `matrix_experiment.py`
- `experiments/09_matrix_multiplication_search.py`

Current status:
- Implemented for the matrix benchmark.

Next refinements:
- Extend symbolic verification to additional arithmetic families.
- Add adversarial counterexample recycling when a candidate fails verification.

## 7. Better Search Pressure

Goal:
- Improve the evolutionary search without seeding a full solution.

Why this matters:
- The DNA-like system should discover structure, not replay hand-written answers.

Current repo seam:
- `evolution.py`
- `colony.py`
- `replication.py`
- `matrix_experiment.py`

Current status:
- Partially implemented.

Next refinements:
- Use adaptive curriculum switching.
- Add diversity-aware survivor selection.
- Tune mutation rates by lineage performance.
- Keep founder bias to motifs only, not complete programs.

## 8. Broader Math Ecology

Goal:
- Expand the problem ecology from arithmetic to small linear algebra and symbolic regimes.

Why this matters:
- It is the path from a toy benchmark to a real mathematical adaptation environment.

Current repo seam:
- `math_ecology.py`
- `tasks.py`
- `experiments/12_math_task_ecology.py`

Current status:
- Partially implemented.

Next refinements:
- Add 2x2 matrix families.
- Add symbolic equation families.
- Add mixed-family curricula with random context drift.

## Practical Priority Order

If implementing in sequence, do it in this order:

1. Codon degeneracy and neutrality
2. Modular motifs and local inheritance
3. Energy-based survival and reproduction
4. Contextual and changing environments
5. Archive, lineage, and transfer tracking
6. Symbolic or exact verification
7. Better search pressure
8. Broader math ecology

## Current Baseline

Already working in the repo:
- Self-replication
- Contextual task streams
- Rotating selective niches
- Matrix benchmark with symbolic verification
- Math ecology with multiple task families

The main open work is not basic functionality anymore. It is improving the search so the system can discover more general reusable structure without being seeded with near-solutions.
