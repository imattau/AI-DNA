# Inter-Cell Gene Expression Experiment Design

Date: 2026-05-28

## Summary

Design an experiment where two cells evolve to coordinate on a shared task using gene-level communication through cooperative chemistry. All cells start from random genomes with no fixed sender/receiver roles — specialisation emerges from selection pressure. The fitness signal is combined output accuracy on the multiply task. Gene blocks and regulatory codons are the communication channel: cell A's gene expression raises a shared signal, cell B's PROMOTER responds and activates its own gene block, producing output. The experiment demonstrates the full AI-DNA language stack — cooperative chemistry + gene blocks + regulatory codons — working end-to-end.

## What Already Exists (do not re-implement)

- `cooperative_chemistry.py` — `CooperativeChemistrySystem`, shared `ChemistryContext`, inbox/outbox broadcast
- `colony.py` — `Colony.advance()`, `select_top_diverse`, `mutate_genome`, `crossover_genomes`
- `evolution.py` — `EvolutionConfig`, `Evaluation`, fitness scoring infrastructure
- `tasks.py` — multiply `TaskBundle` with anti-shortcut cases
- Regulatory codons: PROMOTER, GATE, IF_S* conditions, END_BLOCK (implemented)
- Gene blocks: GENE_START/GENE_END, CALL_0-7, two-pass transcription (implemented)
- `experiments/13_cooperative_chemistry.py` — existing cooperative experiment (do not modify)

## Design

### Approach: Free population with cooperative evaluation

All cells start from random genomes. Each generation, cells are evaluated in random pairs using `CooperativeChemistrySystem`. Pair score = combined output accuracy. Both cells in a pair receive the same fitness. Role specialisation (sender/receiver) is an emergent observable outcome, not a designed-in constraint.

### Section 1 — Task and Scoring

Task: multiply — same benchmark as experiment 01. Each pair receives input `(x, y)`, scored on how close their combined output is to `x * y`.

Combined output calculation:
- If both cells produce output: `combined = mean(cell_A.output, cell_B.output)`
- If only one produces output: `combined = that cell's output`
- If neither produces output: `combined = 0.0` (maximum penalty)

Combined error: `abs(combined - x * y)`.

Each cell's individual fitness = the pair score. No separate sender/receiver reward.

Anti-shortcut cases from the existing `TaskBundle` are reused. Results are directly comparable to single-cell baseline (experiment 01).

Observable specialisation signal: one cell's output stabilises near a signal value (encoding), the other's tracks the correct answer (decoding) — visible in traces without requiring fixed role labels.

### Section 2 — Evolution Loop

Each generation:

1. **Pair randomly** — shuffle live population, pair adjacent cells. Odd cell evaluates solo (no cooperative bonus).
2. **Evaluate pairs** — run each pair through `CooperativeChemistrySystem` on multiply task bundle. Record combined output score. Both cells receive the same fitness score.
3. **Select survivors** — `select_top_diverse` keyed on pair score. Diversity deduplication prevents population collapse.
4. **Reproduce** — survivors spawn offspring via `mutate_genome` and `crossover_genomes`. Standard `EvolutionConfig` rates.
5. **Track specialisation** — record solo vs cooperative score gap and specialisation index per generation.

Config parameter `cooperative_fraction: float = 1.0` — fraction of evaluations that are cooperative vs solo. If population degenerates (no cell functions alone), reduce to 0.5.

Uses `Colony.advance()` with a custom cooperative scorer wrapping `CooperativeChemistrySystem`. No new colony infrastructure needed.

### Section 3 — Gene-Level Communication Channel

The communication pathway that evolution discovers (not seeded):

```
Cell A genome (example of what may evolve):
  GENE_START, GENE_ID_0, RULE_EMIT_S1, GENE_END   ← gene 0: emits signal[1]
  CALL_0                                            ← always express gene 0

Cell B genome (example of what may evolve):
  GENE_START, GENE_ID_2, RULE_OUTPUT_IF1Z, GENE_END ← gene 2: outputs based on signal
  PROMOTER, IF_S1_GT, CALL_2, END_BLOCK             ← express gene 2 only when signal[1] high
```

Cell A gene 0 fires → raises signal[1] in shared context → delivered to cell B's inbox → cell B's PROMOTER reads signal[1] at next transcription → activates gene 2 → cell B produces output.

The experiment does not seed this pattern. Starting genomes are random. Selection pressure causes it to evolve if the language stack is working correctly.

New experiment file: `experiments/27_intercell_gene_expression.py`. Output line starts with `cooperative_stream:`.

### Section 4 — Observability and Metrics

Four metrics reported each generation:

**1. Solo vs cooperative score gap**
Evaluate every cell both alone and in a random pair each generation. Report `solo_mean_error` and `cooperative_mean_error`. Widening gap (cooperative drops faster) = communication contributing to fitness.

**2. Specialisation index**
Per cell: `solo_output_variance / cooperative_output_variance` (clamped, avoiding division by zero). Low solo variance + high cooperative variance = receiver. High solo variance + stable cooperative output = sender. Report mean specialisation index across population — rising index = emerging roles.

**3. Dominant signal channel**
Track mean absolute value per signal slot in cooperative vs solo runs. Report `dominant_channel: signal[N]` — the slot with largest cooperative/solo variance ratio. If slot 1 dominates in cooperative runs, it's the inter-cell channel.

**4. Gene block activation frequency**
Track which gene IDs fire most often across a generation. Report `top_gene_A: N, top_gene_B: M` for the two most divergent gene usage patterns. Role specialisation visible at gene level.

All metrics computed from existing `CellState.signals`, `cell.output`, and active rule lists. No new infrastructure required.

### Section 5 — Testing Strategy

**Level 1 — Cooperative scorer (unit)**
- Hand-crafted genome pair (cell A emits signal[1], cell B reads signal[1] and outputs): combined score lower error than either cell alone
- Random genome pair: scorer runs without error, returns a float
- Odd population (one unpaired cell): solo evaluation runs, no crash

**Level 2 — Evolution loop (unit)**
- Population 6, 3 generations: population size stable, pair scores recorded each generation
- `cooperative_fraction=0.0`: all solo evaluations, identical to baseline colony behaviour
- `cooperative_fraction=1.0`: all cooperative evaluations, pair scores used throughout

**Level 3 — Specialisation metrics (unit)**
- Specialisation index = 0.0 for identical cell pairs (no role difference)
- Specialisation index > 0.0 for hand-crafted sender/receiver pair
- `dominant_channel` correctly identifies signal slot with most cooperative/solo variance

**Level 4 — Integration**
- Experiment 27 runs end-to-end, exits 0, stdout contains `cooperative_stream:`
- Over 20 generations with population 12: `cooperative_mean_error` trends downward
- Experiment 13 (existing cooperative chemistry) runs unchanged

## Files Changed

| File | Change |
|---|---|
| `cooperative_chemistry.py` | Add `CooperativePairScorer` — wraps `CooperativeChemistrySystem`, computes combined output, returns `Evaluation` |
| `experiments/27_intercell_gene_expression.py` | New experiment: cooperative evolution loop, specialisation metrics, output starting with `cooperative_stream:` |
| `tests/test_intercell_gene_expression.py` | New file: Level 1, 2, 3, 4 tests |
| `tests/test_experiments.py` | Add `"27_intercell_gene_expression.py"` to script list; add `"cooperative_stream:"` to recognised prefixes |

## Out of Scope

- Fixed sender/receiver roles (emergent only)
- Island populations or paired-genome individuals
- Tasks other than multiply for the initial experiment
- Inter-cell GATE conditions (runtime gating of partner's genes — future extension)
- More than two cells per cooperative group (future extension)

## Success Criteria

1. All existing tests pass unchanged
2. Hand-crafted sender/receiver genome pair scores lower error cooperatively than solo
3. Over 20 generations, `cooperative_mean_error` trends downward relative to generation 1
4. Specialisation index rises above 0.1 in at least one generation
5. Experiment 27 exits 0 and produces `cooperative_stream:` output
6. Experiment 13 output unchanged
