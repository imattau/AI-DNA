# Epistasis Colony Experiment Design

Date: 2026-05-29

## Summary

A long-running experiment (`experiments/30_epistasis_colony.py`) demonstrating emergent specialisation in a two-cell colony solving split-input multiplication. Neither cell alone receives the full input; they must cooperate via peer signals and an epistasis mixing gate to produce the correct product. Success is visible when the mixing weight stabilises away from 0.5 and the two cells compute measurably different things.

This experiment exercises: cooperative chemistry, SENSE_PEER_N, epistasis (SCALE_BY_Sn), and the reading layer — all composing in a single run.

## Task

**Split-input multiplication:** input `(a, b)` where `a, b ∈ {1, 2, 3}` (9 cases total).

- Cell 1 receives `signal[0] = a / 3.0`, `signal[1] = 0.0`
- Cell 2 receives `signal[0] = 0.0`, `signal[1] = b / 3.0`

Target output = `(a * b) / 9.0` (normalised to [0, 1]).

Neither cell holds both values. Peer signal exchange via SENSE_PEER_N is the only mechanism by which each cell can access the other's input channel.

## Colony Output and Mixing Gate

Colony output = `cell_1.output × w + cell_2.output × (1 − w)`

where `w = cell_1.signals[3]` — the learned mixing weight.

`w` is not set externally. It emerges from the genome: a `SCALE_BY_S3` codon on an output-writing rule, combined with signal[3] being driven by some internal computation, is one way `w` can be shaped. Evolution finds whatever works.

Fitness = mean squared error over all 9 cases: `mean((colony_output - target)^2)`. Lower is better.

## Experiment Structure

**File:** `experiments/30_epistasis_colony.py`

**Parameters:**
- 200 generations
- Population of 8 genomes
- Each genome evaluated as a 2-cell colony (both cells use the same genome — homogeneous)
- Uses existing `CooperativePairScorer.sample_pair()` with a custom split-input task bundle
- Survivor count: 3, siblings per survivor: 2

**Per-generation output (prefix `epistasis_colony:`):**
```
epistasis_colony: gen={g} mean_error={e:.6f} best_error={b:.6f} mixing_weight={w:.4f} cell1_out={c1:.4f} cell2_out={c2:.4f}
```

`mixing_weight`, `cell1_out`, `cell2_out` are sampled on the best genome using case `(2, 3)` — the canonical probe case.

**Every 20 generations:** also print a specialisation snapshot:
```
epistasis_colony: snapshot gen={g} cell1_signals={...} cell2_signals={...}
```

## Implementation Approach

### Custom task bundle

The standard `build_multiply_bundle()` sets both input signals on a single cell. This experiment needs a `SplitMultiplyBundle` that, when evaluated as a pair, sets:
- Cell 1: `signals[0] = a/3`, `signals[1] = 0.0`
- Cell 2: `signals[0] = 0.0`, `signals[1] = b/3`

The simplest implementation: subclass or wrap `TaskBundle` with a custom scorer that intercepts cell initialisation. Alternatively, implement a standalone `score_split_pair(cell1, cell2, a, b)` function inside the experiment script that bypasses `CooperativePairScorer` entirely and directly runs chemistry on both cells with the split initialisation.

The standalone scorer approach is simpler and keeps all experiment-specific logic in the experiment file.

### Standalone pair evaluator

```python
def score_split_pair(
    genome: CellGenome,
    system: CooperativeChemistrySystem,
    rng: Random,
) -> tuple[float, float, float, float]:
    """Returns (mean_error, mixing_weight, cell1_out, cell2_out) over all 9 cases."""
    ...
```

For each `(a, b)` in `{1,2,3} × {1,2,3}`:
1. Initialise cell1 with `signals = [a/3, 0, 0, 0, 0]`, cell2 with `signals = [0, b/3, 0, 0, 0]`
2. Run N chemistry rounds (e.g., 6) with peer vector exchange between rounds
3. Colony output = `cell1.output * cell1.signals[3] + cell2.output * (1 - cell1.signals[3])`
4. Error = `(colony_output - (a*b)/9.0) ** 2`

Mean error over 9 cases = fitness. Also return `cell1.signals[3]`, `cell1.output`, `cell2.output` for the `(2, 3)` probe case.

### Evolution loop

Standard tournament selection using `mean_error` as fitness. Mutation via `evolution.mutate_genome()`. No crossover needed (single-genome colony).

## What Success Looks Like

By generation 100+, the experiment is interesting if:
1. `mean_error` is below 0.05 (vs. ~0.11 expected for random output)
2. `mixing_weight` has moved away from 0.5 and stabilised (cells have differentiated)
3. `cell1_out` and `cell2_out` differ by > 0.1 on probe case `(2, 3)` — cells are computing different things

The experiment does not need to fully solve the task to be scientifically interesting. Partial specialisation with declining error is the target observation.

## Files Changed

| File | Change |
|---|---|
| `experiments/30_epistasis_colony.py` | New experiment script |
| `tests/test_experiments.py` | Add `30_epistasis_colony.py` to script list; add `"epistasis_colony:"` to recognised output prefixes |

## Out of Scope

- Heterogeneous colony (two different genomes co-evolving) — future extension
- Writing layer integration in this experiment — the writing layer experiment comes after
- More than 2 cells — 2 is sufficient to demonstrate the principle
