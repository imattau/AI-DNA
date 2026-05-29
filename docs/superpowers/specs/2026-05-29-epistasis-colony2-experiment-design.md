# Epistasis Colony 2 Experiment Design

Date: 2026-05-29

## Summary

A redesigned two-cell colony experiment (`experiments/31_epistasis_colony2.py`) that fixes the degenerate local optimum found in experiment 30. Two independent populations (A and B) co-evolve under pair fitness with forced equal mixing — no learned gate. This removes the `w=0` attractor and forces both cells to contribute meaningfully to the colony output.

## What Failed in Experiment 30

The mixing gate (`w = cell1.signals[3]`) collapsed to 0.0 at generation 0 and never moved. Colony fitness = cell2 output only. Cell1 was never selected to do anything. The degenerate attractor was wider and easier to find than the cooperation basin.

## Design

### Populations

- **Population A** (8 genomes): always evaluated as cell1. Receives `signal[0] = a/3, signal[1] = 0.0`.
- **Population B** (8 genomes): always evaluated as cell2. Receives `signal[0] = 0.0, signal[1] = b/3`.

Both cells can read each other via SENSE_PEER_N. Roles are fixed by evaluation protocol, not genome content.

### Colony Output

```
colony_output = (cell1.output + cell2.output) / 2
```

No learned gate. Equal mixing is enforced. This eliminates the `w=0` attractor: a genome that produces no output contributes 0.0 to the average and is directly penalised.

### Task

Split-input multiplication: input `(a, b)` where `a, b ∈ {1, 2, 3}` (9 cases).
- Cell 1: `signals = [a/3, 0, 0, 0, 0]`
- Cell 2: `signals = [0, b/3, 0, 0, 0]`
- Target: `(a * b) / 9.0`
- Fitness: mean squared error over all 9 cases

### Individual Fitness (Best-of-3 Sampling)

Each genome is paired with 3 randomly sampled partners from the opposing population. Individual fitness = mean colony MSE across those 3 pairings.

This gives a stable enough signal without the O(n²) cost of round-robin, and mirrors natural co-evolution (organisms don't meet every possible partner).

### Evolution Loop

Each generation:
1. For each genome in A, sample 3 partners from B (with replacement); compute mean colony error → individual fitness
2. For each genome in B, sample 3 partners from A (with replacement); compute mean colony error → individual fitness
3. Select top 3 survivors from A independently; select top 3 survivors from B independently
4. Mutate survivors to refill each population to 8

Both populations accumulate independent mutations. They can diverge in strategy — one may learn to preprocess its input channel, the other to compute a scaled output — as long as their combined average approaches the target.

### Parameters

- 200 generations
- Population A: 8 genomes, Population B: 8 genomes
- 3 partner samples per genome per generation
- Chemistry rounds: 6 per pair evaluation
- Survivors: 3 per population, 2 mutations per survivor + 1 survivor carried over

### Output

Per-generation prefix `epistasis_colony2:`:
```
epistasis_colony2: gen={g} mean_error={e:.6f} best_error={b:.6f} cell1_out={c1:.4f} cell2_out={c2:.4f} lineage_a={la} lineage_b={lb}
```

`best_error` = colony error of the best A/B pair found this generation (min over all 3×8 pairings sampled).
`cell1_out`, `cell2_out` = outputs of the best pair on probe case (2,3).
`lineage_a`, `lineage_b` = lineage IDs of the best pair.

Every 20 generations, snapshot:
```
epistasis_colony2: snapshot gen={g} cell1_signals=[...] cell2_signals=[...]
```

### File

`experiments/31_epistasis_colony2.py`

## Success Criteria

By generation 100+:
1. `mean_error` drops below 0.02 (better than single-cell constant baseline ~0.071)
2. `cell1_out` and `cell2_out` differ consistently on probe (2,3) — cells are computing different things
3. Signal snapshots show cell1 and cell2 developing distinct internal signal patterns

## Files Changed

| File | Change |
|---|---|
| `experiments/31_epistasis_colony2.py` | New experiment script |
| `tests/test_experiments.py` | Add script to list; add `"epistasis_colony2:"` to recognised prefixes |

## Out of Scope

- Re-introducing the mixing gate (experiment 30 tested that; force mixing is the diagnostic step)
- More than 2 cells
- Heterozygous populations (A and B could eventually merge into one population with role tagging)
