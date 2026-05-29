# Epistasis Colony 4 Experiment Design

Date: 2026-05-29

## Summary

A warm-start experiment (`experiments/33_epistasis_colony4.py`) that continues from experiment 32's final populations and re-introduces the epistasis gate as the colony output mechanism. The curriculum (experiments 30–32) taught cells to produce meaningful, role-appropriate outputs. This experiment tests whether 100 additional generations of gate-fitness pressure can assemble the product computation `cell1.out × cell1.signals[3] = a*b/9`.

## Why This Should Work Now

Experiment 32 established:
- Cell1 outputs ≈ a/3 (its own input channel)
- Cell2 outputs ≈ b/3 (its own input channel)
- Cells have divergent signal patterns (cell1 stable at gen 60, cell2 developed at gen 100)
- SENSE_PEER exchange is active (cell2's signal[1] = b/3 is readable by cell1)

For `colony_out = cell1.out × cell1.signals[3]` to equal `a*b/9`:
- Cell1 must output a/3 ✓ (already learned)
- Cell1 must set signal[3] = b/3 via SENSE_PEER from cell2 (new step)

The gate computation is one SENSE_PEER + one SCALE_BY_S3 away from the product.

## Warm Start

Experiment 33 re-runs the experiment 32 curriculum internally (seed=32, 200 generations) to recover the final `pop_a` and `pop_b`. Deterministic — same seed = same result. No serialization needed.

## Gate Fitness

### Colony output
```
colony_out = cell1.output × cell1.signals[3]
```

### Cell1 fitness (population A)
```
gate_error = (colony_out - a*b/9)²
```
Optimises the full product computation.

### Cell2 fitness (population B)
```
echo_error = (cell2.output - b/3)²
```
Keeps cell2's b/3 output available for cell1 to read via SENSE_PEER. Without this, cell2 loses the value cell1 needs for its gate.

### Blended individual fitness
- Population A: `gate_error` (full product pressure)
- Population B: `echo_error` (maintain b/3 output)

Best-of-3 partner sampling, same as experiments 31–32.

## Parameters

- Warm-up: 200 generations (experiment 32 curriculum, seed=32)
- Gate phase: 100 generations
- Population: 8 per population, 3 survivors, refill by mutation
- Chemistry rounds: 6

## Output

Per-generation prefix `epistasis_colony4:`:
```
epistasis_colony4: gen={g} gate_err={e:.6f} cell1_out={c1:.4f} cell1_s3={s3:.4f} cell2_out={c2:.4f} colony_out={co:.4f}
```

`cell1_s3` = cell1.signals[3] for probe (2,3). If working: should approach 1.0 (= b/3 for b=3).
`colony_out` = cell1.out × cell1.s3 for probe (2,3). Target = 6/9 = 0.667.

Every 20 generations: snapshot signal states.

## Success Criteria

By generation 80+:
1. `gate_err` < 0.02 (better than exp 32's stage3_err=0.053)
2. `cell1_s3` approaching b/3 = 1.0 for probe (2,3)
3. `colony_out` approaching target 0.667 for probe (2,3)

## Files Changed

| File | Change |
|---|---|
| `experiments/33_epistasis_colony4.py` | New experiment script |
| `tests/test_experiments.py` | Add script + `"epistasis_colony4:"` prefix |

## Out of Scope

- Warm start via serialized genome files — deterministic re-run is simpler
- Reintroducing curriculum blending in the gate phase — jump straight to gate fitness
- More than 100 gate-phase generations — sufficient to test whether the mechanism is findable
