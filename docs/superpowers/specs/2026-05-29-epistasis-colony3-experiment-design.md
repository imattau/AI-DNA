# Epistasis Colony 3 Experiment Design

Date: 2026-05-29

## Summary

A curriculum-learning two-cell colony experiment (`experiments/32_epistasis_colony3.py`) that fixes the zero-output attractor by gradually shifting the fitness target from simple echo tasks to full cooperative multiplication. The curriculum uses a soft blend (no hard stage transitions) so the population always has a reachable gradient.

## What Failed in Experiments 30 and 31

Both experiments converged to degenerate zero-output or constant-output attractors. The fitness landscape has no gradient rewarding "getting closer to non-zero output" — any non-zero output that isn't near the target increases MSE. Zero is locally stable from generation 0.

## Curriculum Design

### Three Stages

**Stage 1 — Echo own input** (`α = 0`):
- Cell 1 target: `a/3` (echo its own signal[0])
- Cell 2 target: `b/3` (echo its own signal[1])
- No cooperation required. Just learn to produce output tied to input.

**Stage 2 — Read partner** (`α = 0.5`):
- Cell 1 target: `b/3` (must read cell 2's input via SENSE_PEER)
- Cell 2 target: `a/3` (must read cell 1's input via SENSE_PEER)
- Forces SENSE_PEER exchange to become useful.

**Stage 3 — Multiply** (`α = 1.0`):
- Colony target: `a*b/9` with forced equal mixing `(cell1.out + cell2.out) / 2`
- Full split-input multiplication.

### Blended Fitness

`α` increases linearly from 0 to 1 over 200 generations.

- Gens 0–99: blend between stage 1 and stage 2. `α_local = generation / 99`
- Gens 100–199: blend between stage 2 and stage 3. `α_local = (generation - 100) / 99`

Blended error at gen `g`:
- If g < 100: `(1 - α_local) * stage1_error + α_local * stage2_error`
- If g >= 100: `(1 - α_local) * stage2_error + α_local * stage3_error`

### Individual Fitness

Two independent populations (A and B), each scored via best-of-3 partner sampling (same as experiment 31).

Stage errors per genome are computed as:
- **Stage 1 error**: mean `(cell_role_output - own_input_target)²` over 9 cases
- **Stage 2 error**: mean `(cell_role_output - peer_input_target)²` over 9 cases
- **Stage 3 error**: mean `((cell1.out + cell2.out)/2 - a*b/9)²` over 9 cases

Each genome is scored on the blended error. Stage 3 error is a colony-level metric — both genomes in a pair share the same stage 3 error.

### Populations and Evolution

- Population A: 8 genomes, always evaluated as cell1 (`signal[0]=a/3`)
- Population B: 8 genomes, always evaluated as cell2 (`signal[1]=b/3`)
- Best-of-3 partner sampling per genome per generation
- Survivors: 3 per population, refill to 8 via mutation
- Mutation via `mutate_genome(parent, rng, lineage_id=child_id)`

## Output

Per-generation prefix `epistasis_colony3:`:
```
epistasis_colony3: gen={g} alpha={a:.3f} stage1_err={e1:.4f} stage2_err={e2:.4f} stage3_err={e3:.4f} blended_err={eb:.4f} cell1_out={c1:.4f} cell2_out={c2:.4f}
```

All three stage errors reported every generation. `blended_err` is what selection uses. `cell1_out` and `cell2_out` from the probe case (2,3) on the best pair.

Every 20 generations, snapshot the best pair's signal states on probe (2,3).

## Success Criteria

By generation 150+:
1. `stage1_err` < 0.01 — cells learned to echo their own input
2. `stage2_err` < 0.05 — peer signal exchange is functional
3. `stage3_err` declining — colony approaching multiplication
4. `cell1_out` and `cell2_out` differ on probe — role specialisation visible

## Files Changed

| File | Change |
|---|---|
| `experiments/32_epistasis_colony3.py` | New experiment script |
| `tests/test_experiments.py` | Add script + `"epistasis_colony3:"` prefix |

## Out of Scope

- Adaptive curriculum pacing (advancing faster if population is ready) — fixed schedule is sufficient for diagnosis
- More than 2 cells
- Re-introducing epistasis mixing gate — forced equal mixing until cooperation is proven
