# Epistasis Colony 6 Design

Date: 2026-05-30

## Summary

Experiment 35 (`35_epistasis_colony6.py`) tests whether persisted motifs from exp33 give a three-cell colony a meaningful head start. Populations are hybrid-seeded — some genomes initialised from stored gate/echo circuits, the rest random. A purpose-built two-stage curriculum (matching the exp32 blended-α pattern) grows three-cell coordination from scratch rather than transferring warm populations.

## Goal

Validate the open-ended complexity growth hypothesis: if motif seeding accelerates discovery of three-cell multiply compared to exp34 (which started fully random and converged to a two-cell attractor), then the motif store is genuinely useful as a cross-experiment building block.

## Architecture

### Three-cell roles

| Cell | Input | Role | Output |
|------|-------|------|--------|
| cell3 | c/3 in signals[2] | Echo: output c/3 | c/3 |
| cell2 | b/3 in signals[1] | Gate: read c/3 from cell3 via SENSE_PEER into s3, output (b/3)×s3 = b*c/9 | b*c/9 |
| cell1 | a/3 in signals[0] | Gate: read b*c/9 from cell2 via SENSE_PEER into s3, output (a/3)×s3 = a*b*c/27 | a*b*c/27 |

Colony output = cell1.output × cell1.signals[3]. Target = a*b*c/27 for all 27 cases (a,b,c ∈ {1,2,3}).

### Hybrid seeding

`MotifStore().query(role="gate", task="multiply_2cell")` and `role="echo"` are loaded at startup. `GenomeWriter().compose([motif], lineage_id=...)` converts the stored rule-name pattern back to a valid `CellGenome` via `CellGenome.from_rule_names()`.

- pop_a: 3 seeded from gate motifs, 5 random
- pop_b: 3 seeded from gate motifs, 5 random
- pop_c: 3 seeded from echo motifs, 5 random
- If `data/motifs.db` is absent or has fewer motifs than needed, fall back to all-random for that population

### Two-stage blended curriculum

Matches the exp32 pattern: α blends linearly between fitness targets.

**Stage 1 (generations 0–149):** α = gen / 149

| Cell | Score |
|------|-------|
| cell3 | `echo_c_err = mean((c3 − c/3)²)` over all 27 cases, paired with random cell2/cell1 partners |
| cell2 | `(1−α) × echo_b_err + α × gate_bc_err` — blend echo_b toward two-cell gate; paired with cell3 partners only |
| cell1 | `echo_a_err = mean((c1 − a/3)²)` — not yet exposed to three-cell pressure |

**Stage 2 (generations 150–399):** α2 = (gen − 150) / 249

| Cell | Score |
|------|-------|
| cell3 | `echo_c_err` (unchanged) |
| cell2 | `gate_bc_err` (two-cell with cell3, unchanged) |
| cell1 | `(1−α2) × echo_a_err + α2 × gate_abc_err` — blend echo_a toward full three-cell product |

Where:
- `echo_b_err = mean((c2 − b/3)²)`
- `gate_bc_err = mean((c2 × c2_s3 − b*c/9)²)` — cell2 paired with cell3
- `echo_a_err = mean((c1 − a/3)²)`
- `gate_abc_err = mean((c1 × c1_s3 − a*b*c/27)²)` — cell1 paired with cell2+cell3

### Fitness evaluation

Each genome is scored against `PARTNER_SAMPLES=3` random partners drawn from the relevant partner populations. In stage 1, cell2 is scored against cell3 only (two-cell sub-task). In stage 2, cell1 is scored against cell2+cell3 (three-cell).

`_run_trio()` evaluates one (genome_a, genome_b, genome_c) triple across all 27 cases and returns individual component errors plus probe values.

### Probe and output

Probe case: a=2, b=2, c=2, target=8/27≈0.2963.

Each generation prints:

```
epistasis_colony6: gen={g} colony_err={e:.6f} colony_out={o:.4f} probe_target=0.2963 cell1_out={x:.4f} cell1_s3={s:.4f} cell2_s3={t:.4f} lineage_a=... lineage_b=... lineage_c=...
```

Every 20 generations, a snapshot prints all three cells' signal vectors for the best trio.

### Parameters

| Parameter | Value |
|-----------|-------|
| POPULATION_SIZE | 8 |
| SURVIVORS | 3 |
| PARTNER_SAMPLES | 3 |
| CHEMISTRY_ROUNDS | 6 |
| TOTAL_GENS | 400 |
| Seed | 35 |

### Files

| Action | File |
|--------|------|
| Create | `experiments/35_epistasis_colony6.py` |
| Modify | `tests/test_experiments.py` — add script name and `"epistasis_colony6:"` prefix |

## Success criteria

1. Experiment runs to completion without error and prints `epistasis_colony6:` output each generation
2. If exp33 has been run (data/motifs.db exists), seeded genomes are used and confirmed in startup print
3. Ideally: colony_err < 0.05 within 400 gens (not guaranteed, but the comparison with exp34's 0.019916 attractor is the scientific result either way)

## Out of scope

- Motif capture from exp35 (no gate_err threshold triggered for three-cell circuit yet)
- Curriculum tuning or hyperparameter search
- Changing the MotifStore schema
