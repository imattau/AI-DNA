# Temporal Unfolding Design

Date: 2026-05-29

## Summary

Add a dedicated stage accumulator signal (signal[4]) to the chemistry VM so genomes can encode developmental programs that express different gene sets at different lifecycle stages. The VM increments signal[4] by a fixed rate each chemistry round. Genomes gate gene block expression using existing IF_S4_GT/IF_S4_LT condition codons. This is the morphogen gradient model from real developmental biology — a signal that accumulates over time and crosses thresholds that switch gene expression waves. No new codon types beyond IF_S4_GT and IF_S4_LT. Backward compatible.

## What Already Exists (do not re-implement)

- `cell.py` — CellState with 4-slot signals list, reset() method
- `chemistry.py` — ChemistryContext with time field, run() loop, _condition_passes() for IF_S* ops
- `codons.py` — IF_S0_GT through IF_S2_LT established pattern (IF_S3_GT/LT missing — add those too for completeness)
- Regulatory codons: PROMOTER/GATE/END_BLOCK use _condition_passes() — no changes needed to that layer
- Gene blocks: CALL_0-7 — no changes needed

## Design

### Approach: Stage accumulator in ChemistryContext

signal[4] is the dedicated stage slot. The VM increments it unconditionally each round before rules fire. Genomes written before this change have 4-slot signal access — they continue to work unchanged because signal[4] is never referenced by existing rules. The only new things: 5-slot signals, stage_increment on ChemistryContext, IF_S4_GT/IF_S4_LT in the codon table and _condition_passes().

### Section 1 — Signal Slot Extension

`CellState.signals` extended from 4 slots to 5 slots:
- signal[0]: x input (existing)
- signal[1]: y input / computation (existing)
- signal[2]: computation (existing)
- signal[3]: computation (existing)
- signal[4]: stage accumulator (new, dedicated, VM-controlled)

Default: `[0.0, 0.0, 0.0, 0.0, 0.0]`

`reset(x, y)` initialises as `[x, y, 0.0, 0.0, 0.0]` — signal[4] always starts at 0.0 each episode.

`_record()` type hint updated from `tuple[float,float,float,float]` to `tuple[float,float,float,float,float]`.

Two new condition codons added to codon table (3-4 synonyms each):
- `IF_S3_GT` — signal[3] > 0.5 (was missing, add for completeness)
- `IF_S3_LT` — signal[3] < 0.5
- `IF_S4_GT` — signal[4] > 0.5 (stage gate: fires after ~5 rounds)
- `IF_S4_LT` — signal[4] < 0.5 (early-stage gate: fires before ~5 rounds)

### Section 2 — VM Stage Increment

`ChemistryContext` gains one field:
```python
stage_increment: float = 0.1
```

In `ChemistrySystem.run()` (or `step()`), at the **start of each round before any rules fire**:
```python
cell.signals[4] = min(cell.signals[4] + context.stage_increment, 1.0)
```

Clamped at 1.0 — signal[4] saturates, does not overflow.

Timeline with default rate:
- Round 1: signal[4] = 0.1
- Round 5: signal[4] = 0.5 → IF_S4_GT threshold crossed
- Round 10: signal[4] = 1.0 (saturated)

Episode reset: `cell.reset()` sets signal[4] = 0.0. Each new task case starts from stage 0.

Experiments can set `stage_increment=0.0` to disable temporal unfolding entirely (backward compatibility mode).

### Section 3 — Developmental Gene Expression

What this enables — a genome with two developmental phases:

```
GENE_START, GENE_ID_0, RULE_EMIT_X, GENE_END          ← early gene: loads x input
GENE_START, GENE_ID_1, RULE_ADD0_IF1, GENE_END        ← late gene: accumulates
CALL_0                                                  ← always express early gene
PROMOTER, IF_S4_GT, CALL_1, END_BLOCK                 ← late gene only after stage 0.5
```

Rounds 1-4: only gene 0 fires — pattern establishment phase.
Round 5+: gene 1 activates — computation phase.

Same genome, two developmental phases, no new mechanisms. Composes naturally with:
- GATE conditions: a runtime gate on IF_S4_GT rechecks each round (fires immediately when threshold crossed)
- Cooperative chemistry: stage signal[4] is cell-local, not shared via ChemistryContext outbox
- Gene networks: gene expressed in early phase can produce signal that gates late-phase gene differently per cell

### Section 4 — Testing Strategy

**Level 1 — Signal slot extension (unit)**
- `CellState()` default has 5 slots, signal[4] = 0.0
- `cell.reset(2, 3)` sets signals = [2, 3, 0, 0, 0]
- IF_S4_GT and IF_S4_LT present in codon table with synonyms

**Level 2 — VM increment (unit)**
- After 1 round: signal[4] = 0.1
- After 5 rounds: signal[4] = 0.5
- After 10 rounds: signal[4] = 1.0 (clamped, not 1.1)
- `stage_increment=0.0`: signal[4] stays 0.0 throughout episode
- Episode reset: signal[4] = 0.0 after cell.reset()

**Level 3 — Stage gating (unit)**
- PROMOTER + IF_S4_GT + CALL_1 + END_BLOCK: gene 1 absent in rounds 1-4, present from round 5
- GATE + IF_S4_LT + CALL_0 + END_BLOCK: gene 0 fires in rounds 1-4, stops from round 5
- Two-phase genome (Section 3 example): early gene active from round 1, late gene from round 5

**Level 4 — Regression**
- Experiment 01 (multiply benchmark): `stage_increment=0.0` default in existing configs → signal[4] stays 0.0 → identical output to pre-change baseline
- All existing tests pass with 5-slot signals (4-slot references unchanged)

## Files Changed

| File | Change |
|---|---|
| `cell.py` | Extend signals to 5 slots; update reset() and type hints |
| `chemistry.py` | Add stage_increment to ChemistryContext; increment signal[4] each round; add IF_S3_GT/LT/IF_S4_GT/LT to _condition_passes(); update _record() type hint |
| `codons.py` | Add IF_S3_GT, IF_S3_LT, IF_S4_GT, IF_S4_LT with 3-4 synonyms each |
| `tests/test_temporal_unfolding.py` | New file: Level 1, 2, 3, 4 tests |

## Out of Scope

- Multiple stage accumulators (one is sufficient for developmental waves)
- Genome-encoded stage rate (fixed rate is simpler and sufficient)
- Inter-cell stage synchronisation (cells develop independently)
- Stage decay or oscillation (accumulate-and-saturate is the simplest model)

## Success Criteria

1. All existing tests pass unchanged (with stage_increment=0.0 default in existing configs)
2. signal[4] increments correctly each round and clamps at 1.0
3. IF_S4_GT correctly gates gene expression from round 5 onward
4. Two-phase genome (Section 3 example) produces different active rule sets in rounds 1-4 vs 5+
5. Experiment 01 output identical to pre-change baseline
