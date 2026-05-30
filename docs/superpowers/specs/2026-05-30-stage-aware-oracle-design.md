# Stage-Aware Oracle Design (Exp 38)

**Date:** 2026-05-30

## Goal

Extend `CodonOracle` with stage detection so tinyllama proposes codons matched to the current developmental stage of the evolving colony, rather than proposing the same codon class regardless of context.

## Background

Exp37 (blind oracle) showed tinyllama correctly identifies `SENSE_PEER_S_TO_D` as the right codon class but proposes it even when cell2 can't yet output proportional values — the prerequisite skill. Exp38 adds stage awareness so the oracle scaffolds skills in the right order.

## Stage Detection

Three stages derived from per-generation metrics:

| Stage | Condition | Meaning |
|---|---|---|
| `echo` | echo_err > 0.05 | cell2 can't produce proportional output (b/3) |
| `sense` | echo_err ≤ 0.05 AND cell2_s3 < 0.1 | can output b/3 but not reading c/3 from peer |
| `scale` | echo_err ≤ 0.05 AND cell2_s3 ≥ 0.1 | reading peer but gate product still wrong |

**Metrics definitions:**
- `echo_err`: mean((cell2_output - b/3)²) over all 9 (b,c) cases — measures cell2's ability to echo its own input
- `gate_err`: mean((cell2_output × cell2_s3 - b×c/9)²) over all 9 cases — measures gate product accuracy
- `cell2_s3`: mean of cell2.signals[3] over all 9 cases — measures peer-signal uptake

## Stage-Constrained Template Menu

Each stage narrows the template menu sent to tinyllama:

| Stage | Templates offered |
|---|---|
| `echo` | `COPY_S_TO_D` only — routes signals within cell2 |
| `sense` | `SENSE_PEER_S_TO_D` only — reads from peer cell |
| `scale` | `SCALE_BY_SN`, `IF_SN_GT`, `IF_SN_LT` — refines gate product |

## Changes to `codon_oracle.py`

Two new methods added to `CodonOracle`:

```python
def detect_stage(self, echo_err: float, gate_err: float, cell2_s3_mean: float) -> str:
    if echo_err > 0.05:
        return "echo"
    if cell2_s3_mean < 0.1:
        return "sense"
    return "scale"

def build_stage_prompt(
    self,
    stage: str,
    codon_names: list[str],
    fitness_history: list[float],
    best_motif: list[str],
) -> str:
    # stage-specific menu and framing
    ...
```

`try_inject` gains three new parameters: `echo_err: float`, `gate_err: float`, `cell2_s3_mean: float`. It calls `detect_stage` and `build_stage_prompt` instead of `build_prompt`.

## New File: `experiments/38_stage_oracle.py`

Runs the same bc gate evolution as exp37 with:
- Separate echo_err and gate_err computed each generation
- `cell2_s3_mean` averaged over all 9 test cases
- Stage reported in output: `stage_oracle: gen=N stage=echo gate_err=... echo_err=...`
- Oracle fires every 50 stagnating gens (same window as exp37)
- GATE_GENS = 600
- Output prefix: `stage_oracle:`

## Echo Error Measurement

New helper `_echo_err_b(genome_b, genome_c, system)` computes cell2's echo error:

```python
def _echo_err_b(genome_b, genome_c, system):
    err = 0.0
    s3_sum = 0.0
    for b, c in SPLIT_CASES:
        # run pair, measure cell2_out vs b/3 and cell2.signals[3]
        err += (cell2_out - b/3)**2
        s3_sum += cell2.signals[3]
    return err / len(SPLIT_CASES), s3_sum / len(SPLIT_CASES)
```

## Test Coverage

Add to `tests/test_codon_oracle.py`:
- `test_detect_stage_echo` — high echo_err → "echo"
- `test_detect_stage_sense` — low echo_err, low s3 → "sense"  
- `test_detect_stage_scale` — low echo_err, high s3 → "scale"
- `test_build_stage_prompt_echo_has_copy_template` — echo stage prompt contains COPY_S_TO_D
- `test_build_stage_prompt_sense_has_sense_peer` — sense stage prompt contains SENSE_PEER_S_TO_D
- `test_build_stage_prompt_scale_has_scale_by` — scale stage prompt contains SCALE_BY_SN

## Test Suite Registration

Add `"38_stage_oracle.py"` to `tests/test_experiments.py` scripts list and `"stage_oracle:"` to recognised prefixes.
