# Epistasis Design

Date: 2026-05-29

## Summary

Add analog output modulation to the AI-DNA rule VM. A new family of `SCALE_BY_Sn` codons annotates the immediately following rule so that all its signal writes are multiplied by the current value of `signal[n]`. This is epistasis: one gene's signal product continuously scales another gene's output magnitude, without blocking it entirely.

This completes the regulatory hierarchy:
- `PROMOTER` — binary enable/disable at transcription time
- `GATE` / `IF_*` — binary enable/disable at runtime
- `SCALE_BY_Sn` — analog scaling at runtime (this feature)

Complexity comes from composition, not richer opcodes. `SCALE_BY_S2, EMIT_S0` with signal[2]=0.5 halves the emission; signal[2]=0.0 suppresses it to zero; signal[2]=1.0 leaves it unchanged.

## What Already Exists (do not re-implement)

- `PROMOTER` — transcription-time gating via promoter stack in `genome.py`
- `GateRule` dataclass in `genome.py` — runtime binary gate wrapper
- `_condition_passes()` in `chemistry.py` — evaluates IF_* conditions
- Signal clamp logic throughout `chemistry.py` — all signal writes already clamped to [0, 1]

## What Is Missing

### 1. New codons in `codons.py`

Four new codon names, one per modulatable signal slot:

```
SCALE_BY_S0
SCALE_BY_S1
SCALE_BY_S2
SCALE_BY_S3
```

Each gets 3 byte synonyms in the codon table, consistent with all existing codons. They appear in the `ALL_CODONS` list and the byte→name mapping dict.

These are modifier codons — they carry no runtime semantics themselves. Their only effect is to annotate the next rule during transcription.

### 2. `ScaleRule` dataclass in `genome.py`

```python
@dataclass(frozen=True, slots=True)
class ScaleRule:
    signal_slot: int          # 0–3, which signal to read as modulator
    inner: str | GateRule     # the rule being modulated
```

`ScaleRule` is a wrapper — it never appears standalone in a rule list. It wraps one `inner` rule and records which signal slot provides the scale factor.

### 3. Transcription in `genome.py`

During `_transcribe_rules()`, a `SCALE_BY_Sn` codon sets a one-shot pending modifier (an integer 0–3, or `None`). The next rule encountered is wrapped in a `ScaleRule(signal_slot=n, inner=rule)` before being appended to the output list. The pending modifier is then cleared.

**Edge cases:**
- Two consecutive `SCALE_BY` codons: second overwrites first (last writer wins).
- `SCALE_BY` at genome end with no following rule: silently dropped.
- `SCALE_BY` followed by another `SCALE_BY` (no rule between): last wins, applied to the first real rule encountered.

### 4. VM execution in `chemistry.py`

When the VM encounters a `ScaleRule` in the rule list:

1. Read `modulator = cell.signals[scale_rule.signal_slot]` once at fire time.
2. Execute `scale_rule.inner` rule normally through the existing dispatch path.
3. After each signal write that the inner rule performs, apply: `new_val = clamp(written_val * modulator)`.

The modulator is read once and held constant for the rule's full execution. The modulator signal slot itself is never written by this operation.

**All signal writes** in the inner rule are scaled — there is no distinction between primary and secondary outputs.

`clamp` = the existing `max(0.0, min(1.0, val))` used throughout `chemistry.py`.

Implementation approach: the VM's inner rule dispatch already returns or applies signal deltas. The cleanest implementation is to temporarily replace `cell.signals` with a proxy during inner rule execution, or to apply the scale factor immediately after the inner rule's writes by diffing cell state before and after. The diff approach is simpler: snapshot `cell.signals` before, execute inner rule, then for each slot that changed, re-apply `signals[i] = clamp(signals[i] * modulator + pre[i] * (1 - modulator))`.

Wait — that formula is wrong. The correct approach: snapshot before, execute inner, then for each slot `i` where `signals[i] != pre[i]`, set `signals[i] = clamp(pre[i] + (signals[i] - pre[i]) * modulator)`. This scales the *delta*, not the absolute value, which is the correct semantics: the rule's contribution is scaled, not the existing signal level.

### 5. Testing

- Unit: `SCALE_BY_S0, EMIT_S0` — with signal[0]=0.5 set before execution, assert emitted delta is halved vs unscaled baseline.
- Unit: `SCALE_BY_S2, EMIT_S0` with signal[2]=0.0 — rule fires, writes 0.0 delta (full suppression without blocking).
- Unit: dangling `SCALE_BY_S0` at end of codon sequence — `declare_rules()` returns same rules as without it, no crash.
- Unit: `SCALE_BY_S0, SCALE_BY_S1, EMIT_S0` — assert signal[1] (last writer) is used as modulator.
- Integration: genome containing `SCALE_BY_S2` modulating an EMIT evolves on a task — `evaluate()` returns a finite error, no crash.

## Files Changed

| File | Change |
|---|---|
| `codons.py` | Add `SCALE_BY_S0–S3` to `ALL_CODONS` and byte→name table (12 byte slots, 3 per name) |
| `genome.py` | Add `ScaleRule` dataclass; add pending-modifier logic to `_transcribe_rules()` |
| `chemistry.py` | Add `ScaleRule` dispatch to VM: snapshot → execute inner → scale deltas |
| `tests/test_epistasis.py` | New file: unit + integration tests |

## Out of Scope

- Scaling by signal[4] (stage accumulator) — signal[4] is the temporal stage counter; modulating by it is temporal unfolding (already implemented via IF_S4_GT/LT)
- Scaling by peer signals — inter-cell modulation is already handled via SENSE_PEER_N
- Chained scaling (two SCALE_BY wrapping the same rule) — last writer wins is sufficient

## Success Criteria

1. All existing tests pass.
2. A rule wrapped in `ScaleRule` with modulator 0.5 produces exactly half the signal delta of the same rule unwrapped.
3. A rule wrapped in `ScaleRule` with modulator 0.0 fires but produces zero delta.
4. Dangling `SCALE_BY` codons do not crash transcription.
5. Genomes containing `SCALE_BY` codons evaluate without error on existing tasks.
