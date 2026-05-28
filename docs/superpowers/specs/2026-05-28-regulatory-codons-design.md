# Regulatory Codons Design

Date: 2026-05-28

## Summary

Add two independent regulatory layers to the codon/chemistry architecture: genome-level promoters (transcription-time gating) and chemistry-level GATE codons (runtime gating). Both share the same signal slots so regulation and computation interact directly. The same structural idiom — a control codon opens a block, condition codons define the test, END_BLOCK closes it — is used at both layers. This is the first step toward hierarchical, context-sensitive genomes that can encode conditional programs rather than flat rule sequences.

## What Already Exists (do not re-implement)

- `codons.py` — CodonTable mapping int values to op-name strings, synonym mapping, mutation helpers
- `genome.py` — CellGenome, flat codon list, declare_rules() transcription, Motif
- `chemistry.py` — chemistry VM, Rule functions `(CellState, TaskCase, ChemistryContext) -> str | None`, signal slots signal[0..3]
- `cell.py` — CellState with signal slots
- Neutral mutation (synonym codons) — must be preserved for new op-names

## Design

### Approach: Option A — Flat regulatory codons

New codon op-names added to the existing table, distributed throughout the genome like real regulatory sequences. Same transcription reader, extended to handle control codons. Backward compatible — genomes without regulatory codons behave identically.

### Section 1 — Codon Table Extensions

Eight new op-names, each with 3-4 synonym codon values for neutral drift:

| Op-name | Meaning |
|---|---|
| `PROMOTER` | Start a transcription-time condition block |
| `GATE` | Start a runtime condition block |
| `IF_S0_GT` | Condition: signal[0] > 0.5 |
| `IF_S1_GT` | Condition: signal[1] > 0.5 |
| `IF_S2_GT` | Condition: signal[2] > 0.5 |
| `IF_S0_LT` | Condition: signal[0] < 0.5 |
| `IF_S1_LT` | Condition: signal[1] < 0.5 |
| `END_BLOCK` | Close the current condition block |

Threshold fixed at 0.5 — high/low switch, matching biological promoter on/off behaviour. Extendable later.

### Section 2 — Transcription (Genome-Level Promoters)

`declare_rules()` in `genome.py` walks the codon list and maintains a condition stack (max depth 4):

1. `PROMOTER` → evaluate the next codon as a condition against `cell.signals` at transcription time
   - True: include rules that follow
   - False: skip all codons until matching `END_BLOCK`
2. `GATE` → emit a `GateRule(condition, block)` placeholder into the active rule list (evaluated at runtime)
3. `IF_S*` codon → used as condition test immediately after `PROMOTER` or `GATE`
4. `END_BLOCK` → close current block, resume normal reading
5. All other codons → decoded as rules exactly as today

Stack depth capped at 4. End-of-genome treated as implicit `END_BLOCK`. Unknown condition codon defaults to **true** (constitutive expression — backward compatible).

Promoters read `cell.signals` at transcription time. Since transcription happens at the start of each episode, signals from the previous episode shape which rules activate in the next. This creates the transcription-time feedback loop.

### Section 3 — Runtime Gating (Chemistry-Level)

The chemistry VM in `chemistry.py` processes `GateRule` placeholders in the active rule list each chemistry round:

1. `GateRule(condition, block)` → evaluate condition against **current** `cell.signals` this round
   - True: execute rules inside the block this round
   - False: skip block entirely this round
2. Re-evaluated fresh every chemistry round — can flip open and closed within a single episode

Key difference from promoters: promoters decide once per episode at transcription time; gates decide once per chemistry round at runtime.

**The feedback loop:**
```
Rule fires → updates signal[1] →
GATE checks signal[1] → enables inner rules →
inner rules update signal[2] →
next episode PROMOTER checks signal[2] →
activates different rule block
```

### Section 4 — Mutation and Neutral Drift

Three properties preserved:

1. **Synonym neutrality** — 3-4 synonym values per op-name. Mutations between synonyms are neutral.

2. **Block integrity** — unbalanced blocks (missing END_BLOCK) are valid genomes. End-of-genome = implicit END_BLOCK. Stack depth cap (4) limits pathological structures. Selection pressure disfavors bad structures without needing validation.

3. **Insertion at block boundaries** — a codon inserted between PROMOTER and its condition becomes the new condition. If it's not an IF_S* op, it defaults to true (constitutive expression). Creates meaningful variation, not crashes.

**Biological parallels:**
- Unbalanced blocks → pseudogenes (inert, not harmful)
- Unknown condition → constitutive expression (always-on gene)
- Insertion at regulatory boundary → promoter mutation (changes when, not what)

### Section 5 — Testing Strategy

**Level 1 — Codon table (unit)**
- New op-names present with correct synonym counts
- Mutations between synonyms of IF_S0_GT produce identical decoded ops
- Round-trip encode/decode preserves op-names

**Level 2 — Transcription (unit)**
- PROMOTER + IF_S0_GT + RULE_EMIT_X + END_BLOCK with signal[0]=0.8 → RULE_EMIT_X in active rules
- Same genome with signal[0]=0.2 → RULE_EMIT_X absent
- Unbalanced genome (missing END_BLOCK) → completes without error
- Unknown condition codon → defaults to true, block activates

**Level 3 — Runtime gating (unit)**
- GateRule(IF_S1_GT, [RULE_EMIT_X]) with signal[1]=0.8 → rule fires this round
- Same with signal[1]=0.2 → rule does not fire
- Signal crosses threshold mid-episode → gate opens from that round forward

**Level 4 — Integration**
- Genome with both promoter and gate on same signal slot: promoter activates block at transcription, gate controls round-by-round execution within block
- Experiment 01 (multiply benchmark) runs unchanged — no regulatory codons, identical behaviour

## Files Changed

| File | Change |
|---|---|
| `codons.py` | Add 8 new op-names with synonym values |
| `genome.py` | Extend `declare_rules()` with promoter stack logic; add `GateRule` dataclass |
| `chemistry.py` | VM loop handles `GateRule` placeholders with per-round condition evaluation |
| `cell.py` | No changes needed |
| `tests/test_codons.py` | Level 1 tests |
| `tests/test_regulatory_codons.py` | New file: Level 2, 3, 4 tests |

## Out of Scope

- Threshold values other than 0.5 (extend later)
- Stack depth > 4 (extend later)
- Regulatory signals separate from computation signals (user chose shared)
- Nested GATE inside PROMOTER inside GATE beyond depth cap

## Success Criteria

1. All existing tests pass unchanged
2. A genome with PROMOTER blocks activates different rules depending on signal state at transcription time
3. A genome with GATE blocks toggles rule execution within an episode as signals change
4. Neutral drift rate (Phase A metric) is non-zero for regulatory genomes
5. Experiment 01 output is identical to pre-change baseline
