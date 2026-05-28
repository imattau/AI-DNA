
# Gene Blocks Design

Date: 2026-05-28

## Summary

Add named, reusable gene declaration blocks to the genome. A genome can declare named codon sequences (`GENE_START <id> ... GENE_END`) and call them from elsewhere in the same genome (`CALL_<id>`). Two-pass transcription collects declarations first, then resolves calls during rule list construction. Combined with regulatory codons (PROMOTER/GATE), this enables gene networks: rule blocks that activate each other through shared signal slots, producing complex time-varying behavior from simple genomes. This is the gene→protein step in the AI-DNA language model.

## What Already Exists (do not re-implement)

- `codons.py` — CodonTable, synonym mapping, mutation helpers
- `genome.py` — CellGenome, declare_rules() (single-pass, to be extended), Motif (passive bookkeeping — unchanged)
- `chemistry.py` — chemistry VM, GateRule handling (from regulatory codons)
- Regulatory codons: PROMOTER, GATE, IF_S* condition codons, END_BLOCK, stack depth 4

## Design

### Approach: Option A — Gene blocks as genome regions

Gene declarations live in the genome itself. Two-pass transcription. Numeric IDs encoded as codon ops. CALL codons reference declarations by ID. Fully genome-local, order-independent, backward compatible.

### Section 1 — New Codon Ops

Added to the codon table, each with 3-4 synonym integer values:

| Op-name | Meaning |
|---|---|
| `GENE_START` | Begin a gene declaration block |
| `GENE_ID_0` | Gene identity: 0 |
| `GENE_ID_1` | Gene identity: 1 |
| `GENE_ID_2` | Gene identity: 2 |
| `GENE_ID_3` | Gene identity: 3 |
| `GENE_ID_4` | Gene identity: 4 |
| `GENE_ID_5` | Gene identity: 5 |
| `GENE_ID_6` | Gene identity: 6 |
| `GENE_ID_7` | Gene identity: 7 |
| `GENE_END` | Close the current gene declaration |
| `CALL_0` | Call gene 0 at this position |
| `CALL_1` | Call gene 1 at this position |
| `CALL_2` | Call gene 2 at this position |
| `CALL_3` | Call gene 3 at this position |
| `CALL_4` | Call gene 4 at this position |
| `CALL_5` | Call gene 5 at this position |
| `CALL_6` | Call gene 6 at this position |
| `CALL_7` | Call gene 7 at this position |

8 gene IDs allows complex gene networks without an unwieldy codon table. Declaration syntax: `GENE_START, GENE_ID_3, <rule codons...>, GENE_END`. Call syntax: `CALL_3` (optionally wrapped in PROMOTER/GATE conditions).

### Section 2 — Two-Pass Transcription

`declare_rules()` in `genome.py` becomes two-pass. Signature unchanged externally — `signals` parameter added (backward compatible, defaults to None meaning all blocks active).

**Pass 1 — collect declarations:**
Walk full codon list. On `GENE_START`: read next codon as gene ID, collect all codons until `GENE_END` into `gene_table: dict[int, tuple[str,...]]` (storing decoded op-names, not raw codon values). Nested `GENE_START` inside a declaration closes the outer declaration implicitly and starts a new one. End-of-genome inside a declaration = implicit `GENE_END`.

**Pass 2 — build active rule list:**
Walk codon list again. Skip all `GENE_START ... GENE_END` regions (definitions, not execution). On `CALL_<id>`: look up `gene_table[id]`, recursively decode those op-names into rules and insert at current position. If ID not in gene_table: silent skip (pseudogene — inert, not an error). PROMOTER and GATE conditions wrap CALL codons normally.

**Recursion depth cap:** gene calls inside gene declarations are followed (genes can call genes), capped at depth 4. Matches regulatory codon stack depth cap.

**Backward compatible:** genomes with no gene codons pass through both phases identically to current behavior.

### Section 3 — Gene Networks

With declarations, calls, and regulatory codons composing, gene networks emerge without new mechanisms:

```
GENE_START, GENE_ID_0, RULE_EMIT_S1, GENE_END      ← gene 0: emits into signal[1]
GENE_START, GENE_ID_1, RULE_INHIBIT_S0, GENE_END   ← gene 1: inhibits signal[0]
CALL_0                                               ← always express gene 0
PROMOTER, IF_S1_GT, CALL_1, END_BLOCK               ← express gene 1 only when signal[1] high
```

Gene 0 fires → raises signal[1] → next episode PROMOTER activates → gene 1 fires → inhibits signal[0] → gene 0 weakens. Negative feedback loop from three lines of regulatory structure.

### Section 4 — Mutation Behaviour

**Declaration mutations:**
- Mutation inside a declaration changes what the gene does at all call sites simultaneously
- `GENE_ID_3` mutating to `GENE_ID_5` inside a declaration renames the gene; all `CALL_3` become dangling (silent skip), `CALL_5` gains a new target
- Deletion of `GENE_END` merges declarations or absorbs following rules into the gene body — degenerate but valid

**Call mutations:**
- `CALL_3` mutating to `CALL_5` rewires which gene fires at that position — declarations unchanged
- `CALL` inserted by mutation with no matching declaration is immediately a silent pseudogene call — inert until a matching declaration appears through crossover or motif injection

**Crossover:**
- Crossover inside a declaration can split it across offspring — both receive partial declarations (valid, treated as ending at genome end)
- Crossover can unite a declaration from one parent with a call from the other — gene network rewiring through recombination, a major source of novelty
- With Phase B motif-aware crossover: crossover points prefer motif boundaries; gene declarations are natural motif candidates, so declarations tend to be preserved intact across recombination

### Section 5 — Documented Future Extension: Genome-Level Repair

When `CALL_<id>` has no matching declaration (dangling reference from mutation), current behavior is silent skip. A future repair layer will:
1. Detect dangling CALLs during transcription
2. Search the cell's motif library for a motif whose codon pattern could serve as a gene declaration for that ID
3. Reconstruct a `GENE_START, GENE_ID_<id>, <motif codons>, GENE_END` block and insert it into the genome

This closes the loop between evolutionary memory (motifs as passive records) and active genome structure (gene declarations). It is the AI-DNA equivalent of DNA repair mechanisms. To be specced as a separate feature after gene blocks are stable and producing measurable gene network behavior.

### Section 6 — Testing Strategy

**Level 1 — Codon table (unit)**
- All 20 new op-names present with correct synonym counts
- `GENE_ID_0` through `GENE_ID_7` all distinct, each 3-4 synonyms
- `CALL_0` through `CALL_7` all distinct, each 3-4 synonyms
- Round-trip encode/decode preserves op-names

**Level 2 — Declaration collection, Pass 1 (unit)**
- Genome with `GENE_START, GENE_ID_2, RULE_EMIT_X, GENE_END` → `gene_table[2]` contains `RULE_EMIT_X`
- Two declarations → both collected correctly
- Nested `GENE_START` inside declaration → outer closes implicitly, inner starts fresh
- No gene codons → empty gene_table, no error

**Level 3 — Call resolution, Pass 2 (unit)**
- `CALL_2` with `gene_table[2]` containing `RULE_EMIT_X` → `RULE_EMIT_X` in active rules at call position
- `CALL_5` with no entry in gene_table → silent skip, no error, execution continues
- Declaration after call (order-independent) → still resolved correctly
- Gene calls gene at depth ≤ 4 → resolved; depth > 4 → stops without error

**Level 4 — Regulatory composition (unit)**
- `PROMOTER, IF_S0_GT, CALL_2, END_BLOCK` with signal[0]=0.8 → gene 2 rules in active list
- Same with signal[0]=0.2 → gene 2 absent
- `GATE` wrapping `CALL_2` → gene 2 rules gated per chemistry round

**Level 5 — Gene network integration**
- Genome encoding negative feedback loop (Section 3) → signal[1] rises then stabilises across episodes
- Experiment 01 (multiply benchmark) runs unchanged — no gene codons in existing genomes, identical output

## Files Changed

| File | Change |
|---|---|
| `codons.py` | Add 20 new op-names (GENE_START, GENE_ID_0-7, GENE_END, CALL_0-7) with synonyms |
| `genome.py` | Extend `declare_rules()` to two-pass; add `gene_table` collection in pass 1; resolve CALLs in pass 2; recursion depth cap 4 |
| `chemistry.py` | No changes needed — GateRule handling already in place from regulatory codons |
| `tests/test_codons.py` | Level 1 tests |
| `tests/test_gene_blocks.py` | New file: Level 2, 3, 4, 5 tests |

## Out of Scope

- More than 8 gene IDs per genome (extend later)
- Recursion depth > 4 (extend later)
- Genome-level repair (documented above, specced separately)
- Inter-cell gene expression (future: cell A activates gene in cell B via shared chemistry)
- Gene expression tracking / reporting (future: which genes fired in which episodes)

## Success Criteria

1. All existing tests pass unchanged
2. A genome with gene declarations and CALL codons executes the declared rules at call positions
3. CALL with no matching declaration is silently skipped — no error
4. Gene network (Section 3 example) produces signal[1] rise and stabilisation across episodes
5. PROMOTER/GATE conditions correctly gate CALL codons
6. Experiment 01 output identical to pre-change baseline
