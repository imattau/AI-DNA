# Writing Layer Design

Date: 2026-05-29

## Summary

Add a writing layer to AI-DNA that allows external agents (or meta-learners) to compose new genomes from existing motifs. The layer sits above the evolutionary substrate: it queries a library of captured motifs and concatenates their codon sequences into a valid `CellGenome`. This is the complement to the reading layer (`reading.py`) — read observes what a genome is doing; write constructs what a genome should do.

The writing intent is motif-level. Individual codons are too low-level (random search does that better); whole-genome rewrite discards structure. Motifs are the natural unit — they are named, task-tagged, and already produced by the existing `attach_success_motif()` pathway.

## What Already Exists (do not re-implement)

- `Motif` dataclass in `genome.py` — `name`, `codons`, `task_origin`, `lineage_id`, `reuse_count`
- `attach_success_motif()` in `evolution.py` — captures motifs at evaluation time
- `MotifLibrary` query stubs in `genome.py` — `find_by_task()`, `find_by_lineage()`
- `CellReader` in `reading.py` — `observe()` returns 6-float signal vector
- `cell.to_vector()` in `cell.py` — base 6-float vector (signal[0-4] + output)

## What Is Missing

### 1. `Motif.origin_signals` field

`Motif` has no record of the cell's signal state at the moment the motif was captured. Without this, signal-vector similarity queries are impossible.

**Change:** Add `origin_signals: tuple[float, ...] = ()` to the `Motif` dataclass in `genome.py`.

**Populate:** In `attach_success_motif()` in `evolution.py`, call `cell.to_vector()` immediately before constructing the `Motif` and pass the result as `origin_signals`.

This is backward-compatible: existing motifs (empty tuple) score 0 in cosine queries and are returned last.

### 2. `MotifLibrary` and `MotifQuery` in `writing.py`

A query API for selecting motifs from a collection.

```python
@dataclass
class MotifQuery:
    task_origin: str | None = None
    lineage_prefix: str | None = None
    signal_vector: tuple[float, ...] = ()
    top_k: int = 4
```

```python
class MotifLibrary:
    def __init__(self, motifs: Iterable[Motif]) -> None: ...
    def query(self, q: MotifQuery) -> list[Motif]: ...
```

**Query pipeline:**

1. If `task_origin` is set, filter to motifs whose `task_origin` matches exactly.
2. If `lineage_prefix` is set, filter to motifs whose `lineage_id` starts with the prefix.
3. If `signal_vector` is non-empty, rank remaining motifs by cosine similarity between `signal_vector` and `motif.origin_signals`. Motifs with empty `origin_signals` score 0 and sort last.
4. Return the top-k results. If fewer than k remain after filtering, return all.

Filters compose: each narrows the pool before the next applies. An empty query returns up to top-k motifs in insertion order.

**Cosine similarity:** standard dot-product / (|a| * |b|), clamped to [0, 1]. If either vector is zero-length, similarity = 0.

### 3. `GenomeWriter` in `writing.py`

```python
class GenomeWriter:
    def compose(self, motifs: Sequence[Motif], lineage_id: str = "") -> CellGenome:
        ...
```

Concatenates the `codons` sequences of the supplied motifs in order, then calls `declare_rules()` to transcribe the codon sequence into a valid `CellGenome`. Sets the genome's `lineage_id` field.

No mutation, no crossover — those remain in `evolution.py`. The writer only composes. Callers who want variation apply mutation after composition.

If `motifs` is empty, `compose()` raises `ValueError`. If the concatenated codon sequence produces no rules after transcription (degenerate genome), the genome is returned as-is — the caller decides whether to retry.

### 4. `writing.py` module layout

```
writing.py
├── _cosine(a, b) -> float          # internal helper
├── MotifQuery                       # dataclass
├── MotifLibrary                     # query engine
└── GenomeWriter                     # composition engine
```

No dependency on `task_stream.py`, `evolution.py`, or `spatial.py`. Only imports: `genome.py` (Motif, CellGenome, declare_rules), `cell.py` (not needed directly — origin_signals is pre-computed).

## Files Changed

| File | Change |
|---|---|
| `genome.py` | Add `origin_signals: tuple[float, ...]= ()` to `Motif` dataclass |
| `evolution.py` | Populate `origin_signals` in `attach_success_motif()` |
| `writing.py` | New file: `MotifQuery`, `MotifLibrary`, `GenomeWriter` |
| `tests/test_writing.py` | New file: unit and integration tests |

## Out of Scope

- Motif mutation during composition — use `evolution.mutate_genome()` after composing
- Cross-genome crossover at write time — already in `evolution.crossover_genomes()`
- Experiment script — a later experiment will demonstrate motif-guided search
- Persisting the motif library to disk — callers manage the `Motif` collection

## Success Criteria

1. All existing tests pass.
2. `MotifLibrary.query()` returns correct subsets for task, lineage, and signal filters independently and combined.
3. `GenomeWriter.compose()` produces a `CellGenome` whose rule count is > 0 for any non-trivial motif input.
4. Round-trip integration: motifs captured from a short stream run → `MotifLibrary.query()` → `GenomeWriter.compose()` → `evaluate()` returns a finite error.
