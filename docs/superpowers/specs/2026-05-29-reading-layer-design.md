
# Reading Layer Design

Date: 2026-05-29

## Summary

Add a compact float vector encoding of cell state that serves as the machine-native reading protocol for the AI-DNA language. A `to_vector()` method on `CellState` returns a 6-float tuple (signal[0-4] + output) consumable directly by neural networks or other AI systems without parsing. Two consumers share one protocol: inter-cell reading via new `SENSE_PEER_N` codon ops in cooperative chemistry, and external agent observation via a thin `CellReader` wrapper in a new `reading.py` module. This is the bridge that makes AI-DNA a language rather than just a computation substrate.

## What Already Exists (do not re-implement)

- `cell.py` — CellState with 5-slot signals, output field, reset()
- `cooperative_chemistry.py` — CooperativeChemistrySystem, shared ChemistryContext, inbox/outbox
- `tracing.py` — ExperimentReport, human-readable output (unchanged — this layer is parallel, not a replacement)
- `genome.py` — CellGenome, Motif, declare_rules(), signature()
- Gene blocks: active gene IDs available from declare_rules() pass 2
- Motifs: local_motifs on CellGenome

## Design

### Approach: `to_vector()` on CellState, one protocol two consumers

Minimal — one method, fixed-length vector, same format for inter-cell and external agent consumption. Optional extended mode appends gene activation and motif hash without breaking base readers.

### Section 1 — `to_vector()` on CellState

Add to `cell.py`:

```python
def to_vector(
    self,
    *,
    extended: bool = False,
    active_gene_ids: frozenset[int] = frozenset(),
    motif_hash: float = 0.0,
) -> tuple[float, ...]:
    out = self.output if self.output is not None else 0.0
    base = (*self.signals, out)  # 6 floats: signal[0-4] + output
    if not extended:
        return base
    gene_flags = tuple(1.0 if i in active_gene_ids else 0.0 for i in range(8))
    return base + gene_flags + (motif_hash,)  # 15 floats
```

**Vector layout (base, 6 floats):**
- [0] signal[0] — x input / computation
- [1] signal[1] — y input / computation
- [2] signal[2] — computation
- [3] signal[3] — computation
- [4] signal[4] — developmental stage
- [5] output — primary result (0.0 if not yet produced)

**Extended layout (15 floats):**
- [0-5] base vector
- [6-13] gene activation flags (1.0 if gene N fired this episode, else 0.0)
- [14] motif hash — `(hash(tuple(m.origin_lineage for m in genome.local_motifs)) % 1000) / 1000.0`

Fixed length, fixed order, no parsing. Directly consumable by neural networks as a 6- or 15-dimensional observation vector. Deterministic — same cell state always produces same vector.

Active gene IDs and motif hash are passed in by the caller (chemistry VM or CellReader) rather than stored on CellState — keeps CellState minimal.

### Section 2 — Inter-Cell Reading

Two new codon ops added to `codons.py` (3-4 synonyms each):

| Op-name | Meaning |
|---|---|
| `SENSE_PEER_0` | Load peer 0's base vector into own signals[0-4] |
| `SENSE_PEER_1` | Load peer 1's base vector into own signals[0-4] |

In `cooperative_chemistry.py`, after each cell runs its chemistry step, store its `to_vector()` in `ChemistryContext` as `peer_vectors: dict[int, tuple[float,...]]` keyed by cell index.

When `SENSE_PEER_N` fires as a rule:
- If peer N vector available in context: copy vector[0-4] into cell.signals[0-4]
- If not available (solo context or peer index out of range): no-op

This enables true gene-level reading: cell B expresses a gene containing `SENSE_PEER_0` → loads cell A's signal state into its own signals → its next PROMOTER/GATE conditions gate on what it read from cell A.

**Backward compatible:** `SENSE_PEER_N` is a no-op when no peer vector exists. Solo cells, existing experiments unaffected. `peer_vectors` field added to `ChemistryContext` as `dict[int, tuple[float,...]]` with default empty dict.

### Section 3 — External Agent Interface

New file `reading.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from cell import CellState


@dataclass(frozen=True, slots=True)
class CellReader:
    def observe(self, cell: CellState) -> tuple[float, ...]:
        return cell.to_vector()

    def observe_extended(
        self,
        cell: CellState,
        *,
        active_gene_ids: frozenset[int] = frozenset(),
        motif_hash: float = 0.0,
    ) -> tuple[float, ...]:
        return cell.to_vector(extended=True, active_gene_ids=active_gene_ids, motif_hash=motif_hash)

    def observe_population(self, cells: Sequence[CellState]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.observe(cell) for cell in cells)

    def observe_colony(self, cells: Sequence[CellState]) -> tuple[float, ...]:
        vecs = self.observe_population(cells)
        if not vecs:
            return (0.0,) * 6
        return tuple(sum(v[i] for v in vecs) / len(vecs) for i in range(6))
```

Four reading modes:
- **`observe(cell)`** — 6-float snapshot of one cell
- **`observe_extended(cell, ...)`** — 15-float snapshot with gene activation and motif identity
- **`observe_population(cells)`** — tuple of N vectors, one per cell
- **`observe_colony(cells)`** — single 6-float mean vector across population (fixed size regardless of population count)

The colony mean is the most useful for external agents: encodes collective population state in a fixed-size vector. An RL agent observing a colony reads one 6-float vector regardless of how many cells exist.

### Section 4 — Testing Strategy

**Level 1 — `to_vector()` base (unit)**
- CellState with signals [1.0, 2.0, 0.5, 0.0, 0.3] and output 4.5 → vector = (1.0, 2.0, 0.5, 0.0, 0.3, 4.5)
- Cell with output=None → vector[5] = 0.0
- Base vector length always 6; extended always 15
- Same cell called twice returns identical vector (deterministic)

**Level 2 — `to_vector(extended=True)` (unit)**
- active_gene_ids={2, 5} → vector[8] = 1.0 (gene 2), vector[11] = 1.0 (gene 5), others 0.0
- motif_hash in range [0.0, 1.0)
- Extended length = 15

**Level 3 — Inter-cell reading (unit)**
- SENSE_PEER_0 with peer 0 vector available → cell signals[0-4] updated to peer's signal values
- SENSE_PEER_0 with no peer 0 (solo context) → no-op, signals unchanged
- Two cells in cooperative context: cell B with SENSE_PEER_0 reads cell A's signals correctly after cell A's step

**Level 4 — CellReader (unit)**
- `observe(cell)` returns 6-float tuple
- `observe_population([cell_a, cell_b])` returns 2-tuple of 6-float vectors
- `observe_colony([cell_a, cell_b])` returns mean vector — verified by hand calculation
- `observe_extended` returns 15-float tuple

**Level 5 — Regression**
- Experiment 01 output unchanged — `to_vector()` additive, no existing paths modified
- Experiment 13 (cooperative chemistry) output unchanged — SENSE_PEER_N only fires if codon present in genome

## Files Changed

| File | Change |
|---|---|
| `cell.py` | Add `to_vector(extended, active_gene_ids, motif_hash)` method |
| `codons.py` | Add `SENSE_PEER_0`, `SENSE_PEER_1` with 3-4 synonyms each |
| `chemistry.py` | Add `peer_vectors: dict[int, tuple[float,...]]` to ChemistryContext; implement SENSE_PEER_N rule handling |
| `cooperative_chemistry.py` | Store each cell's `to_vector()` in `context.peer_vectors` after each cell step |
| `reading.py` | New file: CellReader with observe/observe_extended/observe_population/observe_colony |
| `tests/test_reading_layer.py` | New file: Level 1-5 tests |

## Out of Scope

- Writing layer (AI agent writing into a genome — future)
- Vector normalisation beyond [0,1] clamping (signals already in this range by convention)
- More than 2 peer slots (SENSE_PEER_0/1 sufficient for initial cooperative pairs)
- Streaming observation (observe over time, not just snapshot)
- Vector compression or encoding schemes

## Success Criteria

1. All existing tests pass unchanged
2. `cell.to_vector()` returns deterministic 6-float tuple from any CellState
3. `SENSE_PEER_0` in a cooperative context correctly loads peer's signals into reading cell
4. `CellReader.observe_colony()` returns fixed-size 6-float mean vector regardless of population size
5. Experiment 01 and experiment 13 output identical to pre-change baseline
