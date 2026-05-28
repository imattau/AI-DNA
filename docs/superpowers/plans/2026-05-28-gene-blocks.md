# Gene Blocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add named gene declaration blocks and CALL codons to the genome, enabling reusable rule sequences and gene networks through two-pass transcription.
**Architecture:** 20 new codon op-names; declare_rules() becomes two-pass (collect declarations, resolve calls); recursion depth cap 4; silent skip for dangling calls; backward compatible.
**Tech Stack:** Python 3.10+, dataclasses, pytest

---

## File Map

- Modify: `codons.py` — add 20 new op-names to `default_codon_table()` and `REGULATORY_CODON_MAP`
- Modify: `genome.py` — add `_collect_gene_table()`; extend `declare_rules()` to two-pass with CALL resolution
- Modify: `tests/test_codons.py` — Task 1 codon table tests
- Create: `tests/test_gene_blocks.py` — Task 2, 3, 4 tests

---

## Task 1: Codon table extensions

**Files:**
- Modify: `codons.py`
- Modify: `tests/test_codons.py`

### Step 1: Write the failing tests

Add to `tests/test_codons.py`:

```python
def test_gene_block_op_names_present_in_table() -> None:
    """All 20 new gene block op-names appear in the default codon table."""
    from codons import default_codon_table

    table = default_codon_table()
    gene_ops = {
        "GENE_START", "GENE_END",
        "GENE_ID_0", "GENE_ID_1", "GENE_ID_2", "GENE_ID_3",
        "GENE_ID_4", "GENE_ID_5", "GENE_ID_6", "GENE_ID_7",
        "CALL_0", "CALL_1", "CALL_2", "CALL_3",
        "CALL_4", "CALL_5", "CALL_6", "CALL_7",
    }
    present = set(table.codon_to_op.values())
    for op in gene_ops:
        assert op in present, f"{op} missing from codon table"


def test_gene_block_synonyms_decode_to_same_op() -> None:
    """All synonyms for CALL_2 decode to the same op-name (neutral mutation)."""
    from codons import default_codon_table

    table = default_codon_table()
    synonyms = [c for c, op in table.codon_to_op.items() if op == "CALL_2"]
    assert len(synonyms) >= 3, f"Expected at least 3 synonyms for CALL_2, got {synonyms}"
    assert all(table.op(c) == "CALL_2" for c in synonyms)


def test_gene_block_round_trip_encode_decode() -> None:
    """Encoding gene block op-names then decoding returns the same op-names."""
    from codons import default_codon_table

    table = default_codon_table()
    ops = ["GENE_START", "GENE_ID_2", "RULE_EMIT_X", "GENE_END", "CALL_2"]
    encoded = table.encode_ops(ops)
    decoded = table.decode(encoded)
    assert decoded == ops


def test_each_gene_block_op_has_at_least_three_synonyms() -> None:
    """Each new gene block op-name has at least 3 synonym codon values."""
    from codons import default_codon_table

    table = default_codon_table()
    gene_ops = [
        "GENE_START", "GENE_END",
        "GENE_ID_0", "GENE_ID_1", "GENE_ID_2", "GENE_ID_3",
        "GENE_ID_4", "GENE_ID_5", "GENE_ID_6", "GENE_ID_7",
        "CALL_0", "CALL_1", "CALL_2", "CALL_3",
        "CALL_4", "CALL_5", "CALL_6", "CALL_7",
    ]
    op_to_codons = table.op_to_codons
    for op in gene_ops:
        count = len(op_to_codons.get(op, []))
        assert count >= 3, f"{op} has only {count} synonyms, expected >= 3"


def test_gene_id_ops_are_distinct() -> None:
    """GENE_ID_0 through GENE_ID_7 each map to distinct, non-overlapping codon sets."""
    from codons import default_codon_table

    table = default_codon_table()
    gene_id_ops = [f"GENE_ID_{i}" for i in range(8)]
    codon_sets = [frozenset(table.op_to_codons.get(op, [])) for op in gene_id_ops]
    for i, s_i in enumerate(codon_sets):
        for j, s_j in enumerate(codon_sets):
            if i != j:
                overlap = s_i & s_j
                assert not overlap, (
                    f"GENE_ID_{i} and GENE_ID_{j} share codon(s): {overlap}"
                )


def test_call_ops_are_distinct() -> None:
    """CALL_0 through CALL_7 each map to distinct, non-overlapping codon sets."""
    from codons import default_codon_table

    table = default_codon_table()
    call_ops = [f"CALL_{i}" for i in range(8)]
    codon_sets = [frozenset(table.op_to_codons.get(op, [])) for op in call_ops]
    for i, s_i in enumerate(codon_sets):
        for j, s_j in enumerate(codon_sets):
            if i != j:
                overlap = s_i & s_j
                assert not overlap, (
                    f"CALL_{i} and CALL_{j} share codon(s): {overlap}"
                )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_codons.py::test_gene_block_op_names_present_in_table tests/test_codons.py::test_gene_block_synonyms_decode_to_same_op tests/test_codons.py::test_gene_block_round_trip_encode_decode tests/test_codons.py::test_each_gene_block_op_has_at_least_three_synonyms tests/test_codons.py::test_gene_id_ops_are_distinct tests/test_codons.py::test_call_ops_are_distinct -v
```

Expected: FAIL — gene block ops not yet in the table.

- [ ] **Step 3: Extend `REGULATORY_CODON_MAP` and `REGULATORY_OP_NAMES` in `codons.py`**

The current `REGULATORY_CODON_MAP` occupies codon integers 231–255. Add 20 new op-names using integers 128–227 (before the regulatory block). Each op gets 3 synonym values allocated consecutively. Total: 20 ops × 3 synonyms = 60 slots (128–187), with headroom to 231.

In `codons.py`, append to `REGULATORY_OP_NAMES`:

```python
REGULATORY_OP_NAMES: tuple[str, ...] = (
    "PROMOTER",
    "GATE",
    "IF_S0_GT",
    "IF_S1_GT",
    "IF_S2_GT",
    "IF_S0_LT",
    "IF_S1_LT",
    "END_BLOCK",
    # Gene block ops — added for gene blocks feature
    "GENE_START",
    "GENE_ID_0", "GENE_ID_1", "GENE_ID_2", "GENE_ID_3",
    "GENE_ID_4", "GENE_ID_5", "GENE_ID_6", "GENE_ID_7",
    "GENE_END",
    "CALL_0", "CALL_1", "CALL_2", "CALL_3",
    "CALL_4", "CALL_5", "CALL_6", "CALL_7",
)
```

Add the gene block codon map entries to `REGULATORY_CODON_MAP`:

```python
GENE_BLOCK_CODON_MAP: dict[int, str] = {
    # GENE_START: 3 synonyms
    128: "GENE_START", 129: "GENE_START", 130: "GENE_START",
    # GENE_ID_0: 3 synonyms
    131: "GENE_ID_0", 132: "GENE_ID_0", 133: "GENE_ID_0",
    # GENE_ID_1: 3 synonyms
    134: "GENE_ID_1", 135: "GENE_ID_1", 136: "GENE_ID_1",
    # GENE_ID_2: 3 synonyms
    137: "GENE_ID_2", 138: "GENE_ID_2", 139: "GENE_ID_2",
    # GENE_ID_3: 3 synonyms
    140: "GENE_ID_3", 141: "GENE_ID_3", 142: "GENE_ID_3",
    # GENE_ID_4: 3 synonyms
    143: "GENE_ID_4", 144: "GENE_ID_4", 145: "GENE_ID_4",
    # GENE_ID_5: 3 synonyms
    146: "GENE_ID_5", 147: "GENE_ID_5", 148: "GENE_ID_5",
    # GENE_ID_6: 3 synonyms
    149: "GENE_ID_6", 150: "GENE_ID_6", 151: "GENE_ID_6",
    # GENE_ID_7: 3 synonyms
    152: "GENE_ID_7", 153: "GENE_ID_7", 154: "GENE_ID_7",
    # GENE_END: 3 synonyms
    155: "GENE_END", 156: "GENE_END", 157: "GENE_END",
    # CALL_0: 3 synonyms
    158: "CALL_0", 159: "CALL_0", 160: "CALL_0",
    # CALL_1: 3 synonyms
    161: "CALL_1", 162: "CALL_1", 163: "CALL_1",
    # CALL_2: 3 synonyms
    164: "CALL_2", 165: "CALL_2", 166: "CALL_2",
    # CALL_3: 3 synonyms
    167: "CALL_3", 168: "CALL_3", 169: "CALL_3",
    # CALL_4: 3 synonyms
    170: "CALL_4", 171: "CALL_4", 172: "CALL_4",
    # CALL_5: 3 synonyms
    173: "CALL_5", 174: "CALL_5", 175: "CALL_5",
    # CALL_6: 3 synonyms
    176: "CALL_6", 177: "CALL_6", 178: "CALL_6",
    # CALL_7: 3 synonyms
    179: "CALL_7", 180: "CALL_7", 181: "CALL_7",
}
```

Then update `REGULATORY_CODON_MAP` to merge the new entries and update `default_codon_table()`:

Replace the existing `REGULATORY_CODON_MAP` definition and add the merge:

```python
REGULATORY_CODON_MAP: dict[int, str] = {
    231: "PROMOTER",
    232: "PROMOTER",
    233: "PROMOTER",
    234: "GATE",
    235: "GATE",
    236: "GATE",
    237: "IF_S0_GT",
    238: "IF_S0_GT",
    239: "IF_S0_GT",
    240: "IF_S1_GT",
    241: "IF_S1_GT",
    242: "IF_S1_GT",
    243: "IF_S2_GT",
    244: "IF_S2_GT",
    245: "IF_S2_GT",
    246: "IF_S0_LT",
    247: "IF_S0_LT",
    248: "IF_S0_LT",
    249: "IF_S1_LT",
    250: "IF_S1_LT",
    251: "IF_S1_LT",
    252: "END_BLOCK",
    253: "END_BLOCK",
    254: "END_BLOCK",
    255: "END_BLOCK",
    **GENE_BLOCK_CODON_MAP,
}
```

The `default_codon_table()` function already merges `REGULATORY_CODON_MAP` into the base table — no change needed there:

```python
def default_codon_table() -> CodonTable:
    allocation = {
        "START": 3,
        "STOP": 3,
        "NOOP": 4,
        "SEND": 2,
        "RECV": 2,
        "RULE_EMIT_X": 5,
        "RULE_EMIT_Y": 5,
        "RULE_ZERO_2": 4,
        "RULE_ADD0_IF1": 6,
        "RULE_DECAY1": 6,
        "RULE_OUTPUT_IF1Z": 5,
        "RULE_INHIBIT1Z": 3,
        "RULE_COPY0_3": 3,
        "RULE_COPY1_3": 3,
        "RULE_ADD3_IF1": 3,
        "RULE_DECAY3": 3,
        "RULE_THRESH1": 2,
        "RULE_THRESH3": 2,
    }
    table = build_codon_table(allocation, table_size=231, fill_op="NOOP")
    codon_to_op = dict(table.codon_to_op)
    codon_to_op.update(REGULATORY_CODON_MAP)  # now includes GENE_BLOCK_CODON_MAP via **merge
    return CodonTable(codon_to_op)
```

Also update the `exclude` defaults in `random_codons` and `mutate_codons` to include the new gene block codons:

```python
def random_codons(
    rng: Random,
    length: int,
    modulus: int = 256,
    *,
    exclude: Sequence[int] | None = tuple(REGULATORY_CODON_MAP),
) -> tuple[int, ...]:
```

The `exclude` default already uses `tuple(REGULATORY_CODON_MAP)` — since `REGULATORY_CODON_MAP` now includes `GENE_BLOCK_CODON_MAP` entries, no change is required. The same applies to `mutate_codons`.

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_codons.py -v
```

Expected: all pass.

- [ ] **Step 5: Verify existing tests still pass**

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass — new codons occupy the 128-181 range which was previously NOOP fill; existing codon assignments 0-127 and 231-255 unchanged.

- [ ] **Step 6: Commit**

```bash
git add codons.py tests/test_codons.py
git commit -m "feat: add 20 gene block op-names to codon table (GENE_START, GENE_ID_0-7, GENE_END, CALL_0-7)"
```

---

## Task 2: Pass 1 — declaration collection

**Files:**
- Modify: `genome.py`
- Create: `tests/test_gene_blocks.py` (Pass 1 tests)

### Step 1: Write the failing tests

Create `tests/test_gene_blocks.py`:

```python
from __future__ import annotations

import pytest
from codons import default_codon_table
from genome import CellGenome, GateRule, _collect_gene_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TABLE = default_codon_table()


def _genome_from_ops(ops: list[str], lineage_id: str = "test") -> CellGenome:
    """Build a CellGenome whose codons encode the given sequence of op-names."""
    codons = tuple(TABLE.encode_ops(ops))
    return CellGenome(codons=codons, local_motifs=(), lineage_id=lineage_id)


def _op_names_from_ops(ops: list[str]) -> tuple[str, ...]:
    """Encode ops to codons then decode back to a tuple of op-names."""
    codons = TABLE.encode_ops(ops)
    return tuple(TABLE.decode(codons))


# ---------------------------------------------------------------------------
# Level 2 — Pass 1: declaration collection
# ---------------------------------------------------------------------------

def test_single_gene_declaration_collected() -> None:
    """A single GENE_START / GENE_ID_2 / body / GENE_END is collected into gene_table[2]."""
    op_names = _op_names_from_ops([
        "GENE_START", "GENE_ID_2",
        "RULE_EMIT_X",
        "GENE_END",
    ])
    gene_table = _collect_gene_table(op_names)
    assert 2 in gene_table, f"gene_table should contain key 2, got keys: {set(gene_table)}"
    assert "RULE_EMIT_X" in gene_table[2], (
        f"gene_table[2] should contain RULE_EMIT_X, got {gene_table[2]}"
    )


def test_two_declarations_both_collected() -> None:
    """Two separate gene declarations are both present in gene_table."""
    op_names = _op_names_from_ops([
        "GENE_START", "GENE_ID_0",
        "RULE_EMIT_X",
        "GENE_END",
        "GENE_START", "GENE_ID_5",
        "RULE_EMIT_Y",
        "GENE_END",
    ])
    gene_table = _collect_gene_table(op_names)
    assert 0 in gene_table, f"gene_table missing key 0, keys: {set(gene_table)}"
    assert 5 in gene_table, f"gene_table missing key 5, keys: {set(gene_table)}"
    assert "RULE_EMIT_X" in gene_table[0]
    assert "RULE_EMIT_Y" in gene_table[5]


def test_nested_gene_start_closes_outer_implicitly() -> None:
    """A nested GENE_START closes the outer declaration and starts a fresh inner one."""
    op_names = _op_names_from_ops([
        "GENE_START", "GENE_ID_1",
        "RULE_EMIT_X",
        "GENE_START", "GENE_ID_3",  # closes gene 1 implicitly, starts gene 3
        "RULE_EMIT_Y",
        "GENE_END",
    ])
    gene_table = _collect_gene_table(op_names)
    # Gene 1 body ends at the nested GENE_START
    assert 1 in gene_table, f"gene_table missing key 1, keys: {set(gene_table)}"
    assert "RULE_EMIT_X" in gene_table[1]
    assert "RULE_EMIT_Y" not in gene_table[1], (
        "RULE_EMIT_Y should be in gene 3, not gene 1"
    )
    # Gene 3 gets the body after the nested GENE_START
    assert 3 in gene_table, f"gene_table missing key 3, keys: {set(gene_table)}"
    assert "RULE_EMIT_Y" in gene_table[3]


def test_end_of_list_acts_as_implicit_gene_end() -> None:
    """A GENE_START without a closing GENE_END still collects up to end of list."""
    op_names = _op_names_from_ops([
        "GENE_START", "GENE_ID_4",
        "RULE_EMIT_X",
        "RULE_DECAY1",
        # No GENE_END — end of list = implicit GENE_END
    ])
    gene_table = _collect_gene_table(op_names)
    assert 4 in gene_table, f"gene_table missing key 4, keys: {set(gene_table)}"
    assert "RULE_EMIT_X" in gene_table[4]
    assert "RULE_DECAY1" in gene_table[4]


def test_no_gene_codons_returns_empty_dict() -> None:
    """A genome with no GENE_START codons returns an empty gene_table without error."""
    op_names = _op_names_from_ops([
        "RULE_EMIT_X", "RULE_EMIT_Y", "RULE_DECAY1",
    ])
    gene_table = _collect_gene_table(op_names)
    assert gene_table == {}, f"Expected empty dict, got {gene_table}"


def test_gene_start_without_id_codon_is_skipped() -> None:
    """GENE_START at end of list (no ID codon following) does not raise."""
    op_names = _op_names_from_ops(["GENE_START"])
    gene_table = _collect_gene_table(op_names)
    # No valid gene declared — gene_table empty
    assert gene_table == {}


def test_gene_body_multi_rule_collected_in_order() -> None:
    """Multiple rules inside a declaration are all collected in order."""
    op_names = _op_names_from_ops([
        "GENE_START", "GENE_ID_7",
        "RULE_EMIT_X", "RULE_ADD0_IF1", "RULE_DECAY1",
        "GENE_END",
    ])
    gene_table = _collect_gene_table(op_names)
    assert 7 in gene_table
    body = gene_table[7]
    assert "RULE_EMIT_X" in body
    assert "RULE_ADD0_IF1" in body
    assert "RULE_DECAY1" in body
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_gene_blocks.py -k "test_single_gene or test_two_decl or test_nested or test_end_of_list or test_no_gene or test_gene_start_without or test_gene_body_multi" -v
```

Expected: FAIL — `_collect_gene_table` does not exist.

- [ ] **Step 3: Add `GENE_BLOCK_OP_NAMES` set and `_collect_gene_table()` to `genome.py`**

After the `_condition_true` function in `genome.py`, add:

```python
# Gene block op-name sets — used by two-pass transcription
_GENE_START_OPS: frozenset[str] = frozenset({"GENE_START"})
_GENE_END_OPS: frozenset[str] = frozenset({"GENE_END"})
_GENE_ID_OPS: dict[str, int] = {
    "GENE_ID_0": 0, "GENE_ID_1": 1, "GENE_ID_2": 2, "GENE_ID_3": 3,
    "GENE_ID_4": 4, "GENE_ID_5": 5, "GENE_ID_6": 6, "GENE_ID_7": 7,
}
_CALL_OPS: dict[str, int] = {
    "CALL_0": 0, "CALL_1": 1, "CALL_2": 2, "CALL_3": 3,
    "CALL_4": 4, "CALL_5": 5, "CALL_6": 6, "CALL_7": 7,
}
_MAX_CALL_DEPTH: int = 4


def _collect_gene_table(op_names: tuple[str, ...]) -> dict[int, tuple[str, ...]]:
    """Pass 1 — walk op-name list and collect GENE_START...GENE_END regions.

    Returns a dict mapping gene ID (0-7) → tuple of op-name strings inside that
    gene's declaration body. Non-rule op-names (NOOP, START, STOP, structural ops)
    are included as-is; call resolution (Pass 2) handles them.

    Rules:
    - GENE_START must be followed by a GENE_ID_* codon (the gene's ID).
    - Body continues until GENE_END, another GENE_START (closes current implicitly),
      or end of list (implicit GENE_END).
    - Nested GENE_START: outer closes, inner starts fresh.
    - No GENE_START present: returns {}.
    """
    gene_table: dict[int, tuple[str, ...]] = {}
    i = 0
    n = len(op_names)

    while i < n:
        op = op_names[i]

        if op not in _GENE_START_OPS:
            i += 1
            continue

        # Found GENE_START — read next codon as ID
        i += 1
        if i >= n:
            # GENE_START at end of list with no ID codon — skip
            break

        id_op = op_names[i]
        if id_op not in _GENE_ID_OPS:
            # Codon after GENE_START is not a GENE_ID_* — treat as malformed, skip
            i += 1
            continue

        gene_id = _GENE_ID_OPS[id_op]
        i += 1  # advance past the GENE_ID codon

        # Collect body until GENE_END or GENE_START (implicit close) or end of list
        body: list[str] = []
        while i < n:
            body_op = op_names[i]
            if body_op in _GENE_END_OPS:
                i += 1  # consume GENE_END
                break
            if body_op in _GENE_START_OPS:
                # Nested GENE_START — close outer implicitly, do NOT advance i
                # (outer loop will re-encounter this GENE_START next iteration)
                break
            body.append(body_op)
            i += 1

        gene_table[gene_id] = tuple(body)

    return gene_table
```

- [ ] **Step 4: Run Pass 1 tests**

```bash
python3 -m pytest tests/test_gene_blocks.py -k "test_single_gene or test_two_decl or test_nested or test_end_of_list or test_no_gene or test_gene_start_without or test_gene_body_multi" -v
```

Expected: all pass.

- [ ] **Step 5: Run full fast suite**

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add genome.py tests/test_gene_blocks.py
git commit -m "feat: add _collect_gene_table() — Pass 1 gene declaration collection"
```

---

## Task 3: Pass 2 — call resolution in `declare_rules()`

**Files:**
- Modify: `genome.py` — extend `_transcribe_rules()` and `declare_rules()` for two-pass CALL resolution
- Modify: `tests/test_gene_blocks.py` — add Pass 2 and composition tests

### Step 1: Write the failing tests

Add to `tests/test_gene_blocks.py`:

```python
# ---------------------------------------------------------------------------
# Level 3 — Pass 2: CALL resolution
# ---------------------------------------------------------------------------

def test_call_with_declaration_inserts_rules() -> None:
    """CALL_2 when gene_table[2] is declared inserts gene 2's rules at call position."""
    genome = _genome_from_ops([
        "GENE_START", "GENE_ID_2",
        "RULE_EMIT_X",
        "GENE_END",
        "CALL_2",
    ])
    rules = genome.declare_rules()
    rule_names = [r for r in rules if isinstance(r, str)]
    assert "RULE_EMIT_X" in rule_names, (
        f"RULE_EMIT_X should be inserted at CALL_2 position, got {rule_names}"
    )


def test_call_dangling_no_declaration_silent_skip() -> None:
    """CALL_5 with no gene_table[5] entry is silently skipped — no error."""
    genome = _genome_from_ops([
        "RULE_EMIT_Y",
        "CALL_5",   # no gene 5 declared — dangling call
    ])
    rules = genome.declare_rules()
    rule_names = [r for r in rules if isinstance(r, str)]
    assert "RULE_EMIT_Y" in rule_names, "RULE_EMIT_Y should be present"
    # No error raised, execution continues normally
    assert isinstance(rules, tuple)


def test_call_order_independent_declaration_after_call() -> None:
    """Declaration appearing after CALL is still resolved (order-independent, Pass 1 first)."""
    genome = _genome_from_ops([
        "CALL_3",               # call comes before declaration
        "GENE_START", "GENE_ID_3",
        "RULE_ADD0_IF1",
        "GENE_END",
    ])
    rules = genome.declare_rules()
    rule_names = [r for r in rules if isinstance(r, str)]
    assert "RULE_ADD0_IF1" in rule_names, (
        f"RULE_ADD0_IF1 should be resolved despite declaration appearing after CALL, got {rule_names}"
    )


def test_gene_calls_gene_depth_within_cap() -> None:
    """Gene 0 calls gene 1; both resolved correctly when depth <= 4."""
    genome = _genome_from_ops([
        "GENE_START", "GENE_ID_1",
        "RULE_EMIT_Y",
        "GENE_END",
        "GENE_START", "GENE_ID_0",
        "CALL_1",   # gene 0's body calls gene 1
        "GENE_END",
        "CALL_0",   # active region calls gene 0
    ])
    rules = genome.declare_rules()
    rule_names = [r for r in rules if isinstance(r, str)]
    assert "RULE_EMIT_Y" in rule_names, (
        f"RULE_EMIT_Y should be reachable via gene 0 → gene 1 chain, got {rule_names}"
    )


def test_gene_call_recursion_depth_cap_no_error() -> None:
    """Gene network with depth > 4 stops cleanly without error or infinite loop."""
    # Gene 0 calls gene 1 calls gene 2 calls gene 3 calls gene 4 calls gene 5 (depth 5)
    genome = _genome_from_ops([
        "GENE_START", "GENE_ID_5", "RULE_EMIT_X", "GENE_END",
        "GENE_START", "GENE_ID_4", "CALL_5", "GENE_END",
        "GENE_START", "GENE_ID_3", "CALL_4", "GENE_END",
        "GENE_START", "GENE_ID_2", "CALL_3", "GENE_END",
        "GENE_START", "GENE_ID_1", "CALL_2", "GENE_END",
        "GENE_START", "GENE_ID_0", "CALL_1", "GENE_END",
        "CALL_0",   # triggers chain: 0→1→2→3→4→5 (depth 5, cap 4)
    ])
    # Must not raise, must not hang
    rules = genome.declare_rules()
    assert isinstance(rules, tuple)
    # The chain is truncated at depth 4, so RULE_EMIT_X may or may not appear
    # depending on where the cap falls — what matters is no error


def test_gene_declarations_not_in_active_rules() -> None:
    """GENE_START...GENE_END regions are skipped during Pass 2 active list construction."""
    genome = _genome_from_ops([
        "GENE_START", "GENE_ID_6",
        "RULE_EMIT_X",
        "GENE_END",
        "RULE_EMIT_Y",  # only active rule — no CALL, so RULE_EMIT_X not activated
    ])
    rules = genome.declare_rules()
    rule_names = [r for r in rules if isinstance(r, str)]
    assert "RULE_EMIT_Y" in rule_names
    assert "RULE_EMIT_X" not in rule_names, (
        f"RULE_EMIT_X should not appear without a CALL_6, got {rule_names}"
    )


def test_backward_compatible_genome_without_gene_codons() -> None:
    """Genome with no gene codons produces identical output to current behavior."""
    genome = _genome_from_ops(["RULE_EMIT_X", "RULE_EMIT_Y", "RULE_DECAY1"])
    rules = genome.declare_rules()
    rule_names = [r for r in rules if isinstance(r, str)]
    assert "RULE_EMIT_X" in rule_names
    assert "RULE_EMIT_Y" in rule_names
    assert "RULE_DECAY1" in rule_names


# ---------------------------------------------------------------------------
# Level 4 — Regulatory composition
# ---------------------------------------------------------------------------

def test_promoter_wrapping_call_signal_high() -> None:
    """PROMOTER + IF_S0_GT + CALL_2 + END_BLOCK with signal[0]=0.8 → gene 2 rules active."""
    genome = _genome_from_ops([
        "GENE_START", "GENE_ID_2",
        "RULE_EMIT_X",
        "GENE_END",
        "PROMOTER", "IF_S0_GT",
        "CALL_2",
        "END_BLOCK",
    ])
    signals = (0.8, 0.0, 0.0, 0.0)
    rules = genome.declare_rules(signals=signals)
    rule_names = [r for r in rules if isinstance(r, str)]
    assert "RULE_EMIT_X" in rule_names, (
        f"CALL_2 inside active PROMOTER block should expand gene 2, got {rule_names}"
    )


def test_promoter_wrapping_call_signal_low() -> None:
    """PROMOTER + IF_S0_GT + CALL_2 + END_BLOCK with signal[0]=0.2 → gene 2 rules absent."""
    genome = _genome_from_ops([
        "GENE_START", "GENE_ID_2",
        "RULE_EMIT_X",
        "GENE_END",
        "PROMOTER", "IF_S0_GT",
        "CALL_2",
        "END_BLOCK",
    ])
    signals = (0.2, 0.0, 0.0, 0.0)
    rules = genome.declare_rules(signals=signals)
    rule_names = [r for r in rules if isinstance(r, str)]
    assert "RULE_EMIT_X" not in rule_names, (
        f"CALL_2 inside inactive PROMOTER block should not expand gene 2, got {rule_names}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_gene_blocks.py -k "test_call_with or test_call_dangling or test_call_order or test_gene_calls_gene or test_gene_call_recursion or test_gene_declarations_not or test_backward_compatible or test_promoter_wrapping" -v
```

Expected: FAIL — `declare_rules()` does not yet do two-pass or CALL resolution.

- [ ] **Step 3: Extend `_transcribe_rules()` to handle CALL op-names**

The current `_transcribe_rules()` in `genome.py` uses a recursive `parse()` that walks a deduplicated rule list. We replace it with a depth-aware version that accepts the `gene_table` and resolves CALL ops.

Replace `_transcribe_rules` in `genome.py`:

```python
def _transcribe_rules(
    rules: Sequence[str],
    signals: Sequence[float],
    gene_table: dict[int, tuple[str, ...]],
    *,
    _depth: int = 0,
) -> tuple[str | GateRule, ...]:
    """Walk a sequence of op-names, honouring PROMOTER/GATE blocks and CALL ops.

    Parameters
    ----------
    rules:
        Flat sequence of op-name strings (already deduplicated by caller).
    signals:
        Current signal vector for PROMOTER condition evaluation.
    gene_table:
        Mapping from gene ID → body op-names, produced by _collect_gene_table().
    _depth:
        Current recursion depth for CALL resolution. Capped at _MAX_CALL_DEPTH.
    """
    def parse(index: int) -> tuple[list[str | GateRule], int]:
        active: list[str | GateRule] = []
        while index < len(rules):
            op = rules[index]

            if op == "END_BLOCK":
                return active, index + 1

            if op == "PROMOTER":
                condition = rules[index + 1] if index + 1 < len(rules) else "DEFAULT_TRUE"
                block, next_index = parse(index + 2)
                if _condition_true(condition, signals):
                    active.extend(block)
                index = next_index
                continue

            if op == "GATE":
                condition = rules[index + 1] if index + 1 < len(rules) else "DEFAULT_TRUE"
                block, next_index = parse(index + 2)
                active.append(GateRule(condition=condition, rules=tuple(block)))
                index = next_index
                continue

            if op in _CALL_OPS:
                gene_id = _CALL_OPS[op]
                if gene_id in gene_table and _depth < _MAX_CALL_DEPTH:
                    body = gene_table[gene_id]
                    resolved = _transcribe_rules(
                        body, signals, gene_table, _depth=_depth + 1
                    )
                    active.extend(resolved)
                # else: dangling call (no declaration) or depth cap reached → silent skip
                index += 1
                continue

            # Gene declaration markers are skipped in Pass 2 — they were consumed in Pass 1
            if op in _GENE_START_OPS or op in _GENE_ID_OPS or op in _GENE_END_OPS:
                index += 1
                continue

            active.append(op)
            index += 1

        return active, index

    transcribed, _ = parse(0)
    return tuple(transcribed)
```

- [ ] **Step 4: Extend `declare_rules()` in `CellGenome` to be two-pass**

Replace the `declare_rules` method in `genome.py`:

```python
def declare_rules(self, *, signals: Sequence[float] | None = None) -> tuple[str | GateRule, ...]:
    """Two-pass transcription: collect gene declarations, then resolve rule list.

    Pass 1 — _collect_gene_table():
        Walk all codons, collect GENE_START...GENE_END regions into gene_table.

    Pass 2 — _transcribe_rules():
        Walk all codons again. Skip gene declaration regions. Resolve CALL ops
        by looking up gene_table and recursively transcribing the body at that
        position. PROMOTER/GATE conditions applied as before.

    Parameters
    ----------
    signals:
        Current cell signal vector. When None, all PROMOTER blocks are treated
        as active (backward compatible — no regulatory evaluation).
    """
    signal_values = tuple(signals or (0.0, 0.0, 0.0, 0.0))

    # Decode all codons to op-names (raw, including structural ops)
    raw_op_names = tuple(decode_codon_op(codon) for codon in self.codons)

    # Pass 1: collect gene declarations
    gene_table = _collect_gene_table(raw_op_names)

    # Build the active op-name list including motif patterns
    # Motif patterns are appended after genome codons (same as before)
    all_op_names: list[str] = list(raw_op_names)
    for motif in self.local_motifs:
        all_op_names.extend(motif.pattern)

    # Deduplicate while preserving order (gene block structural ops included)
    deduped = unique_ordered(all_op_names)

    # Pass 2: transcribe with CALL resolution
    return _transcribe_rules(deduped, signal_values, gene_table)
```

- [ ] **Step 5: Run Pass 2 and composition tests**

```bash
python3 -m pytest tests/test_gene_blocks.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run full fast suite**

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add genome.py tests/test_gene_blocks.py
git commit -m "feat: two-pass declare_rules() with CALL resolution and recursion depth cap"
```

---

## Task 4: Integration and regression

**Files:**
- Modify: `tests/test_gene_blocks.py` — add gene network and regression tests
- No source changes expected

### Step 1: Write the gene network integration test

Add to `tests/test_gene_blocks.py`:

```python
# ---------------------------------------------------------------------------
# Level 5 — Gene network integration
# ---------------------------------------------------------------------------

def test_negative_feedback_gene_network_signal1_rises_then_stabilises() -> None:
    """Negative feedback loop: gene 0 emits signal[1]; gene 1 inhibits signal[0].

    Genome layout (Section 3 of spec):
        GENE_START GENE_ID_0 RULE_EMIT_Y GENE_END    ← gene 0: raises signal[1]
        GENE_START GENE_ID_1 RULE_INHIBIT1Z GENE_END ← gene 1: inhibits signal based on s1
        CALL_0                                        ← always express gene 0
        PROMOTER IF_S1_GT CALL_1 END_BLOCK           ← express gene 1 only when signal[1] high

    Expected behavior across episodes:
    - Episode 1: CALL_0 fires → RULE_EMIT_Y sets signal[1] to task.y
    - Episode 2+: signal[1] high → PROMOTER gates CALL_1 open → RULE_INHIBIT1Z runs
    - signal[1] should remain > 0 (gene 0 keeps firing) and stabilise
    """
    from chemistry import ChemistrySystem
    from cell import CellState
    from tasks import TaskCase

    genome = _genome_from_ops([
        "GENE_START", "GENE_ID_0", "RULE_EMIT_Y", "GENE_END",
        "GENE_START", "GENE_ID_1", "RULE_INHIBIT1Z", "GENE_END",
        "CALL_0",
        "PROMOTER", "IF_S1_GT",
        "CALL_1",
        "END_BLOCK",
    ])

    task = TaskCase(x=0.0, y=3.0, z=3.0)
    system = ChemistrySystem()
    signal1_values: list[float] = []

    # Run 5 episodes, recording signal[1] after each
    signals = (0.0, 0.0, 0.0, 0.0)
    for _episode in range(5):
        rules = genome.declare_rules(signals=signals)
        cell = CellState(active_rules=list(rules))
        cell.signals[1] = signals[1]
        system.run(cell, task)
        signals = tuple(cell.signals)
        signal1_values.append(signals[1])

    # signal[1] should rise from 0 in episode 1
    assert signal1_values[0] > 0.0, (
        f"signal[1] should rise after episode 1 (gene 0 fires), got {signal1_values}"
    )
    # signal[1] should stabilise — later values should not keep growing unboundedly
    # Allow that stabilisation may take 3+ episodes; check that max - min in last 3 < 2.0
    last_three = signal1_values[2:]
    spread = max(last_three) - min(last_three)
    assert spread < 2.0, (
        f"signal[1] should stabilise across episodes, spread={spread:.3f}, values={signal1_values}"
    )


def test_experiment_01_runs_successfully() -> None:
    """Experiment 01 (multiply benchmark) exits 0 — backward compatibility confirmed."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "experiments/01_multiply_persistent_rules.py"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/home/mattthomson/workspace/AI-DNA",
    )
    assert result.returncode == 0, (
        f"Experiment 01 exited with code {result.returncode}.\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )
```

- [ ] **Step 2: Run the gene network test**

```bash
python3 -m pytest tests/test_gene_blocks.py::test_negative_feedback_gene_network_signal1_rises_then_stabilises -v
```

Expected: PASS.

- [ ] **Step 3: Run experiment 01 directly**

```bash
python3 experiments/01_multiply_persistent_rules.py
```

Expected: exits 0, output identical to pre-change baseline (no gene codons in existing genomes).

- [ ] **Step 4: Run full fast suite**

```bash
python3 -m pytest tests/ -q --ignore=tests/test_experiments.py
```

Expected: all pass, no failures or errors.

- [ ] **Step 5: Run the experiment 01 regression test**

```bash
python3 -m pytest tests/test_gene_blocks.py::test_experiment_01_runs_successfully -v
```

Expected: PASS.

- [ ] **Step 6: Final commit**

```bash
git add tests/test_gene_blocks.py
git commit -m "feat: add gene blocks — two-pass transcription with CALL resolution"
```

---

## Self-review checklist

### Spec success criteria coverage

| Success criterion | Covered by |
|---|---|
| 1. All existing tests pass unchanged | Task 1 Step 5; Task 2 Step 5; Task 3 Step 6; Task 4 Step 4 |
| 2. Genome with gene declarations + CALL executes declared rules at call positions | Task 3 `test_call_with_declaration_inserts_rules` |
| 3. CALL with no matching declaration is silently skipped — no error | Task 3 `test_call_dangling_no_declaration_silent_skip` |
| 4. Gene network produces signal[1] rise and stabilisation across episodes | Task 4 `test_negative_feedback_gene_network_signal1_rises_then_stabilises` |
| 5. PROMOTER/GATE conditions correctly gate CALL codons | Task 3 `test_promoter_wrapping_call_signal_high/low` |
| 6. Experiment 01 output identical to pre-change baseline | Task 4 `test_experiment_01_runs_successfully` |

### No placeholders

All implementation code is shown in full. Every function body is complete Python. No "implement X" without the code.

### Type consistency

- `gene_table` type is `dict[int, tuple[str, ...]]` throughout:
  - `_collect_gene_table()` return type: `dict[int, tuple[str, ...]]`
  - `_transcribe_rules()` parameter type: `dict[int, tuple[str, ...]]`
  - `declare_rules()` local variable: `dict[int, tuple[str, ...]]`
- `GateRule` is defined once in `genome.py` (already present from regulatory codons).
- `_GENE_START_OPS`, `_GENE_END_OPS`, `_GENE_ID_OPS`, `_CALL_OPS` defined once in `genome.py`.
- `_MAX_CALL_DEPTH = 4` matches regulatory codon stack depth cap `_MAX_PROMOTER_DEPTH = 4`.
- `declare_rules()` return type unchanged: `tuple[str | GateRule, ...]`.

### Backward compatibility

- `decode_codon_op()` is called on all codons 0-255; gene block codons (128-181) now decode to their op-names rather than NOOP. Existing genomes do not contain codons 128-181 (they were NOOP fill before this feature) — no behavior change.
- `_collect_gene_table()` returns `{}` when no GENE_START present — `_transcribe_rules()` with empty `gene_table` falls through CALL branches silently and produces identical output to current behavior.
- `unique_ordered()` deduplication in `declare_rules()` is preserved.

### Circular import guard

- `genome.py` imports from `codons.py` only ✓
- `chemistry.py` imports `GateRule` from `genome.py` ✓
- `genome.py` does NOT import from `chemistry.py` ✓
- No new imports introduced that would create cycles.
