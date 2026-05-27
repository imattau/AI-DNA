from __future__ import annotations

from random import Random

from codons import build_codon_table, decode_rule_name, mutate_codons_neutral, mutate_codons_with_table, synonymous_codons


def test_synonymous_codons_share_rule_identity() -> None:
    codon = 7
    synonyms = synonymous_codons(codon)
    assert synonyms
    assert all(decode_rule_name(candidate) == decode_rule_name(codon) for candidate in synonyms)


def test_neutral_mutation_preserves_rule_identity_when_forced() -> None:
    rng = Random(13)
    original = (3, 4, 5, 6)
    mutated = mutate_codons_neutral(original, rng, mutation_rate=1.0, synonym_rate=1.0)
    assert len(mutated) == len(original)
    assert all(decode_rule_name(left) == decode_rule_name(right) for left, right in zip(original, mutated, strict=False))


def test_table_based_neutral_mutation_stays_in_same_opcode_family() -> None:
    rng = Random(19)
    table = build_codon_table({"NOOP": 2, "ADD": 2, "MUL": 2}, table_size=8, fill_op="NOOP")
    original = (0, 2, 4, 6)
    mutated = mutate_codons_with_table(original, rng, table, mutation_rate=1.0, synonym_rate=1.0)
    assert len(mutated) == len(original)
    assert all(table.op(left) == table.op(right) for left, right in zip(original, mutated, strict=False))
