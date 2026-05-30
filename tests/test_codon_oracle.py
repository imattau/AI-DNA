from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from chemistry import ChemistryContext, Rule, build_rulebook
from codon_oracle import CodonOracle


def _make_rulebook() -> dict:
    return build_rulebook()


def _make_codon_map() -> dict:
    from codons import REGULATORY_CODON_MAP
    return dict(REGULATORY_CODON_MAP)


def _make_op_names() -> list[str]:
    from codons import REGULATORY_OP_NAMES
    return list(REGULATORY_OP_NAMES)


def test_is_stagnating_flat_history():
    oracle = CodonOracle()
    history = [0.25] * 50
    assert oracle.is_stagnating(history) is True


def test_is_stagnating_improving_history():
    oracle = CodonOracle()
    history = [0.25 - i * 0.005 for i in range(50)]
    assert oracle.is_stagnating(history) is False


def test_is_stagnating_short_history_returns_false():
    oracle = CodonOracle()
    assert oracle.is_stagnating([0.25] * 10) is False


def test_inject_valid_sense_peer_proposal():
    oracle = CodonOracle()
    rulebook = _make_rulebook()
    codon_map = _make_codon_map()
    op_names = _make_op_names()

    proposal = {"template": "SENSE_PEER_S_TO_D", "s": 3, "d": 4}
    result = oracle.inject(proposal, rulebook, codon_map, op_names)

    assert result == "SENSE_PEER_3_TO_4"
    assert "SENSE_PEER_3_TO_4" in rulebook
    assert "SENSE_PEER_3_TO_4" in op_names
    # 3 numeric IDs assigned
    new_ids = [k for k, v in codon_map.items() if v == "SENSE_PEER_3_TO_4"]
    assert len(new_ids) == 3


def test_inject_duplicate_codon_returns_none():
    oracle = CodonOracle()
    rulebook = _make_rulebook()
    codon_map = _make_codon_map()
    op_names = _make_op_names()

    proposal = {"template": "SENSE_PEER_S_TO_D", "s": 2, "d": 3}
    # SENSE_PEER_2_TO_3 already exists
    result = oracle.inject(proposal, rulebook, codon_map, op_names)
    assert result is None


def test_inject_invalid_template_returns_none():
    oracle = CodonOracle()
    rulebook = _make_rulebook()
    codon_map = _make_codon_map()
    op_names = _make_op_names()

    proposal = {"template": "UNKNOWN_TEMPLATE", "s": 1, "d": 2}
    result = oracle.inject(proposal, rulebook, codon_map, op_names)
    assert result is None


def test_build_prompt_contains_required_fields():
    oracle = CodonOracle()
    codon_names = ["SENSE_PEER_0", "SCALE_BY_S3"]
    fitness_history = [0.25, 0.24, 0.24, 0.24]
    best_motif = ["RULE_COPY1_2", "SENSE_PEER_2_TO_3"]
    prompt = oracle.build_prompt(codon_names, fitness_history, best_motif)

    assert "SENSE_PEER_0" in prompt
    assert "0.25" in prompt or "0.24" in prompt
    assert "RULE_COPY1_2" in prompt
    assert "SENSE_PEER_S_TO_D" in prompt  # template menu present
