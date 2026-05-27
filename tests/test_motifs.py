from __future__ import annotations

from random import Random

from genome import CellGenome, Motif, format_motif, motif_statistics


def test_format_motif_includes_task_lineage_and_reuse() -> None:
    motif = Motif(pattern=("RULE_EMIT_X", "RULE_COPY0_3"), origin_lineage="L1", origin_task="multiply", reuse_count=2)
    assert format_motif(motif) == "multiply:L1:RULE_EMIT_X|RULE_COPY0_3#reuse=2"


def test_motif_statistics_reports_origin_sets_and_reuse_total() -> None:
    motifs = (
        Motif(pattern=("A",), origin_lineage="L1", origin_task="multiply", reuse_count=1),
        Motif(pattern=("B",), origin_lineage="L2", origin_task="max", reuse_count=3),
    )
    stats = motif_statistics(motifs)
    assert stats["motif_count"] == 2
    assert stats["motif_reuse_total"] == 4
    assert stats["motif_origin_tasks"] == ("max", "multiply")
    assert stats["motif_origin_lineages"] == ("L1", "L2")


def test_cell_genome_motif_mutation_increments_reuse() -> None:
    genome = CellGenome.from_rule_names(["RULE_EMIT_X"], lineage_id="motif-test")
    genome = genome.attach_motif(Motif(pattern=("RULE_ADD0_IF1",), origin_lineage="motif-test", origin_task="multiply"))
    mutated = genome.mutate(Random(5), mutation_rate=0.0, motif_mutation_rate=1.0)
    assert mutated.local_motifs
    assert mutated.local_motifs[0].reuse_count == 1
