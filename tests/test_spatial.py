from __future__ import annotations

from genome import CellGenome
from spatial import (
    MorphogenField,
    SpatialArena,
    SpatialCell,
    build_spatial_demo_genome,
    build_spatial_genome,
    decode_spatial_op,
    run_spatial_development,
)


def test_spatial_op_decoder_uses_shared_codons() -> None:
    assert decode_spatial_op(0) == "NOOP"
    assert decode_spatial_op(1) == "GET_X"
    assert decode_spatial_op(7) == "DIVIDE_EAST"


def test_morphogen_diffusion_spreads_signal() -> None:
    field = MorphogenField(width=3, height=3, decay=0.0, diffusion=0.25)
    field.emit(1, 1, 8.0)
    field.diffuse()
    assert field.sense(1, 1) < 8.0
    assert field.sense(0, 1) > 0.0


def test_spatial_development_can_divide_and_differentiate() -> None:
    genome = build_spatial_genome(
        ("EMIT_0", "SENSE_0", "SET_TYPE_HIGH", "DIVIDE_EAST", "HALT"),
        lineage_id="Z1",
    )
    arena = SpatialArena(width=5, height=5)
    assert arena.place(SpatialCell(genome=genome, x=2, y=2))
    report = arena.run(steps=4)
    assert len(report.cells) >= 2
    assert report.morphogen_peak > 0.0
    assert report.type_histogram


def test_spatial_demo_report_runs() -> None:
    report = run_spatial_development(width=6, height=6, steps=6, seed=17, genome=build_spatial_demo_genome())
    assert report.cells
    assert report.occupied_positions
    assert report.format_text().startswith("spatial_development:")
