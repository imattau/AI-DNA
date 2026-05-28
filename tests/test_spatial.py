from __future__ import annotations

from genome import CellGenome
from spatial import (
    MorphogenField,
    SpatialArena,
    SpatialCell,
    build_spatial_adhesion_demo_genome,
    build_spatial_demo_genome,
    build_spatial_genome,
    build_spatial_roaming_demo_genome,
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


def test_spatial_roaming_cell_moves_through_space() -> None:
    arena = SpatialArena(width=6, height=6)
    assert arena.place(SpatialCell(genome=build_spatial_roaming_demo_genome(), x=2, y=2))
    arena.run(steps=2)
    assert len(arena.cells) == 1
    assert (2, 2) not in arena.cells
    assert arena.cells


def test_spatial_adhesion_cell_clusters_on_signal() -> None:
    arena = SpatialArena(width=5, height=5)
    mover = SpatialCell(genome=build_spatial_adhesion_demo_genome(), x=1, y=2)
    anchor = SpatialCell(genome=build_spatial_genome(("HALT",), lineage_id="A0"), x=3, y=2)
    assert arena.place(mover)
    assert arena.place(anchor)
    arena.morphogen_field.emit(1, 2, 1.0)
    before_neighbors = {
        position
        for position in arena.cells
        if abs(position[0] - anchor.x) + abs(position[1] - anchor.y) == 1
    }
    assert before_neighbors == set()
    arena.run(steps=2)
    assert (2, 2) in arena.cells
    assert len(arena.cells) == 2
    after_neighbors = {
        position
        for position in arena.cells
        if position != (3, 2) and abs(position[0] - anchor.x) + abs(position[1] - anchor.y) == 1
    }
    assert after_neighbors == {(2, 2)}
    assert any(entry["op"] == "ADHERE" for entry in arena.cells[(2, 2)].trace)


def test_spatial_adhesion_requires_partner_signal() -> None:
    solo_arena = SpatialArena(width=6, height=5)
    solo = SpatialCell(
        genome=build_spatial_genome(("SENSE_0", "ADHERE", "SENSE_0", "ADHERE", "HALT"), lineage_id="ASOLO"),
        x=2,
        y=2,
    )
    assert solo_arena.place(solo)
    solo_arena.run(steps=3)
    assert (2, 2) in solo_arena.cells
    assert len(solo_arena.cells) == 1

    cooperative_arena = SpatialArena(width=6, height=5)
    receiver = SpatialCell(
        genome=build_spatial_genome(("SENSE_0", "ADHERE", "SENSE_0", "ADHERE", "HALT"), lineage_id="ARECV"),
        x=2,
        y=2,
    )
    sender = SpatialCell(
        genome=build_spatial_genome(("EMIT_0", "EMIT_0", "HALT"), lineage_id="ASEND"),
        x=4,
        y=2,
    )
    assert cooperative_arena.place(receiver)
    assert cooperative_arena.place(sender)
    cooperative_arena.run(steps=5)
    assert (3, 2) in cooperative_arena.cells
    assert (4, 2) in cooperative_arena.cells
    assert any(entry["op"] == "ADHERE" and "adhere=" in entry["note"] for entry in cooperative_arena.cells[(3, 2)].trace)
    assert cooperative_arena.cells[(3, 2)].trace
