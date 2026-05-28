from __future__ import annotations

from spatial3d import (
    MorphogenField3D,
    Spatial3DArena,
    Spatial3DCell,
    build_spatial3d_demo_genome,
    build_spatial3d_genome,
    decode_spatial3d_op,
    run_spatial3d_development,
)


def test_spatial3d_op_decoder_uses_shared_codons() -> None:
    assert decode_spatial3d_op(0) == "NOOP"
    assert decode_spatial3d_op(1) == "GET_X"
    assert decode_spatial3d_op(13) == "DIVIDE_UP"


def test_morphogen3d_diffusion_spreads_signal() -> None:
    field = MorphogenField3D(width=3, height=3, depth=3, decay=0.0, diffusion=0.2)
    field.emit(1, 1, 1, 8.0)
    field.diffuse()
    assert field.sense(1, 1, 1) < 8.0
    assert field.sense(1, 1, 0) > 0.0


def test_spatial3d_development_can_divide_and_differentiate() -> None:
    genome = build_spatial3d_genome(
        ("EMIT_0", "SENSE_0", "SET_TYPE_HIGH", "DIVIDE_EAST", "DIVIDE_UP", "HALT"),
        lineage_id="V1",
    )
    arena = Spatial3DArena(width=4, height=4, depth=4)
    assert arena.place(Spatial3DCell(genome=genome, x=2, y=2, z=2))
    report = arena.run(steps=5)
    assert len(report.cells) >= 2
    assert report.morphogen_peak > 0.0
    assert report.type_histogram


def test_spatial3d_demo_report_runs() -> None:
    report = run_spatial3d_development(width=4, height=4, depth=4, steps=6, seed=29, genome=build_spatial3d_demo_genome())
    assert report.cells
    assert report.occupied_positions
    assert report.format_text().startswith("spatial3d_development:")


def test_spatial3d_wander_moves_cell() -> None:
    genome = build_spatial3d_genome(("WANDER", "HALT"), lineage_id="VW")
    arena = Spatial3DArena(width=4, height=4, depth=4)
    assert arena.place(Spatial3DCell(genome=genome, x=2, y=2, z=2))
    arena.run(steps=1)
    assert (2, 2, 2) not in arena.cells
    assert len(arena.cells) == 1


def test_spatial3d_adhesion_cell_clusters_on_signal() -> None:
    arena = Spatial3DArena(width=4, height=4, depth=4)
    mover = Spatial3DCell(
        genome=build_spatial3d_genome(("SENSE_0", "ADHERE", "SENSE_0", "ADHERE", "HALT"), lineage_id="VA"),
        x=1,
        y=2,
        z=2,
    )
    anchor = Spatial3DCell(genome=build_spatial3d_genome(("HALT",), lineage_id="VA0"), x=3, y=2, z=2)
    assert arena.place(mover)
    assert arena.place(anchor)
    arena.morphogen_field.emit(1, 2, 2, 1.0)
    before_neighbors = {
        position
        for position in arena.cells
        if abs(position[0] - anchor.x) + abs(position[1] - anchor.y) + abs(position[2] - anchor.z) == 1
    }
    assert before_neighbors == set()
    arena.run(steps=5)
    assert (2, 2, 2) in arena.cells
    assert len(arena.cells) == 2
    after_neighbors = {
        position
        for position in arena.cells
        if position != (3, 2, 2) and abs(position[0] - anchor.x) + abs(position[1] - anchor.y) + abs(position[2] - anchor.z) == 1
    }
    assert after_neighbors == {(2, 2, 2)}
    assert any(entry["op"] == "ADHERE" for entry in arena.cells[(2, 2, 2)].trace)
