from __future__ import annotations

from spatial import SpatialArena, SpatialCell, build_spatial_repair_demo_genome, decode_spatial_op


def test_spatial_repair_op_decoder_appends_without_breaking_existing_mapping() -> None:
    assert decode_spatial_op(7) == "DIVIDE_EAST"
    assert decode_spatial_op(15) == "REPAIR_EAST"


def test_spatial_repair_regenerates_missing_neighbor() -> None:
    genome = build_spatial_repair_demo_genome(lineage_id="R1")
    arena = SpatialArena(width=5, height=5)
    assert arena.place(SpatialCell(genome=genome, x=2, y=2))
    arena.step()
    assert (3, 2) in arena.cells
    arena.cells.pop((3, 2), None)
    for _ in range(4):
        arena.step()
    assert (3, 2) in arena.cells


def test_spatial_repair_demo_runs() -> None:
    genome = build_spatial_repair_demo_genome(lineage_id="R2")
    arena = SpatialArena(width=5, height=5)
    assert arena.place(SpatialCell(genome=genome, x=2, y=2))
    report = arena.run(steps=5)
    assert report.cells
    assert report.occupied_positions
