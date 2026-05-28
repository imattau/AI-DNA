from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial import SpatialArena, SpatialCell, build_spatial_adhesion_demo_genome, build_spatial_genome


def main() -> int:
    arena = SpatialArena(width=5, height=5)
    mover = SpatialCell(genome=build_spatial_adhesion_demo_genome(lineage_id="A1"), x=1, y=2)
    anchor = SpatialCell(genome=build_spatial_genome(("HALT",), lineage_id="A0"), x=3, y=2)
    arena.place(mover)
    arena.place(anchor)
    arena.morphogen_field.emit(1, 2, 1.0)
    report = arena.run(steps=2)
    mover_cell = arena.cells.get((2, 2))
    trace = tuple(entry["note"] for entry in (mover_cell.trace if mover_cell else ()))
    print(
        "experiment: spatial_adhesion "
        f"occupied={report.occupied_positions} "
        f"morphogen_peak={report.morphogen_peak:.4f} "
        f"mover_trace={trace}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
