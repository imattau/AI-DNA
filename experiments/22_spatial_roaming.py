from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial import SpatialArena, SpatialCell, build_spatial_roaming_demo_genome


def main() -> None:
    genome = build_spatial_roaming_demo_genome(lineage_id="W22")
    arena = SpatialArena(width=6, height=6)
    walker = SpatialCell(genome=genome, x=2, y=2)
    arena.place(walker)
    arena.run(steps=5)
    report = arena.report(steps=5)
    print(
        "\n".join(
            [
                "experiment: 22_spatial_roaming",
                f"cells: {len(report.cells)}",
                f"occupied_positions: {report.occupied_positions}",
                f"types: {report.type_histogram}",
                f"lineages: {report.lineages}",
                "trace_examples:",
            ]
        )
    )
    for cell in report.cells[:2]:
        for trace in cell.trace[:3]:
            print(
                f"  - {cell.lineage_id}@{(cell.x, cell.y)}:{trace['op']} "
                f"{list(trace['before'])} -> {list(trace['after'])}"
                + (f" ({trace['note']})" if trace.get("note") else "")
            )


if __name__ == "__main__":
    main()
