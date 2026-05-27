from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial import SpatialArena, SpatialCell, build_spatial_repair_demo_genome


def main() -> None:
    genome = build_spatial_repair_demo_genome(lineage_id="R20")
    arena = SpatialArena(width=5, height=5)
    parent = SpatialCell(genome=genome, x=2, y=2)
    arena.place(parent)
    arena.step()
    removed = arena.cells.pop((3, 2), None) is not None
    for _ in range(4):
        arena.step()
    report = arena.report(steps=5)
    repaired = (3, 2) in arena.cells
    print(
        "\n".join(
            [
                "experiment: 20_spatial_self_repair",
                f"removed_neighbor: {removed}",
                f"repaired_neighbor: {repaired}",
                f"cells: {len(report.cells)}",
                f"occupied_positions: {report.occupied_positions}",
                f"types: {report.type_histogram}",
                f"lineages: {report.lineages}",
                "trace_examples:",
            ]
        )
    )
    for entry in report.cells[:2]:
        for trace in entry.trace[:3]:
            print(
                f"  - {entry.lineage_id}@{(entry.x, entry.y)}:{trace['op']} "
                f"{list(trace['before'])} -> {list(trace['after'])}"
                + (f" ({trace['note']})" if trace.get("note") else "")
            )


if __name__ == "__main__":
    main()
