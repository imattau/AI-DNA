from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cell import CellState
from chemistry import ChemistryContext
from cooperative_chemistry import CooperativeChemistrySystem
from tasks import TaskCase


def main() -> None:
    sender = CellState(active_rules=["SEND"])
    sender.reset(x=0.0, y=0.0)
    sender.output = 13.0

    receiver = CellState(active_rules=["RECV", "RULE_ADD3_IF1", "RULE_DECAY1", "RULE_OUTPUT_IF1Z"])
    receiver.reset(x=0.0, y=1.0)

    arena = CooperativeChemistrySystem()
    result = arena.run(
        [sender, receiver],
        TaskCase(0.0, 1.0, 0.0, "cooperative_exchange"),
        context=ChemistryContext(),
        max_time=4.0,
    )

    print("experiment: 13_cooperative_chemistry")
    print(f"messages_delivered: {result.messages_delivered}")
    print(f"steps: {result.steps}")
    print(f"sender_output: {result.cells[0].output}")
    print(f"receiver_output: {result.cells[1].output}")
    print(f"receiver_signal3: {result.cells[1].signals[3]}")
    print("trace_examples:")
    for cell_index, cell in enumerate(result.cells):
        for event in cell.trace[:4]:
            print(f"  - cell{cell_index}:{event['round']} {event['rule']} {event['note']}")
    if result.messages_delivered <= 0 or result.cells[1].output != 13.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
