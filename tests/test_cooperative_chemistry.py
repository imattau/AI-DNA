from __future__ import annotations

from cell import CellState
from chemistry import ChemistryContext
from cooperative_chemistry import CooperativeChemistrySystem
from tasks import TaskCase


def test_cells_can_pass_messages_through_shared_context() -> None:
    sender = CellState(active_rules=["SEND"])
    sender.reset(x=4.0, y=0.0)
    sender.output = 4.0

    receiver = CellState(active_rules=["RECV"])
    receiver.reset(x=0.0, y=0.0)

    arena = CooperativeChemistrySystem()
    result = arena.run(
        [sender, receiver],
        TaskCase(4.0, 0.0, 0.0, "cooperative"),
        context=ChemistryContext(),
        max_time=4.0,
    )

    assert result.messages_delivered > 0
    assert receiver.signals[3] == 4.0
    assert any(entry["rule"] == "SEND" for entry in sender.trace)
    assert any(entry["rule"] == "RECV" for entry in receiver.trace)
