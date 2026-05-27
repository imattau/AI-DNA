from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from cell import CellState
from chemistry import ChemistryContext, ChemistrySystem
from tasks import TaskCase


@dataclass(slots=True)
class CooperativeRunResult:
    cells: tuple[CellState, ...]
    context: ChemistryContext
    steps: int
    messages_delivered: int


@dataclass(slots=True)
class CooperativeChemistrySystem:
    chemistry: ChemistrySystem = field(default_factory=ChemistrySystem)

    def _deliver_messages(self, context: ChemistryContext, *, broadcast: bool = True) -> int:
        delivered = len(context.outbox)
        if not context.outbox:
            return 0
        if broadcast:
            context.inbox.extend(context.outbox)
        else:
            context.inbox.append(context.outbox[-1])
        context.outbox.clear()
        return delivered

    def run(
        self,
        cells: Sequence[CellState],
        task: TaskCase,
        *,
        context: ChemistryContext | None = None,
        max_time: float | None = None,
    ) -> CooperativeRunResult:
        context = context or ChemistryContext()
        time_limit = max_time if max_time is not None else self.chemistry.max_time
        event_index = 0
        quiet_ticks = 0
        messages_delivered = 0
        step_count = 0
        cell_list = list(cells)

        while context.time <= time_limit:
            changed = False
            messages_delivered += self._deliver_messages(context, broadcast=True)
            for cell in cell_list:
                cell_changed, event_index = self.chemistry.step(cell, task, context, event_index_start=event_index)
                changed = changed or cell_changed
                messages_delivered += self._deliver_messages(context, broadcast=True)
            if all(cell.output is not None for cell in cell_list) and not changed:
                quiet_ticks += 1
            else:
                quiet_ticks = 0
            if quiet_ticks >= self.chemistry.quiescence_steps:
                break
            context.time += self.chemistry.dt
            step_count += 1

        for cell in cell_list:
            if cell.output is None:
                cell.output = cell.signals[2]
        return CooperativeRunResult(
            cells=tuple(cell_list),
            context=context,
            steps=step_count,
            messages_delivered=messages_delivered,
        )
