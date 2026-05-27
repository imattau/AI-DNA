from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Callable, Iterable

from cell import CellState
from tasks import TaskCase
from tracing import TraceEvent


@dataclass(slots=True)
class ChemistryContext:
    inbox: list[float] = field(default_factory=list)
    outbox: list[float] = field(default_factory=list)
    time: float = 0.0
    events: list[dict[str, float | str]] = field(default_factory=list)

    def schedule(self, kind: str, *, when: float, payload: float | str | None = None) -> None:
        entry: dict[str, float | str] = {"kind": kind, "when": when}
        if payload is not None:
            entry["payload"] = payload
        self.events.append(entry)


RuleFn = Callable[[CellState, TaskCase, ChemistryContext], str | None]


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    apply: RuleFn


def _record(
    state: CellState,
    event_index: int,
    rule_name: str,
    before: tuple[float, float, float, float],
    note: str = "",
) -> None:
    state.trace.append(
        {
            "round": event_index,
            "rule": rule_name,
            "before": before,
            "after": tuple(state.signals),
            "note": note,
        }
    )


def build_rulebook() -> dict[str, Rule]:
    def emit_x(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[0] != task.x:
            state.signals[0] = task.x
            return "emit_x"
        return None

    def emit_y(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[1] != task.y:
            state.signals[1] = task.y
            return "emit_y"
        return None

    def zero_2(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[2] != 0.0:
            state.signals[2] = 0.0
            return "zero_2"
        return None

    def add0_if1(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[1] > 0:
            state.signals[2] += state.signals[0]
            context.schedule("accumulate", when=context.time + 1.0, payload=state.signals[2])
            return "added"
        return None

    def decay1(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[1] > 0:
            state.signals[1] = max(0.0, state.signals[1] - 1.0)
            context.schedule("decay", when=context.time + 1.0, payload=state.signals[1])
            return "decay1"
        return None

    def output_if1z(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[1] == 0:
            if state.output != state.signals[2]:
                state.output = state.signals[2]
                return "output"
        return None

    def inhibit1z(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[1] == 0 and state.signals[3] != 0.0:
            state.signals[3] = 0.0
            return "inhibit"
        return None

    def copy0_3(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[3] != state.signals[0]:
            state.signals[3] = state.signals[0]
            return "copy0"
        return None

    def copy1_3(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[3] != state.signals[1]:
            state.signals[3] = state.signals[1]
            return "copy1"
        return None

    def add3_if1(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[1] > 0:
            state.signals[2] += state.signals[3]
            context.schedule("accumulate", when=context.time + 1.0, payload=state.signals[2])
            return "added3"
        return None

    def decay3(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if state.signals[3] > 0:
            state.signals[3] = max(0.0, state.signals[3] - 1.0)
            context.schedule("decay", when=context.time + 1.0, payload=state.signals[3])
            return "decay3"
        return None

    def thresh1(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        value = 1.0 if state.signals[1] > 0 else 0.0
        if state.signals[1] != value:
            state.signals[1] = value
            return "thresh1"
        return None

    def thresh3(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        value = 1.0 if state.signals[3] > 0 else 0.0
        if state.signals[3] != value:
            state.signals[3] = value
            return "thresh3"
        return None

    def send(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        payload = state.output if state.output is not None else state.signals[2]
        if not context.outbox or context.outbox[-1] != payload:
            context.outbox.append(payload)
            context.schedule("send", when=context.time, payload=payload)
            return "send"
        return None

    def recv(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if context.inbox:
            state.signals[3] = context.inbox.pop(0)
            context.schedule("recv", when=context.time, payload=state.signals[3])
            return "recv"
        return None

    return {
        "RULE_EMIT_X": Rule("RULE_EMIT_X", emit_x),
        "RULE_EMIT_Y": Rule("RULE_EMIT_Y", emit_y),
        "RULE_ZERO_2": Rule("RULE_ZERO_2", zero_2),
        "RULE_ADD0_IF1": Rule("RULE_ADD0_IF1", add0_if1),
        "RULE_DECAY1": Rule("RULE_DECAY1", decay1),
        "RULE_OUTPUT_IF1Z": Rule("RULE_OUTPUT_IF1Z", output_if1z),
        "RULE_INHIBIT1Z": Rule("RULE_INHIBIT1Z", inhibit1z),
        "RULE_COPY0_3": Rule("RULE_COPY0_3", copy0_3),
        "RULE_COPY1_3": Rule("RULE_COPY1_3", copy1_3),
        "RULE_ADD3_IF1": Rule("RULE_ADD3_IF1", add3_if1),
        "RULE_DECAY3": Rule("RULE_DECAY3", decay3),
        "RULE_THRESH1": Rule("RULE_THRESH1", thresh1),
        "RULE_THRESH3": Rule("RULE_THRESH3", thresh3),
        "SEND": Rule("SEND", send),
        "RECV": Rule("RECV", recv),
    }


@dataclass(slots=True)
class ChemistrySystem:
    rulebook: dict[str, Rule] = field(default_factory=build_rulebook)
    max_time: float = 32.0
    dt: float = 1.0
    quiescence_steps: int = 2

    def run(self, cell: CellState, task: TaskCase, *, context: ChemistryContext | None = None) -> CellState:
        context = context or ChemistryContext()
        quiet_ticks = 0
        event_index = 0
        while context.time <= self.max_time:
            changed = False
            for rule_name in cell.active_rules:
                rule = self.rulebook[rule_name]
                before = tuple(cell.signals)
                note = rule.apply(cell, task, context)
                after = tuple(cell.signals)
                if after != before or note:
                    changed = True
                    _record(cell, event_index, rule_name, before, note or "")
                    event_index += 1
            if cell.output is not None and not changed:
                quiet_ticks += 1
            else:
                quiet_ticks = 0
            if quiet_ticks >= self.quiescence_steps:
                break
            context.time += self.dt
        if cell.output is None:
            cell.output = cell.signals[2]
        return cell


def run_cell(
    active_rules: Iterable[str],
    task: TaskCase,
    *,
    max_time: float = 32.0,
    dt: float = 1.0,
    context: ChemistryContext | None = None,
) -> CellState:
    cell = CellState(active_rules=list(active_rules))
    cell.reset(x=task.x, y=task.y)
    ChemistrySystem(max_time=max_time, dt=dt).run(cell, task, context=context)
    return cell
