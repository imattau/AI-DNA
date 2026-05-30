from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Callable, Iterable

from cell import CellState
from genome import GateRule, ScaleRule
from tasks import TaskCase
from tracing import TraceEvent


@dataclass(slots=True)
class ChemistryContext:
    inbox: list[float] = field(default_factory=list)
    outbox: list[float] = field(default_factory=list)
    time: float = 0.0
    stage_increment: float = 0.0
    peer_vectors: dict[int, tuple[float, ...]] = field(default_factory=dict)
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


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _record(
    state: CellState,
    event_index: int,
    rule_name: str,
    before: tuple[float, float, float, float, float],
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

    def _sense_peer(state: CellState, context: ChemistryContext, peer_index: int) -> str | None:
        peer_vector = context.peer_vectors.get(peer_index)
        if not peer_vector:
            return None
        if peer_index >= len(peer_vector) or peer_index >= len(state.signals):
            return None
        value = float(peer_vector[peer_index])
        changed = state.signals[peer_index] != value
        if changed:
            state.signals[peer_index] = value
        return f"sense_peer_{peer_index}" if changed else None

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
        "SENSE_PEER_0": Rule("SENSE_PEER_0", lambda state, task, context: _sense_peer(state, context, 0)),
        "SENSE_PEER_1": Rule("SENSE_PEER_1", lambda state, task, context: _sense_peer(state, context, 1)),
        "SENSE_PEER_2": Rule("SENSE_PEER_2", lambda state, task, context: _sense_peer(state, context, 2)),
    }


def _condition_passes(condition: str, cell: CellState) -> bool:
    def signal(index: int) -> float:
        return cell.signals[index] if index < len(cell.signals) else 0.0

    if condition == "IF_S0_GT":
        return signal(0) > 0.5
    if condition == "IF_S1_GT":
        return signal(1) > 0.5
    if condition == "IF_S2_GT":
        return signal(2) > 0.5
    if condition == "IF_S2_LT":
        return signal(2) < 0.5
    if condition == "IF_S3_GT":
        return signal(3) > 0.5
    if condition == "IF_S3_LT":
        return signal(3) < 0.5
    if condition == "IF_S4_GT":
        return signal(4) > 0.5
    if condition == "IF_S4_LT":
        return signal(4) < 0.5
    if condition == "IF_S0_LT":
        return signal(0) < 0.5
    if condition == "IF_S1_LT":
        return signal(1) < 0.5
    return True


@dataclass(slots=True)
class ChemistrySystem:
    rulebook: dict[str, Rule] = field(default_factory=build_rulebook)
    max_time: float = 32.0
    dt: float = 1.0
    quiescence_steps: int = 2

    def step(
        self,
        cell: CellState,
        task: TaskCase,
        context: ChemistryContext,
        *,
        event_index_start: int = 0,
    ) -> tuple[bool, int]:
        changed = False
        event_index = event_index_start

        def execute(rule_entry: str | GateRule | ScaleRule) -> bool:
            nonlocal changed, event_index
            if isinstance(rule_entry, ScaleRule):
                modulator = cell.signals[rule_entry.signal_slot] if rule_entry.signal_slot < len(cell.signals) else 0.0
                before = tuple(cell.signals)
                inner_changed = execute(rule_entry.inner)
                after_inner = tuple(cell.signals)
                if inner_changed or after_inner != before:
                    for index, pre_value in enumerate(before):
                        if index == rule_entry.signal_slot:
                            continue
                        delta = cell.signals[index] - pre_value
                        if delta != 0.0:
                            cell.signals[index] = _clamp(pre_value + delta * modulator)
                    after = tuple(cell.signals)
                    if after != before:
                        changed = True
                        _record(
                            cell,
                            event_index,
                            f"SCALE_BY_S{rule_entry.signal_slot}",
                            before,
                            f"scale={modulator:.3f}",
                        )
                        event_index += 1
                return changed
            if isinstance(rule_entry, GateRule):
                before = tuple(cell.signals)
                gate_name = f"GATE[{rule_entry.condition}]"
                if _condition_passes(rule_entry.condition, cell):
                    _record(cell, event_index, gate_name, before, "open")
                    event_index += 1
                    for nested in rule_entry.rules:
                        execute(nested)
                else:
                    _record(cell, event_index, gate_name, before, "closed")
                    event_index += 1
                return changed

            rule = self.rulebook.get(rule_entry)
            if rule is None:
                return changed
            before = tuple(cell.signals)
            note = rule.apply(cell, task, context)
            after = tuple(cell.signals)
            if after != before or note:
                changed = True
                _record(cell, event_index, rule_entry, before, note or "")
                event_index += 1
            return changed

        for rule_name in cell.active_rules:
            execute(rule_name)
        return changed, event_index

    def run(self, cell: CellState, task: TaskCase, *, context: ChemistryContext | None = None) -> CellState:
        context = context or ChemistryContext()
        quiet_ticks = 0
        event_index = 0
        while context.time <= self.max_time:
            if context.stage_increment > 0.0 and len(cell.signals) > 4:
                cell.signals[4] = min(cell.signals[4] + context.stage_increment, 1.0)
            changed, event_index = self.step(cell, task, context, event_index_start=event_index)
            if context.stage_increment > 0.0:
                quiet_ticks = 0
            elif cell.output is not None and not changed:
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
