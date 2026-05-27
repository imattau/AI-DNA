from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class TraceEvent:
    round_index: int
    rule_name: str
    before: tuple[float, float, float, float]
    after: tuple[float, float, float, float]
    note: str = ""


@dataclass(slots=True)
class ExperimentReport:
    experiment: str
    train_error: float
    full_validation_error: float
    cell_count: int
    active_rules: tuple[str, ...]
    lineage_tree: str
    motifs_per_cell: tuple[str, ...]
    shortcut_hits: int
    trace_examples: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)

    def format_text(self) -> str:
        lines = [
            f"experiment: {self.experiment}",
            f"train_error: {self.train_error:.6f}",
            f"full_validation_error: {self.full_validation_error:.6f}",
            f"cell_count: {self.cell_count}",
            f"active_rules: {', '.join(self.active_rules) if self.active_rules else '<none>'}",
            f"lineage_tree: {self.lineage_tree}",
            f"motifs_per_cell: {', '.join(self.motifs_per_cell) if self.motifs_per_cell else '<none>'}",
            f"shortcut_hits: {self.shortcut_hits}",
            "trace_examples:",
        ]
        lines.extend(f"  - {trace}" for trace in self.trace_examples or ("<none>",))
        if self.extra:
            lines.append("extra:")
            for key, value in self.extra.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def format_trace(events: Iterable[TraceEvent]) -> tuple[str, ...]:
    formatted = []
    for event in events:
        formatted.append(
            f"r{event.round_index}:{event.rule_name} {list(event.before)} -> {list(event.after)}{f' ({event.note})' if event.note else ''}"
        )
    return tuple(formatted)


def lineage_tree_text(nodes: Sequence[tuple[str, Sequence[str]]]) -> str:
    if not nodes:
        return "<empty>"
    return " | ".join(f"{node}<-{','.join(parents) if parents else 'root'}" for node, parents in nodes)
