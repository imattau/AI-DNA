from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CellState:
    active_rules: list[str]
    signals: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    output: float | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def reset(self, *, x: float, y: float) -> None:
        self.signals = [float(x), float(y), 0.0, 0.0]
        self.output = None
        self.trace.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_rules": list(self.active_rules),
            "signals": list(self.signals),
            "output": self.output,
        }

    @property
    def cell_count(self) -> int:
        return 1
