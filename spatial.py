from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from random import Random
from typing import Iterable

from genome import CellGenome


SPATIAL_OPS: tuple[str, ...] = (
    "NOOP",
    "GET_X",
    "GET_Y",
    "SENSE_0",
    "SENSE_GRAD_X_0",
    "SENSE_GRAD_Y_0",
    "EMIT_0",
    "DIVIDE_EAST",
    "DIVIDE_WEST",
    "DIVIDE_NORTH",
    "DIVIDE_SOUTH",
    "SET_TYPE_LOW",
    "SET_TYPE_HIGH",
    "KILL",
    "HALT",
    "REPAIR_EAST",
    "REPAIR_WEST",
    "REPAIR_NORTH",
    "REPAIR_SOUTH",
    "ADHERE",
    "WANDER",
)


def encode_spatial_op(op_name: str) -> int:
    try:
        return SPATIAL_OPS.index(op_name)
    except ValueError as exc:
        raise KeyError(op_name) from exc


def decode_spatial_op(codon: int) -> str:
    return SPATIAL_OPS[codon % len(SPATIAL_OPS)]


def build_spatial_genome(
    op_names: Iterable[str],
    *,
    lineage_id: str,
) -> CellGenome:
    return CellGenome(
        codons=tuple(encode_spatial_op(op_name) for op_name in op_names),
        local_motifs=(),
        lineage_id=lineage_id,
    )


@dataclass(slots=True)
class MorphogenField:
    width: int
    height: int
    decay: float = 0.08
    diffusion: float = 0.22
    toroidal: bool = True
    values: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.values = [[0.0 for _ in range(self.width)] for _ in range(self.height)]

    def _wrap(self, x: int, y: int) -> tuple[int, int] | None:
        if self.toroidal:
            return x % self.width, y % self.height
        if 0 <= x < self.width and 0 <= y < self.height:
            return x, y
        return None

    def sense(self, x: int, y: int) -> float:
        wrapped = self._wrap(x, y)
        if wrapped is None:
            return 0.0
        wx, wy = wrapped
        return self.values[wy][wx]

    def emit(self, x: int, y: int, amount: float) -> None:
        wrapped = self._wrap(x, y)
        if wrapped is None:
            return
        wx, wy = wrapped
        self.values[wy][wx] += amount

    def gradient_x(self, x: int, y: int) -> float:
        return self.sense(x + 1, y) - self.sense(x - 1, y)

    def gradient_y(self, x: int, y: int) -> float:
        return self.sense(x, y + 1) - self.sense(x, y - 1)

    def diffuse(self) -> None:
        new_values = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                center = self.values[y][x]
                north = self.sense(x, y - 1)
                south = self.sense(x, y + 1)
                east = self.sense(x + 1, y)
                west = self.sense(x - 1, y)
                laplacian = north + south + east + west - 4.0 * center
                new_value = center + self.diffusion * laplacian
                new_value *= 1.0 - self.decay
                new_values[y][x] = max(0.0, new_value)
        self.values = new_values

    def peak(self) -> float:
        return max((value for row in self.values for value in row), default=0.0)


@dataclass(slots=True)
class SpatialCell:
    genome: CellGenome
    x: int
    y: int
    pc: int = 0
    registers: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    type_id: int = 0
    output: float | None = None
    alive: bool = True
    halted: bool = False
    age: int = 0
    trace: list[dict[str, object]] = field(default_factory=list)

    @property
    def lineage_id(self) -> str:
        return self.genome.lineage_id


@dataclass(slots=True)
class SpatialDevelopmentReport:
    width: int
    height: int
    steps: int
    cells: tuple[SpatialCell, ...]
    morphogen_peak: float
    occupied_positions: tuple[tuple[int, int], ...]
    type_histogram: tuple[tuple[int, int], ...]
    lineages: tuple[str, ...]

    def format_text(self) -> str:
        return (
            f"spatial_development: {self.width}x{self.height} steps={self.steps} "
            f"cells={len(self.cells)} morphogen_peak={self.morphogen_peak:.4f} "
            f"occupied={self.occupied_positions} types={self.type_histogram} "
            f"lineages={self.lineages}"
        )


@dataclass(slots=True)
class SpatialArena:
    width: int
    height: int
    morphogen_field: MorphogenField = field(init=False)
    cells: dict[tuple[int, int], SpatialCell] = field(default_factory=dict)
    lineage_counter: int = 0

    def __post_init__(self) -> None:
        self.morphogen_field = MorphogenField(self.width, self.height)

    def _wrap(self, x: int, y: int) -> tuple[int, int] | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return x, y
        return None

    def place(self, cell: SpatialCell) -> bool:
        location = self._wrap(cell.x, cell.y)
        if location is None or location in self.cells:
            return False
        self.cells[location] = cell
        return True

    def spawn_child(self, parent: SpatialCell, dx: int, dy: int) -> bool:
        location = self._wrap(parent.x + dx, parent.y + dy)
        if location is None or location in self.cells:
            return False
        self.lineage_counter += 1
        child = SpatialCell(
            genome=parent.genome.with_lineage(f"{parent.lineage_id}.s{self.lineage_counter}"),
            x=location[0],
            y=location[1],
            registers=list(parent.registers),
            type_id=parent.type_id,
        )
        self.cells[location] = child
        return True

    def move_cell(self, cell: SpatialCell, dx: int, dy: int) -> bool:
        source = (cell.x, cell.y)
        target = self._wrap(cell.x + dx, cell.y + dy)
        if target is None or target in self.cells:
            return False
        self.cells.pop(source, None)
        cell.x, cell.y = target
        self.cells[target] = cell
        return True

    def _wander(self, cell: SpatialCell) -> tuple[int, int]:
        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        offset = int(abs(sum(cell.registers)) + cell.age + cell.pc) % len(directions)
        for attempt in range(len(directions)):
            dx, dy = directions[(offset + attempt) % len(directions)]
            if self.move_cell(cell, dx, dy):
                return dx, dy
        return 0, 0

    def _adhere(self, cell: SpatialCell) -> tuple[int, int]:
        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        if cell.registers[2] <= 0.0:
            return 0, 0
        source = (cell.x, cell.y)
        others = [position for position in self.cells if position != source]
        if not others:
            return 0, 0

        def adjacent_occupied_count(x: int, y: int) -> int:
            count = 0
            for dx, dy in directions:
                neighbor = self._wrap(x + dx, y + dy)
                if neighbor is not None and neighbor in self.cells and neighbor != source:
                    count += 1
            return count

        offset = int(abs(sum(cell.registers)) + cell.age + cell.pc) % len(directions)
        for attempt in range(len(directions)):
            dx, dy = directions[(offset + attempt) % len(directions)]
            target = self._wrap(cell.x + dx, cell.y + dy)
            if target is None or target in self.cells:
                continue
            if adjacent_occupied_count(*target) > 0:
                if self.move_cell(cell, dx, dy):
                    return dx, dy

        best_move: tuple[int, int] | None = None
        best_distance: int | None = None
        for attempt in range(len(directions)):
            dx, dy = directions[(offset + attempt) % len(directions)]
            target = self._wrap(cell.x + dx, cell.y + dy)
            if target is None or target in self.cells:
                continue
            distance = min(abs(target[0] - ox) + abs(target[1] - oy) for ox, oy in others)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_move = (dx, dy)
        if best_move is not None and self.move_cell(cell, *best_move):
            return best_move
        return 0, 0

    def _execute(self, cell: SpatialCell) -> None:
        if not cell.alive or cell.halted or not cell.genome.codons:
            return
        op = decode_spatial_op(cell.genome.codons[cell.pc % len(cell.genome.codons)])
        before = tuple(cell.registers)
        note = ""
        if op == "NOOP":
            pass
        elif op == "GET_X":
            cell.registers[0] = float(cell.x)
        elif op == "GET_Y":
            cell.registers[1] = float(cell.y)
        elif op == "SENSE_0":
            cell.registers[2] = self.morphogen_field.sense(cell.x, cell.y)
        elif op == "SENSE_GRAD_X_0":
            cell.registers[3] = self.morphogen_field.gradient_x(cell.x, cell.y)
        elif op == "SENSE_GRAD_Y_0":
            cell.registers[3] = self.morphogen_field.gradient_y(cell.x, cell.y)
        elif op == "EMIT_0":
            amount = max(0.25, cell.registers[2] if cell.registers[2] != 0.0 else 1.0)
            self.morphogen_field.emit(cell.x, cell.y, amount)
            note = f"emit={amount:.2f}"
        elif op == "DIVIDE_EAST":
            note = "divide_east" if self.spawn_child(cell, 1, 0) else "divide_fail"
        elif op == "DIVIDE_WEST":
            note = "divide_west" if self.spawn_child(cell, -1, 0) else "divide_fail"
        elif op == "DIVIDE_NORTH":
            note = "divide_north" if self.spawn_child(cell, 0, -1) else "divide_fail"
        elif op == "DIVIDE_SOUTH":
            note = "divide_south" if self.spawn_child(cell, 0, 1) else "divide_fail"
        elif op == "SET_TYPE_LOW":
            cell.type_id = 0 if cell.registers[2] <= 0.5 else 1
            note = f"type={cell.type_id}"
        elif op == "SET_TYPE_HIGH":
            cell.type_id = 1 if cell.registers[2] > 0.5 else 0
            note = f"type={cell.type_id}"
        elif op == "KILL":
            cell.alive = False
            note = "kill"
        elif op == "HALT":
            cell.halted = True
            note = "halt"
        elif op == "REPAIR_EAST":
            note = "repair_east" if self.spawn_child(cell, 1, 0) else "repair_fail"
        elif op == "REPAIR_WEST":
            note = "repair_west" if self.spawn_child(cell, -1, 0) else "repair_fail"
        elif op == "REPAIR_NORTH":
            note = "repair_north" if self.spawn_child(cell, 0, -1) else "repair_fail"
        elif op == "REPAIR_SOUTH":
            note = "repair_south" if self.spawn_child(cell, 0, 1) else "repair_fail"
        elif op == "ADHERE":
            dx, dy = self._adhere(cell)
            note = f"adhere={dx},{dy}" if (dx or dy) else "adhere_fail"
        elif op == "WANDER":
            dx, dy = self._wander(cell)
            note = f"wander={dx},{dy}" if (dx or dy) else "wander_fail"
        else:
            pass
        if tuple(cell.registers) != before or note:
            cell.trace.append(
                {
                    "step": cell.age,
                    "op": op,
                    "before": before,
                    "after": tuple(cell.registers),
                    "note": note,
                    "position": (cell.x, cell.y),
                }
            )
        if not cell.halted and cell.alive:
            cell.pc += 1
        cell.age += 1

    def step(self) -> None:
        for position in sorted(list(self.cells)):
            cell = self.cells.get(position)
            if cell is None:
                continue
            self._execute(cell)
        dead_positions = [position for position, cell in self.cells.items() if not cell.alive]
        for position in dead_positions:
            self.cells.pop(position, None)
        self.morphogen_field.diffuse()

    def run(self, *, steps: int) -> SpatialDevelopmentReport:
        for _ in range(steps):
            self.step()
        return self.report(steps=steps)

    def report(self, *, steps: int) -> SpatialDevelopmentReport:
        cells = tuple(sorted(self.cells.values(), key=lambda cell: (cell.y, cell.x, cell.lineage_id)))
        occupied_positions = tuple(sorted(self.cells))
        type_histogram = tuple(sorted(Counter(cell.type_id for cell in cells).items()))
        lineages = tuple(sorted(cell.lineage_id for cell in cells))
        return SpatialDevelopmentReport(
            width=self.width,
            height=self.height,
            steps=steps,
            cells=cells,
            morphogen_peak=self.morphogen_field.peak(),
            occupied_positions=occupied_positions,
            type_histogram=type_histogram,
            lineages=lineages,
        )


def build_spatial_demo_genome(*, lineage_id: str = "Z") -> CellGenome:
    return build_spatial_genome(
        (
            "GET_X",
            "EMIT_0",
            "SENSE_0",
            "SET_TYPE_HIGH",
            "DIVIDE_EAST",
            "DIVIDE_SOUTH",
            "HALT",
        ),
        lineage_id=lineage_id,
    )


def build_spatial_repair_demo_genome(*, lineage_id: str = "R") -> CellGenome:
    return build_spatial_genome(
        (
            "REPAIR_EAST",
            "REPAIR_SOUTH",
            "SENSE_0",
        ),
        lineage_id=lineage_id,
    )


def build_spatial_roaming_demo_genome(*, lineage_id: str = "W") -> CellGenome:
    return build_spatial_genome(
        (
            "WANDER",
            "GET_X",
            "GET_Y",
            "WANDER",
            "WANDER",
            "HALT",
        ),
        lineage_id=lineage_id,
    )


def build_spatial_adhesion_demo_genome(*, lineage_id: str = "A") -> CellGenome:
    return build_spatial_genome(
        (
            "SENSE_0",
            "ADHERE",
            "HALT",
        ),
        lineage_id=lineage_id,
    )


def run_spatial_development(
    *,
    width: int = 6,
    height: int = 6,
    steps: int = 8,
    seed: int = 11,
    genome: CellGenome | None = None,
) -> SpatialDevelopmentReport:
    rng = Random(seed)
    arena = SpatialArena(width=width, height=height)
    chosen_genome = genome or build_spatial_demo_genome(lineage_id=f"Z{seed}")
    center = SpatialCell(
        genome=chosen_genome,
        x=width // 2,
        y=height // 2,
        registers=[rng.random(), rng.random(), 0.0, 0.0],
    )
    arena.place(center)
    return arena.run(steps=steps)
