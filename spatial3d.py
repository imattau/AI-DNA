from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from random import Random
from typing import Iterable

from genome import CellGenome


SPATIAL3D_OPS: tuple[str, ...] = (
    "NOOP",
    "GET_X",
    "GET_Y",
    "GET_Z",
    "SENSE_0",
    "SENSE_GRAD_X_0",
    "SENSE_GRAD_Y_0",
    "SENSE_GRAD_Z_0",
    "EMIT_0",
    "DIVIDE_EAST",
    "DIVIDE_WEST",
    "DIVIDE_NORTH",
    "DIVIDE_SOUTH",
    "DIVIDE_UP",
    "DIVIDE_DOWN",
    "SET_TYPE_LOW",
    "SET_TYPE_HIGH",
    "KILL",
    "HALT",
    "REPAIR_EAST",
    "REPAIR_WEST",
    "REPAIR_NORTH",
    "REPAIR_SOUTH",
    "REPAIR_UP",
    "REPAIR_DOWN",
    "ADHERE",
    "WANDER",
)


def encode_spatial3d_op(op_name: str) -> int:
    try:
        return SPATIAL3D_OPS.index(op_name)
    except ValueError as exc:
        raise KeyError(op_name) from exc


def decode_spatial3d_op(codon: int) -> str:
    return SPATIAL3D_OPS[codon % len(SPATIAL3D_OPS)]


def build_spatial3d_genome(
    op_names: Iterable[str],
    *,
    lineage_id: str,
) -> CellGenome:
    return CellGenome(
        codons=tuple(encode_spatial3d_op(op_name) for op_name in op_names),
        local_motifs=(),
        lineage_id=lineage_id,
    )


@dataclass(slots=True)
class MorphogenField3D:
    width: int
    height: int
    depth: int
    decay: float = 0.08
    diffusion: float = 0.18
    toroidal: bool = True
    values: list[list[list[float]]] = field(init=False)

    def __post_init__(self) -> None:
        self.values = [
            [[0.0 for _ in range(self.width)] for _ in range(self.height)]
            for _ in range(self.depth)
        ]

    def _wrap(self, x: int, y: int, z: int) -> tuple[int, int, int] | None:
        if self.toroidal:
            return x % self.width, y % self.height, z % self.depth
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            return x, y, z
        return None

    def sense(self, x: int, y: int, z: int) -> float:
        wrapped = self._wrap(x, y, z)
        if wrapped is None:
            return 0.0
        wx, wy, wz = wrapped
        return self.values[wz][wy][wx]

    def emit(self, x: int, y: int, z: int, amount: float) -> None:
        wrapped = self._wrap(x, y, z)
        if wrapped is None:
            return
        wx, wy, wz = wrapped
        self.values[wz][wy][wx] += amount

    def gradient_x(self, x: int, y: int, z: int) -> float:
        return self.sense(x + 1, y, z) - self.sense(x - 1, y, z)

    def gradient_y(self, x: int, y: int, z: int) -> float:
        return self.sense(x, y + 1, z) - self.sense(x, y - 1, z)

    def gradient_z(self, x: int, y: int, z: int) -> float:
        return self.sense(x, y, z + 1) - self.sense(x, y, z - 1)

    def diffuse(self) -> None:
        new_values = [
            [[0.0 for _ in range(self.width)] for _ in range(self.height)]
            for _ in range(self.depth)
        ]
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    center = self.values[z][y][x]
                    neighbors = (
                        self.sense(x + 1, y, z),
                        self.sense(x - 1, y, z),
                        self.sense(x, y + 1, z),
                        self.sense(x, y - 1, z),
                        self.sense(x, y, z + 1),
                        self.sense(x, y, z - 1),
                    )
                    laplacian = sum(neighbors) - 6.0 * center
                    new_value = (center + self.diffusion * laplacian) * (1.0 - self.decay)
                    new_values[z][y][x] = max(0.0, new_value)
        self.values = new_values

    def peak(self) -> float:
        return max((value for plane in self.values for row in plane for value in row), default=0.0)


@dataclass(slots=True)
class Spatial3DCell:
    genome: CellGenome
    x: int
    y: int
    z: int
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
class Spatial3DDevelopmentReport:
    width: int
    height: int
    depth: int
    steps: int
    cells: tuple[Spatial3DCell, ...]
    morphogen_peak: float
    occupied_positions: tuple[tuple[int, int, int], ...]
    type_histogram: tuple[tuple[int, int], ...]
    lineages: tuple[str, ...]

    def format_text(self) -> str:
        return (
            f"spatial3d_development: {self.width}x{self.height}x{self.depth} steps={self.steps} "
            f"cells={len(self.cells)} morphogen_peak={self.morphogen_peak:.4f} "
            f"occupied={self.occupied_positions} types={self.type_histogram} "
            f"lineages={self.lineages}"
        )


@dataclass(slots=True)
class Spatial3DArena:
    width: int
    height: int
    depth: int
    morphogen_field: MorphogenField3D = field(init=False)
    cells: dict[tuple[int, int, int], Spatial3DCell] = field(default_factory=dict)
    lineage_counter: int = 0

    def __post_init__(self) -> None:
        self.morphogen_field = MorphogenField3D(self.width, self.height, self.depth)

    def _wrap(self, x: int, y: int, z: int) -> tuple[int, int, int] | None:
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            return x, y, z
        return None

    def place(self, cell: Spatial3DCell) -> bool:
        location = self._wrap(cell.x, cell.y, cell.z)
        if location is None or location in self.cells:
            return False
        self.cells[location] = cell
        return True

    def spawn_child(self, parent: Spatial3DCell, dx: int, dy: int, dz: int) -> bool:
        location = self._wrap(parent.x + dx, parent.y + dy, parent.z + dz)
        if location is None or location in self.cells:
            return False
        self.lineage_counter += 1
        child = Spatial3DCell(
            genome=parent.genome.with_lineage(f"{parent.lineage_id}.v{self.lineage_counter}"),
            x=location[0],
            y=location[1],
            z=location[2],
            registers=list(parent.registers),
            type_id=parent.type_id,
        )
        self.cells[location] = child
        return True

    def move_cell(self, cell: Spatial3DCell, dx: int, dy: int, dz: int) -> bool:
        source = (cell.x, cell.y, cell.z)
        target = self._wrap(cell.x + dx, cell.y + dy, cell.z + dz)
        if target is None or target in self.cells:
            return False
        self.cells.pop(source, None)
        cell.x, cell.y, cell.z = target
        self.cells[target] = cell
        return True

    def _wander(self, cell: Spatial3DCell) -> tuple[int, int, int]:
        directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1))
        offset = int(abs(sum(cell.registers)) + cell.age + cell.pc) % len(directions)
        for attempt in range(len(directions)):
            dx, dy, dz = directions[(offset + attempt) % len(directions)]
            if self.move_cell(cell, dx, dy, dz):
                return dx, dy, dz
        return 0, 0, 0

    def _adhere(self, cell: Spatial3DCell) -> tuple[int, int, int]:
        directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1))
        if cell.registers[3] <= 0.0:
            return 0, 0, 0
        source = (cell.x, cell.y, cell.z)
        others = [position for position in self.cells if position != source]
        if not others:
            return 0, 0, 0

        def adjacent_occupied_count(x: int, y: int, z: int) -> int:
            count = 0
            for dx, dy, dz in directions:
                neighbor = self._wrap(x + dx, y + dy, z + dz)
                if neighbor is not None and neighbor in self.cells and neighbor != source:
                    count += 1
            return count

        offset = int(abs(sum(cell.registers)) + cell.age + cell.pc) % len(directions)
        for attempt in range(len(directions)):
            dx, dy, dz = directions[(offset + attempt) % len(directions)]
            target = self._wrap(cell.x + dx, cell.y + dy, cell.z + dz)
            if target is None or target in self.cells:
                continue
            if adjacent_occupied_count(*target) > 0:
                if self.move_cell(cell, dx, dy, dz):
                    return dx, dy, dz

        best_move: tuple[int, int, int] | None = None
        best_distance: int | None = None
        for attempt in range(len(directions)):
            dx, dy, dz = directions[(offset + attempt) % len(directions)]
            target = self._wrap(cell.x + dx, cell.y + dy, cell.z + dz)
            if target is None or target in self.cells:
                continue
            distance = min(
                abs(target[0] - ox) + abs(target[1] - oy) + abs(target[2] - oz)
                for ox, oy, oz in others
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_move = (dx, dy, dz)
        if best_move is not None and self.move_cell(cell, *best_move):
            return best_move
        return 0, 0, 0

    def _execute(self, cell: Spatial3DCell) -> None:
        if not cell.alive or cell.halted or not cell.genome.codons:
            return
        op = decode_spatial3d_op(cell.genome.codons[cell.pc % len(cell.genome.codons)])
        before = tuple(cell.registers)
        note = ""
        if op == "NOOP":
            pass
        elif op == "GET_X":
            cell.registers[0] = float(cell.x)
        elif op == "GET_Y":
            cell.registers[1] = float(cell.y)
        elif op == "GET_Z":
            cell.registers[2] = float(cell.z)
        elif op == "SENSE_0":
            cell.registers[3] = self.morphogen_field.sense(cell.x, cell.y, cell.z)
        elif op == "SENSE_GRAD_X_0":
            cell.registers[3] = self.morphogen_field.gradient_x(cell.x, cell.y, cell.z)
        elif op == "SENSE_GRAD_Y_0":
            cell.registers[3] = self.morphogen_field.gradient_y(cell.x, cell.y, cell.z)
        elif op == "SENSE_GRAD_Z_0":
            cell.registers[3] = self.morphogen_field.gradient_z(cell.x, cell.y, cell.z)
        elif op == "EMIT_0":
            amount = max(0.25, cell.registers[3] if cell.registers[3] != 0.0 else 1.0)
            self.morphogen_field.emit(cell.x, cell.y, cell.z, amount)
            note = f"emit={amount:.2f}"
        elif op == "DIVIDE_EAST":
            note = "divide_east" if self.spawn_child(cell, 1, 0, 0) else "divide_fail"
        elif op == "DIVIDE_WEST":
            note = "divide_west" if self.spawn_child(cell, -1, 0, 0) else "divide_fail"
        elif op == "DIVIDE_NORTH":
            note = "divide_north" if self.spawn_child(cell, 0, -1, 0) else "divide_fail"
        elif op == "DIVIDE_SOUTH":
            note = "divide_south" if self.spawn_child(cell, 0, 1, 0) else "divide_fail"
        elif op == "DIVIDE_UP":
            note = "divide_up" if self.spawn_child(cell, 0, 0, 1) else "divide_fail"
        elif op == "DIVIDE_DOWN":
            note = "divide_down" if self.spawn_child(cell, 0, 0, -1) else "divide_fail"
        elif op == "SET_TYPE_LOW":
            cell.type_id = 0 if cell.registers[3] <= 0.5 else 1
            note = f"type={cell.type_id}"
        elif op == "SET_TYPE_HIGH":
            cell.type_id = 1 if cell.registers[3] > 0.5 else 0
            note = f"type={cell.type_id}"
        elif op == "KILL":
            cell.alive = False
            note = "kill"
        elif op == "HALT":
            cell.halted = True
            note = "halt"
        elif op == "REPAIR_EAST":
            note = "repair_east" if self.spawn_child(cell, 1, 0, 0) else "repair_fail"
        elif op == "REPAIR_WEST":
            note = "repair_west" if self.spawn_child(cell, -1, 0, 0) else "repair_fail"
        elif op == "REPAIR_NORTH":
            note = "repair_north" if self.spawn_child(cell, 0, -1, 0) else "repair_fail"
        elif op == "REPAIR_SOUTH":
            note = "repair_south" if self.spawn_child(cell, 0, 1, 0) else "repair_fail"
        elif op == "REPAIR_UP":
            note = "repair_up" if self.spawn_child(cell, 0, 0, 1) else "repair_fail"
        elif op == "REPAIR_DOWN":
            note = "repair_down" if self.spawn_child(cell, 0, 0, -1) else "repair_fail"
        elif op == "ADHERE":
            dx, dy, dz = self._adhere(cell)
            note = f"adhere={dx},{dy},{dz}" if (dx or dy or dz) else "adhere_fail"
        elif op == "WANDER":
            dx, dy, dz = self._wander(cell)
            note = f"wander={dx},{dy},{dz}" if (dx or dy or dz) else "wander_fail"
        if tuple(cell.registers) != before or note:
            cell.trace.append(
                {
                    "step": cell.age,
                    "op": op,
                    "before": before,
                    "after": tuple(cell.registers),
                    "note": note,
                    "position": (cell.x, cell.y, cell.z),
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

    def run(self, *, steps: int) -> Spatial3DDevelopmentReport:
        for _ in range(steps):
            self.step()
        return self.report(steps=steps)

    def report(self, *, steps: int) -> Spatial3DDevelopmentReport:
        cells = tuple(sorted(self.cells.values(), key=lambda cell: (cell.z, cell.y, cell.x, cell.lineage_id)))
        occupied_positions = tuple(sorted(self.cells))
        type_histogram = tuple(sorted(Counter(cell.type_id for cell in cells).items()))
        lineages = tuple(sorted(cell.lineage_id for cell in cells))
        return Spatial3DDevelopmentReport(
            width=self.width,
            height=self.height,
            depth=self.depth,
            steps=steps,
            cells=cells,
            morphogen_peak=self.morphogen_field.peak(),
            occupied_positions=occupied_positions,
            type_histogram=type_histogram,
            lineages=lineages,
        )


def build_spatial3d_demo_genome(*, lineage_id: str = "V") -> CellGenome:
    return build_spatial3d_genome(
        (
            "GET_X",
            "GET_Z",
            "EMIT_0",
            "SENSE_0",
            "SET_TYPE_HIGH",
            "DIVIDE_EAST",
            "DIVIDE_UP",
            "DIVIDE_SOUTH",
            "HALT",
        ),
        lineage_id=lineage_id,
    )


def build_spatial3d_adhesion_demo_genome(*, lineage_id: str = "AV") -> CellGenome:
    return build_spatial3d_genome(
        (
            "SENSE_0",
            "ADHERE",
            "HALT",
        ),
        lineage_id=lineage_id,
    )


def run_spatial3d_development(
    *,
    width: int = 4,
    height: int = 4,
    depth: int = 4,
    steps: int = 10,
    seed: int = 23,
    genome: CellGenome | None = None,
) -> Spatial3DDevelopmentReport:
    rng = Random(seed)
    arena = Spatial3DArena(width=width, height=height, depth=depth)
    chosen_genome = genome or build_spatial3d_demo_genome(lineage_id=f"V{seed}")
    center = Spatial3DCell(
        genome=chosen_genome,
        x=width // 2,
        y=height // 2,
        z=depth // 2,
        registers=[rng.random(), rng.random(), rng.random(), 0.0],
    )
    arena.place(center)
    return arena.run(steps=steps)
