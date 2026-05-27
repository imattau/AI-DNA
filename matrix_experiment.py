from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import Callable, Sequence

from codons import build_codon_table, crossover_codons, insert_delete_codons, mutate_codons, random_codons, unique_ordered
from genome import Motif
from tracing import ExperimentReport, lineage_tree_text


MATRIX_OPS: tuple[str, ...] = (
    "NOOP",
    "HALT",
    "DUP",
    "SWAP",
    "POP",
    "ADD",
    "SUB",
    "MUL",
    "LOAD_A_0",
    "LOAD_A_1",
    "LOAD_A_2",
    "LOAD_A_3",
    "LOAD_A_4",
    "LOAD_A_5",
    "LOAD_A_6",
    "LOAD_A_7",
    "LOAD_A_8",
    "LOAD_B_0",
    "LOAD_B_1",
    "LOAD_B_2",
    "LOAD_B_3",
    "LOAD_B_4",
    "LOAD_B_5",
    "LOAD_B_6",
    "LOAD_B_7",
    "LOAD_B_8",
    "STORE_C_0",
    "STORE_C_1",
    "STORE_C_2",
    "STORE_C_3",
    "STORE_C_4",
    "STORE_C_5",
    "STORE_C_6",
    "STORE_C_7",
    "STORE_C_8",
)

MATRIX_CODON_ALLOCATION: dict[str, int] = {
    "NOOP": 6,
    "HALT": 2,
    "DUP": 3,
    "SWAP": 3,
    "POP": 2,
    "ADD": 5,
    "SUB": 5,
    "MUL": 8,
    "LOAD_A_0": 2,
    "LOAD_A_1": 2,
    "LOAD_A_2": 2,
    "LOAD_A_3": 1,
    "LOAD_A_4": 1,
    "LOAD_A_5": 1,
    "LOAD_A_6": 1,
    "LOAD_A_7": 1,
    "LOAD_A_8": 1,
    "LOAD_B_0": 2,
    "LOAD_B_1": 2,
    "LOAD_B_2": 2,
    "LOAD_B_3": 1,
    "LOAD_B_4": 1,
    "LOAD_B_5": 1,
    "LOAD_B_6": 1,
    "LOAD_B_7": 1,
    "LOAD_B_8": 1,
    "STORE_C_0": 2,
    "STORE_C_1": 2,
    "STORE_C_2": 2,
    "STORE_C_3": 1,
    "STORE_C_4": 1,
    "STORE_C_5": 1,
    "STORE_C_6": 1,
    "STORE_C_7": 1,
    "STORE_C_8": 1,
}

MATRIX_CODON_TABLE = build_codon_table(MATRIX_CODON_ALLOCATION, table_size=64, fill_op="NOOP")

SAFE_MATRIX_MOTIFS: tuple[tuple[str, ...], ...] = (
    ("LOAD_A_0", "LOAD_B_0", "MUL", "STORE_C_0", "HALT"),
    ("LOAD_A_1", "LOAD_B_4", "MUL", "STORE_C_4", "HALT"),
    ("LOAD_A_2", "LOAD_B_8", "MUL", "STORE_C_8", "HALT"),
    ("LOAD_A_0", "LOAD_B_1", "MUL", "ADD", "STORE_C_1", "HALT"),
    ("LOAD_A_3", "LOAD_B_3", "MUL", "STORE_C_3", "HALT"),
    ("LOAD_A_6", "LOAD_B_6", "MUL", "STORE_C_6", "HALT"),
)


def decode_matrix_op(value: int) -> str:
    return MATRIX_CODON_TABLE.op(value % len(MATRIX_CODON_TABLE.ops))


def encode_matrix_op(op_name: str) -> int:
    try:
        return MATRIX_CODON_TABLE.op_to_codons[op_name][0]
    except KeyError as exc:
        raise KeyError(op_name) from exc


def random_codon_for_op(op_name: str, rng: Random) -> int:
    return rng.choice(MATRIX_CODON_TABLE.op_to_codons[op_name])


def codons_from_matrix_ops(ops: Sequence[str], rng: Random) -> tuple[int, ...]:
    return tuple(MATRIX_CODON_TABLE.encode_ops(list(ops)))


def matrix_flat(*rows: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for row in rows for value in row)


def matrix_zero() -> tuple[float, ...]:
    return tuple(0.0 for _ in range(9))


def matrix_add(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(a) + float(b) for a, b in zip(left, right, strict=True))


def matrix_sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(a) - float(b) for a, b in zip(left, right, strict=True))


def matrix_transpose(values: Sequence[float]) -> tuple[float, ...]:
    grid = list(values)
    return tuple(grid[row + col * 3] for row in range(3) for col in range(3))


def matrix_product(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    a = list(left)
    b = list(right)
    out: list[float] = [0.0] * 9
    for row in range(3):
        for col in range(3):
            total = 0.0
            for k in range(3):
                total += a[row * 3 + k] * b[k * 3 + col]
            out[row * 3 + col] = total
    return tuple(out)


Monomial = tuple[int, ...]


def _poly_simplify(terms: dict[Monomial, int]) -> dict[Monomial, int]:
    return {monomial: coefficient for monomial, coefficient in terms.items() if coefficient != 0}


def _poly_var(index: int) -> dict[Monomial, int]:
    return {(index,): 1}


def _poly_const(value: int) -> dict[Monomial, int]:
    return {(): value} if value != 0 else {}


def _poly_add(left: dict[Monomial, int], right: dict[Monomial, int]) -> dict[Monomial, int]:
    out = dict(left)
    for monomial, coefficient in right.items():
        out[monomial] = out.get(monomial, 0) + coefficient
        if out[monomial] == 0:
            del out[monomial]
    return out


def _poly_sub(left: dict[Monomial, int], right: dict[Monomial, int]) -> dict[Monomial, int]:
    out = dict(left)
    for monomial, coefficient in right.items():
        out[monomial] = out.get(monomial, 0) - coefficient
        if out[monomial] == 0:
            del out[monomial]
    return out


def _poly_mul(left: dict[Monomial, int], right: dict[Monomial, int]) -> dict[Monomial, int]:
    out: dict[Monomial, int] = {}
    for left_monomial, left_coeff in left.items():
        for right_monomial, right_coeff in right.items():
            monomial = tuple(sorted(left_monomial + right_monomial))
            out[monomial] = out.get(monomial, 0) + left_coeff * right_coeff
            if out[monomial] == 0:
                del out[monomial]
    return out


def _poly_equals(left: dict[Monomial, int], right: dict[Monomial, int]) -> bool:
    return _poly_simplify(left) == _poly_simplify(right)


def _poly_to_text(poly: dict[Monomial, int]) -> str:
    if not poly:
        return "0"
    parts: list[str] = []
    for monomial, coefficient in sorted(poly.items(), key=lambda item: (len(item[0]), item[0])):
        if monomial:
            factor = "*".join(("a" if index < 9 else "b") + str(index if index < 9 else index - 9) for index in monomial)
        else:
            factor = "1"
        parts.append(f"{coefficient}*{factor}")
    return " + ".join(parts)


def matrix_symbolic_inputs() -> tuple[dict[Monomial, int], ...]:
    return tuple(_poly_var(index) for index in range(18))


def matrix_symbolic_target() -> tuple[dict[Monomial, int], ...]:
    outputs: list[dict[Monomial, int]] = []
    for out_index in range(9):
        row, col = divmod(out_index, 3)
        terms: dict[Monomial, int] = {}
        for k in range(3):
            left_index = row * 3 + k
            right_index = 9 + k * 3 + col
            terms = _poly_add(terms, {tuple(sorted((left_index, right_index))): 1})
        outputs.append(terms)
    return tuple(outputs)


def symbolic_verify_matrix_program(genome: "MatrixGenome") -> tuple[bool, tuple[dict[Monomial, int], ...]]:
    stack: list[dict[Monomial, int]] = []
    outputs: list[dict[Monomial, int]] = [_poly_const(0) for _ in range(9)]
    a_inputs = matrix_symbolic_inputs()[:9]
    b_inputs = matrix_symbolic_inputs()[9:]
    for op in genome.declare_ops():
        if op == "NOOP":
            continue
        if op == "HALT":
            break
        if op == "DUP":
            stack.append(stack[-1] if stack else _poly_const(0))
            continue
        if op == "SWAP":
            if len(stack) >= 2:
                stack[-1], stack[-2] = stack[-2], stack[-1]
            continue
        if op == "POP":
            if stack:
                stack.pop()
            continue
        if op == "ADD":
            right = stack.pop() if stack else _poly_const(0)
            left = stack.pop() if stack else _poly_const(0)
            stack.append(_poly_add(left, right))
            continue
        if op == "SUB":
            right = stack.pop() if stack else _poly_const(0)
            left = stack.pop() if stack else _poly_const(0)
            stack.append(_poly_sub(left, right))
            continue
        if op == "MUL":
            right = stack.pop() if stack else _poly_const(0)
            left = stack.pop() if stack else _poly_const(0)
            stack.append(_poly_mul(left, right))
            continue
        if op.startswith("LOAD_A_"):
            index = int(op.rsplit("_", 1)[1])
            stack.append(a_inputs[index])
            continue
        if op.startswith("LOAD_B_"):
            index = int(op.rsplit("_", 1)[1])
            stack.append(b_inputs[index])
            continue
        if op.startswith("STORE_C_"):
            index = int(op.rsplit("_", 1)[1])
            outputs[index] = stack.pop() if stack else _poly_const(0)
            continue
    target = matrix_symbolic_target()
    return all(_poly_equals(output, expected) for output, expected in zip(outputs, target, strict=True)), tuple(outputs)


@dataclass(frozen=True, slots=True)
class MatrixCase:
    a: tuple[float, ...]
    b: tuple[float, ...]
    target: tuple[float, ...]
    task_name: str


@dataclass(frozen=True, slots=True)
class MatrixBundle:
    name: str
    train: tuple[MatrixCase, ...]
    validation: tuple[MatrixCase, ...]
    anti_shortcuts: tuple[MatrixCase, ...]
    shortcut_checks: tuple[Callable[[MatrixCase], tuple[float, ...]], ...]


@dataclass(frozen=True, slots=True)
class MatrixMotif:
    pattern: tuple[str, ...]
    origin_lineage: str
    origin_task: str
    reuse_count: int = 0

    def inherit(self) -> "MatrixMotif":
        return replace(self, reuse_count=self.reuse_count + 1)


@dataclass(frozen=True, slots=True)
class MatrixGenome:
    codons: tuple[int, ...]
    local_motifs: tuple[MatrixMotif, ...]
    lineage_id: str

    @classmethod
    def from_ops(
        cls,
        ops: Sequence[str],
        *,
        lineage_id: str,
        rng: Random,
        local_motifs: Sequence[MatrixMotif] | None = None,
    ) -> "MatrixGenome":
        return cls(
            codons=codons_from_matrix_ops(ops, rng),
            local_motifs=tuple(local_motifs or ()),
            lineage_id=lineage_id,
        )

    def declare_ops(self) -> tuple[str, ...]:
        declared = [decode_matrix_op(codon) for codon in self.codons]
        for motif in self.local_motifs:
            declared.extend(motif.pattern)
        return tuple(declared)

    def with_lineage(self, lineage_id: str) -> "MatrixGenome":
        return replace(self, lineage_id=lineage_id)

    def attach_motif(self, motif: MatrixMotif) -> "MatrixGenome":
        return replace(self, local_motifs=self.local_motifs + (motif,))

    def mutate(
        self,
        rng: Random,
        *,
        mutation_rate: float = 0.08,
        insertion_rate: float = 0.05,
        deletion_rate: float = 0.03,
        motif_mutation_rate: float = 0.2,
    ) -> "MatrixGenome":
        codons = mutate_codons(self.codons, rng, mutation_rate=mutation_rate)
        codons = insert_delete_codons(
            codons,
            rng,
            insertion_rate=insertion_rate,
            deletion_rate=deletion_rate,
        )
        motifs = list(self.local_motifs)
        if motifs and rng.random() < motif_mutation_rate:
            motifs[rng.randrange(len(motifs))] = motifs[rng.randrange(len(motifs))].inherit()
        return replace(self, codons=codons, local_motifs=tuple(motifs))

    def signature(self) -> tuple[int, ...]:
        return self.codons


@dataclass(frozen=True, slots=True)
class MatrixEvaluation:
    genome: MatrixGenome
    train_error: float
    validation_error: float
    exact_output_positions: int
    symbolic_verified: bool
    shortcut_hits: int
    mul_count: int
    case_errors: tuple[float, ...]
    trace_examples: tuple[str, ...]

    @property
    def score(self) -> float:
        if self.exact_output_positions < 9:
            structural_penalty = (9 - self.exact_output_positions) * 1000.0
            return structural_penalty + self.validation_error + self.train_error + 5.0 * self.shortcut_hits + 0.05 * self.mul_count
        verification_bonus = 0.0 if self.symbolic_verified else 100.0
        return verification_bonus + 0.01 * self.mul_count + 5.0 * self.shortcut_hits


@dataclass(slots=True)
class MatrixRunState:
    stack: list[float]
    outputs: list[float]
    trace: list[str]
    mul_count: int = 0


def build_reference_ops() -> tuple[str, ...]:
    ops: list[str] = []
    for out_index in range(9):
        row, col = divmod(out_index, 3)
        for k in range(3):
            ops.append(f"LOAD_A_{row * 3 + k}")
            ops.append(f"LOAD_B_{k * 3 + col}")
            ops.append("MUL")
            if k > 0:
                ops.append("ADD")
        ops.append(f"STORE_C_{out_index}")
    ops.append("HALT")
    return tuple(ops)


def build_reference_genome(*, lineage_id: str, rng: Random) -> MatrixGenome:
    return MatrixGenome.from_ops(build_reference_ops(), lineage_id=lineage_id, rng=rng)


def build_random_genome(*, lineage_id: str, rng: Random, length: int = 48) -> MatrixGenome:
    return MatrixGenome(
        codons=random_codons(rng, length),
        local_motifs=(),
        lineage_id=lineage_id,
    )


def build_biased_founder(*, lineage_id: str, rng: Random) -> MatrixGenome:
    genome = build_random_genome(lineage_id=lineage_id, rng=rng, length=rng.randrange(24, 40))
    motif_count = 1 if rng.random() < 0.7 else 2
    motifs = []
    for _ in range(motif_count):
        motifs.append(
            MatrixMotif(
                pattern=tuple(rng.choice(SAFE_MATRIX_MOTIFS)),
                origin_lineage=lineage_id,
                origin_task="matrix_founder_bias",
            )
        )
    for motif in motifs:
        genome = genome.attach_motif(motif)
    return genome


def build_mixed_population(
    rng: Random,
    *,
    size: int,
    lineage_prefix: str = "M",
) -> list[MatrixGenome]:
    population: list[MatrixGenome] = []
    while len(population) < size:
        if len(population) < max(1, size // 4):
            population.append(
                build_biased_founder(
                    lineage_id=f"{lineage_prefix}B{len(population)}",
                    rng=rng,
                )
            )
        else:
            population.append(
                build_random_genome(
                    lineage_id=f"{lineage_prefix}R{len(population)}",
                    rng=rng,
                    length=rng.randrange(32, 64),
                )
            )
    return population


def _push(state: MatrixRunState, value: float) -> None:
    state.stack.append(float(value))


def _pop_or_zero(state: MatrixRunState) -> float:
    if state.stack:
        return float(state.stack.pop())
    return 0.0


def run_matrix_program(
    genome: MatrixGenome,
    case: MatrixCase,
    *,
    max_steps: int = 256,
) -> MatrixRunState:
    state = MatrixRunState(stack=[], outputs=list(matrix_zero()), trace=[])
    a = case.a
    b = case.b
    for step_index, op in enumerate(genome.declare_ops()):
        if step_index >= max_steps:
            state.trace.append(f"s{step_index}:MAX_STEPS")
            break
        before_stack = list(state.stack)
        before_outputs = list(state.outputs)
        note = ""
        if op == "NOOP":
            note = "noop"
        elif op == "HALT":
            note = "halt"
            state.trace.append(
                f"s{step_index}:{op} stack={before_stack} out={before_outputs} -> stack={state.stack} out={state.outputs}"
            )
            break
        elif op == "DUP":
            _push(state, state.stack[-1] if state.stack else 0.0)
            note = "dup"
        elif op == "SWAP":
            if len(state.stack) >= 2:
                state.stack[-1], state.stack[-2] = state.stack[-2], state.stack[-1]
                note = "swap"
            else:
                note = "swap-empty"
        elif op == "POP":
            _pop_or_zero(state)
            note = "pop"
        elif op == "ADD":
            right = _pop_or_zero(state)
            left = _pop_or_zero(state)
            _push(state, left + right)
            note = "add"
        elif op == "SUB":
            right = _pop_or_zero(state)
            left = _pop_or_zero(state)
            _push(state, left - right)
            note = "sub"
        elif op == "MUL":
            right = _pop_or_zero(state)
            left = _pop_or_zero(state)
            _push(state, left * right)
            state.mul_count += 1
            note = "mul"
        elif op.startswith("LOAD_A_"):
            index = int(op.rsplit("_", 1)[1])
            _push(state, a[index])
            note = f"load_a[{index}]"
        elif op.startswith("LOAD_B_"):
            index = int(op.rsplit("_", 1)[1])
            _push(state, b[index])
            note = f"load_b[{index}]"
        elif op.startswith("STORE_C_"):
            index = int(op.rsplit("_", 1)[1])
            value = _pop_or_zero(state)
            outputs = list(state.outputs)
            outputs[index] = float(value)
            state.outputs = outputs
            note = f"store_c[{index}]"
        else:
            note = "unknown"
        state.trace.append(
            f"s{step_index}:{op} {before_stack} -> {state.stack} | {before_outputs} -> {state.outputs} ({note})"
        )
    return state


def extract_motif_from_ops(
    declared_ops: Sequence[str],
    *,
    origin_lineage: str,
    origin_task: str,
) -> MatrixMotif:
    return MatrixMotif(
        pattern=unique_ordered(declared_ops),
        origin_lineage=origin_lineage,
        origin_task=origin_task,
    )


def shortcut_prediction_functions() -> tuple[Callable[[MatrixCase], tuple[float, ...]], ...]:
    return (
        lambda case: case.a,
        lambda case: case.b,
        lambda case: matrix_add(case.a, case.b),
        lambda case: matrix_sub(case.a, case.b),
        lambda case: matrix_transpose(case.a),
        lambda case: matrix_zero(),
    )


def _make_case(name: str, a: tuple[float, ...], b: tuple[float, ...]) -> MatrixCase:
    return MatrixCase(a=a, b=b, target=matrix_product(a, b), task_name=name)


def build_matrix_bundle(
    *,
    train_cases: Sequence[tuple[tuple[float, ...], tuple[float, ...]]] | None = None,
    validation_cases: Sequence[tuple[tuple[float, ...], tuple[float, ...]]] | None = None,
) -> MatrixBundle:
    if train_cases is None:
        train_cases = (
            (
                matrix_flat((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                matrix_flat((2, 1, 0), (0, 1, 3), (4, 0, -1)),
            ),
            (
                matrix_flat((0, 1, 0), (1, 0, 0), (0, 0, 1)),
                matrix_flat((1, 2, 3), (4, 5, 6), (7, 8, 9)),
            ),
            (
                matrix_flat((2, 0, 1), (1, 1, 0), (0, 3, 1)),
                matrix_flat((1, 0, 2), (0, 1, 1), (3, 2, 0)),
            ),
            (
                matrix_flat((1, 2, 1), (0, 1, 0), (2, 0, 1)),
                matrix_flat((3, 1, 0), (0, 2, 1), (1, 0, 2)),
            ),
        )
    if validation_cases is None:
        rng = Random(19)
        validation_cases = []
        for _ in range(8):
            a = tuple(float(rng.randint(-2, 3)) for _ in range(9))
            b = tuple(float(rng.randint(-2, 3)) for _ in range(9))
            validation_cases.append((a, b))
        validation_cases = tuple(validation_cases)

    anti_shortcuts = (
        _make_case("matrix_multiply", matrix_flat((1, 0, 0), (0, 1, 0), (0, 0, 1)), matrix_flat((0, 1, 2), (3, 4, 5), (6, 7, 8))),
        _make_case("matrix_multiply", matrix_flat((0, 1, 2), (1, 0, 1), (2, 1, 0)), matrix_flat((2, 0, 1), (1, 1, 0), (0, 2, 1))),
        _make_case("matrix_multiply", matrix_flat((2, 1, 0), (0, 1, 2), (1, 0, 1)), matrix_flat((1, 2, 3), (0, 1, 0), (3, 0, 1))),
        _make_case("matrix_multiply", matrix_flat((1, 1, 1), (0, 2, 0), (3, 0, 1)), matrix_flat((0, 1, 0), (2, 0, 1), (1, 3, 2))),
    )

    train = tuple(_make_case("matrix_multiply", a, b) for a, b in train_cases)
    validation = tuple(_make_case("matrix_multiply", a, b) for a, b in validation_cases)
    return MatrixBundle(
        name="matrix_multiplication_3x3",
        train=train,
        validation=validation,
        anti_shortcuts=anti_shortcuts,
        shortcut_checks=shortcut_prediction_functions(),
    )


def build_basis_matrix_bundle() -> MatrixBundle:
    identity = matrix_flat((1, 0, 0), (0, 1, 0), (0, 0, 1))
    basis_matrices = (
        matrix_flat((1, 0, 0), (0, 0, 0), (0, 0, 0)),
        matrix_flat((0, 1, 0), (0, 0, 0), (0, 0, 0)),
        matrix_flat((0, 0, 1), (0, 0, 0), (0, 0, 0)),
        matrix_flat((0, 0, 0), (1, 0, 0), (0, 0, 0)),
        matrix_flat((0, 0, 0), (0, 1, 0), (0, 0, 0)),
        matrix_flat((0, 0, 0), (0, 0, 1), (0, 0, 0)),
        matrix_flat((0, 0, 0), (0, 0, 0), (1, 0, 0)),
        matrix_flat((0, 0, 0), (0, 0, 0), (0, 1, 0)),
        matrix_flat((0, 0, 0), (0, 0, 0), (0, 0, 1)),
    )
    train_pairs = tuple((basis, identity) for basis in basis_matrices) + tuple((identity, basis) for basis in basis_matrices)
    validation_pairs = tuple((basis_matrices[index], basis_matrices[(index + 1) % len(basis_matrices)]) for index in range(len(basis_matrices)))
    anti_shortcuts = tuple(
        _make_case("matrix_multiply", basis, identity if index % 2 == 0 else basis_matrices[(index + 3) % len(basis_matrices)])
        for index, basis in enumerate(basis_matrices[:4])
    )
    return MatrixBundle(
        name="matrix_multiplication_basis",
        train=tuple(_make_case("matrix_multiply", a, b) for a, b in train_pairs),
        validation=tuple(_make_case("matrix_multiply", a, b) for a, b in validation_pairs),
        anti_shortcuts=anti_shortcuts,
        shortcut_checks=shortcut_prediction_functions(),
    )


def shortcut_hit(case: MatrixCase, prediction: Sequence[float], shortcut_checks: Sequence[Callable[[MatrixCase], tuple[float, ...]]]) -> bool:
    if tuple(float(value) for value in prediction) == case.target:
        return False
    return any(tuple(float(value) for value in prediction) == check(case) for check in shortcut_checks)


def _case_error(prediction: Sequence[float], target: Sequence[float]) -> float:
    return sum(abs(pred - goal) for pred, goal in zip(prediction, target, strict=True))


def _exact_output_positions(predictions: Sequence[Sequence[float]], cases: Sequence[MatrixCase]) -> int:
    exact = 0
    for index in range(9):
        if all(abs(prediction[index] - case.target[index]) < 1e-9 for prediction, case in zip(predictions, cases, strict=True)):
            exact += 1
    return exact


def evaluate_genome(
    genome: MatrixGenome,
    bundle: MatrixBundle,
    *,
    max_steps: int = 256,
) -> MatrixEvaluation:
    train_error = 0.0
    validation_error = 0.0
    shortcut_hits = 0
    mul_count = 0
    traces: list[str] = []
    train_predictions: list[tuple[float, ...]] = []
    validation_predictions: list[tuple[float, ...]] = []
    train_case_errors: list[float] = []
    showcase_cases = [case for case in bundle.validation if any(abs(value) > 0.0 for value in case.a)][:2]

    def run_case(case: MatrixCase) -> MatrixRunState:
        return run_matrix_program(genome, case, max_steps=max_steps)

    for case in bundle.train:
        state = run_case(case)
        mul_count += state.mul_count
        prediction = tuple(state.outputs)
        train_predictions.append(prediction)
        case_error = _case_error(prediction, case.target)
        train_case_errors.append(case_error)
        train_error += case_error

    for case in showcase_cases:
        state = run_case(case)
        traces.extend(state.trace[:4])

    for case in bundle.validation:
        state = run_case(case)
        mul_count += state.mul_count
        prediction = tuple(state.outputs)
        validation_predictions.append(prediction)
        validation_error += _case_error(prediction, case.target)

    for case in bundle.anti_shortcuts:
        state = run_case(case)
        if shortcut_hit(case, state.outputs, bundle.shortcut_checks):
            shortcut_hits += 1

    exact_output_positions = _exact_output_positions(train_predictions + validation_predictions, bundle.train + bundle.validation)
    symbolic_verified = False
    if train_error == 0.0 and validation_error == 0.0:
        symbolic_verified, _symbolic_outputs = symbolic_verify_matrix_program(genome)

    return MatrixEvaluation(
        genome=genome,
        train_error=train_error / max(1, len(bundle.train)),
        validation_error=validation_error / max(1, len(bundle.validation)),
        exact_output_positions=exact_output_positions,
        symbolic_verified=symbolic_verified,
        shortcut_hits=shortcut_hits,
        mul_count=mul_count,
        case_errors=tuple(train_case_errors),
        trace_examples=tuple(traces[:8]),
    )


def build_lineage_report(genome: MatrixGenome) -> str:
    motifs = tuple(motif.origin_lineage for motif in genome.local_motifs)
    return lineage_tree_text(((genome.lineage_id, motifs),))


def clone_population(genomes: Sequence[MatrixGenome]) -> list[MatrixGenome]:
    return list(genomes)


@dataclass(frozen=True, slots=True)
class MatrixSearchConfig:
    mutation_rate: float = 0.08
    insertion_rate: float = 0.05
    deletion_rate: float = 0.03
    motif_mutation_rate: float = 0.2
    crossover_rate: float = 0.35
    population_size: int = 10
    restarts: int = 4
    generations: int = 8
    survivor_count: int = 3
    siblings_per_survivor: int = 3
    island_count: int = 4
    migration_interval: int = 3
    migration_size: int = 1
    lexicase_epsilon: float = 1e-9
    basis_phase_generations: int = 4


def _mate(left: MatrixGenome, right: MatrixGenome, rng: Random, *, lineage_id: str) -> MatrixGenome:
    return MatrixGenome(
        codons=crossover_codons(left.codons, right.codons, rng),
        local_motifs=left.local_motifs + right.local_motifs,
        lineage_id=lineage_id,
    )


def select_lexicase(population: Sequence[MatrixGenome], evaluations: Sequence[MatrixEvaluation], rng: Random, *, epsilon: float) -> MatrixGenome:
    if not population:
        raise ValueError("population must not be empty")
    if len(population) == 1:
        return population[0]
    candidates = list(range(len(population)))
    case_order = list(range(len(evaluations[0].case_errors)))
    rng.shuffle(case_order)
    for case_index in case_order:
        case_scores = [evaluations[index].case_errors[case_index] for index in candidates]
        best_score = min(case_scores)
        candidates = [index for index in candidates if evaluations[index].case_errors[case_index] <= best_score + epsilon]
        if len(candidates) <= 1:
            break
    return population[rng.choice(candidates)]


def migrate_islands(islands: list[list[MatrixGenome]], evaluations: list[list[MatrixEvaluation]], *, migration_size: int) -> None:
    if len(islands) <= 1 or migration_size <= 0:
        return
    top_per_island = [
        [genome for genome, _evaluation in sorted(zip(population, island_evaluations), key=lambda pair: pair[1].score)[:migration_size]]
        for population, island_evaluations in zip(islands, evaluations, strict=True)
    ]
    for index, population in enumerate(islands):
        incoming = top_per_island[index - 1]
        if not incoming:
            continue
        for migrate_index, genome in enumerate(incoming):
            population[-(migrate_index + 1)] = genome.with_lineage(f"{genome.lineage_id}.m")


def run_matrix_experiment(
    *,
    experiment_name: str,
    seed: int,
    bundle: MatrixBundle | None = None,
    config: MatrixSearchConfig | None = None,
    require_perfect: bool = False,
) -> ExperimentReport:
    bundle = bundle or build_matrix_bundle()
    basis_bundle = build_basis_matrix_bundle()
    config = config or MatrixSearchConfig()
    best_evaluation: MatrixEvaluation | None = None
    best_genome: MatrixGenome | None = None

    for restart in range(config.restarts):
        rng = Random(seed + restart * 101)
        islands = [
            build_mixed_population(rng, size=config.population_size, lineage_prefix=f"R{restart}I{island_index}")
            for island_index in range(config.island_count)
        ]

        for generation in range(config.generations):
            active_bundle = basis_bundle if generation < config.basis_phase_generations else bundle
            island_evaluations = [[evaluate_genome(genome, active_bundle) for genome in population] for population in islands]
            for population, evaluations in zip(islands, island_evaluations, strict=True):
                generation_best_genome, generation_best_eval = min(zip(population, evaluations), key=lambda pair: pair[1].score)
                final_generation_eval = evaluate_genome(generation_best_genome, bundle)
                if best_evaluation is None or final_generation_eval.score < best_evaluation.score:
                    best_evaluation = final_generation_eval
                    best_genome = generation_best_genome
                if final_generation_eval.validation_error == 0.0 and final_generation_eval.train_error == 0.0 and final_generation_eval.symbolic_verified:
                    best_evaluation = final_generation_eval
                    best_genome = generation_best_genome
                    break
            else:
                if config.migration_interval > 0 and generation > 0 and generation % config.migration_interval == 0:
                    migrate_islands(islands, island_evaluations, migration_size=config.migration_size)

                next_islands: list[list[MatrixGenome]] = []
                for island_index, (population, evaluations) in enumerate(zip(islands, island_evaluations, strict=True)):
                    ranked = sorted(zip(population, evaluations), key=lambda pair: pair[1].score)
                    survivors = [genome for genome, _evaluation in ranked[: config.survivor_count]]
                    next_population: list[MatrixGenome] = [survivors[0]]
                    for survivor in survivors:
                        for _sibling in range(config.siblings_per_survivor):
                            if len(next_population) >= config.population_size:
                                break
                            parent_a = select_lexicase(population, evaluations, rng, epsilon=config.lexicase_epsilon)
                            if len(survivors) >= 2 and rng.random() < config.crossover_rate:
                                parent_b = select_lexicase(population, evaluations, rng, epsilon=config.lexicase_epsilon)
                                child = _mate(parent_a, parent_b, rng, lineage_id=f"R{restart}G{generation}I{island_index}.{len(next_population)}")
                            else:
                                child = parent_a.mutate(
                                    rng,
                                    mutation_rate=config.mutation_rate,
                                    insertion_rate=config.insertion_rate,
                                    deletion_rate=config.deletion_rate,
                                    motif_mutation_rate=config.motif_mutation_rate,
                                ).with_lineage(f"R{restart}G{generation}I{island_index}.{len(next_population)}")
                            next_population.append(child)
                    while len(next_population) < config.population_size:
                        next_population.append(
                            build_random_genome(
                                lineage_id=f"R{restart}G{generation}I{island_index}.R{len(next_population)}",
                                rng=rng,
                                length=rng.randrange(32, 64),
                            )
                        )
                    next_islands.append(clone_population(next_population))
                islands = next_islands
                continue
            break

        final_evaluations = [[evaluate_genome(genome, bundle) for genome in population] for population in islands]
        for population, evaluations in zip(islands, final_evaluations, strict=True):
            restart_best_genome, restart_best_eval = min(zip(population, evaluations), key=lambda pair: pair[1].score)
            if best_evaluation is None or restart_best_eval.score < best_evaluation.score:
                best_evaluation = restart_best_eval
                best_genome = restart_best_genome
        if best_evaluation is not None and best_evaluation.validation_error == 0.0 and best_evaluation.train_error == 0.0 and best_evaluation.symbolic_verified:
            break

    assert best_evaluation is not None
    assert best_genome is not None

    solved = best_evaluation.validation_error == 0.0 and best_evaluation.train_error == 0.0 and best_evaluation.symbolic_verified
    motif_summary: tuple[str, ...] = ()
    if solved:
        motif = extract_motif_from_ops(
            best_genome.declare_ops()[:12],
            origin_lineage=best_genome.lineage_id,
            origin_task=bundle.name,
        )
        motif_summary = (f"{motif.origin_lineage}:{'|'.join(motif.pattern)}#reuse={motif.reuse_count}",)

    report = ExperimentReport(
        experiment=experiment_name,
        train_error=best_evaluation.train_error,
        full_validation_error=best_evaluation.validation_error,
        cell_count=1,
        active_rules=best_genome.declare_ops(),
        lineage_tree=build_lineage_report(best_genome),
        motifs_per_cell=motif_summary,
        shortcut_hits=best_evaluation.shortcut_hits,
        trace_examples=best_evaluation.trace_examples,
        extra={
            "genome_codons": best_genome.signature(),
            "mul_count": best_evaluation.mul_count,
            "solved": solved,
            "symbolic_verified": best_evaluation.symbolic_verified,
            "exact_output_positions": best_evaluation.exact_output_positions,
            "island_count": config.island_count,
            "migration_interval": config.migration_interval,
            "migration_size": config.migration_size,
            "lexicase_epsilon": config.lexicase_epsilon,
            "basis_phase_generations": config.basis_phase_generations,
            "task": bundle.name,
            "seeded_reference": False,
        },
    )
    if require_perfect and not solved:
        raise SystemExit(1)
    return report
