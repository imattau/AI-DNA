from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from random import Random
from cell import CellState
from chemistry import ChemistryContext, build_rulebook
from codons import REGULATORY_CODON_MAP, REGULATORY_OP_NAMES, random_codons
from cooperative_chemistry import CooperativeChemistrySystem
from codon_oracle import CodonOracle
from evolution import mutate_genome
from genome import CellGenome
from tasks import TaskCase
from motif_store import MotifStore

SPLIT_CASES: tuple[tuple[int, int], ...] = tuple(
    (b, c) for b in (1, 2, 3) for c in (1, 2, 3)
)
CHEMISTRY_ROUNDS = 6
POPULATION_SIZE = 4
SURVIVORS = 2
PARTNER_SAMPLES = 1
GATE_GENS = 3  # short for smoke-test; full research run would use 600+


def _make_case(b: int, c: int) -> TaskCase:
    return TaskCase(x=float(b), y=float(c), target=(b * c) / 9.0, task_name="multiply")


def _gate_errors(
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[float, list[float]]:
    c2_outs, c2_s3s = [], []
    probe = [0.0, 0.0, 0.0, 0.0]
    for b, c in SPLIT_CASES:
        case = _make_case(b, c)
        cell2 = CellState(active_rules=list(genome_b.declare_rules()))
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        cell3 = CellState(active_rules=list(genome_c.declare_rules()))
        cell3.signals = [0.0, 0.0, c / 3.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell2, cell3], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c2 = cell2.output if cell2.output is not None else cell2.signals[2]
        s3 = max(0.0, min(1.0, cell2.signals[3]))
        c2_outs.append(c2)
        c2_s3s.append(s3)
        if b == 2 and c == 3:
            c3 = cell3.output if cell3.output is not None else cell3.signals[2]
            probe = [c2, c3, s3, c2 * s3]
    n = len(SPLIT_CASES)
    gate_err = sum(
        (c2 * s3 - (b * c) / 9.0) ** 2
        for c2, s3, (b, c) in zip(c2_outs, c2_s3s, SPLIT_CASES)
    ) / n
    return gate_err, probe


def _score_b(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    return sum(_gate_errors(genome, rng.choice(partners), system)[0] for _ in range(PARTNER_SAMPLES)) / PARTNER_SAMPLES


def _score_c(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    errors = []
    for _ in range(PARTNER_SAMPLES):
        partner = rng.choice(partners)
        case = _make_case(2, 3)
        cell2 = CellState(active_rules=list(partner.declare_rules()))
        cell2.signals = [0.0, 2 / 3.0, 0.0, 0.0, 0.0]
        cell3 = CellState(active_rules=list(genome.declare_rules()))
        cell3.signals = [0.0, 0.0, 1.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell2, cell3], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c3 = cell3.output if cell3.output is not None else cell3.signals[2]
        errors.append((c3 - 1.0) ** 2)
    return sum(errors) / len(errors)


def _random_genome(rng: Random, lineage_id: str) -> CellGenome:
    return CellGenome(codons=tuple(random_codons(rng, rng.randint(6, 20))), local_motifs=(), lineage_id=lineage_id)


def main() -> None:
    system = CooperativeChemistrySystem()
    rng = Random(37)
    oracle = CodonOracle()

    # Use the live rulebook from the chemistry system so injected codons take effect
    live_rulebook = system.chemistry.rulebook
    live_codon_map = dict(REGULATORY_CODON_MAP)
    live_op_names = list(REGULATORY_OP_NAMES)

    pop_b = [_random_genome(rng, f"B{i+1}") for i in range(POPULATION_SIZE)]
    pop_c = [_random_genome(rng, f"C{i+1}") for i in range(POPULATION_SIZE)]

    store = MotifStore()
    motif_captured = False
    gate_err_history: list[float] = []

    for generation in range(GATE_GENS):
        scores_b = sorted([(_score_b(g, pop_c, system, rng), g) for g in pop_b], key=lambda x: x[0])
        scores_c = sorted([(_score_c(g, pop_b, system, rng), g) for g in pop_c], key=lambda x: x[0])

        gen_best = float("inf")
        probe_best = [0.0, 0.0, 0.0, 0.0]
        for _, gb in scores_b[:3]:
            for _, gc in scores_c[:3]:
                ge, probe = _gate_errors(gb, gc, system)
                if ge < gen_best:
                    gen_best = ge
                    probe_best = probe

        gate_err_history.append(gen_best)
        c2, c3, s3, colony = probe_best
        print(
            f"llm_oracle: gen={generation} gate_err={gen_best:.6f} "
            f"cell2_out={c2:.4f} cell2_s3={s3:.4f} cell3_out={c3:.4f} colony_out={colony:.4f}"
        )

        best_motif = [str(r) for r in scores_b[0][1].declare_rules() if isinstance(r, str)]
        oracle.try_inject(
            generation, gate_err_history,
            live_op_names, best_motif,
            live_rulebook, live_codon_map, live_op_names,
        )

        survivors_b = [g for _, g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            next_b.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation+1}.{len(next_b)}"))
        pop_b = next_b

        survivors_c = [g for _, g in scores_c[:SURVIVORS]]
        next_c: list[CellGenome] = list(survivors_c)
        while len(next_c) < POPULATION_SIZE:
            parent = rng.choice(survivors_c)
            next_c.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation+1}.{len(next_c)}"))
        pop_c = next_c


if __name__ == "__main__":
    main()
