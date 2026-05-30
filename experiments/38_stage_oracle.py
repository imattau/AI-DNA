from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from random import Random
from cell import CellState
from chemistry import ChemistryContext, build_rulebook
from codons import REGULATORY_CODON_MAP, REGULATORY_OP_NAMES
from cooperative_chemistry import CooperativeChemistrySystem
from codon_oracle import CodonOracle
from evolution import mutate_genome
from genome import CellGenome, extract_motif_from_rules
from codons import random_codons
from tasks import TaskCase
from motif_store import MotifStore

SPLIT_CASES: tuple[tuple[int, int], ...] = tuple(
    (b, c) for b in (1, 2, 3) for c in (1, 2, 3)
)
CHEMISTRY_ROUNDS = 6
POPULATION_SIZE = 8
SURVIVORS = 3
PARTNER_SAMPLES = 3
GATE_GENS = 600
ORACLE_WINDOW = 50


def _make_case(b: int, c: int) -> TaskCase:
    return TaskCase(x=float(b), y=float(c), target=(b * c) / 9.0, task_name="multiply")


def _run_pair(
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[float, float, float, list[float]]:
    """Return (gate_err, echo_err_b, cell2_s3_mean, probe) over all 9 cases."""
    gate_errs, echo_errs, s3s = [], [], []
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
        c3 = cell3.output if cell3.output is not None else cell3.signals[2]
        s3 = max(0.0, min(1.0, cell2.signals[3]))
        gate_errs.append((c2 * s3 - (b * c) / 9.0) ** 2)
        echo_errs.append((c2 - b / 3.0) ** 2)
        s3s.append(s3)
        if b == 2 and c == 3:
            probe = [c2, c3, s3, c2 * s3]
    n = len(SPLIT_CASES)
    return sum(gate_errs) / n, sum(echo_errs) / n, sum(s3s) / n, probe


def _score_b(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    return sum(_run_pair(genome, rng.choice(partners), system)[0] for _ in range(PARTNER_SAMPLES)) / PARTNER_SAMPLES


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
    rng = Random(38)
    oracle = CodonOracle()

    live_rulebook = system.rulebook if hasattr(system, 'rulebook') else build_rulebook()
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

        gen_best_gate = float("inf")
        gen_best_echo = 0.5
        gen_best_s3 = 0.0
        probe_best = [0.0, 0.0, 0.0, 0.0]
        for _, gb in scores_b[:3]:
            for _, gc in scores_c[:3]:
                ge, ee, s3m, probe = _run_pair(gb, gc, system)
                if ge < gen_best_gate:
                    gen_best_gate = ge
                    gen_best_echo = ee
                    gen_best_s3 = s3m
                    probe_best = probe

        gate_err_history.append(gen_best_gate)
        c2, c3, s3, colony = probe_best
        stage = oracle.detect_stage(gen_best_echo, gen_best_gate, gen_best_s3)
        print(
            f"stage_oracle: gen={generation} stage={stage} gate_err={gen_best_gate:.6f} "
            f"echo_err={gen_best_echo:.6f} cell2_s3={gen_best_s3:.4f} "
            f"cell2_out={c2:.4f} colony_out={colony:.4f}"
        )

        best_motif = [str(r) for r in scores_b[0][1].declare_rules() if isinstance(r, str)]
        oracle.try_inject(
            generation, gate_err_history,
            live_op_names, best_motif,
            live_rulebook, live_codon_map, live_op_names,
            echo_err=gen_best_echo,
            gate_err=gen_best_gate,
            cell2_s3_mean=gen_best_s3,
        )

        if not motif_captured and gen_best_gate < 0.05:
            store.save(
                extract_motif_from_rules(
                    [r for r in scores_b[0][1].declare_rules() if isinstance(r, str)],
                    origin_lineage=scores_b[0][1].lineage_id,
                    origin_task="multiply_3cell_bc",
                    origin_signals=(0.0, 2/3.0, 0.0, 0.0, 0.0),
                ),
                role="gate_bc", task="multiply_3cell_bc",
                gate_err=gen_best_gate, generation=generation, experiment="stage_oracle",
            )
            print(f"stage_oracle: motif_captured gen={generation} gate_err={gen_best_gate:.6f}")
            motif_captured = True

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
