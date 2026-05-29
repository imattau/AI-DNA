from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from random import Random
from cell import CellState
from chemistry import ChemistryContext
from cooperative_chemistry import CooperativeChemistrySystem
from evolution import mutate_genome
from genome import CellGenome
from codons import random_codons
from tasks import TaskCase


SPLIT_CASES: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in (1, 2, 3) for b in (1, 2, 3)
)
CHEMISTRY_ROUNDS = 6
POPULATION_SIZE = 8
SURVIVORS = 3
PARTNER_SAMPLES = 3
GENERATIONS = 200


def _make_case(a: int, b: int) -> TaskCase:
    return TaskCase(x=float(a), y=float(b), target=(a * b) / 9.0, task_name="multiply")


def _run_pair(
    genome_a: CellGenome,
    genome_b: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[float, float, float, float, float]:
    """Return (mean_mse, probe_c1_out, probe_c2_out, probe_c1_s3, probe_c2_s3)."""
    total_error = 0.0
    probe_c1 = 0.0
    probe_c2 = 0.0
    probe_c1_s3 = 0.0
    probe_c2_s3 = 0.0

    for a, b in SPLIT_CASES:
        case = _make_case(a, b)
        cell1 = CellState(active_rules=list(genome_a.declare_rules()))
        cell1.signals = [a / 3.0, 0.0, 0.0, 0.0, 0.0]
        cell2 = CellState(active_rules=list(genome_b.declare_rules()))
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell1, cell2], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c1 = cell1.output if cell1.output is not None else cell1.signals[2]
        c2 = cell2.output if cell2.output is not None else cell2.signals[2]
        colony = (c1 + c2) / 2.0
        total_error += (colony - case.target) ** 2
        if a == 2 and b == 3:
            probe_c1 = c1
            probe_c2 = c2
            probe_c1_s3 = cell1.signals[3]
            probe_c2_s3 = cell2.signals[3]

    return total_error / len(SPLIT_CASES), probe_c1, probe_c2, probe_c1_s3, probe_c2_s3


def _score_genome(
    genome: CellGenome,
    role: str,
    partners: list[CellGenome],
    system: CooperativeChemistrySystem,
    rng: Random,
) -> float:
    """Mean colony MSE across PARTNER_SAMPLES random partners."""
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    errors = []
    for partner in sampled:
        if role == "A":
            err, *_ = _run_pair(genome, partner, system)
        else:
            err, *_ = _run_pair(partner, genome, system)
        errors.append(err)
    return sum(errors) / len(errors)


def _random_genome(rng: Random, lineage_id: str) -> CellGenome:
    length = rng.randint(6, 20)
    return CellGenome(
        codons=tuple(random_codons(rng, length)),
        local_motifs=(),
        lineage_id=lineage_id,
    )


def main() -> None:
    rng = Random(31)
    system = CooperativeChemistrySystem()
    pop_a = [_random_genome(rng, f"A{i+1}") for i in range(POPULATION_SIZE)]
    pop_b = [_random_genome(rng, f"B{i+1}") for i in range(POPULATION_SIZE)]

    best_a_lineage = pop_a[0].lineage_id
    best_b_lineage = pop_b[0].lineage_id

    for generation in range(GENERATIONS):
        # Score population A
        scores_a = [
            (_score_genome(g, "A", pop_b, system, rng), g)
            for g in pop_a
        ]
        scores_a.sort(key=lambda x: x[0])

        # Score population B
        scores_b = [
            (_score_genome(g, "B", pop_a, system, rng), g)
            for g in pop_b
        ]
        scores_b.sort(key=lambda x: x[0])

        # Find best pair this generation
        gen_best_error = float("inf")
        gen_best_c1 = 0.0
        gen_best_c2 = 0.0
        for _, ga in scores_a[:3]:
            for _, gb in scores_b[:3]:
                pair_err, c1, c2, _, _ = _run_pair(ga, gb, system)
                if pair_err < gen_best_error:
                    gen_best_error = pair_err
                    gen_best_c1 = c1
                    gen_best_c2 = c2
                    best_a_lineage = ga.lineage_id
                    best_b_lineage = gb.lineage_id

        mean_error = (
            sum(s[0] for s in scores_a) + sum(s[0] for s in scores_b)
        ) / (len(scores_a) + len(scores_b))

        print(
            f"epistasis_colony2: gen={generation} "
            f"mean_error={mean_error:.6f} best_error={gen_best_error:.6f} "
            f"cell1_out={gen_best_c1:.4f} cell2_out={gen_best_c2:.4f} "
            f"lineage_a={best_a_lineage} lineage_b={best_b_lineage}"
        )

        if generation % 20 == 0:
            best_a = scores_a[0][1]
            best_b = scores_b[0][1]
            probe_case = _make_case(2, 3)
            cell1 = CellState(active_rules=list(best_a.declare_rules()))
            cell1.signals = [2 / 3.0, 0.0, 0.0, 0.0, 0.0]
            cell2 = CellState(active_rules=list(best_b.declare_rules()))
            cell2.signals = [0.0, 1.0, 0.0, 0.0, 0.0]
            context = ChemistryContext()
            system.run([cell1, cell2], probe_case, context=context, max_time=float(CHEMISTRY_ROUNDS))
            print(
                f"epistasis_colony2: snapshot gen={generation} "
                f"cell1_signals={[round(s, 3) for s in cell1.signals]} "
                f"cell2_signals={[round(s, 3) for s in cell2.signals]}"
            )

        # Reproduce population A
        survivors_a = [g for _, g in scores_a[:SURVIVORS]]
        next_a: list[CellGenome] = list(survivors_a)
        while len(next_a) < POPULATION_SIZE:
            parent = rng.choice(survivors_a)
            child_id = f"{parent.lineage_id}.{generation+1}.{len(next_a)}"
            next_a.append(mutate_genome(parent, rng, lineage_id=child_id))
        pop_a = next_a

        # Reproduce population B
        survivors_b = [g for _, g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            child_id = f"{parent.lineage_id}.{generation+1}.{len(next_b)}"
            next_b.append(mutate_genome(parent, rng, lineage_id=child_id))
        pop_b = next_b


if __name__ == "__main__":
    main()
