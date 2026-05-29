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
) -> dict[str, float]:
    s1_err_a = 0.0
    s1_err_b = 0.0
    s2_err_a = 0.0
    s2_err_b = 0.0
    s3_err = 0.0
    probe_c1 = 0.0
    probe_c2 = 0.0

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

        s1_err_a += (c1 - a / 3.0) ** 2
        s1_err_b += (c2 - b / 3.0) ** 2
        s2_err_a += (c1 - b / 3.0) ** 2
        s2_err_b += (c2 - a / 3.0) ** 2
        s3_err += ((c1 + c2) / 2.0 - (a * b) / 9.0) ** 2

        if a == 2 and b == 3:
            probe_c1 = c1
            probe_c2 = c2

    n = len(SPLIT_CASES)
    return {
        "s1_a": s1_err_a / n,
        "s1_b": s1_err_b / n,
        "s2_a": s2_err_a / n,
        "s2_b": s2_err_b / n,
        "s3": s3_err / n,
        "probe_c1": probe_c1,
        "probe_c2": probe_c2,
    }


def _alpha(generation: int) -> tuple[float, str, str]:
    if generation < 100:
        return generation / 99.0, "s1", "s2"
    return (generation - 100) / 99.0, "s2", "s3"


def _blended_error(metrics: dict[str, float], role: str, alpha: float, stage_from: str, stage_to: str) -> float:
    def stage_err(stage: str) -> float:
        if stage == "s1":
            return metrics[f"s1_{role}"]
        if stage == "s2":
            return metrics[f"s2_{role}"]
        return metrics["s3"]

    return (1.0 - alpha) * stage_err(stage_from) + alpha * stage_err(stage_to)


def _score_genome(
    genome: CellGenome,
    role: str,
    partners: list[CellGenome],
    system: CooperativeChemistrySystem,
    rng: Random,
    alpha: float,
    stage_from: str,
    stage_to: str,
) -> tuple[float, dict[str, float]]:
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    errors = []
    last_metrics: dict[str, float] = {}
    for partner in sampled:
        if role == "A":
            m = _run_pair(genome, partner, system)
        else:
            m = _run_pair(partner, genome, system)
        errors.append(_blended_error(m, role.lower(), alpha, stage_from, stage_to))
        last_metrics = m
    return sum(errors) / len(errors), last_metrics


def _random_genome(rng: Random, lineage_id: str) -> CellGenome:
    length = rng.randint(6, 20)
    return CellGenome(
        codons=tuple(random_codons(rng, length)),
        local_motifs=(),
        lineage_id=lineage_id,
    )


def main() -> None:
    rng = Random(32)
    system = CooperativeChemistrySystem()

    pop_a = [_random_genome(rng, f"A{i+1}") for i in range(POPULATION_SIZE)]
    pop_b = [_random_genome(rng, f"B{i+1}") for i in range(POPULATION_SIZE)]

    best_a_lineage = pop_a[0].lineage_id
    best_b_lineage = pop_b[0].lineage_id

    for generation in range(GENERATIONS):
        alpha, stage_from, stage_to = _alpha(generation)

        scores_a = [
            (_score_genome(g, "A", pop_b, system, rng, alpha, stage_from, stage_to), g)
            for g in pop_a
        ]
        scores_a.sort(key=lambda x: x[0][0])

        scores_b = [
            (_score_genome(g, "B", pop_a, system, rng, alpha, stage_from, stage_to), g)
            for g in pop_b
        ]
        scores_b.sort(key=lambda x: x[0][0])

        gen_best_blended = float("inf")
        gen_best_metrics: dict[str, float] = {}
        for (_, _ma), ga in scores_a[:3]:
            for (_, _mb), gb in scores_b[:3]:
                m = _run_pair(ga, gb, system)
                blended = (
                    _blended_error(m, "a", alpha, stage_from, stage_to)
                    + _blended_error(m, "b", alpha, stage_from, stage_to)
                ) / 2.0
                if blended < gen_best_blended:
                    gen_best_blended = blended
                    gen_best_metrics = m
                    best_a_lineage = ga.lineage_id
                    best_b_lineage = gb.lineage_id

        mean_blended = (
            sum(s[0][0] for s in scores_a) + sum(s[0][0] for s in scores_b)
        ) / (len(scores_a) + len(scores_b))

        s1 = (gen_best_metrics.get("s1_a", 0) + gen_best_metrics.get("s1_b", 0)) / 2
        s2 = (gen_best_metrics.get("s2_a", 0) + gen_best_metrics.get("s2_b", 0)) / 2
        s3 = gen_best_metrics.get("s3", 0)
        c1 = gen_best_metrics.get("probe_c1", 0)
        c2 = gen_best_metrics.get("probe_c2", 0)

        print(
            f"epistasis_colony3: gen={generation} alpha={alpha:.3f} "
            f"stage1_err={s1:.4f} stage2_err={s2:.4f} stage3_err={s3:.4f} "
            f"blended_err={gen_best_blended:.6f} "
            f"cell1_out={c1:.4f} cell2_out={c2:.4f} "
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
                f"epistasis_colony3: snapshot gen={generation} "
                f"cell1_signals={[round(s, 3) for s in cell1.signals]} "
                f"cell2_signals={[round(s, 3) for s in cell2.signals]}"
            )

        survivors_a = [g for (_, _), g in scores_a[:SURVIVORS]]
        next_a: list[CellGenome] = list(survivors_a)
        while len(next_a) < POPULATION_SIZE:
            parent = rng.choice(survivors_a)
            child_id = f"{parent.lineage_id}.{generation+1}.{len(next_a)}"
            next_a.append(mutate_genome(parent, rng, lineage_id=child_id))
        pop_a = next_a

        survivors_b = [g for (_, _), g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            child_id = f"{parent.lineage_id}.{generation+1}.{len(next_b)}"
            next_b.append(mutate_genome(parent, rng, lineage_id=child_id))
        pop_b = next_b


if __name__ == "__main__":
    main()
