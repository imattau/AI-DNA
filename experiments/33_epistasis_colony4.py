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
from genome import CellGenome, extract_motif_from_rules
from codons import random_codons
from tasks import TaskCase
from motif_store import MotifStore


SPLIT_CASES: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in (1, 2, 3) for b in (1, 2, 3)
)
CHEMISTRY_ROUNDS = 6
POPULATION_SIZE = 8
SURVIVORS = 3
PARTNER_SAMPLES = 3
WARMUP_GENS = 200
GATE_GENS = 400


def _make_case(a: int, b: int) -> TaskCase:
    return TaskCase(x=float(a), y=float(b), target=(a * b) / 9.0, task_name="multiply")


def _run_pair_raw(
    genome_a: CellGenome,
    genome_b: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (c1_outs, c2_outs, c1_s3s, probe) for all 9 cases."""
    c1_outs, c2_outs, c1_s3s = [], [], []
    probe = [0.0, 0.0, 0.0, 0.0]  # c1, c2, s3, colony

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
        s3 = max(0.0, min(1.0, cell1.signals[3]))
        c1_outs.append(c1)
        c2_outs.append(c2)
        c1_s3s.append(s3)

        if a == 2 and b == 3:
            probe = [c1, c2, s3, c1 * s3]

    return c1_outs, c2_outs, c1_s3s, probe


def _gate_errors(
    genome_a: CellGenome,
    genome_b: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[float, float, list[float]]:
    """Return (gate_err_a, echo_err_b, probe)."""
    c1_outs, c2_outs, c1_s3s, probe = _run_pair_raw(genome_a, genome_b, system)
    n = len(SPLIT_CASES)
    gate_err = sum(
        (c1 * s3 - (a * b) / 9.0) ** 2
        for (c1, s3, (a, b)) in zip(c1_outs, c1_s3s, SPLIT_CASES)
    ) / n
    echo_err = sum(
        (c2 - b / 3.0) ** 2
        for (c2, (_, b)) in zip(c2_outs, SPLIT_CASES)
    ) / n
    return gate_err, echo_err, probe


def _score_a(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    return sum(_gate_errors(genome, p, system)[0] for p in sampled) / PARTNER_SAMPLES


def _score_b(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    return sum(_gate_errors(p, genome, system)[1] for p in sampled) / PARTNER_SAMPLES


def _random_genome(rng: Random, lineage_id: str) -> CellGenome:
    length = rng.randint(6, 20)
    return CellGenome(
        codons=tuple(random_codons(rng, length)),
        local_motifs=(),
        lineage_id=lineage_id,
    )


# ── Curriculum warm-up (experiment 32 logic, same seed) ──────────────────────

def _warmup_alpha(generation: int) -> tuple[float, str, str]:
    if generation < 100:
        return generation / 99.0, "s1", "s2"
    return (generation - 100) / 99.0, "s2", "s3"


def _warmup_run_pair(
    genome_a: CellGenome,
    genome_b: CellGenome,
    system: CooperativeChemistrySystem,
) -> dict[str, float]:
    s1_a = s1_b = s2_a = s2_b = s3_err = 0.0
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
        s1_a += (c1 - a / 3.0) ** 2
        s1_b += (c2 - b / 3.0) ** 2
        s2_a += (c1 - b / 3.0) ** 2
        s2_b += (c2 - a / 3.0) ** 2
        s3_err += ((c1 + c2) / 2.0 - (a * b) / 9.0) ** 2
    n = len(SPLIT_CASES)
    return {"s1_a": s1_a/n, "s1_b": s1_b/n, "s2_a": s2_a/n, "s2_b": s2_b/n, "s3": s3_err/n}


def _warmup_blended(metrics: dict[str, float], role: str, alpha: float, sf: str, st: str) -> float:
    def e(s: str) -> float:
        return metrics.get(f"{s}_{role}" if s != "s3" else "s3", 0.0)
    return (1.0 - alpha) * e(sf) + alpha * e(st)


def _warmup_score(genome: CellGenome, role: str, partners: list[CellGenome],
                  system: CooperativeChemistrySystem, rng: Random,
                  alpha: float, sf: str, st: str) -> float:
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    errors = []
    for partner in sampled:
        m = _warmup_run_pair(genome, partner, system) if role == "A" else _warmup_run_pair(partner, genome, system)
        errors.append(_warmup_blended(m, role.lower(), alpha, sf, st))
    return sum(errors) / len(errors)


def _run_warmup(system: CooperativeChemistrySystem) -> tuple[list[CellGenome], list[CellGenome]]:
    """Re-run experiment 32 curriculum with seed=32 to recover final populations."""
    rng = Random(32)
    pop_a = [_random_genome(rng, f"A{i+1}") for i in range(POPULATION_SIZE)]
    pop_b = [_random_genome(rng, f"B{i+1}") for i in range(POPULATION_SIZE)]
    print("epistasis_colony4: warmup starting (200 curriculum gens)...")

    for generation in range(WARMUP_GENS):
        alpha, sf, st = _warmup_alpha(generation)
        scores_a = sorted(
            [(_warmup_score(g, "A", pop_b, system, rng, alpha, sf, st), g) for g in pop_a],
            key=lambda x: x[0]
        )
        scores_b = sorted(
            [(_warmup_score(g, "B", pop_a, system, rng, alpha, sf, st), g) for g in pop_b],
            key=lambda x: x[0]
        )
        survivors_a = [g for _, g in scores_a[:SURVIVORS]]
        next_a: list[CellGenome] = list(survivors_a)
        while len(next_a) < POPULATION_SIZE:
            parent = rng.choice(survivors_a)
            next_a.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.{generation+1}.{len(next_a)}"))
        pop_a = next_a

        survivors_b = [g for _, g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            next_b.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.{generation+1}.{len(next_b)}"))
        pop_b = next_b

    print("epistasis_colony4: warmup complete")
    return pop_a, pop_b


# ── Gate phase ────────────────────────────────────────────────────────────────

def main() -> None:
    system = CooperativeChemistrySystem()
    pop_a, pop_b = _run_warmup(system)
    rng = Random(33)
    store = MotifStore()
    motif_captured = False

    best_a_lineage = pop_a[0].lineage_id
    best_b_lineage = pop_b[0].lineage_id

    for generation in range(GATE_GENS):
        scores_a = sorted([(_score_a(g, pop_b, system, rng), g) for g in pop_a], key=lambda x: x[0])
        scores_b = sorted([(_score_b(g, pop_a, system, rng), g) for g in pop_b], key=lambda x: x[0])

        gen_best_gate = float("inf")
        probe_best = [0.0, 0.0, 0.0, 0.0]
        for _, ga in scores_a[:3]:
            for _, gb in scores_b[:3]:
                ge, _, probe = _gate_errors(ga, gb, system)
                if ge < gen_best_gate:
                    gen_best_gate = ge
                    probe_best = probe
                    best_a_lineage = ga.lineage_id
                    best_b_lineage = gb.lineage_id

        mean_gate = sum(s for s, _ in scores_a) / len(scores_a)
        c1, c2, s3, colony = probe_best

        print(
            f"epistasis_colony4: gen={generation} gate_err={gen_best_gate:.6f} "
            f"cell1_out={c1:.4f} cell1_s3={s3:.4f} cell2_out={c2:.4f} colony_out={colony:.4f} "
            f"lineage_a={best_a_lineage} lineage_b={best_b_lineage}"
        )

        if not motif_captured and gen_best_gate < 0.05:
            best_a = scores_a[0][1]
            best_b = scores_b[0][1]
            probe_case = _make_case(2, 3)
            cell1 = CellState(active_rules=list(best_a.declare_rules()))
            cell1.signals = [2 / 3.0, 0.0, 0.0, 0.0, 0.0]
            cell2 = CellState(active_rules=list(best_b.declare_rules()))
            cell2.signals = [0.0, 1.0, 0.0, 0.0, 0.0]
            context = ChemistryContext()
            system.run([cell1, cell2], probe_case, context=context, max_time=float(CHEMISTRY_ROUNDS))
            motif_a = extract_motif_from_rules(
                [r for r in best_a.declare_rules() if isinstance(r, str)],
                origin_lineage=best_a.lineage_id,
                origin_task="multiply_2cell",
                origin_signals=tuple(cell1.signals),
            )
            motif_b = extract_motif_from_rules(
                [r for r in best_b.declare_rules() if isinstance(r, str)],
                origin_lineage=best_b.lineage_id,
                origin_task="multiply_2cell",
                origin_signals=tuple(cell2.signals),
            )
            store.save(motif_a, role="gate", task="multiply_2cell",
                       gate_err=gen_best_gate, generation=generation, experiment="epistasis_colony4")
            print(f"epistasis_colony4: motif_captured role=gate gen={generation} gate_err={gen_best_gate:.6f}")
            store.save(motif_b, role="echo", task="multiply_2cell",
                       gate_err=gen_best_gate, generation=generation, experiment="epistasis_colony4")
            print(f"epistasis_colony4: motif_captured role=echo gen={generation} gate_err={gen_best_gate:.6f}")
            motif_captured = True

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
                f"epistasis_colony4: snapshot gen={generation} "
                f"cell1_signals={[round(s, 3) for s in cell1.signals]} "
                f"cell2_signals={[round(s, 3) for s in cell2.signals]}"
            )

        survivors_a = [g for _, g in scores_a[:SURVIVORS]]
        next_a: list[CellGenome] = list(survivors_a)
        while len(next_a) < POPULATION_SIZE:
            parent = rng.choice(survivors_a)
            next_a.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation+1}.{len(next_a)}"))
        pop_a = next_a

        survivors_b = [g for _, g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            next_b.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.g{generation+1}.{len(next_b)}"))
        pop_b = next_b


if __name__ == "__main__":
    main()
