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


# Analogous to exp33 but for the b*c/9 sub-task:
# cell2: receives b/3 in signals[1], must discover SENSE_PEER to read c/3 from cell3
#        then output (b/3) * s3 = b*c/9
# cell3: receives c/3 in signals[2], must echo c/3

SPLIT_CASES: tuple[tuple[int, int], ...] = tuple(
    (b, c) for b in (1, 2, 3) for c in (1, 2, 3)
)
CHEMISTRY_ROUNDS = 6
POPULATION_SIZE = 8
SURVIVORS = 3
PARTNER_SAMPLES = 3
WARMUP_GENS = 200
GATE_GENS = 400


def _make_case(b: int, c: int) -> TaskCase:
    return TaskCase(x=float(b), y=float(c), target=(b * c) / 9.0, task_name="multiply")


def _run_pair_raw(
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (c2_outs, c3_outs, c2_s3s, probe) for all 9 cases."""
    c2_outs, c3_outs, c2_s3s = [], [], []
    probe = [0.0, 0.0, 0.0, 0.0]  # c2, c3, s3, colony

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
        c2_outs.append(c2)
        c3_outs.append(c3)
        c2_s3s.append(s3)

        if b == 2 and c == 3:
            probe = [c2, c3, s3, c2 * s3]

    return c2_outs, c3_outs, c2_s3s, probe


def _gate_errors(
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> tuple[float, float, list[float]]:
    """Return (gate_err_b, echo_err_c, probe)."""
    c2_outs, c3_outs, c2_s3s, probe = _run_pair_raw(genome_b, genome_c, system)
    n = len(SPLIT_CASES)
    gate_err = sum(
        (c2 * s3 - (b * c) / 9.0) ** 2
        for (c2, s3, (b, c)) in zip(c2_outs, c2_s3s, SPLIT_CASES)
    ) / n
    echo_err = sum(
        (c3 - c / 3.0) ** 2
        for (c3, (_, c)) in zip(c3_outs, SPLIT_CASES)
    ) / n
    return gate_err, echo_err, probe


def _score_b(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    return sum(_gate_errors(genome, p, system)[0] for p in sampled) / PARTNER_SAMPLES


def _score_c(genome: CellGenome, partners: list[CellGenome], system: CooperativeChemistrySystem, rng: Random) -> float:
    sampled = [rng.choice(partners) for _ in range(PARTNER_SAMPLES)]
    return sum(_gate_errors(p, genome, system)[1] for p in sampled) / PARTNER_SAMPLES


def _random_genome(rng: Random, lineage_id: str) -> CellGenome:
    length = rng.randint(6, 20)
    return CellGenome(
        codons=tuple(random_codons(rng, length)),
        local_motifs=(),
        lineage_id=lineage_id,
    )


# ── Curriculum warm-up (mirrors exp32 pattern for b/c roles) ─────────────────

def _warmup_alpha(generation: int) -> tuple[float, str, str]:
    if generation < 100:
        return generation / 99.0, "s1", "s2"
    return (generation - 100) / 99.0, "s2", "s3"


def _warmup_run_pair(
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> dict[str, float]:
    s1_b = s1_c = s2_b = s2_c = s3_err = 0.0
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
        s1_b += (c2 - b / 3.0) ** 2   # echo own
        s1_c += (c3 - c / 3.0) ** 2   # echo own
        s2_b += (c2 - c / 3.0) ** 2   # echo peer
        s2_c += (c3 - b / 3.0) ** 2   # echo peer
        s3_err += ((c2 + c3) / 2.0 - (b * c) / 9.0) ** 2
    n = len(SPLIT_CASES)
    return {"s1_b": s1_b/n, "s1_c": s1_c/n, "s2_b": s2_b/n, "s2_c": s2_c/n, "s3": s3_err/n}


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
        m = _warmup_run_pair(genome, partner, system) if role == "B" else _warmup_run_pair(partner, genome, system)
        errors.append(_warmup_blended(m, role.lower(), alpha, sf, st))
    return sum(errors) / len(errors)


def _run_warmup(system: CooperativeChemistrySystem) -> tuple[list[CellGenome], list[CellGenome]]:
    rng = Random(36)
    pop_b = [_random_genome(rng, f"B{i+1}") for i in range(POPULATION_SIZE)]
    pop_c = [_random_genome(rng, f"C{i+1}") for i in range(POPULATION_SIZE)]
    print("epistasis_bc_gate: warmup starting (200 curriculum gens)...")

    for generation in range(WARMUP_GENS):
        alpha, sf, st = _warmup_alpha(generation)
        scores_b = sorted(
            [(_warmup_score(g, "B", pop_c, system, rng, alpha, sf, st), g) for g in pop_b],
            key=lambda x: x[0]
        )
        scores_c = sorted(
            [(_warmup_score(g, "C", pop_b, system, rng, alpha, sf, st), g) for g in pop_c],
            key=lambda x: x[0]
        )
        survivors_b = [g for _, g in scores_b[:SURVIVORS]]
        next_b: list[CellGenome] = list(survivors_b)
        while len(next_b) < POPULATION_SIZE:
            parent = rng.choice(survivors_b)
            next_b.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.{generation+1}.{len(next_b)}"))
        pop_b = next_b

        survivors_c = [g for _, g in scores_c[:SURVIVORS]]
        next_c: list[CellGenome] = list(survivors_c)
        while len(next_c) < POPULATION_SIZE:
            parent = rng.choice(survivors_c)
            next_c.append(mutate_genome(parent, rng, lineage_id=f"{parent.lineage_id}.{generation+1}.{len(next_c)}"))
        pop_c = next_c

    print("epistasis_bc_gate: warmup complete")
    return pop_b, pop_c


# ── Gate phase ────────────────────────────────────────────────────────────────

def main() -> None:
    system = CooperativeChemistrySystem()
    pop_b, pop_c = _run_warmup(system)
    rng = Random(361)

    store = MotifStore()
    motif_captured = False

    best_b_lineage = pop_b[0].lineage_id
    best_c_lineage = pop_c[0].lineage_id

    for generation in range(GATE_GENS):
        scores_b = sorted([(_score_b(g, pop_c, system, rng), g) for g in pop_b], key=lambda x: x[0])
        scores_c = sorted([(_score_c(g, pop_b, system, rng), g) for g in pop_c], key=lambda x: x[0])

        gen_best_gate = float("inf")
        probe_best = [0.0, 0.0, 0.0, 0.0]
        for _, gb in scores_b[:3]:
            for _, gc in scores_c[:3]:
                ge, _, probe = _gate_errors(gb, gc, system)
                if ge < gen_best_gate:
                    gen_best_gate = ge
                    probe_best = probe
                    best_b_lineage = gb.lineage_id
                    best_c_lineage = gc.lineage_id

        c2, c3, s3, colony = probe_best
        print(
            f"epistasis_bc_gate: gen={generation} gate_err={gen_best_gate:.6f} "
            f"cell2_out={c2:.4f} cell2_s3={s3:.4f} cell3_out={c3:.4f} colony_out={colony:.4f} "
            f"lineage_b={best_b_lineage} lineage_c={best_c_lineage}"
        )

        if not motif_captured and gen_best_gate < 0.05:
            best_b = scores_b[0][1]
            best_c = scores_c[0][1]
            probe_case = _make_case(2, 3)
            cell2 = CellState(active_rules=list(best_b.declare_rules()))
            cell2.signals = [0.0, 2 / 3.0, 0.0, 0.0, 0.0]
            cell3 = CellState(active_rules=list(best_c.declare_rules()))
            cell3.signals = [0.0, 0.0, 1.0, 0.0, 0.0]
            context = ChemistryContext()
            system.run([cell2, cell3], probe_case, context=context, max_time=float(CHEMISTRY_ROUNDS))

            motif_b = extract_motif_from_rules(
                [r for r in best_b.declare_rules() if isinstance(r, str)],
                origin_lineage=best_b.lineage_id,
                origin_task="multiply_3cell_bc",
                origin_signals=tuple(cell2.signals),
            )
            motif_c = extract_motif_from_rules(
                [r for r in best_c.declare_rules() if isinstance(r, str)],
                origin_lineage=best_c.lineage_id,
                origin_task="multiply_3cell_bc",
                origin_signals=tuple(cell3.signals),
            )
            store.save(motif_b, role="gate_bc", task="multiply_3cell_bc",
                       gate_err=gen_best_gate, generation=generation, experiment="epistasis_bc_gate")
            print(f"epistasis_bc_gate: motif_captured role=gate_bc gen={generation} gate_err={gen_best_gate:.6f}")
            store.save(motif_c, role="echo_c", task="multiply_3cell_bc",
                       gate_err=gen_best_gate, generation=generation, experiment="epistasis_bc_gate")
            print(f"epistasis_bc_gate: motif_captured role=echo_c gen={generation} gate_err={gen_best_gate:.6f}")
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
