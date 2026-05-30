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
GATE_GENS = 800


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


# ── Curriculum warm-up: 3-phase scaffold ─────────────────────────────────────
# Phase 1 (gens 0-99):   solo echo — cell2 echoes b/3, cell3 echoes c/3, NO partner runs
# Phase 2 (gens 100-149): paired echo — both cells run together, scored on own-echo
# Phase 3 (gens 150-199): blended gate — blend echo_own → gate product


def _solo_echo_score_b(genome: CellGenome, system: CooperativeChemistrySystem) -> float:
    """Score cell2 on echoing b/3 from signals[1], run solo (no partner)."""
    err = 0.0
    for b, c in SPLIT_CASES:
        case = _make_case(b, c)
        cell2 = CellState(active_rules=list(genome.declare_rules()))
        cell2.signals = [0.0, b / 3.0, 0.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell2], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c2 = cell2.output if cell2.output is not None else cell2.signals[2]
        err += (c2 - b / 3.0) ** 2
    return err / len(SPLIT_CASES)


def _solo_echo_score_c(genome: CellGenome, system: CooperativeChemistrySystem) -> float:
    """Score cell3 on echoing c/3 from signals[2], run solo (no partner)."""
    err = 0.0
    for b, c in SPLIT_CASES:
        case = _make_case(b, c)
        cell3 = CellState(active_rules=list(genome.declare_rules()))
        cell3.signals = [0.0, 0.0, c / 3.0, 0.0, 0.0]
        context = ChemistryContext()
        system.run([cell3], case, context=context, max_time=float(CHEMISTRY_ROUNDS))
        c3 = cell3.output if cell3.output is not None else cell3.signals[2]
        err += (c3 - c / 3.0) ** 2
    return err / len(SPLIT_CASES)


def _paired_run(
    genome_b: CellGenome,
    genome_c: CellGenome,
    system: CooperativeChemistrySystem,
) -> dict[str, float]:
    """Run cell2+cell3 paired; return echo_own and gate_product metrics."""
    echo_b = echo_c = gate_err = 0.0
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
        echo_b += (c2 - b / 3.0) ** 2
        echo_c += (c3 - c / 3.0) ** 2
        gate_err += (c2 * s3 - (b * c) / 9.0) ** 2
    n = len(SPLIT_CASES)
    return {"echo_b": echo_b / n, "echo_c": echo_c / n, "gate": gate_err / n}


def _run_warmup(system: CooperativeChemistrySystem) -> tuple[list[CellGenome], list[CellGenome]]:
    rng = Random(36)
    pop_b = [_random_genome(rng, f"B{i+1}") for i in range(POPULATION_SIZE)]

    # Seed cell3 population from motifs.db echo motifs if available.
    # Motif patterns are stored as string rule names; embed them as local_motifs
    # on random-codon genomes so the cell can express them via declare_rules().
    store = MotifStore()
    echo_motifs = store.query(role="echo", task="multiply_2cell", top_k=POPULATION_SIZE)
    if echo_motifs:
        pop_c: list[CellGenome] = []
        for i, motif in enumerate(echo_motifs):
            base = _random_genome(rng, f"C{i+1}_seeded")
            pop_c.append(CellGenome(
                codons=base.codons,
                local_motifs=(motif,),
                lineage_id=f"C{i+1}_seeded",
            ))
        while len(pop_c) < POPULATION_SIZE:
            pop_c.append(_random_genome(rng, f"C{len(pop_c)+1}"))
        print(f"epistasis_bc_gate: seeded {len(echo_motifs)} cell3 genomes from motifs.db echo motifs")
    else:
        pop_c = [_random_genome(rng, f"C{i+1}") for i in range(POPULATION_SIZE)]
        print("epistasis_bc_gate: no echo motifs found in motifs.db, using random cell3 genomes")

    print("epistasis_bc_gate: warmup starting (200 curriculum gens, 3-phase)...")

    for generation in range(WARMUP_GENS):
        if generation < 100:
            # Phase 1: solo echo — no partner interaction
            scores_b = sorted(
                [(_solo_echo_score_b(g, system), g) for g in pop_b],
                key=lambda x: x[0]
            )
            scores_c = sorted(
                [(_solo_echo_score_c(g, system), g) for g in pop_c],
                key=lambda x: x[0]
            )
        elif generation < 150:
            # Phase 2: paired echo — run together but score on own echo
            scores_b = sorted(
                [(sum(_paired_run(g, rng.choice(pop_c), system)["echo_b"]
                      for _ in range(PARTNER_SAMPLES)) / PARTNER_SAMPLES, g)
                 for g in pop_b],
                key=lambda x: x[0]
            )
            scores_c = sorted(
                [(sum(_paired_run(rng.choice(pop_b), g, system)["echo_c"]
                      for _ in range(PARTNER_SAMPLES)) / PARTNER_SAMPLES, g)
                 for g in pop_c],
                key=lambda x: x[0]
            )
        else:
            # Phase 3: blended gate — blend echo_own → gate_product
            alpha = (generation - 150) / 49.0  # 0→1 over gens 150-199
            scores_b = sorted(
                [(sum(
                    (1.0 - alpha) * _paired_run(g, rng.choice(pop_c), system)["echo_b"]
                    + alpha * _paired_run(g, rng.choice(pop_c), system)["gate"]
                    for _ in range(PARTNER_SAMPLES)) / PARTNER_SAMPLES, g)
                 for g in pop_b],
                key=lambda x: x[0]
            )
            scores_c = sorted(
                [(sum(
                    (1.0 - alpha) * _paired_run(rng.choice(pop_b), g, system)["echo_c"]
                    + alpha * _paired_run(rng.choice(pop_b), g, system)["gate"]
                    for _ in range(PARTNER_SAMPLES)) / PARTNER_SAMPLES, g)
                 for g in pop_c],
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
