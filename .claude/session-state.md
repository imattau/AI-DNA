# Session State Checkpoint
Generated: 2026-05-28
Reason: Context at 60% — handing off Phase B spec writing

## Execution Mode
**Mode**: interactive
**Auto-Continue**: false

## Current Task
Write the Phase B — Search Quality spec for the AI-DNA project, then save it to `docs/superpowers/specs/2026-05-28-phase-b-search-quality-design.md` and commit it.

## Progress Summary

### Completed this session
1. Full project review conducted — see review summary below
2. `tests/test_experiments.py` fixed — assert now accepts `spatial_development:` and `spatial3d_development:` prefixes (line 43)
3. Phase A spec written: `docs/superpowers/specs/2026-05-28-phase-a-metrics-design.md`
4. Phase A implementation plan written: `docs/superpowers/plans/2026-05-28-phase-a-metrics.md`
5. Session state file created (this file)

### Not yet committed
- `tests/test_experiments.py` fix
- The two spec/plan docs

## Project Context

**Repo:** `/home/mattthomson/workspace/AI-DNA`
**What it is:** DNA-inspired evolutionary computation platform. Genomes = codon sequences → chemistry VM → cells evolve under ecological pressure. 24 experiments across arithmetic, self-replication, cooperative chemistry, spatial development, routing.

**Three-phase roadmap agreed with user:**
- Phase A — Observable Metrics (spec + plan done, not yet implemented)
- Phase B — Search Quality (spec needed NOW)
- Phase C — Spatial-Ecology Integration (spec to come after B)

## Phase B Scope (what to write the spec for)

Phase B adds four search improvements to `evolution.py` and `colony.py`. The goal is to make the evolver smarter for hard targets (matrix multiply exp 09, adaptive ecology exp 14) without seeding full solutions.

### Key files to understand before writing:

**`evolution.py`** — key functions:
- `crossover_genomes(left, right, rng, lineage_id)` — currently does raw codon splice via `crossover_codons`. No motif boundary awareness.
- `select_top_diverse(evaluations, k)` — deduplicates by genome signature but no fitness sharing / crowding penalty
- `mutate_genome(...)` — fixed `mutation_rate` per call, no per-lineage adaptation
- `estimate_neutrality(genome, evaluator, rng, trials)` — exists, returns fraction of neutral mutations
- `EvolutionConfig` dataclass — holds `mutation_rate`, `synonym_rate`, etc.

**`colony.py`** — `Colony.advance()` calls `select_top_diverse` then spawns siblings. No diversity pressure beyond dedup.

**`task_stream.py`** — `StreamConfig` holds evolution params. `TaskStreamColony` manages stream-mode evolution. Archive exists.

**`genome.py`** — `CellGenome` has `local_motifs: tuple[Motif, ...]`. `Motif` has `origin_lineage`, `origin_task`. `declare_rules()` returns rule names. `signature()` returns tuple of ints.

### Four deliverables for Phase B:

1. **Fitness sharing in `select_top_diverse`** — penalise candidates whose phenotype (error score) is close to already-selected survivors. Add a `sharing_radius: float` parameter. If two candidates have `|score_a - score_b| < sharing_radius`, the second one's effective score is penalised. This keeps multiple niches alive.

2. **Motif-aware crossover in `crossover_genomes`** — instead of random codon splice, identify motif boundaries in both parents (motifs are stored as `local_motifs` on `CellGenome`) and prefer crossover points that fall between motifs rather than inside them. Preserves functional units across recombination.

3. **Adaptive mutation rate per lineage in `StreamConfig`/`task_stream.py`** — lineages with high `neutral_drift_rate` (from Phase A) get a higher mutation rate multiplier; high-performing lineages get a lower one. Add `adaptive_mutation: bool = False` flag to `StreamConfig`. When enabled, compute per-cell mutation rate before spawning offspring.

4. **Quality-Diversity (MAP-Elites style) archive in `task_stream.py`** — maintain a behavioural grid alongside the score archive. Grid axes: (error_bucket, diversity_bucket). Each cell holds the best genome for that behavioural niche. Replaces the simple energy-ranked archive. Benchmark: run exp 09 and exp 14, compare convergence speed.

### Design constraints:
- All changes backward-compatible (new params have defaults that preserve current behaviour)
- No new files needed — modify `evolution.py`, `colony.py`, `task_stream.py`
- Each improvement independently toggleable via config flags
- Benchmarks: exp 09 (`matrix_multiplication_search`) and exp 14 (`adaptive_math_ecology`) are the reference hard cases

## Continuation Instructions

1. Read `docs/superpowers/specs/2026-05-28-phase-a-metrics-design.md` for spec style reference
2. Read `evolution.py` (full) and `colony.py` (full) to ground the spec in actual code
3. Write `docs/superpowers/specs/2026-05-28-phase-b-search-quality-design.md` following the same structure as the Phase A spec
4. Run spec self-review (placeholder scan, consistency check)
5. Commit all uncommitted work: the test fix + Phase A docs + Phase B spec
6. Report back to user: "Phase B spec written at [path]. Please review before I write the implementation plan."

## Files Changed (uncommitted)
- `tests/test_experiments.py` — assert fix (line 43)
- `docs/superpowers/specs/2026-05-28-phase-a-metrics-design.md` — new
- `docs/superpowers/plans/2026-05-28-phase-a-metrics.md` — new
- `.claude/session-state.md` — this file
