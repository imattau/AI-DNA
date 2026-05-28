# AI-DNA

AI-DNA is a DNA-inspired research playground for evolving small programs, colonies of cells, and task-adaptive chemistries.

The project is built around a simple idea:

- genomes are encoded as codons
- codons declare rules or opcodes
- rules are executed in a chemistry layer
- cells survive, reproduce, and compete under environmental pressure
- lineages carry motifs, drift through neutral variation, and adapt across changing tasks

This is not a biologically literal simulator. It is a search system that borrows the most useful parts of DNA, evolution, and ecology to explore program space.

## What This Project Is For

The codebase is a testbed for:

- evolving arithmetic programs
- running changing task curricula
- studying codon degeneracy and neutral drift
- tracking motif inheritance and lineage transfer
- simulating energy-based survival and reproduction
- experimenting with self-replication and cooperative chemistry
- searching for matrix-multiplication and other math programs without seeding a full solution

The current emphasis is not on a single toy result. It is on building a reusable platform for long-running adaptive search.

## Quick Start

1. Clone the repo and enter it.
2. Use Python 3.10+.
3. Run the test suite:

```bash
python3 -m pytest -q
```

4. Run the main baseline experiment:

```bash
python3 experiments/01_multiply_persistent_rules.py
```

5. Try the changing-environment runs:

```bash
python3 experiments/12_math_task_ecology.py
python3 experiments/14_adaptive_math_ecology.py
```

6. Explore the special-purpose demos:

```bash
python3 experiments/07_self_replication.py
python3 experiments/13_cooperative_chemistry.py
python3 experiments/09_matrix_multiplication_search.py
```

## Core Model

The current architecture has these layers:

1. **Codons**
   - integer encoded genome substrate
   - many codons can map to the same semantic rule
   - synonymous mutation creates neutral or near-neutral drift

2. **Genomes**
   - a genome is a codon sequence
   - genomes carry lineage IDs and local motifs
   - genomes declare rules, but do not execute them directly

3. **Chemistry**
   - rules run across reaction rounds
   - signals are updated over time
   - cells can emit, receive, copy, inhibit, decay, threshold, and output signals

4. **Cells**
   - a cell has a genome, active rules, signals, output, and trace
   - cells can persist across episodes
   - cells can cooperate through shared chemistry contexts

5. **Colony and evolution**
   - cells reproduce under maintenance cost and resource pressure
   - survivors spawn mutated offspring and occasional immigrants
   - lineage history and motif inheritance are tracked

6. **Task ecology**
   - the environment changes over time
   - tasks can be contextual, noisy, rescaled, or resource constrained
   - adaptive curricula select harder or easier tasks based on recent performance

7. **Verification**
   - experiments report train error, validation error, traces, motifs, lineage trees, and neutrality estimates
   - some benchmarks include exact or symbolic verification

## Repository Layout

Key files:

- `codons.py`
  - codon table, synonym mapping, mutation helpers
- `genome.py`
  - genome model, motifs, formatting, motif statistics
- `cell.py`
  - cell state used by the chemistry VM
- `chemistry.py`
  - chemistry system, rulebook, signal updates, traces
- `colony.py`
  - colony-level survival and reproduction
- `evolution.py`
  - mutation, crossover, founder populations, neutrality estimates
- `tasks.py`
  - task bundles, anti-shortcut cases, contextual tasks
- `task_stream.py`
  - changing task streams, adaptive ecology, archive and lineage tracking
- `math_ecology.py`
  - task family generation and adaptive curriculum selection
- `replication.py`
  - VM-level self-replication experiments
- `matrix_experiment.py`
  - matrix-program synthesis benchmark with symbolic verification
- `cooperative_chemistry.py`
  - shared-context multi-cell signaling arena
- `tracing.py`
  - trace formatting and experiment report helpers
- `experiments/`
  - runnable experiment entrypoints
- `tests/`
  - regression coverage for the major systems
- `dna_chem_vm/`
  - compatibility package that re-exports the main modules

## Main Concepts

### Codon degeneracy

Many different codons can encode the same rule. That gives the search neutral space to drift through instead of breaking on every mutation.

### Local motifs

Successful short patterns are stored as lineage-local motifs. They are inherited by offspring and tracked in reports. There is no global motif library that injects solutions back into the population.

### Energy-based ecology

Cells pay maintenance cost. Reproduction costs energy. Environmental resources refill over time. Cells die when they cannot sustain themselves.

### Changing environments

The system is designed to run across task streams rather than a single fixed objective. Task changes can be abrupt or contextual.

### Cooperation

Cells can communicate through shared chemistry. This is a small step toward multicellular behavior, but still grounded in the same genome-to-chemistry model.

### Self-replication

The repository includes a VM-level replication path that can build offspring genomes internally using replication instructions.

### Verification

Some experiments are empirical, some are exact, and some are symbolic. The matrix benchmark includes a verifier so candidate programs can be checked against the true algebraic target.

## Running the Experiments

All experiments are plain Python entrypoints.

### Basic arithmetic benchmark

```bash
python3 experiments/01_multiply_persistent_rules.py
```

### Other single-task arithmetic runs

```bash
python3 experiments/02_max_persistent_rules.py
python3 experiments/03_abs_persistent_rules.py
python3 experiments/04_conditional_persistent_rules.py
python3 experiments/05_exponentiation_persistent_rules.py
```

### Task-stream and ecology experiments

```bash
python3 experiments/06_task_stream_adaptation.py
python3 experiments/08_rotating_selective_niches.py
python3 experiments/11_contextual_task_stream.py
python3 experiments/12_math_task_ecology.py
python3 experiments/14_adaptive_math_ecology.py
```

### Self-replication experiments

```bash
python3 experiments/07_self_replication.py
python3 experiments/10_evolve_self_replication.py
```

### Cooperative chemistry

```bash
python3 experiments/13_cooperative_chemistry.py
```

### Matrix synthesis benchmark

```bash
python3 experiments/09_matrix_multiplication_search.py
```

### Spatial development and routing

```bash
python3 experiments/15_spatial_development.py
python3 experiments/16_spatial3d_development.py
python3 experiments/17_spatial_body_plan_search.py
python3 experiments/18_spatial_matrix_fabric.py
python3 experiments/19_spatial_matrix_fabric_stream.py
python3 experiments/20_spatial_self_repair.py
python3 experiments/21_spatial_matrix_fabric_solve.py
python3 experiments/22_spatial_roaming.py
python3 experiments/23_spatial_adhesion.py
python3 experiments/24_spatial_routing.py
```

## Testing

Run the test suite with:

```bash
python3 -m pytest -q
```

Useful focused tests:

```bash
python3 -m pytest -q tests/test_codons.py
python3 -m pytest -q tests/test_tasks.py
python3 -m pytest -q tests/test_motifs.py
python3 -m pytest -q tests/test_chemistry_and_colony.py
python3 -m pytest -q tests/test_task_stream.py
python3 -m pytest -q tests/test_contextual_task_stream.py
python3 -m pytest -q tests/test_math_ecology.py
python3 -m pytest -q tests/test_matrix_experiment.py
python3 -m pytest -q tests/test_replication.py
python3 -m pytest -q tests/test_cooperative_chemistry.py
python3 -m pytest -q tests/test_spatial.py
python3 -m pytest -q tests/test_spatial3d.py
python3 -m pytest -q tests/test_spatial_body_plan.py
python3 -m pytest -q tests/test_spatial_matrix_fabric.py
python3 -m pytest -q tests/test_spatial_matrix_fabric_solve.py
python3 -m pytest -q tests/test_spatial_matrix_fabric_stream.py
python3 -m pytest -q tests/test_spatial_routing.py
python3 -m pytest -q tests/test_spatial_self_repair.py
```

## Import Surface

The `dna_chem_vm` package provides a compatibility layer that re-exports the main pieces of the system:

- codon utilities
- chemistry and cell state
- colony and evolution helpers
- genome and motif types
- replication helpers
- task definitions

That means the package can be used as a small library as well as a script-driven research project.

## What the Current System Can Do

- solve the basic multiply benchmark exactly
- run changing task streams
- track archive snapshots and lineage trees
- estimate neutrality
- preserve and report motifs
- run a VM-level replication demo
- evolve a self-replication candidate
- run a cooperative signaling arena
- search matrix programs under a symbolic verifier
- run a broader math ecology with changing contexts

## What It Does Not Yet Do

- it does not provide a general mathematical reasoning engine
- it does not use a stochastic Gillespie-style chemistry simulator
- it does not have a full developmental biology model
- it does not have a global motif memory, by design
- it does not guarantee discovery of minimal matrix-multiplication circuits on demand

## Test Results And Implications

This section records the current measured outcomes from the regression suite and the most informative experiment runs.

### Measured Regression Results

| Test block | Actual result | What it suggests |
| --- | --- | --- |
| `tests/test_codons.py` + `tests/test_tasks.py` + `tests/test_motifs.py` + `tests/test_chemistry_and_colony.py` | `12 passed in 7.94s` | The codon substrate, task bundles, motif bookkeeping, and chemistry/colony plumbing are internally consistent. |
| `tests/test_task_stream.py` + `tests/test_contextual_task_stream.py` + `tests/test_math_ecology.py` + `tests/test_replication.py` + `tests/test_cooperative_chemistry.py` | `7 passed in 65.69s` | The system can survive changing tasks, contextual noise, replication, and shared chemistries in the same overall architecture. |
| `tests/test_matrix_experiment.py -k 'not script'` | `3 passed, 1 deselected in 0.40s` | The matrix synthesis core and symbolic verifier are functioning without relying on the slower smoke-style script path. |
| `tests/test_spatial.py` + `tests/test_spatial3d.py` + `tests/test_spatial_routing.py` | `15 passed in 1.06s` | The spatial substrate now supports development, roaming, adhesion, and signal routing in both 2D and 3D. |
| `tests/test_spatial_body_plan.py` | `3 passed in 0.87s` | The 3D body-plan benchmark is exact and reproducible. |
| `tests/test_spatial_self_repair.py` | `3 passed in 0.30s` | Local repair/regeneration works as an internal spatial behavior. |
| `tests/test_spatial_matrix_fabric.py` | `3 passed in 2.95s` | The spatial fabric benchmark scaffold and scoring are valid. |
| `tests/test_spatial_matrix_fabric_solve.py` | `1 passed in 0.21s` | The matrix-fabric solve harness executes cleanly even though the benchmark itself is still partial. |
| `tests/test_spatial_matrix_fabric_stream.py` | `1 passed in 2.75s` | The changing spatial-task stream path is operational. |

### Representative Experiment Outputs

| Experiment | Actual outcome | Implication |
| --- | --- | --- |
| `experiments/01_multiply_persistent_rules.py` | exact multiply benchmark reaches zero validation error | The persistent-rule chemistry can solve a small arithmetic task exactly. |
| `experiments/09_matrix_multiplication_search.py` | exploratory matrix search remains partial but runs without a seeded full solution | The platform is genuinely searching algorithm space rather than replaying an answer. |
| `experiments/10_evolve_self_replication.py` | exact replication is reached by evolution | Self-replication can emerge through search. |
| `experiments/13_cooperative_chemistry.py` | one cell sends and another receives in a shared context | Cells can coordinate through a common chemical environment. |
| `experiments/17_spatial_body_plan_search.py` | exact body plan match | A single genome can grow a precise 3D spatial arrangement. |
| `experiments/20_spatial_self_repair.py` | a removed neighbor is regenerated | The spatial substrate has internal repair, not just passive robustness. |
| `experiments/23_spatial_adhesion.py` | signal-gated clustering occurs | Cells can join locally when the cue is present, without genome fusion. |
| `experiments/24_spatial_routing.py` | exact routing solution | The spatial substrate can already solve a simple cooperative transport task end to end. |

### What The Results Mean

- The project now has a **single genetic substrate** but multiple phenotypes: chemistry, ecology, replication, cooperation, and spatial development.
- The tests show the model is not only a toy arithmetic solver. It can also:
  - adapt across changing tasks
  - replicate
  - cooperate
  - repair
  - develop spatial structure
  - route signals through space
- The hardest open problems are now clearly visible:
  - general mathematical reasoning
  - exact spatial matrix fabrication
  - richer spatial neighborhoods and developmental regulation

### Focused Regression Coverage

The repository also includes focused tests for the main subsystems:

- `tests/test_codons.py`
- `tests/test_motifs.py`
- `tests/test_task_stream.py`
- `tests/test_contextual_task_stream.py`
- `tests/test_math_ecology.py`
- `tests/test_matrix_experiment.py`
- `tests/test_replication.py`
- `tests/test_cooperative_chemistry.py`
- `tests/test_spatial.py`
- `tests/test_spatial3d.py`
- `tests/test_spatial_body_plan.py`
- `tests/test_spatial_matrix_fabric.py`
- `tests/test_spatial_matrix_fabric_solve.py`
- `tests/test_spatial_matrix_fabric_stream.py`
- `tests/test_spatial_routing.py`
- `tests/test_spatial_self_repair.py`

## Design Principles

The implementation is intentionally biased toward the following principles:

- **Neutrality over brittleness**
  - synonymous codons and drift are useful
- **Local inheritance over global memory**
  - motifs should spread by lineage, not by manager-side injection
- **Selection over shortcuts**
  - useful lineages should survive because they work, not because they are preserved unchanged
- **Ecology over fixed caps**
  - survival should depend on resources and maintenance cost
- **Context over static evaluation**
  - a genome should be judged across changing tasks and environments
- **Verification over guesswork**
  - when a candidate is close enough, check it exactly

## Current Status

The project is beyond the stub stage and now includes:

- codon degeneracy and neutral mutation
- local motif inheritance and reuse tracking
- energy-based reproduction and survival
- contextual task streams
- archive, lineage, and transfer metrics
- cooperative chemistry
- self-replication
- adaptive math ecology
- a matrix benchmark with symbolic verification
- a set of spatial developmental benchmarks
- exact spatial routing
- local adhesion and repair

The remaining work is mostly about making the search better, the environments richer, and the spatial organisms more capable, not about making the core system exist.

## Roadmap

The most valuable remaining extensions are:

1. richer epigenetic or regulatory state
2. more spatially local cell interactions
3. stronger repair and robustness mechanisms
4. broader math ecologies
5. better search pressure without seeding full solutions
6. more informative lineage and forgetting metrics
7. richer spatial neighborhoods and task-linked multicell coordination

## Notes on the Checklist

The file `DNA_IMPLEMENTATION_CHECKLIST.md` is a working implementation map, not the main project document.

This README is meant to be the user-facing overview that explains:

- what the project is
- how the pieces fit together
- how to run the experiments
- what is already working
- where the remaining gaps are

## License

See [`LICENSE`](LICENSE).
