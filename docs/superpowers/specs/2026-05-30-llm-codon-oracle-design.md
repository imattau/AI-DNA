# LLM Codon Oracle Design

**Date:** 2026-05-30

## Goal

Integrate a local LLM (tinyllama via ollama) as a periodic codon proposer that expands the genome's instruction set when evolution stagnates — analogous to gene duplication and mutation in real DNA.

## Architecture

Four components wired into `experiments/37_llm_codon_oracle.py`, running the bc gate evolution (exp36 base) with LLM-driven genome expansion.

### 1. Stagnation Detector

Monitors gate_err history with a sliding window. Fires when improvement < 1% over the last 50 generations. Implemented as a simple function `is_stagnating(history: list[float], window: int = 50, threshold: float = 0.01) -> bool`.

### 2. Context Builder

Assembles a ~200-token prompt snapshot:
- Current codon names in the registry (REGULATORY_OP_NAMES)
- Last 10 gate_err values
- Best motif rule sequence (from top genome's declare_rules())
- Template menu with examples

### 3. LLM Oracle

Calls tinyllama via ollama HTTP API: `POST http://localhost:11434/api/generate`

Prompt instructs the model to return a single JSON object selecting one template + parameters. Uses `stream=false`. Timeout: 10 seconds. On failure (timeout, bad JSON, invalid template), logs and skips — no retry.

Expected response format:
```json
{"template": "SENSE_PEER_S_TO_D", "s": 2, "d": 4}
```

Template menu (constrained choices):
| Template key | Parameters | Example codon name |
|---|---|---|
| `SENSE_PEER_S_TO_D` | s: 0-4, d: 0-4, s≠d | `SENSE_PEER_2_TO_4` |
| `COPY_S_TO_D` | s: 0-4, d: 0-4, s≠d | `COPY_1_TO_4` |
| `SCALE_BY_SN` | n: 0-4 | `SCALE_BY_S4` |
| `IF_SN_GT` | n: 0-4 | `IF_S4_GT` |
| `IF_SN_LT` | n: 0-4 | `IF_S4_LT` |

### 4. Codon Injector

Validates the proposal:
- Template must be in the menu
- Parameters must be in-range integers
- Resulting codon name must not already exist in REGULATORY_OP_NAMES

On valid proposal:
- Constructs the Rule function from a template factory in `codon_oracle.py`
- Inserts into `chemistry.py`'s live rulebook dict
- Assigns next free numeric ID in REGULATORY_CODON_MAP (3 entries per codon, matching existing pattern)
- Appends name to REGULATORY_OP_NAMES (runtime mutation of the tuple → list conversion)

## New File: `codon_oracle.py`

```python
class CodonOracle:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        ...
    def is_stagnating(self, history: list[float], window: int = 50, threshold: float = 0.01) -> bool:
        ...
    def build_prompt(self, codon_names: list[str], fitness_history: list[float], best_motif: list[str]) -> str:
        ...
    def query(self, prompt: str, timeout: float = 10.0) -> dict | None:
        # POST to ollama, parse JSON, return raw dict or None
        ...
    def inject(self, proposal: dict, rulebook: dict, codon_map: dict, op_names: list[str]) -> str | None:
        # validate, build Rule, insert, return codon name or None
        ...
```

Template factory (inside `codon_oracle.py`):
```python
def _build_rule(template: str, params: dict) -> Rule:
    if template == "SENSE_PEER_S_TO_D":
        s, d = params["s"], params["d"]
        return Rule(f"SENSE_PEER_{s}_TO_{d}", lambda state, task, ctx, _s=s, _d=d: _sense_peer_cross(state, ctx, _s, _d))
    if template == "COPY_S_TO_D":
        s, d = params["s"], params["d"]
        return Rule(f"COPY_{s}_TO_{d}", lambda state, task, ctx, _s=s, _d=d: _copy_signals(state, _s, _d))
    # ... SCALE_BY_SN, IF_SN_GT, IF_SN_LT
```

The `_sense_peer_cross` and `_copy_signals` helpers are imported from `chemistry.py` (already exist or are extracted there).

## Experiment 37

`experiments/37_llm_codon_oracle.py` runs the same bc gate evolution as exp36 with:
- `oracle = CodonOracle()`
- After each generation: `if oracle.is_stagnating(gate_err_history): oracle.try_inject(...)`
- Output prefix: `llm_oracle:`
- Injection events logged: `llm_oracle: gen=150 proposed=SENSE_PEER_3_TO_4 injected=True`

## Testing

`tests/test_codon_oracle.py`:
- `test_is_stagnating_detects_plateau` — flat history triggers, improving history doesn't
- `test_inject_valid_proposal` — mock ollama response, verify codon added to rulebook + map
- `test_inject_duplicate_skipped` — proposing existing codon name returns None
- `test_inject_invalid_template_skipped` — unknown template returns None
- `test_build_prompt_contains_required_fields` — prompt includes codon names and fitness values

## Integration with test suite

Add `"37_llm_codon_oracle.py"` to `tests/test_experiments.py` scripts list with prefix `"llm_oracle:"`.

Note: the test runner will need ollama running locally. The experiment should gracefully skip LLM calls if ollama is unreachable (log warning, continue evolution without injection).

## Data flow summary

```
evolve N gens
    → stagnation check
    → [stagnating] build prompt → tinyllama → parse JSON → inject codon
    → evolution continues with expanded codon set
    → [gate_err < 0.05] capture motif to motifs.db
```
