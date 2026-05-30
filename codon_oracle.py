from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from chemistry import ChemistryContext, Rule, _copy_slot_impl, _sense_peer_cross_impl
from cell import CellState
from tasks import TaskCase

_TEMPLATES = {
    "SENSE_PEER_S_TO_D",
    "COPY_S_TO_D",
    "SCALE_BY_SN",
    "IF_SN_GT",
    "IF_SN_LT",
}


def _make_scale_by_sn(n: int) -> Rule:
    name = f"SCALE_BY_S{n}"

    def apply(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if n >= len(state.signals):
            return None
        scale = state.signals[n]
        if scale == 0.0:
            return None
        if state.output is not None:
            state.output = max(0.0, min(1.0, state.output * scale))
        return f"scale_by_s{n}"

    return Rule(name, apply)


def _make_if_sn(n: int, gt: bool) -> Rule:
    op = "GT" if gt else "LT"
    name = f"IF_S{n}_{op}"

    def apply(state: CellState, task: TaskCase, context: ChemistryContext) -> str | None:
        if n >= len(state.signals):
            return None
        if gt and state.signals[n] <= 0.5:
            return "skip"
        if not gt and state.signals[n] >= 0.5:
            return "skip"
        return None

    return Rule(name, apply)


class CodonOracle:
    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        self._url = ollama_url

    def is_stagnating(self, history: list[float], window: int = 50, threshold: float = 0.01) -> bool:
        if len(history) < window:
            return False
        recent = history[-window:]
        improvement = (recent[0] - recent[-1]) / (recent[0] + 1e-9)
        return improvement < threshold

    def build_prompt(
        self,
        codon_names: list[str],
        fitness_history: list[float],
        best_motif: list[str],
    ) -> str:
        history_str = ", ".join(f"{v:.4f}" for v in fitness_history[-10:])
        motif_str = " → ".join(best_motif) if best_motif else "none"
        names_str = ", ".join(codon_names[:20])
        return (
            f"You are a genome expander for an evolutionary system. "
            f"Current codons: {names_str}. "
            f"Recent fitness (lower=better): [{history_str}]. "
            f"Best circuit: {motif_str}. "
            f"Evolution is stagnating. Propose ONE new codon using exactly one of these templates: "
            f"SENSE_PEER_S_TO_D (params: s, d in 0-4), "
            f"COPY_S_TO_D (params: s, d in 0-4), "
            f"SCALE_BY_SN (param: n in 0-4), "
            f"IF_SN_GT (param: n in 0-4), "
            f"IF_SN_LT (param: n in 0-4). "
            f"Reply with ONLY a JSON object, e.g.: "
            f'{{"template": "SENSE_PEER_S_TO_D", "s": 3, "d": 4}}'
        )

    def detect_stage(self, echo_err: float, gate_err: float, cell2_s3_mean: float) -> str:
        if echo_err > 0.05:
            return "echo"
        if cell2_s3_mean < 0.1:
            return "sense"
        return "scale"

    def build_stage_prompt(
        self,
        stage: str,
        codon_names: list[str],
        fitness_history: list[float],
        best_motif: list[str],
    ) -> str:
        history_str = ", ".join(f"{v:.4f}" for v in fitness_history[-10:])
        motif_str = " → ".join(best_motif) if best_motif else "none"
        names_str = ", ".join(codon_names[:20])

        if stage == "echo":
            menu = "COPY_S_TO_D (params: s, d in 0-4, s≠d)"
            framing = "cell2 cannot yet output proportional values — scaffold signal routing within the cell"
            example = '{"template": "COPY_S_TO_D", "s": 1, "d": 2}'
        elif stage == "sense":
            menu = "SENSE_PEER_S_TO_D (params: s, d in 0-4, s≠d)"
            framing = "cell2 outputs correctly but cannot read from its peer — scaffold cross-cell sensing"
            example = '{"template": "SENSE_PEER_S_TO_D", "s": 2, "d": 3}'
        else:  # scale
            menu = "SCALE_BY_SN (param: n in 0-4), IF_SN_GT (param: n in 0-4), IF_SN_LT (param: n in 0-4)"
            framing = "cell2 reads peer but gate product needs refinement — scaffold conditional scaling"
            example = '{"template": "SCALE_BY_SN", "n": 3}'

        return (
            f"You are a genome expander for an evolutionary system. "
            f"Current codons: {names_str}. "
            f"Recent fitness (lower=better): [{history_str}]. "
            f"Best circuit: {motif_str}. "
            f"Stage: {stage} — {framing}. "
            f"Propose ONE new codon using exactly this template: {menu}. "
            f"Reply with ONLY a JSON object, e.g.: {example}"
        )

    def query(self, prompt: str, timeout: float = 10.0) -> dict[str, Any] | None:
        payload = json.dumps({
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.7, "num_predict": 128},
        }).encode()
        req = urllib.request.Request(
            f"{self._url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())
                text = body.get("response", "")
                start = text.find("{")
                end = text.rfind("}") + 1
                if start == -1 or end == 0:
                    return None
                return json.loads(text[start:end])
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return None

    def inject(
        self,
        proposal: dict[str, Any],
        rulebook: dict[str, Rule],
        codon_map: dict[int, str],
        op_names: list[str],
    ) -> str | None:
        template = proposal.get("template", "")
        if not isinstance(template, str):
            return None
        if template not in _TEMPLATES:
            return None

        # flatten {"parameters": {"s": 1, "d": 2}} → {"s": 1, "d": 2, "template": ...}
        if "parameters" in proposal and isinstance(proposal["parameters"], dict):
            flat = dict(proposal)
            flat.update(proposal["parameters"])
            proposal = flat

        try:
            if template == "SENSE_PEER_S_TO_D":
                s, d = int(proposal["s"]), int(proposal["d"])
                if not (0 <= s <= 4 and 0 <= d <= 4 and s != d):
                    return None
                name = f"SENSE_PEER_{s}_TO_{d}"
                rule = Rule(name, lambda state, task, ctx, _s=s, _d=d: _sense_peer_cross_impl(state, ctx, _s, _d))

            elif template == "COPY_S_TO_D":
                s, d = int(proposal["s"]), int(proposal["d"])
                if not (0 <= s <= 4 and 0 <= d <= 4 and s != d):
                    return None
                name = f"COPY_{s}_TO_{d}"
                rule = Rule(name, lambda state, task, ctx, _s=s, _d=d: _copy_slot_impl(state, _s, _d))

            elif template == "SCALE_BY_SN":
                n = int(proposal["n"])
                if not (0 <= n <= 4):
                    return None
                name = f"SCALE_BY_S{n}"
                rule = _make_scale_by_sn(n)

            elif template in ("IF_SN_GT", "IF_SN_LT"):
                n = int(proposal["n"])
                if not (0 <= n <= 4):
                    return None
                gt = template == "IF_SN_GT"
                name = f"IF_S{n}_{'GT' if gt else 'LT'}"
                rule = _make_if_sn(n, gt)

            else:
                return None

        except (KeyError, ValueError, TypeError):
            return None

        if name in op_names:
            return None

        rulebook[name] = rule
        op_names.append(name)
        next_id = max(codon_map.keys()) + 1 if codon_map else 300
        codon_map[next_id] = name
        codon_map[next_id + 1] = name
        codon_map[next_id + 2] = name
        return name

    def try_inject(
        self,
        generation: int,
        gate_err_history: list[float],
        codon_names: list[str],
        best_motif: list[str],
        rulebook: dict[str, Rule],
        codon_map: dict[int, str],
        op_names: list[str],
        echo_err: float = 0.5,
        gate_err: float = 0.5,
        cell2_s3_mean: float = 0.0,
    ) -> None:
        if not self.is_stagnating(gate_err_history):
            return
        stage = self.detect_stage(echo_err, gate_err, cell2_s3_mean)
        prompt = self.build_stage_prompt(stage, codon_names, gate_err_history, best_motif)
        proposal = self.query(prompt)
        if proposal is None:
            print(f"llm_oracle: gen={generation} stage={stage} ollama_unreachable_or_bad_json")
            return
        name = self.inject(proposal, rulebook, codon_map, op_names)
        print(
            f"llm_oracle: gen={generation} stage={stage} proposed={proposal} "
            f"injected={'True' if name else 'False'} codon={name}"
        )
