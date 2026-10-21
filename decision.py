"""
Decision-model backends for the PoC, all behind one interface:

    .choice(instruction, state, options) -> Decision     # a typed, probabilistic pick
    .predicate(instruction, state)       -> Decision     # yes/no -> P(yes)

Three interchangeable backends return the same `Decision`, so they are drop-in swappable:

- `SystemOne`    -- a homemade "decision model" in ~60 lines: take any instruction-tuned
                    causal LM, build a prompt that ends where the answer starts, run ONE
                    forward pass, and softmax the logits of the option labels only. No free
                    text is generated (never a type error) -- but it is still an LLM underneath.
- `Decision3`    -- a REAL open-weight decision model: vLLM Semantic Router's Decision 3.0
                    `d3-edge` (0.59B, Apache-2.0, post-trained from Qwen3.5-0.8B).
                    `model.system_one(...)` answers choice / yes-no / score in one forward
                    pass with 0 output tokens -- the authentic version of the mechanism. Runs
                    on the GPU (MPS) automatically; ~110 ms/call on MPS vs ~2.7 s on CPU, so
                    MPS is essential. trust_remote_code runs code shipped in the model repo
                    (d3_format.py / d3_runtime.py / modeling_d3.py); first run downloads ~1.2 GB.
- `DecisionsAPI` -- the same interface over a HOSTED decision model (the documented OpenAI
                    Decisions API wire shape: POST /v1/decisions on gpt-6-luna, $0.10 / 1M
                    input, output free; set OPENAI_API_KEY). A TypeSafe/Jev or
                    Microsoft-Decision-1 variant is a few lines different (POST /v1/systemone,
                    a `state` + `questions` map) and is noted at the bottom of this file.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Decision:
    choice: str
    probabilities: dict[str, float]
    latency_ms: float
    prompt_tokens: int
    backend: str = "local"
    raw: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.choice]


def _render(state) -> str:
    """The PoC passes a dict or a string; the hosted/d3 `state` is a string."""
    if isinstance(state, str):
        return state
    return "\n\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in state.items())


# --------------------------------------------------------------------------- #
#  SystemOne -- a homemade decision model from any instruction-tuned causal LM
# --------------------------------------------------------------------------- #
class SystemOne:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str | None = None):
        self.model_id = model_id
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16 if self.device != "cpu" else torch.float32)
        self.model.to(self.device).eval()
        # Token ids for " A", " B", ... and "A", "B", ... (both spellings, we take the max).
        self._label_ids = {
            L: sorted({self.tok.encode(L, add_special_tokens=False)[0], self.tok.encode(" " + L, add_special_tokens=False)[-1]})
            for L in LABELS
        }

    def _prompt(self, instruction: str, state: dict | str, options: list[str]) -> str:
        state_txt = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)
        menu = "\n".join(f"{LABELS[i]}. {o}" for i, o in enumerate(options))
        user = (
            f"{instruction}\n\n"
            f"<state>\n{state_txt}\n</state>\n\n"
            f"Options:\n{menu}\n\n"
            "Answer with the letter of the single best option and nothing else."
        )
        msgs = [
            {"role": "system", "content": "You are a decision engine. You only ever answer with one option letter."},
            {"role": "user", "content": user},
        ]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer: "

    @torch.inference_mode()
    def choice(self, instruction: str, state: dict | str, options: list[str]) -> Decision:
        assert 2 <= len(options) <= len(LABELS)
        prompt = self._prompt(instruction, state, options)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        logits = self.model(**ids).logits[0, -1].float()          # ONE forward pass, last position only
        latency = (time.perf_counter() - t0) * 1000
        option_logits = torch.stack([logits[self._label_ids[LABELS[i]]].max() for i in range(len(options))])
        probs = torch.softmax(option_logits, dim=0).tolist()      # restricted to the option tokens
        table = {o: round(p, 4) for o, p in zip(options, probs)}
        best = max(table, key=table.get)
        return Decision(best, table, round(latency, 1), ids["input_ids"].shape[1], backend=f"local:{self.model_id}")

    def predicate(self, instruction: str, state: dict | str) -> Decision:
        """Yes/No question -> P(yes)."""
        return self.choice(instruction, state, ["yes", "no"])


# --------------------------------------------------------------------------- #
#  Decision3 -- the real open-weight decision model, vllm-sr/d3-edge
# --------------------------------------------------------------------------- #
def _parse(answer: dict, options: list[str]) -> tuple[str, dict]:
    """Pull (chosen_option, {option: prob}) out of one question's answer. Defensive:
    the served /v1/systemone shape is {choice, probabilities, confidence}; the local
    call should match, but field names aren't fully documented."""
    probs = answer.get("probabilities") or answer.get("probs") or answer.get("scores") or {}
    probs = {k: float(v) for k, v in probs.items()}
    best = answer.get("choice") or answer.get("answer")
    if best is None and probs:
        best = max(probs, key=probs.get)
    if best is None:
        best = options[0]
    if not probs:                                   # last resort: all weight on `best`
        probs = {o: (1.0 if o == best else 0.0) for o in options}
    return best, {o: round(probs.get(o, 0.0), 4) for o in options}


class Decision3:
    model_id = "vllm-sr/d3-edge"
    device = "local"

    def __init__(self, model_id: str = "vllm-sr/d3-edge", device: str | None = None):
        self.model_id = model_id
        # MUST run on the GPU (MPS): on CPU the reference kernels are ~2.7s/call; on
        # MPS it is ~107ms (25x faster, and faster than the Qwen-1.5B hack).
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        # trust_remote_code: the decision head + system_one() live in the model repo
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        print(f"[d3-edge on {self.device}]", file=sys.stderr)

    def choice(self, instruction: str, state, options: list[str]) -> Decision:
        assert 2 <= len(options) <= 255
        questions = {"verdict": {
            "type": "choice",
            "instructions": instruction,
            "criteria": {o: None for o in options},     # option name -> description (None = just the name)
        }}
        t0 = time.perf_counter()
        out = self.model.system_one(state=_render(state), questions=questions)
        latency = (time.perf_counter() - t0) * 1000
        ans = out["answers"]["verdict"]
        best, probs = _parse(ans, options)
        in_tokens = int(out.get("usage", {}).get("input_tokens", 0))
        return Decision(best, probs, round(latency, 1), in_tokens, backend=f"d3:{self.model_id}", raw=ans)

    def predicate(self, instruction: str, state) -> Decision:
        """Yes/No -> P(yes). (d3 has a native `noul` type; we keep the choice form
        so the interface matches SystemOne exactly.)"""
        return self.choice(instruction, state, ["yes", "no"])


# --------------------------------------------------------------------------- #
#  DecisionsAPI -- the same interface over a HOSTED decision model
# --------------------------------------------------------------------------- #
class DecisionsAPI:
    model_id = "openai:gpt-6-luna (Decisions API)"
    device = "api"

    def __init__(self, model: str = "gpt-6-luna"):
        import httpx
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("set OPENAI_API_KEY to use the hosted DecisionsAPI backend")
        self.model = model
        self.client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )

    def choice(self, instruction: str, state, options: list[str]) -> Decision:
        body = {
            "model": self.model,
            "input": _render(state),
            "questions": [{
                "type": "choice",
                "name": "verdict",
                "instructions": instruction,
                "choices": [{"value": o} for o in options],
            }],
        }
        t0 = time.perf_counter()
        r = self.client.post("/decisions", json=body)
        latency = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        ans = r.json()["answers"][0]
        probs = {p["value"]: p["probability"] for p in ans["probabilities"]}
        return Decision(ans["choice"], probs, round(latency, 1),
                        r.json()["usage"]["input_tokens"], backend=self.model_id)

    def choice_image(self, instruction: str, png_bytes: bytes, options: list[str]) -> Decision:
        """Vision decision: send a screenshot + a choice question (documented image shape).
        Used by the optional NES adapter so the real Decisions API can 'see' the frame."""
        b64 = base64.b64encode(png_bytes).decode()
        body = {
            "model": self.model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                ],
            }],
            "questions": [{
                "type": "choice", "name": "action", "instructions": instruction,
                "choices": [{"value": o} for o in options],
            }],
        }
        t0 = time.perf_counter()
        r = self.client.post("/decisions", json=body)
        latency = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        ans = r.json()["answers"][0]
        probs = {p["value"]: p["probability"] for p in ans["probabilities"]}
        return Decision(ans["choice"], probs, round(latency, 1),
                        r.json()["usage"]["input_tokens"], backend=self.model_id + "+img")


# --- TypeSafe Jev / Microsoft-Decision-1 (same idea, different wire shape) ---
# POST https://api.typesafe.ai/v1/systemone   (or <foundry>/providers/microsoft/v1/systemone)
# { "model": "jev-latest", "state": {...},
#   "questions": { "verdict": { "type": "choice", "instructions": "...",
#                               "criteria": { "safe": None, "injection": None } } } }
# answer: resp["answers"]["verdict"]  ->  {"choice": ..., "probabilities": {...}, "confidence": ...}


if __name__ == "__main__":
    m = SystemOne(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct")
    print(m.choice("Which city is the capital of Japan?", {"hint": "none"}, ["Osaka", "Tokyo", "Kyoto"]))
    print(m.predicate("Is this text spam?", {"sms": "WIN a FREE cruise!!! Reply YES now to claim your prize"}))
