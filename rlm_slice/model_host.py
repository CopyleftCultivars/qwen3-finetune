"""In-process host for the frozen champion, shared by every controller arm.

Why in-process and not vLLM
---------------------------
LOCAL_RLM_SLICE_PLAN.md sec.3.1 allows "vLLM (recommended by the RLM
repository for local models) or a small local wrapper". This slice takes the
wrapper, for three reasons that only became visible once sec.3.4's scoring
requirement was read against the scorer's actual signature:

1. `score_with_trace(model, tokenizer, ...)` in
   `autoresearch/score_mcq_trace_ll.py` takes a live HF model object and reads
   per-letter logits through the frozen `score_symbol`. That is not expressible
   over an OpenAI HTTP surface, so Pass-2 scoring needs an in-process model no
   matter what serves Pass 1. A vLLM server would therefore be a *second* copy
   of the weights, not a replacement for the first.
2. The card is a 12 GB RTX 4070 Ti with other work resident on it. Two copies of
   the champion plus vLLM's KV preallocation is a poor trade for a 0.6B model
   scoring 113 MCQ rows, where throughput was never the bottleneck.
3. vLLM pins torch aggressively. The `qwen3-train` env (torch 2.10.0+cu128) is
   the environment that produced the champion; installing vLLM into it risks
   the training path for no measured gain.

One in-process copy also makes Pass 1 and Pass 2 provably the same weights,
which is what sec.2.3's "directly comparable to
results/mcq_trace_ll_v5_clean.json" actually requires.

`rlm_slice/server.py` puts an OpenAI-compatible HTTP surface *over this same
host* so Arm B's `rlm` LocalREPL can talk to it with backend="openai" without
loading a second model.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlm_slice.guards import DEFAULT_SEED, set_determinism

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "outputs/merged-model-p6-lr5e5-r2"
DEFAULT_DIAGNOSTICS_DIR = REPO_ROOT.parent / "benchmark-eval-workspace" / "diagnostics"


def _import_scorer():
    """Import the trace scorer without copying any of it.

    sec.2.3/2.4: reuse `score_with_trace` rather than reimplementing the
    objective, and do not modify the frozen scorers.
    """
    sys.path.insert(0, str(REPO_ROOT / "autoresearch"))
    import score_mcq_trace_ll as trace_scorer  # noqa: E402

    return trace_scorer


@dataclass
class DecodeStats:
    """Per-item telemetry required by sec.3.4.4 (latency and decode calls)."""

    decode_calls: int = 0
    forward_calls: int = 0
    generated_tokens: int = 0
    latency_ms: float = 0.0
    per_call_ms: list[float] = field(default_factory=list)

    def merge(self, other: "DecodeStats") -> None:
        self.decode_calls += other.decode_calls
        self.forward_calls += other.forward_calls
        self.generated_tokens += other.generated_tokens
        self.latency_ms += other.latency_ms
        self.per_call_ms.extend(other.per_call_ms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decode_calls": self.decode_calls,
            "forward_calls": self.forward_calls,
            "generated_tokens": self.generated_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "per_call_ms": [round(ms, 3) for ms in self.per_call_ms],
        }


class ModelHost:
    """Loads the champion once and lends it to every arm.

    Deliberately thin: generation and scoring both delegate to the frozen /
    trace-scorer helpers so that Arm A reproduces the existing baseline path
    exactly rather than approximating it.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        diagnostics_dir: Path | str = DEFAULT_DIAGNOSTICS_DIR,
        device: str = "cuda",
        seed: int = DEFAULT_SEED,
        full_precision: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.stats = DecodeStats()

        self.scorer = _import_scorer()
        helpers = self.scorer.import_diagnostics_helpers(Path(diagnostics_dir))
        self.model, self.tokenizer = helpers["load_model_and_tokenizer"](
            model_path, full_precision=full_precision, device=device
        )
        self.letter_ids = helpers["get_letter_token_ids"](self.tokenizer)
        self.score_symbol = helpers["score_symbol"]
        set_determinism(seed)

    # -- telemetry ---------------------------------------------------------

    def reset_stats(self) -> None:
        self.stats = DecodeStats()

    def take_stats(self) -> DecodeStats:
        """Return the stats accumulated since the last reset, and clear them."""
        taken = self.stats
        self.stats = DecodeStats()
        return taken

    # -- generation --------------------------------------------------------

    def generate_trace(self, item: dict[str, Any], max_trace_tokens: int = 256):
        """Pass 1 via the trace scorer's own greedy path.

        Returns (raw_generated, trace, truncated, n_new_tokens). Arm A depends on
        this being the *same* call the v5clean baseline made -- do not inline a
        reimplementation here.
        """
        start = time.perf_counter()
        result = self.scorer.generate_trace(
            self.model, self.tokenizer, item, max_trace_tokens, self.device, self.seed
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        self.stats.decode_calls += 1
        self.stats.generated_tokens += int(result[3])
        self.stats.latency_ms += elapsed
        self.stats.per_call_ms.append(elapsed)
        return result

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        enable_thinking: bool = True,
    ) -> tuple[str, int]:
        """Free-form greedy chat generation, for Arms B/C and the HTTP surface.

        Returns (text, n_new_tokens).
        """
        import torch

        torch.manual_seed(self.seed)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return self.generate_raw(prompt, max_new_tokens=max_new_tokens)

    def generate_raw(self, prompt: str, max_new_tokens: int = 256) -> tuple[str, int]:
        """Greedy continuation of an already-templated prompt string."""
        import torch

        start = time.perf_counter()
        torch.manual_seed(self.seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        elapsed = (time.perf_counter() - start) * 1000.0
        self.stats.decode_calls += 1
        self.stats.generated_tokens += int(new_ids.shape[0])
        self.stats.latency_ms += elapsed
        self.stats.per_call_ms.append(elapsed)
        return text, int(new_ids.shape[0])

    # -- scoring -----------------------------------------------------------

    def score_with_trace(self, item: dict[str, Any], trace: str) -> dict[str, float]:
        """Pass 2: option-set-normalized gold-letter LL, via the frozen objective.

        Passing trace="" is the sec.3 delta_trace ablation, exactly as
        score_mcq_trace_ll.main does it.
        """
        start = time.perf_counter()
        logprobs = self.scorer.score_with_trace(
            self.model,
            self.tokenizer,
            item,
            trace,
            self.letter_ids,
            self.score_symbol,
            self.device,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        self.stats.forward_calls += 1
        self.stats.latency_ms += elapsed
        self.stats.per_call_ms.append(elapsed)
        return logprobs


_HOST: ModelHost | None = None


def get_host(**kwargs: Any) -> ModelHost:
    """Process-wide singleton, so the HTTP surface and the arms share weights."""
    global _HOST
    if _HOST is None:
        _HOST = ModelHost(**kwargs)
    return _HOST
