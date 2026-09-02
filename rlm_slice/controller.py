"""Controller arms A, B and C for the local RLM slice (L3/L4/L5, #126-#128).

LOCAL_RLM_SLICE_PLAN.md sec.3.2. Every arm has the same contract: given a
normalized MCQ item, produce a *trace string*, then score the gold letter with
the frozen Pass-2 objective via `ModelHost.score_with_trace`. Keeping the
scoring call identical across arms is what makes them comparable to each other
and to results/mcq_trace_ll_v5_clean.json (sec.2.3).

  Arm A - bare 2-pass. The current Plan A path and the slice's own baseline.
  Arm B - the `rlm` package's LocalREPL at depth 1 (see rlm_arm.py).
  Arm C - self-refine: draft, critique, revise. No tools.

Arms B and C degrade to their Arm A behaviour when the 0.6B model fails to
produce the structure they ask for (sec.6.3). That degradation is recorded per
item in `notes` rather than hidden, because "Arm B silently became Arm A on 70%
of items" is the single most likely way this comparison could mislead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from rlm_slice.model_host import DecodeStats, ModelHost
from rlm_slice.rlm_arm import (
    DEFAULT_BASE_URL,
    build_rlm,
    fetch_server_stats,
    reset_server_stats,
)

DEFAULT_MAX_TRACE_TOKENS = 256
LETTERS = ("A", "B", "C", "D")

ARMS = ("A", "B", "C")


@dataclass
class ArmResult:
    """One arm's output for one item, plus the telemetry sec.3.4.4 requires."""

    question_id: str
    arm: str
    trace: str
    truncated: bool
    logprobs: dict[str, float]
    gold: str
    top_letter: str
    option_set_gold_logprob: float
    correct: bool
    passes: int
    stats: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "arm": self.arm,
            "trace": self.trace,
            "truncated": self.truncated,
            "logprobs": self.logprobs,
            "gold": self.gold,
            "top_letter": self.top_letter,
            "option_set_gold_logprob": self.option_set_gold_logprob,
            "correct": self.correct,
            "passes": self.passes,
            "notes": self.notes,
            **self.stats,
        }


def _finalize(
    host: ModelHost,
    item: dict[str, Any],
    arm: str,
    trace: str,
    truncated: bool,
    passes: int,
    stats: DecodeStats,
    notes: list[str],
) -> ArmResult:
    """Shared Pass-2 scoring tail. Every arm ends here, unchanged."""
    logprobs = host.score_with_trace(item, trace)
    probs = host.scorer._softmax_probs(logprobs)
    gold = item["answer"]
    top_letter = max(LETTERS, key=lambda letter: logprobs[letter])
    stats.merge(host.take_stats())
    return ArmResult(
        question_id=item["question_id"],
        arm=arm,
        trace=trace,
        truncated=truncated,
        logprobs={k: round(float(v), 6) for k, v in logprobs.items()},
        gold=gold,
        top_letter=top_letter,
        option_set_gold_logprob=math.log(probs[gold]),
        correct=top_letter == gold,
        passes=passes,
        stats=stats.as_dict(),
        notes=notes,
    )


def _question_text(item: dict[str, Any]) -> str:
    return (
        f"{item['question']}\n\n"
        f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
    )


# -- Arm A ----------------------------------------------------------------


def arm_a(host: ModelHost, item: dict[str, Any], max_trace_tokens: int = DEFAULT_MAX_TRACE_TOKENS) -> ArmResult:
    """Bare 2-pass (sec.3.2.1).

    Delegates Pass 1 to the trace scorer's own `generate_trace`, so this arm is
    the existing baseline path rather than a lookalike of it.
    """
    host.take_stats()  # start from a clean counter
    stats = DecodeStats()
    _raw, trace, truncated, _n = host.generate_trace(item, max_trace_tokens=max_trace_tokens)
    return _finalize(host, item, "A", trace, truncated, passes=1, stats=stats, notes=[])


# -- Arm B ----------------------------------------------------------------

_ARM_B_INSTRUCTION = (
    "Answer this multiple-choice question. You may use the Python REPL to check "
    "or organize your reasoning before deciding. Finish with your reasoning and "
    "the single best option letter.\n\n"
)


def arm_b(
    host: ModelHost,
    item: dict[str, Any],
    max_trace_tokens: int = DEFAULT_MAX_TRACE_TOKENS,
    rlm_client: Any = None,
    base_url: str = DEFAULT_BASE_URL,
) -> ArmResult:
    """Depth-1 RLM LocalREPL arm (sec.3.2.2), driven by the real `rlm` package.

    The RLM controller reaches the champion over `rlm_slice/server.py`, so its
    decode calls land on the same weights this function then scores with. Pass a
    prebuilt `rlm_client` to avoid reconstructing the controller per item.

    Failure here is degradation, not a crash: a 0.6B model is weak at the REPL
    protocol (sec.6.3), so an empty or errored RLM run falls back to the bare
    2-pass trace and records why in `notes`.
    """
    host.take_stats()
    stats = DecodeStats()
    notes: list[str] = []

    client = rlm_client if rlm_client is not None else build_rlm(host.model_path, base_url=base_url)

    # Arm B generates inside the server process, so its decode telemetry has to
    # be read back over HTTP or it would report as free (sec.3.4.4).
    try:
        reset_server_stats(base_url)
    except Exception:  # noqa: BLE001 -- telemetry must not decide the run
        notes.append("server_stats_unavailable")

    trace = ""
    try:
        result = client.completion(
            _ARM_B_INSTRUCTION + _question_text(item),
            root_prompt=_question_text(item),
        )
        trace = (getattr(result, "response", "") or "").strip()
    except Exception as exc:  # noqa: BLE001 -- an arm failure must not kill the sweep
        notes.append(f"rlm_error:{type(exc).__name__}")

    if "server_stats_unavailable" not in notes:
        try:
            remote = fetch_server_stats(base_url)
            stats.decode_calls += int(remote.get("decode_calls", 0))
            stats.generated_tokens += int(remote.get("generated_tokens", 0))
            stats.latency_ms += float(remote.get("latency_ms", 0.0))
            stats.per_call_ms.extend(remote.get("per_call_ms", []))
        except Exception:  # noqa: BLE001
            notes.append("server_stats_unavailable")

    truncated = False
    if not trace:
        notes.append("empty_rlm_response:fell_back_to_bare_2pass")
        _raw, trace, truncated, _n = host.generate_trace(item, max_trace_tokens=max_trace_tokens)

    return _finalize(host, item, "B", trace, truncated, passes=2, stats=stats, notes=notes)


# -- Arm C ----------------------------------------------------------------

_CRITIQUE_PROMPT = (
    "Critique the reasoning above in two sentences. Name the single biggest "
    "flaw or missing consideration. Do not give a final answer."
)
_REVISE_PROMPT = "Using that critique, write your corrected reasoning."


def arm_c(
    host: ModelHost,
    item: dict[str, Any],
    max_trace_tokens: int = DEFAULT_MAX_TRACE_TOKENS,
) -> ArmResult:
    """Self-refine (sec.3.2.3): draft, critique, revise. Three passes, no tools."""
    host.take_stats()
    stats = DecodeStats()
    notes: list[str] = []
    question = _question_text(item)

    draft_raw, _ = host.generate_chat([{"role": "user", "content": question}], max_new_tokens=max_trace_tokens)
    draft, draft_truncated = host.scorer.extract_trace(draft_raw)

    critique_raw, _ = host.generate_chat(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": draft},
            {"role": "user", "content": _CRITIQUE_PROMPT},
        ],
        max_new_tokens=max_trace_tokens // 2,
        enable_thinking=False,
    )
    critique = critique_raw.strip()
    if not critique:
        # Nothing to revise against -- the draft *is* the trace. Recorded so the
        # comparison cannot mistake a degenerate Arm C for a working one.
        notes.append("empty_critique:draft_used_as_trace")
        return _finalize(host, item, "C", draft, draft_truncated, passes=2, stats=stats, notes=notes)

    revised_raw, _ = host.generate_chat(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": draft},
            {"role": "user", "content": _CRITIQUE_PROMPT},
            {"role": "assistant", "content": critique},
            {"role": "user", "content": _REVISE_PROMPT},
        ],
        max_new_tokens=max_trace_tokens,
    )
    revised, truncated = host.scorer.extract_trace(revised_raw)
    if not revised.strip():
        notes.append("empty_revision:draft_used_as_trace")
        revised, truncated = draft, draft_truncated
    return _finalize(host, item, "C", revised, truncated, passes=3, stats=stats, notes=notes)


ARM_FUNCS = {"A": arm_a, "B": arm_b, "C": arm_c}


def run_arm(arm: str, host: ModelHost, item: dict[str, Any], **kwargs: Any) -> ArmResult:
    if arm not in ARM_FUNCS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_FUNCS)}")
    return ARM_FUNCS[arm](host, item, **kwargs)
