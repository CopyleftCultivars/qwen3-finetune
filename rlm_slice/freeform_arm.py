"""Free-form generation variants of arms A/B/C/R for the L7 dev rubric track (#130).

LOCAL_RLM_SLICE_PLAN.md sec.3.4.3: the deploy target is free-form question/answer,
not MCQ, so this reuses controller.py's arm shapes but drops the MCQ-specific
Pass-2 gold-letter scoring tail -- there is no letter to score here. Each arm
ends with a plain `candidate_answer` string, shaped to drop straight into
eval_freeform/score_rubric.py's `--input` contract (the row shape
autoresearch/generate_freeform_candidates.py already established).

Generation convention: enable_thinking=False throughout, matching
generate_freeform_candidates.py. Every past periodic_rubric_candidate_*.jsonl
and the codexv5 baseline this track pairs against were generated that way, so
switching to think-on here would confound "did the arm help" with "did turning
on thinking help", on top of breaking the pairing.

Arm R uses query_mode="question" only (no model-authored query): the MCQ slice
(#127/#133 findings in rlm_slice/README.md) found the model-authored query does
not beat the raw question and costs an extra decode call, so "question" is the
recommended default there and the only mode implemented here.
"""

from __future__ import annotations

from typing import Any

from rlm_slice.model_host import DecodeStats, ModelHost
from rlm_slice.retrieval import DEFAULT_K, retrieve
from rlm_slice.rlm_arm import (
    DEFAULT_BASE_URL,
    build_rlm,
    fetch_server_stats,
    repl_telemetry,
    reset_server_stats,
)

DEFAULT_MAX_NEW_TOKENS = 400

ARMS = ("A", "B", "C", "R")

# Adapted from rlm_arm.MCQ_SYSTEM_PROMPT: same REPL machinery and submission
# convention, minus the MCQ framing (no options, no letter).
FREEFORM_SYSTEM_PROMPT = """You are answering an open-ended question with the help of a Python REPL.

The variable `context` holds the question as a string. The REPL persists across turns.
Only `print(...)` output is shown back to you.
{custom_tools_section}

To run code, write exactly one block per turn using the ```repl fence:

```repl
print(context)
```

Work in short turns:
1. Print the question and read it carefully.
2. Use the REPL to note relevant facts or organize your reasoning.
3. When you are confident, submit inside a ```repl block:

```repl
answer["content"] = "<your full answer, ready to show the user>"
answer["ready"] = True
```

Always use the ```repl fence when you want code to run -- text outside a fence is
never executed."""

_CRITIQUE_PROMPT = (
    "Critique the answer above in two sentences. Name the single biggest flaw or "
    "missing consideration. Do not give a revised answer yet."
)
_REVISE_PROMPT = "Using that critique, write your corrected, complete answer."


def _question_text(item: dict[str, Any]) -> str:
    return item["question"]


# -- Arm A ------------------------------------------------------------------


def arm_a(host: ModelHost, item: dict[str, Any], max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> dict[str, Any]:
    """Bare single-pass answer: the existing candidate-generation baseline
    (autoresearch/generate_freeform_candidates.py), reused rather than
    reimplemented so this arm is directly comparable to every past rubric run
    it is paired against."""
    host.take_stats()
    text, _n = host.generate_chat(
        [{"role": "user", "content": _question_text(item)}],
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
    )
    stats = host.take_stats()
    return {"candidate_answer": text.strip(), "notes": [], **stats.as_dict()}


# -- Arm B ------------------------------------------------------------------


def arm_b(
    host: ModelHost,
    item: dict[str, Any],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    rlm_client: Any = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Depth-1 RLM LocalREPL arm, free-form variant of controller.arm_b."""
    host.take_stats()
    stats = DecodeStats()
    notes: list[str] = []

    client = rlm_client if rlm_client is not None else build_rlm(
        host.model_path, base_url=base_url, system_prompt=FREEFORM_SYSTEM_PROMPT
    )

    try:
        reset_server_stats(base_url)
    except Exception:  # noqa: BLE001 -- telemetry must not decide the run
        notes.append("server_stats_unavailable")

    answer = ""
    telemetry: dict[str, Any] = {}
    try:
        result = client.completion(_question_text(item), root_prompt=_question_text(item))
        answer = (getattr(result, "response", "") or "").strip()
        telemetry = repl_telemetry(result)
    except Exception as exc:  # noqa: BLE001 -- an arm failure must not kill the sweep
        notes.append(f"rlm_error:{type(exc).__name__}")

    if telemetry and telemetry.get("repl_blocks", 0) == 0:
        notes.append("no_repl_blocks:degenerate_arm_b")
    if telemetry and not telemetry.get("submitted_answer"):
        notes.append("no_submitted_answer:iterations_exhausted")

    if "server_stats_unavailable" not in notes:
        try:
            remote = fetch_server_stats(base_url)
            stats.decode_calls += int(remote.get("decode_calls", 0))
            stats.generated_tokens += int(remote.get("generated_tokens", 0))
            stats.latency_ms += float(remote.get("latency_ms", 0.0))
            stats.per_call_ms.extend(remote.get("per_call_ms", []))
        except Exception:  # noqa: BLE001
            notes.append("server_stats_unavailable")

    if not answer:
        notes.append("empty_rlm_response:fell_back_to_bare_answer")
        answer, _n = host.generate_chat(
            [{"role": "user", "content": _question_text(item)}],
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
        )
        answer = answer.strip()

    stats.merge(host.take_stats())
    out = {"candidate_answer": answer, "notes": notes, **stats.as_dict()}
    out.update(telemetry)
    return out


# -- Arm C ------------------------------------------------------------------


def arm_c(host: ModelHost, item: dict[str, Any], max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> dict[str, Any]:
    """Self-refine: draft, critique, revise. Free-form variant of controller.arm_c."""
    host.take_stats()
    notes: list[str] = []
    question = _question_text(item)

    draft, _n = host.generate_chat(
        [{"role": "user", "content": question}], max_new_tokens=max_new_tokens, enable_thinking=False
    )
    draft = draft.strip()

    critique, _n = host.generate_chat(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": draft},
            {"role": "user", "content": _CRITIQUE_PROMPT},
        ],
        max_new_tokens=max_new_tokens // 2,
        enable_thinking=False,
    )
    critique = critique.strip()
    if not critique:
        notes.append("empty_critique:draft_used_as_answer")
        stats = host.take_stats()
        return {"candidate_answer": draft, "notes": notes, **stats.as_dict()}

    revised, _n = host.generate_chat(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": draft},
            {"role": "user", "content": _CRITIQUE_PROMPT},
            {"role": "assistant", "content": critique},
            {"role": "user", "content": _REVISE_PROMPT},
        ],
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
    )
    revised = revised.strip()
    if not revised:
        notes.append("empty_revision:draft_used_as_answer")
        revised = draft

    stats = host.take_stats()
    return {"candidate_answer": revised, "notes": notes, **stats.as_dict()}


# -- Arm R (RAG) --------------------------------------------------------------


def arm_r(
    host: ModelHost,
    item: dict[str, Any],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    k: int = DEFAULT_K,
    retrieval_mode: str = "search",
) -> dict[str, Any]:
    """Retrieval-augmented single pass. Free-form variant of controller.arm_r,
    query_mode fixed to "question" (see module docstring)."""
    host.take_stats()
    notes: list[str] = []
    question = _question_text(item)

    result = retrieve(question, k=k, mode=retrieval_mode)
    notes.extend(result.notes)
    context = result.as_context()

    if not context:
        notes.append("no_context:degraded_to_bare_answer")
        answer, _n = host.generate_chat(
            [{"role": "user", "content": question}], max_new_tokens=max_new_tokens, enable_thinking=False
        )
        answer = answer.strip()
    else:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Reference material:\n{context}\n\n"
                    f"Using the reference material where relevant, answer:\n{question}"
                ),
            }
        ]
        answer, _n = host.generate_chat(messages, max_new_tokens=max_new_tokens, enable_thinking=False)
        answer = answer.strip()
        if not answer:
            notes.append("empty_answer:fell_back_to_bare_answer")
            answer, _n = host.generate_chat(
                [{"role": "user", "content": question}], max_new_tokens=max_new_tokens, enable_thinking=False
            )
            answer = answer.strip()

    stats = host.take_stats()
    out = {"candidate_answer": answer, "notes": notes, **stats.as_dict()}
    out.update(result.as_dict())
    out["retrieval_query_mode"] = "question"
    return out


ARM_FUNCS = {"A": arm_a, "B": arm_b, "C": arm_c, "R": arm_r}


def run_arm(arm: str, host: ModelHost, item: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    if arm not in ARM_FUNCS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_FUNCS)}")
    return ARM_FUNCS[arm](host, item, **kwargs)
