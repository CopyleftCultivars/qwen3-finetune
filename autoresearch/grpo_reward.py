#!/usr/bin/env python3
"""
grpo_reward.py - Static (no live tool calls) reward functions for GRPO training on
Agriculture-Agent-RL-Training-Data, reimplementing the semantics of the dataset's
own info.reward_components (tool_use_correctness, content_quality,
citation_presence, ethics_redflags) without calling cultivars-mcp/PubChem live.

Why static: TRL GRPOTrainer needs a synchronous, low-latency reward_fn evaluated
thousands of times per run (group_size x steps). Live tool calls in that hot loop
introduce network flakiness/latency that would blow the compute budget
non-deterministically and make iterations non-reproducible for journal.jsonl's
audit trail. See autoresearch/configs/phase23-grpo-*/*_pipeline.md for the
decision record.

Signature matches TRL's reward_funcs convention:
    reward_fn(prompts, completions, **kwargs) -> list[float]
`**kwargs` carries other dataset columns (task, info) as parallel lists.

This is a v0 heuristic proxy, not a reimplementation of the original judge model.
Stage 0 validation (scripts/validate_grpo_reward.py) checks it does NOT reproduce
the `agy` source's self-judging inflation pattern before this ever drives training.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# Tool names observed across the dataset card's category/tool table plus tool
# names referenced inside completion text (e.g. "lookup_gene_evidence"). This is
# a living registry, not a ground-truth schema pulled from cultivars-mcp itself —
# the dataset does not embed per-row tool JSON schemas in the prompt, only a
# natural-language system message, so we validate against name membership + valid
# JSON arguments rather than an argument-level schema. Extend this list (and note
# it in a Tier 4 levers.md entry) if unknown tool names show up in practice.
KNOWN_TOOL_NAMES = frozenset(
    {
        "find_trait_genes",
        "list_trait_categories",
        "lookup_gene",
        "lookup_gene_evidence",
        "translate_trait_to_species",
        "list_maize_nam_founders",
        "lookup_compound",
        "resolve_accession",
        "list_orphan_crop_requests",
    }
)

# Shapes (info.shape) that require at least one tool call per the dataset card's
# system-prompt instructions ("call the relevant tool before answering").
TOOL_REQUIRED_SHAPES = frozenset({"advice_tool"})

ETHICS_REDFLAG_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bguaranteed\s+(cure|fix|result)\b",
        r"\b100%\s+(safe|effective|guaranteed)\b",
        r"\bno\s+side\s+effects\b",
        r"\bdefinitely\s+safe\s+to\s+(ingest|eat|consume)\b",
        r"\bprimitive\b",
        r"\bbackward(s)?\b",
        r"\bmere(ly)?\s+superstition\b",
    ]
]

CITATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"https?://\S+",
        r"\(\s*[A-Z][\w.\-]+(?:\s+et al\.)?,?\s*\d{4}\s*\)",  # (Author, 2020) / (Author et al. 2020)
        r"\baccording to\b",
        r"\bsource:\s*\S+",
        r"\btraditionally documented (by|in)\b",
    ]
]


def _assistant_turns(completion: list[dict]) -> list[dict]:
    return [t for t in completion if t.get("role") == "assistant"]


def _all_tool_calls(completion: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for turn in _assistant_turns(completion):
        calls.extend(turn.get("tool_calls") or [])
    return calls


def _final_text(completion: list[dict]) -> str:
    for turn in reversed(_assistant_turns(completion)):
        content = (turn.get("content") or "").strip()
        if content:
            return content
    return ""


def _shape_of(info: Optional[dict]) -> Optional[str]:
    return (info or {}).get("shape")


def _task_of_row(task: Optional[str], info: Optional[dict]) -> Optional[str]:
    return task or (info or {}).get("topic")


def tool_use_reward(prompts: Any, completions: list[list[dict]], task: list[Optional[str]] = None,
                     info: list[Optional[dict]] = None, **kwargs) -> list[float]:
    """1.0 = correct tool use for the shape, 0.0 = missing/malformed when required."""
    task = task or [None] * len(completions)
    info = info or [None] * len(completions)
    scores = []
    for completion, row_info in zip(completions, info):
        shape = _shape_of(row_info)
        calls = _all_tool_calls(completion)

        if shape in TOOL_REQUIRED_SHAPES:
            if not calls:
                scores.append(0.0)
                continue
            ok = 0
            for call in calls:
                fn = (call.get("function") or {})
                name = fn.get("name")
                args_raw = fn.get("arguments")
                name_ok = name in KNOWN_TOOL_NAMES
                try:
                    args_ok = isinstance(json.loads(args_raw), dict) if args_raw else False
                except (TypeError, json.JSONDecodeError):
                    args_ok = False
                if name_ok and args_ok:
                    ok += 1
            scores.append(ok / len(calls))
        else:
            # tool-optional shapes: don't penalize absence; penalize malformed calls if present
            if not calls:
                scores.append(1.0)
                continue
            ok = sum(
                1
                for call in calls
                if (call.get("function") or {}).get("name") in KNOWN_TOOL_NAMES
            )
            scores.append(ok / len(calls))
    return scores


def citation_reward(prompts: Any, completions: list[list[dict]], task: list[Optional[str]] = None,
                     info: list[Optional[dict]] = None, **kwargs) -> list[float]:
    """Only scored for indigenous_and_traditional_knowledge; 1.0 (not applicable) otherwise."""
    task = task or [None] * len(completions)
    info = info or [None] * len(completions)
    scores = []
    for completion, row_task, row_info in zip(completions, task, info):
        if _task_of_row(row_task, row_info) != "indigenous_and_traditional_knowledge":
            scores.append(1.0)
            continue
        text = _final_text(completion)
        has_citation = any(p.search(text) for p in CITATION_PATTERNS)
        scores.append(1.0 if has_citation else 0.0)
    return scores


def ethics_redflag_reward(prompts: Any, completions: list[list[dict]], **kwargs) -> list[float]:
    """1.0 = clean, 0.0 = at least one red-flag pattern matched."""
    scores = []
    for completion in completions:
        text = _final_text(completion)
        flagged = any(p.search(text) for p in ETHICS_REDFLAG_PATTERNS)
        scores.append(0.0 if flagged else 1.0)
    return scores


def content_quality_proxy_reward(prompts: Any, completions: list[list[dict]], **kwargs) -> list[float]:
    """Length/structure heuristic proxy. NOT a judge-model replacement — see module
    docstring. Real content quality is measured at eval time via the existing
    periodic rubric guard (run_freeform_evaluation.py), not here."""
    scores = []
    for completion in completions:
        text = _final_text(completion)
        if not text:
            scores.append(0.0)
            continue
        length = len(text)
        length_score = 1.0 if 40 <= length <= 4000 else max(0.0, 1.0 - abs(length - 1000) / 4000)
        words = text.split()
        unique_ratio = len(set(words)) / len(words) if words else 0.0
        repetition_score = min(1.0, unique_ratio / 0.4)  # penalize heavy repetition loops
        scores.append(0.5 * length_score + 0.5 * repetition_score)
    return scores


DEFAULT_WEIGHTS = {
    "tool_use_correctness": 0.40,
    "content_quality": 0.30,
    "citation_presence": 0.15,
    "ethics_redflags": 0.15,
}


def combined_reward(
    prompts: Any,
    completions: list[list[dict]],
    task: list[Optional[str]] = None,
    info: list[Optional[dict]] = None,
    weights: dict[str, float] = None,
    **kwargs,
) -> list[float]:
    weights = weights or DEFAULT_WEIGHTS
    tool = tool_use_reward(prompts, completions, task=task, info=info)
    content = content_quality_proxy_reward(prompts, completions)
    citation = citation_reward(prompts, completions, task=task, info=info)
    ethics = ethics_redflag_reward(prompts, completions)

    combined = []
    for t, c, ci, e in zip(tool, content, citation, ethics):
        score = (
            weights["tool_use_correctness"] * t
            + weights["content_quality"] * c
            + weights["citation_presence"] * ci
            + weights["ethics_redflags"] * e
        )
        combined.append(max(0.0, min(1.0, score)))
    return combined
