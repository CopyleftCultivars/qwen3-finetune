"""Deterministic, training-only difficulty signals for curriculum experiments.

The source records do not carry a trusted difficulty annotation. This module
therefore exposes a modest authoring-time proxy rather than pretending that
item difficulty is known. The signal uses only the instruction and target text
already present in a training record; it never reads benchmark or sealed-test
data.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any


CURRICULUM_SIGNAL_VERSION = "length-structure-v1"
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_OPTION_RE = re.compile(r"(?m)^\s*[A-D]\.\s+")
_COMPLEXITY_CUES = re.compile(
    r"\b(?:because|although|unless|except|compare|contrast|first|then|which|"
    r"most|least|best|only|not|if|when|while|therefore|why|how)\b",
    re.IGNORECASE,
)


def _bounded_log_ratio(value: int, reference: int) -> float:
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(reference))


def score_pair(pair: dict[str, Any]) -> dict[str, Any]:
    """Return an explainable, bounded difficulty score for one pair."""
    instruction = str(pair.get("instruction", "")).strip()
    output = str(pair.get("output", "")).strip()
    instruction_words = _WORD_RE.findall(instruction)
    output_words = _WORD_RE.findall(output)
    total_words = len(instruction_words) + len(output_words)
    option_count = len(_OPTION_RE.findall(instruction))
    cue_count = len(_COMPLEXITY_CUES.findall(instruction))
    unique_ratio = (
        len(set(instruction_words)) / len(instruction_words)
        if instruction_words
        else 0.0
    )

    length_signal = _bounded_log_ratio(total_words, 220)
    output_signal = _bounded_log_ratio(len(output_words), 90)
    structure_signal = min(1.0, option_count / 4.0) * 0.6 + min(1.0, cue_count / 6.0) * 0.4
    ambiguity_signal = 1.0 - min(1.0, unique_ratio)
    score = 100.0 * (
        0.50 * length_signal
        + 0.20 * output_signal
        + 0.20 * structure_signal
        + 0.10 * ambiguity_signal
    )
    bucket = "easy" if score < 33.333 else "medium" if score < 66.667 else "hard"
    return {
        "signal_version": CURRICULUM_SIGNAL_VERSION,
        "difficulty_score": round(score, 6),
        "difficulty_bucket": bucket,
        "features": {
            "instruction_words": len(instruction_words),
            "output_words": len(output_words),
            "total_words": total_words,
            "option_count": option_count,
            "complexity_cue_count": cue_count,
            "instruction_unique_ratio": round(unique_ratio, 6),
        },
    }


def _tie_key(pair: dict[str, Any]) -> str:
    return hashlib.sha256(str(pair.get("instruction", "")).encode("utf-8")).hexdigest()


def order_pairs(
    pairs: list[dict[str, Any]],
    order: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Order records and return a JSON-serializable audit summary."""
    if order not in {"none", "easy-to-hard", "hard-to-easy"}:
        raise ValueError(f"unknown curriculum order: {order}")

    scored = [(pair, score_pair(pair)) for pair in pairs]
    if order == "easy-to-hard":
        scored.sort(key=lambda item: (item[1]["difficulty_score"], _tie_key(item[0])))
    elif order == "hard-to-easy":
        scored.sort(key=lambda item: (-item[1]["difficulty_score"], _tie_key(item[0])))

    counts = Counter(item[1]["difficulty_bucket"] for item in scored)
    source_counts = Counter(str(pair.get("_source", "unknown")) for pair, _ in scored)
    scores = [item[1]["difficulty_score"] for item in scored]
    report = {
        "format": "curriculum_signal_report_v1",
        "signal_version": CURRICULUM_SIGNAL_VERSION,
        "order": order,
        "item_count": len(scored),
        "bucket_counts": dict(sorted(counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_mean": round(sum(scores) / len(scores), 6) if scores else None,
        "note": (
            "Heuristic ranking from training instruction/output text only; "
            "not a validated difficulty annotation."
        ),
    }
    return [pair for pair, _ in scored], report
