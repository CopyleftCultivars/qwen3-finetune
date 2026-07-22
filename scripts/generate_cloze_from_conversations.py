#!/usr/bin/env python3
"""Build deterministic fill-in-the-blank SFT pairs from clean chat data.

This is intentionally a conservative, local transformation rather than an LLM
authoring step. It masks one domain phrase or measurable value in a short
assistant sentence and asks the model to recover that exact text. The source
instruction and answer are retained as audit metadata; prepare_dataset_v3.py
strips that metadata before writing train/validation JSONL.

The generated file is *unverified* until it passes
``scripts/check_cloze_contamination.py``. Training should consume only the
audited output of that command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


_DOMAIN_TERMS = (
    "natural farming",
    "organic matter",
    "water retention",
    "soil health",
    "soil structure",
    "microbial activity",
    "beneficial insects",
    "cover crops",
    "root development",
    "nutrient availability",
    "crop rotation",
    "drip irrigation",
    "foliar spray",
    "leaf mold",
    "dynamic accumulator",
    "nitrogen",
    "phosphorus",
    "potassium",
    "calcium",
    "magnesium",
    "sulfur",
    "compost",
    "mulch",
    "drainage",
    "erosion",
    "germination",
    "photosynthesis",
    "fermentation",
    "bioaccumulator",
    "permaculture",
    "agroecology",
    "IMO-4",
    "IMO4",
    "LABs",
    "FPJ",
    "JLF",
    "JMS",
    "FAA",
    "KNF",
    "Jadam",
)

_NUMBER_WITH_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|kg|g|mg|l|ml|liters?|litres?|"
    r"gallons?|weeks?|days?|months?|years?|hours?|minutes?|sq\s*ft|"
    r"square feet?)\b",
    re.IGNORECASE,
)
_RATIO = re.compile(r"\b\d+\s*:\s*\d+\b")
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?](?:\s|$)|$)", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_WORD = re.compile(r"[\w][\w'/-]*", re.UNICODE)
_SYSTEM_PROMPT_MARKERS = (
    "You are a Natural Farming",
    "You are an AI assistant",
    "You are a helpful",
)


def _iter_pairs(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            turns = record.get("conversations", [])
            pairs: list[tuple[str, str]] = []
            for index in range(len(turns) - 1):
                prompt, answer = turns[index], turns[index + 1]
                if prompt.get("from") not in ("human", "user"):
                    continue
                if answer.get("from") not in ("gpt", "assistant"):
                    continue
                instruction = str(prompt.get("value", "")).strip()
                output = str(answer.get("value", "")).strip()
                if instruction and output:
                    pairs.append((instruction, output))

            # Match prepare_dataset_v3.py: the first human turn is a system-like
            # formatting prompt, not a user question and should never become
            # cloze context.
            start = 1 if pairs and any(
                pairs[0][0].startswith(marker) for marker in _SYSTEM_PROMPT_MARKERS
            ) else 0
            for instruction, output in pairs[start:]:
                yield line_number, {"instruction": instruction, "output": output}


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE.finditer(text) if match.group(0).strip()]


def _candidate_spans(sentence: str) -> list[tuple[int, int, str]]:
    """Return conservative spans, longest/highest-information first."""

    def is_quoted(start: int) -> bool:
        # Avoid masking document titles and quoted labels. These are usually
        # provenance noise rather than useful agricultural facts.
        prefix = sentence[:start]
        return prefix.count('"') % 2 == 1 or prefix.count("'") % 2 == 1

    candidates: list[tuple[int, int, str, int]] = []
    patterns = [
        (re.compile(re.escape(term), re.IGNORECASE), 100 + len(term))
        for term in sorted(_DOMAIN_TERMS, key=len, reverse=True)
    ]
    patterns.extend([(_NUMBER_WITH_UNIT, 140), (_RATIO, 145)])
    for pattern, priority in patterns:
        for match in pattern.finditer(sentence):
            value = match.group(0).strip()
            if is_quoted(match.start()):
                continue
            if len(value) < 4 or len(value) > 80:
                continue
            if not _WORD.search(value):
                continue
            candidates.append((match.start(), match.end(), value, priority))

    # Prefer longer, more specific spans; break ties by earlier appearance.
    candidates.sort(key=lambda item: (-item[3], -(item[1] - item[0]), item[0]))
    selected: list[tuple[int, int, str]] = []
    for start, end, value, _priority in candidates:
        if any(start < other_end and end > other_start for other_start, other_end, _ in selected):
            continue
        selected.append((start, end, value))
    return selected


def make_cloze_pair(
    instruction: str,
    output: str,
    source_line: int,
    minimum_answer_chars: int = 8,
    maximum_passage_chars: int = 700,
) -> dict[str, str] | None:
    """Create one deterministic cloze pair or return ``None`` if unsuitable."""
    if _THINK_BLOCK.search(output) or re.search(r"<(?:thinking|output)>", output, re.IGNORECASE):
        return None

    sentence_options: list[tuple[int, str, list[tuple[int, int, str]]]] = []
    for index, sentence in enumerate(_sentences(output)):
        if len(sentence) < 45 or len(sentence) > maximum_passage_chars:
            continue
        spans = [
            span for span in _candidate_spans(sentence)
            if len(span[2]) >= minimum_answer_chars
        ]
        if spans:
            sentence_options.append((index, sentence, spans))
    if not sentence_options:
        return None

    # Stable choice: first suitable sentence and its most specific span.
    sentence_index, sentence, spans = sentence_options[0]
    start, end, answer = spans[0]
    masked = sentence[:start] + "[BLANK]" + sentence[end:]
    source_id = hashlib.sha256(
        f"{instruction}\n\n{output}".encode("utf-8")
    ).hexdigest()[:16]
    prompt = (
        "Complete the missing phrase in this natural-farming answer. "
        "Return only the missing text.\n\n"
        f"Question:\n{instruction}\n\n"
        f"Answer passage:\n{masked}"
    )
    return {
        "instruction": prompt,
        "output": answer,
        "_source": "chat_cloze",
        "_source_record_id": source_id,
        "_source_line": str(source_line),
        "_source_instruction": instruction,
        "_source_answer": output,
        "_cloze_sentence_index": str(sentence_index),
        "_cloze_answer": answer,
        "_contamination_status": "unverified",
    }


def build_items(
    input_path: Path,
    maximum_items: int | None,
    seed: int,
    minimum_answer_chars: int,
    maximum_passage_chars: int,
    maximum_answer_frequency: int,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen_source_ids: set[str] = set()
    for line_number, pair in _iter_pairs(input_path):
        item = make_cloze_pair(
            pair["instruction"],
            pair["output"],
            line_number,
            minimum_answer_chars=minimum_answer_chars,
            maximum_passage_chars=maximum_passage_chars,
        )
        if item is None or item["_source_record_id"] in seen_source_ids:
            continue
        seen_source_ids.add(item["_source_record_id"])
        candidates.append(item)

    candidates.sort(key=lambda item: item["_source_record_id"])
    if maximum_answer_frequency > 0:
        answer_counts: Counter[str] = Counter()
        diverse: list[dict[str, str]] = []
        for item in candidates:
            answer_key = re.sub(r"\s+", " ", item["output"].strip().lower())
            if answer_counts[answer_key] >= maximum_answer_frequency:
                continue
            answer_counts[answer_key] += 1
            diverse.append(item)
        candidates = diverse
    if maximum_items is not None and maximum_items < len(candidates):
        rng = random.Random(seed)
        candidates = sorted(
            rng.sample(candidates, maximum_items),
            key=lambda item: item["_source_record_id"],
        )
    for index, item in enumerate(candidates, start=1):
        item["_cloze_id"] = f"chat-cloze-{index:06d}"
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, metavar="PATH")
    parser.add_argument("--output", required=True, metavar="PATH")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-answer-chars", type=int, default=8)
    parser.add_argument("--maximum-passage-chars", type=int, default=700)
    parser.add_argument(
        "--maximum-answer-frequency",
        type=int,
        default=50,
        help="Cap repeated normalized answer phrases; 0 disables the cap",
    )
    args = parser.parse_args()
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be >= 1")
    if args.minimum_answer_chars < 4:
        parser.error("--minimum-answer-chars must be >= 4")
    if args.maximum_answer_frequency < 0:
        parser.error("--maximum-answer-frequency must be >= 0")

    items = build_items(
        Path(args.input),
        args.max_items,
        args.seed,
        args.minimum_answer_chars,
        args.maximum_passage_chars,
        args.maximum_answer_frequency,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(items)} unverified cloze items → {output_path}")


if __name__ == "__main__":
    main()
