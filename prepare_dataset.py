#!/usr/bin/env python3
"""prepare_dataset.py — Build Qwen3 fine-tune training data from two sources:

  1. Gemma4 conversation dataset (ShareGPT format) — multi-turn NF conversations
  2. MCQ benchmark JSONL — 4 cyclic option shifts per question, think-block output format

Output: data/train.jsonl and data/val.jsonl in {"instruction", "output"} format
compatible with qwen3-finetune/train.py.

*** CONTAMINATED SOURCE WARNING (see MCQ_DIAGNOSTIC_PLAN.md, 2026-07-09) ***
The original usage example for --mcq pointed directly at
benchmark-eval-workspace/benchmark_combined.jsonl — the eval benchmark itself. Running this
script that way trains the model on 100% of the eval set (each question expanded into 5
variants via cyclic option shifts), which is why it must never be repeated: --mcq must point
at an MCQ source that has been verified to have zero question-text overlap with
benchmark_combined.jsonl. This script now refuses to run against a --mcq path whose filename
matches known benchmark filenames (see _refuse_benchmark_mcq below); this is a filename-based
safety net, not a substitute for an actual dedup check against the real benchmark question set.

Usage:
    python qwen3-finetune/prepare_dataset.py \\
        --conversations "$CONVERSATIONS_DIR" \\
        --mcq <path to a verified non-benchmark-overlapping MCQ source> \\
        --output-dir qwen3-finetune/data \\
        --split 0.9 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Generator

_OPTION_LETTERS = ("A", "B", "C", "D")
_SYSTEM_PROMPT_MARKERS = (
    "You are a Natural Farming",
    "You are an AI assistant",
    "You are a helpful",
)


# ---------------------------------------------------------------------------
# Source 1 — ShareGPT conversations
# ---------------------------------------------------------------------------

def _is_system_prompt(text: str) -> bool:
    return any(text.strip().startswith(m) for m in _SYSTEM_PROMPT_MARKERS)


def _load_conversations(path: Path) -> Generator[dict[str, str], None, None]:
    """Flatten ShareGPT conversations to instruction/output pairs.

    Skips the first human/gpt exchange when it is a system-prompt handshake
    (the human turn contains the system prompt and the gpt turn is an acknowledgment).
    Each remaining human→gpt turn pair becomes one instruction/output record.
    """
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc

            turns = rec.get("conversations", [])
            # Pair up human→gpt turns
            pairs: list[tuple[str, str]] = []
            i = 0
            while i < len(turns) - 1:
                a, b = turns[i], turns[i + 1]
                if a.get("from") == "human" and b.get("from") in ("gpt", "assistant"):
                    pairs.append((a["value"].strip(), b["value"].strip()))
                    i += 2
                else:
                    i += 1

            if not pairs:
                continue

            # Drop the first pair if it looks like a system-prompt handshake
            start = 1 if _is_system_prompt(pairs[0][0]) else 0

            for instruction, output in pairs[start:]:
                if instruction and output:
                    yield {"instruction": instruction, "output": output}


# ---------------------------------------------------------------------------
# Source 2 — MCQ benchmark with cyclic option shuffling
# ---------------------------------------------------------------------------

def _cyclic_shift(rec: dict, shift: int) -> tuple[dict[str, str], str]:
    orig = [rec[l] for l in _OPTION_LETTERS]
    shifted = orig[-shift:] + orig[:-shift] if shift > 0 else orig[:]
    opts = {l: shifted[i] for i, l in enumerate(_OPTION_LETTERS)}
    gold_text = rec[rec["answer"].strip().upper()]
    new_gold = _OPTION_LETTERS[shifted.index(gold_text)]
    return opts, new_gold


def _load_mcq(path: Path) -> Generator[dict[str, str], None, None]:
    """Convert MCQ JSONL to instruction/output pairs with 4 cyclic option shifts.

    Each shift produces a variant where the correct answer occupies a different
    position (A/B/C/D), balancing the gold-letter distribution and fixing the
    D-avoidance positional bias observed in Stage A diagnostics.

    Output uses the think-block format so the model learns to produce
    <think>...</think> then just the answer letter — matching extract_answer
    expectations during evaluation.
    """
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc

            required = {"question", "A", "B", "C", "D", "answer"}
            if not required.issubset(rec.keys()):
                continue

            letter = rec["answer"].strip().upper()
            if letter not in _OPTION_LETTERS:
                continue

            question = rec["question"].strip()
            answer_text = rec[letter].strip()

            # Open-ended variant — plain answer without MCQ format
            yield {"instruction": question, "output": answer_text}

            # MCQ variants — 4 cyclic shifts for position-balanced training
            for shift in range(4):
                opts, new_letter = _cyclic_shift(rec, shift)
                new_answer_text = opts[new_letter]
                options_block = "\n".join(f"{l}. {opts[l]}" for l in _OPTION_LETTERS)
                yield {
                    "instruction": f"{question}\n\n{options_block}",
                    "output": (
                        f"<think>\nThe correct answer is {new_letter}. "
                        f"{new_answer_text}\n</think>\n{new_letter}"
                    ),
                }


# ---------------------------------------------------------------------------
# Filtering and deduplication
# ---------------------------------------------------------------------------

def _instruction_hash(pair: dict[str, str]) -> str:
    return hashlib.sha256(pair["instruction"].encode()).hexdigest()


def filter_and_dedup(
    pairs: list[dict[str, str]],
    min_output_chars: int,
    max_total_chars: int,
) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    skipped_short = skipped_long = skipped_dup = 0

    for pair in pairs:
        if len(pair["output"].strip()) < min_output_chars:
            skipped_short += 1
            continue
        total = len(pair["instruction"]) + len(pair["output"])
        if total > max_total_chars:
            skipped_long += 1
            continue
        h = _instruction_hash(pair)
        if h in seen:
            skipped_dup += 1
            continue
        seen.add(h)
        result.append(pair)

    print(
        f"  Filtered: {skipped_short} too-short, "
        f"{skipped_long} too-long, "
        f"{skipped_dup} duplicate instructions",
        file=sys.stderr,
    )
    return result


# ---------------------------------------------------------------------------
# Split and write
# ---------------------------------------------------------------------------

def split_and_write(
    pairs: list[dict[str, str]],
    output_dir: Path,
    train_fraction: float,
    seed: int,
) -> tuple[int, int]:
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * train_fraction))
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    for dest, records in ((output_dir / "train.jsonl", train), (output_dir / "val.jsonl", val)):
        with open(dest, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(train), len(val)


# ---------------------------------------------------------------------------
# Contamination guard — see MCQ_DIAGNOSTIC_PLAN.md
# ---------------------------------------------------------------------------

_KNOWN_BENCHMARK_FILENAMES = frozenset({
    "benchmark_combined.jsonl",
    "comprehensive_agri_benchmark.jsonl",
})


def _is_spanish_transfer_path(path: Path) -> bool:
    """Return whether a path belongs to the evaluation-only Spanish transfer set."""
    resolved = path.expanduser().resolve()
    parts = {part.casefold() for part in resolved.parts}
    return (
        "spanish-benchmark-workspace" in parts
        or resolved.name.casefold() == "english_v5clean_benchmark_es.jsonl"
        or "en_to_es" in parts
    )


def _refuse_benchmark_mcq(mcq_path: Path, parser: argparse.ArgumentParser) -> None:
    """Refuse to build training data directly from a known eval-benchmark file.

    Filename match only — not a substitute for an actual question-text dedup check
    against benchmark_combined.jsonl. This exists because the original --mcq usage
    example pointed straight at the eval benchmark, training the model on 100% of
    the questions it was later scored against. See MCQ_DIAGNOSTIC_PLAN.md.
    """
    if _is_spanish_transfer_path(mcq_path):
        parser.error(
            "--mcq points at the Spanish English-to-Spanish transfer diagnostic. "
            "That benchmark is evaluation-only and must never enter training, "
            "validation, or autoresearch data preparation."
        )
    if mcq_path.name in _KNOWN_BENCHMARK_FILENAMES:
        parser.error(
            f"--mcq points at '{mcq_path.name}', a known eval-benchmark file. "
            "Training on the eval benchmark contaminates every downstream accuracy "
            "number. See MCQ_DIAGNOSTIC_PLAN.md for the redo plan. If this file "
            "genuinely has zero overlap with benchmark_combined.jsonl, rename it "
            "to something else to bypass this filename check."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--conversations",
        metavar="PATH",
        help="ShareGPT conversations JSONL (finetune-gemma4/hf_dataset/train.jsonl)",
    )
    parser.add_argument(
        "--mcq",
        metavar="PATH",
        help="MCQ benchmark JSONL (benchmark_combined.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        metavar="DIR",
        help="Output directory for train.jsonl and val.jsonl (default: data)",
    )
    parser.add_argument("--split", type=float, default=0.9, metavar="F",
                        help="Train fraction (default: 0.9)")
    parser.add_argument("--min-output-chars", type=int, default=10, metavar="N")
    parser.add_argument("--max-total-chars", type=int, default=8192, metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not 0.0 < args.split < 1.0:
        parser.error("--split must be between 0 and 1 (exclusive)")

    if not args.conversations and not args.mcq:
        parser.error("At least one of --conversations or --mcq is required")

    all_pairs: list[dict[str, str]] = []

    if args.conversations:
        conv_path = Path(args.conversations)
        if not conv_path.exists():
            parser.error(f"Conversations file not found: {conv_path}")
        print(f"Loading conversations from {conv_path} ...", file=sys.stderr)
        conv_pairs = list(_load_conversations(conv_path))
        print(f"  Loaded {len(conv_pairs)} instruction/output pairs", file=sys.stderr)
        all_pairs.extend(conv_pairs)

    if args.mcq:
        mcq_path = Path(args.mcq)
        if not mcq_path.exists():
            parser.error(f"MCQ file not found: {mcq_path}")
        _refuse_benchmark_mcq(mcq_path, parser)
        print(f"Loading MCQ benchmark from {mcq_path} ...", file=sys.stderr)
        mcq_pairs = list(_load_mcq(mcq_path))
        print(f"  Loaded {len(mcq_pairs)} pairs ({len(mcq_pairs)//5} questions × 5 variants)", file=sys.stderr)
        all_pairs.extend(mcq_pairs)

    print(f"\nTotal before filtering: {len(all_pairs)}", file=sys.stderr)
    filtered = filter_and_dedup(
        all_pairs,
        min_output_chars=args.min_output_chars,
        max_total_chars=args.max_total_chars,
    )
    print(f"Total after filtering:  {len(filtered)}", file=sys.stderr)

    if not filtered:
        print("ERROR: No pairs remain after filtering.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    n_train, n_val = split_and_write(filtered, output_dir, args.split, args.seed)
    print(f"\nWrote {n_train} train pairs → {output_dir}/train.jsonl", file=sys.stderr)
    print(f"Wrote {n_val}  val pairs   → {output_dir}/val.jsonl", file=sys.stderr)


if __name__ == "__main__":
    main()
