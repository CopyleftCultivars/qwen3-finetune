#!/usr/bin/env python3
"""prepare_dataset_v3.py — Assemble v3 Qwen3 fine-tune training data.

Three source streams:
  1. finetune-gemma4 NF conversations (ShareGPT → instruction/output)
  2. MCQ pairs with genuine CoT rationale (output of scripts/add_rationale.py)
  3. QTMD-generated MCQ pairs (output of scripts/generate_mcq_from_qtmd.py)
     — cyclic shifts applied here, with 2x weight on D-position shift

Optionally filters any source through LLM judge verdicts (judge_verdict="keep").

Stage B findings driving this v3 design:
  - Present-not-surfaced: fill think blocks with genuine rationale (not label restatement)
  - Generation accuracy 56.3% vs logprob 78.3%: format-matched training targets
  - D marginal 16.9% (target 25%): heavier D-position cyclic weight
  - Lower LR intended for train.py (--lr 5e-5 vs prior 1e-4)

Usage:
    python prepare_dataset_v3.py \\
        --conversations data/sources/gemma4/hf_dataset/train.jsonl \\
        --mcq-rationale data/sources/mcq_with_rationale.jsonl \\
        --qtmd-mcq data/sources/qtmd_mcq_generated.jsonl \\
        [--judged-filter data/sources/mcq_with_rationale_judged.jsonl] \\
        [--output-dir data] \\
        [--split 0.9] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Generator

_OPTION_LETTERS = ("A", "B", "C", "D")

# Cyclic shift weights: index = shift amount (0-3).
# shift=3 puts gold answer at D position — gets 2x weight per Stage A findings.
_SHIFT_WEIGHTS = [1, 1, 1, 2]

_MCQ_PATTERN = re.compile(
    r"^(.+?)\n\nA\. (.+)\nB\. (.+)\nC\. (.+)\nD\. (.+)$",
    re.DOTALL,
)
_THINK_CONTENT = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_LETTER = re.compile(r"\n([A-D])\s*$")

_SYSTEM_PROMPT_MARKERS = (
    "You are a Natural Farming",
    "You are an AI assistant",
    "You are a helpful",
)


# ---------------------------------------------------------------------------
# Source 1 — ShareGPT conversations (finetune-gemma4)
# ---------------------------------------------------------------------------

def _is_system_prompt(text: str) -> bool:
    return any(text.strip().startswith(m) for m in _SYSTEM_PROMPT_MARKERS)


def _load_conversations(path: Path) -> Generator[dict[str, str], None, None]:
    """Flatten ShareGPT multi-turn conversations to instruction/output pairs."""
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

            start = 1 if _is_system_prompt(pairs[0][0]) else 0
            for instruction, output in pairs[start:]:
                if instruction and output:
                    yield {"instruction": instruction, "output": output,
                           "_source": "gemma4_conversations"}


# ---------------------------------------------------------------------------
# Source 2 — MCQ pairs already carrying rationale (from add_rationale.py)
# ---------------------------------------------------------------------------

def _load_mcq_with_rationale(path: Path) -> Generator[dict[str, str], None, None]:
    """Pass through instruction/output pairs that already carry rationale.

    These came from the existing benchmark MCQ cyclic-shift variants, so
    position balancing is already done. No re-shifting needed.
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
            if "instruction" in rec and "output" in rec:
                yield {
                    "instruction": rec["instruction"],
                    "output": rec["output"],
                    "_source": rec.get("_source", "mcq_rationale"),
                }


# ---------------------------------------------------------------------------
# Source 3 — QTMD-generated MCQ (from generate_mcq_from_qtmd.py)
# Apply cyclic shifts here with heavier D-position weight.
# ---------------------------------------------------------------------------

def _cyclic_shift_pair(
    pair: dict,
    shift: int,
) -> dict | None:
    """Apply cyclic option shift to a QTMD MCQ pair.

    Parses instruction for question+options, rotates options by `shift`,
    updates the answer letter, keeps rationale content (which explains the
    concept, not the letter label).
    """
    m = _MCQ_PATTERN.match(pair["instruction"].strip())
    if not m:
        return None
    am = _ANSWER_LETTER.search(pair["output"])
    if not am:
        return None

    question = m.group(1).strip()
    orig_opts = [m.group(i).strip() for i in range(2, 6)]
    orig_letter = am.group(1)
    orig_idx = _OPTION_LETTERS.index(orig_letter)

    if shift == 0:
        shifted_opts = orig_opts[:]
        new_idx = orig_idx
    else:
        shifted_opts = orig_opts[-shift:] + orig_opts[:-shift]
        new_idx = (orig_idx + shift) % 4

    new_letter = _OPTION_LETTERS[new_idx]
    options_block = "\n".join(f"{l}. {shifted_opts[i]}"
                              for i, l in enumerate(_OPTION_LETTERS))

    think_m = _THINK_CONTENT.search(pair["output"])
    rationale = think_m.group(1).strip() if think_m else ""
    new_output = f"<think>\n{rationale}\n</think>\n{new_letter}"

    return {
        "instruction": f"{question}\n\n{options_block}",
        "output": new_output,
        "_source": pair.get("_source", "qtmd_mcq"),
        "_source_collection": pair.get("source_collection", ""),
    }


def _load_qtmd_mcq(path: Path) -> Generator[dict[str, str], None, None]:
    """Load QTMD MCQ pairs and emit cyclic-shifted variants with D-position 2x."""
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if "instruction" not in rec or "output" not in rec:
                continue

            for shift, weight in enumerate(_SHIFT_WEIGHTS):
                shifted = _cyclic_shift_pair(rec, shift)
                if shifted is None:
                    continue
                for _ in range(weight):
                    yield shifted


# ---------------------------------------------------------------------------
# Judge filter (optional)
# ---------------------------------------------------------------------------

def _load_keep_instructions(judged_path: Path) -> set[str]:
    """Return set of instruction hashes that received judge verdict 'keep'."""
    keep_hashes: set[str] = set()
    with open(judged_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("judge_verdict") == "keep":
                    h = hashlib.sha256(
                        rec["instruction"].encode()
                    ).hexdigest()
                    keep_hashes.add(h)
            except (json.JSONDecodeError, KeyError):
                continue
    return keep_hashes


# ---------------------------------------------------------------------------
# Deduplication, split, write
# ---------------------------------------------------------------------------

def _instruction_hash(pair: dict[str, str]) -> str:
    return hashlib.sha256(pair["instruction"].encode()).hexdigest()


def filter_and_dedup(
    pairs: list[dict[str, str]],
    min_output_chars: int,
    max_total_chars: int,
    keep_hashes: set[str] | None,
) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    skipped = {"short": 0, "long": 0, "dup": 0, "judge": 0}

    for pair in pairs:
        if len(pair["output"].strip()) < min_output_chars:
            skipped["short"] += 1
            continue
        if len(pair["instruction"]) + len(pair["output"]) > max_total_chars:
            skipped["long"] += 1
            continue
        h = _instruction_hash(pair)
        if h in seen:
            skipped["dup"] += 1
            continue
        if keep_hashes is not None and h not in keep_hashes:
            skipped["judge"] += 1
            continue
        seen.add(h)
        result.append(pair)

    print(
        f"  Filtered: {skipped['short']} too-short, {skipped['long']} too-long, "
        f"{skipped['dup']} duplicates, {skipped['judge']} judge-rejected",
        file=sys.stderr,
    )
    return result


def _letter_distribution(pairs: list[dict[str, str]]) -> dict[str, int]:
    dist: dict[str, int] = {l: 0 for l in _OPTION_LETTERS}
    for pair in pairs:
        am = _ANSWER_LETTER.search(pair["output"])
        if am:
            dist[am.group(1)] += 1
    return dist


def split_and_write(
    pairs: list[dict[str, str]],
    output_dir: Path,
    train_fraction: float,
    seed: int,
    suffix: str = "_v3",
) -> tuple[int, int]:
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * train_fraction))
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)

    clean_fields = {"instruction", "output"}
    for dest_name, records in (
        (f"train{suffix}.jsonl", train),
        (f"val{suffix}.jsonl", val),
    ):
        dest = output_dir / dest_name
        with open(dest, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(
                    {k: v for k, v in rec.items() if k in clean_fields},
                    ensure_ascii=False,
                ) + "\n")

    return len(train), len(val)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--conversations", metavar="PATH",
                    help="gemma4 ShareGPT JSONL (hf_dataset/train.jsonl)")
    ap.add_argument("--mcq-rationale", metavar="PATH",
                    help="MCQ pairs with rationale (data/sources/mcq_with_rationale.jsonl)")
    ap.add_argument("--qtmd-mcq", metavar="PATH",
                    help="QTMD-generated MCQ (data/sources/qtmd_mcq_generated.jsonl)")
    ap.add_argument("--judged-filter", metavar="PATH", default=None,
                    help="Judged JSONL; only keep judge_verdict='keep' instructions")
    ap.add_argument("--output-dir", default="data", metavar="DIR")
    ap.add_argument("--split", type=float, default=0.9, metavar="F")
    ap.add_argument("--min-output-chars", type=int, default=10)
    ap.add_argument("--max-total-chars", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="_v3",
                    help="Suffix for output filenames (default: _v3)")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if not any([args.conversations, args.mcq_rationale, args.qtmd_mcq]):
        ap.error("At least one of --conversations, --mcq-rationale, or --qtmd-mcq is required")
    if not 0.0 < args.split < 1.0:
        ap.error("--split must be between 0 and 1 (exclusive)")

    all_pairs: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}

    if args.conversations:
        path = Path(args.conversations)
        if not path.exists():
            ap.error(f"Conversations file not found: {path}")
        print(f"Loading conversations from {path} ...", file=sys.stderr)
        pairs = list(_load_conversations(path))
        source_counts["gemma4_conversations"] = len(pairs)
        all_pairs.extend(pairs)
        print(f"  → {len(pairs)} pairs", file=sys.stderr)

    if args.mcq_rationale:
        path = Path(args.mcq_rationale)
        if not path.exists():
            ap.error(f"MCQ rationale file not found: {path}")
        print(f"Loading MCQ+rationale pairs from {path} ...", file=sys.stderr)
        pairs = list(_load_mcq_with_rationale(path))
        source_counts["mcq_rationale"] = len(pairs)
        all_pairs.extend(pairs)
        print(f"  → {len(pairs)} pairs", file=sys.stderr)

    if args.qtmd_mcq:
        path = Path(args.qtmd_mcq)
        if not path.exists():
            ap.error(f"QTMD MCQ file not found: {path}")
        print(f"Loading QTMD MCQ from {path} (with cyclic shifts) ...", file=sys.stderr)
        pairs = list(_load_qtmd_mcq(path))
        source_counts["qtmd_mcq"] = len(pairs)
        all_pairs.extend(pairs)
        print(
            f"  → {len(pairs)} pairs "
            f"(shift weights {_SHIFT_WEIGHTS}, D-pos 2x)",
            file=sys.stderr,
        )

    print(f"\nTotal before filtering: {len(all_pairs)}", file=sys.stderr)

    keep_hashes: set[str] | None = None
    if args.judged_filter:
        judge_path = Path(args.judged_filter)
        if not judge_path.exists():
            ap.error(f"Judged filter file not found: {judge_path}")
        print(f"Loading judge filter from {judge_path} ...", file=sys.stderr)
        keep_hashes = _load_keep_instructions(judge_path)
        print(f"  → {len(keep_hashes)} keep-verdict instructions", file=sys.stderr)

    filtered = filter_and_dedup(
        all_pairs,
        min_output_chars=args.min_output_chars,
        max_total_chars=args.max_total_chars,
        keep_hashes=keep_hashes,
    )
    print(f"Total after filtering:  {len(filtered)}", file=sys.stderr)

    if not filtered:
        print("ERROR: No pairs remain after filtering.", file=sys.stderr)
        sys.exit(1)

    letter_dist = _letter_distribution(filtered)
    mcq_total = sum(letter_dist.values())
    print("\nAnswer letter distribution (MCQ pairs only):", file=sys.stderr)
    for letter, count in sorted(letter_dist.items()):
        pct = 100 * count / mcq_total if mcq_total else 0
        print(f"  {letter}: {count:5d}  ({pct:.1f}%)", file=sys.stderr)

    print("\nSource breakdown:", file=sys.stderr)
    for source, count in source_counts.items():
        print(f"  {source}: {count}", file=sys.stderr)

    output_dir = Path(args.output_dir)
    n_train, n_val = split_and_write(
        filtered, output_dir, args.split, args.seed, suffix=args.suffix
    )
    print(f"\nWrote {n_train} train pairs → {output_dir}/train{args.suffix}.jsonl",
          file=sys.stderr)
    print(f"Wrote {n_val}  val pairs   → {output_dir}/val{args.suffix}.jsonl",
          file=sys.stderr)


if __name__ == "__main__":
    main()
