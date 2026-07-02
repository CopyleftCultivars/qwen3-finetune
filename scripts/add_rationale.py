#!/usr/bin/env python3
"""Add genuine rationale to MCQ think blocks using Claude as a teacher LLM.

Reads an instruction/output JSONL (e.g. data/train.jsonl), identifies MCQ pairs
by the presence of A./B./C./D. options in the instruction, and replaces the thin
think-block content ("The correct answer is X. ...") with a genuine 1-3 sentence
rationale that explains why the correct answer is right and why the distractors
are wrong.

Non-MCQ pairs (NF conversations) pass through unchanged.

Uses `claude -p` subprocess so no API key or .env needed — Claude Code OAuth
credentials are used automatically.

Resume behavior:
  - Successfully processed pairs are written with "_src_idx" field.
  - Failed MCQ pairs are NOT written (so --resume retries them).
  - Conversation pass-through pairs are written with "_src_idx" immediately.

Usage:
    python scripts/add_rationale.py \\
        --input data/train.jsonl \\
        --output data/sources/mcq_with_rationale.jsonl \\
        [--model haiku] \\
        [--max-workers 6] \\
        [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

_MCQ_PATTERN = re.compile(
    r"^(.+?)\n\nA\. (.+)\nB\. (.+)\nC\. (.+)\nD\. (.+)$",
    re.DOTALL,
)
_ANSWER_PATTERN = re.compile(r"\n([A-D])\s*$")

_SYSTEM = (
    "You are an expert in natural farming, agroecology, soil science, "
    "permaculture, and sustainable agriculture. Your task is to write a brief "
    "educational rationale for a multiple-choice question answer."
)

_RATIONALE_PROMPT = """\
Given this multiple-choice question and its correct answer, write a concise \
1-3 sentence rationale that explains WHY the correct answer is right and briefly \
distinguishes it from the incorrect options. Be specific and factual. \
Do not start with "The correct answer is" — start directly with the explanation.

Question: {question}

A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Correct answer: {letter}. {correct_text}

Write only the rationale, no preamble:"""

_RETRY_DELAYS = [2, 5, 15]


def _call_claude(prompt: str, model: str, timeout: int = 90) -> str:
    """Call claude -p subprocess and return the result text."""
    full_prompt = f"{_SYSTEM}\n\n{prompt}"
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            result = subprocess.run(
                ["claude", "-p", "--model", model, "--tools", "",
                 "--output-format", "json", full_prompt],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return json.loads(result.stdout)["result"].strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, KeyError) as exc:
            if attempt == len(_RETRY_DELAYS):
                raise
            log.warning("Claude call failed (attempt %d): %s — retrying", attempt + 1, exc)
    raise RuntimeError("All retries exhausted")


def _parse_mcq(pair: dict) -> dict | None:
    """Return parsed MCQ fields or None if not an MCQ pair."""
    m = _MCQ_PATTERN.match(pair["instruction"].strip())
    if not m:
        return None
    am = _ANSWER_PATTERN.search(pair["output"])
    if not am:
        return None
    letter = am.group(1)
    opts = {"A": m.group(2).strip(), "B": m.group(3).strip(),
            "C": m.group(4).strip(), "D": m.group(5).strip()}
    return {
        "question": m.group(1).strip(),
        "options": opts,
        "letter": letter,
        "correct_text": opts[letter],
    }


def _generate_rationale(parsed: dict, model: str) -> str:
    prompt = _RATIONALE_PROMPT.format(
        question=parsed["question"],
        opt_a=parsed["options"]["A"],
        opt_b=parsed["options"]["B"],
        opt_c=parsed["options"]["C"],
        opt_d=parsed["options"]["D"],
        letter=parsed["letter"],
        correct_text=parsed["correct_text"],
    )
    return _call_claude(prompt, model)


def _load_done(output_path: Path) -> set[int]:
    """Return set of _src_idx values already successfully written."""
    done: set[int] = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if "_src_idx" in rec:
                    done.add(rec["_src_idx"])
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d in resume file: %s", i, exc)
    return done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input JSONL (instruction/output pairs)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--model", default="haiku",
                    help="Claude model alias for claude -p (default: haiku)")
    ap.add_argument("--max-workers", type=int, default=6,
                    help="Parallel claude -p subprocesses (default: 6)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip items already successfully written to output file")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as fh:
        pairs = [json.loads(line) for line in fh if line.strip()]
    log.info("Loaded %d pairs from %s", len(pairs), input_path)

    done_indices: set[int] = set()
    if args.resume:
        done_indices = _load_done(output_path)
        log.info("Resuming: %d already done", len(done_indices))

    write_lock = threading.Lock()
    stats = {"mcq": 0, "conv": 0, "errors": 0, "done": 0}

    mode = "a" if args.resume else "w"
    out_fh = open(output_path, mode, encoding="utf-8")  # noqa: WPS515

    def process(idx: int, pair: dict) -> None:
        if idx in done_indices:
            return
        parsed = _parse_mcq(pair)
        if parsed is None:
            # Conversation pair — pass through with _src_idx for resume tracking
            result = {**pair, "_src_idx": idx}
            with write_lock:
                stats["conv"] += 1
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            return
        try:
            rationale = _generate_rationale(parsed, args.model)
            new_output = f"<think>\n{rationale}\n</think>\n{parsed['letter']}"
            result = {"instruction": pair["instruction"], "output": new_output,
                      "_src_idx": idx}
            with write_lock:
                stats["mcq"] += 1
                stats["done"] += 1
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                if stats["done"] % 100 == 0:
                    out_fh.flush()
                    log.info("Progress: %d MCQ rationales added, %d errors",
                             stats["done"], stats["errors"])
        except Exception as exc:
            log.warning("Error on pair %d: %s — will retry on --resume", idx, exc)
            with write_lock:
                stats["errors"] += 1
            # Do NOT write failed pair — omitting _src_idx means --resume will retry it

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process, i, p): i for i, p in enumerate(pairs)}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("Unexpected error on future %d: %s", futures[fut], exc)

    out_fh.flush()
    out_fh.close()

    total = stats["mcq"] + stats["conv"]
    log.info(
        "Done. MCQ with rationale: %d | Conversations: %d | Errors: %d | Total written: %d → %s",
        stats["mcq"], stats["conv"], stats["errors"], total, output_path,
    )
    print(f"\nMCQ pairs upgraded: {stats['mcq']}")
    print(f"Conversation pairs (passthrough): {stats['conv']}")
    print(f"Errors (will retry on --resume): {stats['errors']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
