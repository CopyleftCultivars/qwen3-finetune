#!/usr/bin/env python3
"""Add genuine rationale to MCQ think blocks using Claude as a teacher LLM.

Reads an instruction/output JSONL (e.g. data/train.jsonl), identifies MCQ pairs
by the presence of A./B./C./D. options in the instruction, and replaces the thin
think-block content ("The correct answer is X. ...") with a genuine 1-3 sentence
rationale that explains why the correct answer is right and why the distractors
are wrong.

Non-MCQ pairs (NF conversations) pass through unchanged.

Requires ANTHROPIC_API_KEY in environment or .env.

Usage:
    python scripts/add_rationale.py \
        --input data/train.jsonl \
        --output data/sources/mcq_with_rationale.jsonl \
        [--model claude-haiku-4-5-20251001] \
        [--max-workers 8] \
        [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

_MCQ_PATTERN = re.compile(
    r"^(.+?)\n\nA\. (.+)\nB\. (.+)\nC\. (.+)\nD\. (.+)$",
    re.DOTALL,
)
_ANSWER_PATTERN = re.compile(r"\n([A-D])\s*$")
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

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


def _get_client():
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("pip install anthropic") from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        from pathlib import Path as _P
        env_file = _P(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment or .env")
    return anthropic.Anthropic(api_key=api_key)


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


def _generate_rationale(parsed: dict, client, model: str) -> str:
    prompt = _RATIONALE_PROMPT.format(
        question=parsed["question"],
        opt_a=parsed["options"]["A"],
        opt_b=parsed["options"]["B"],
        opt_c=parsed["options"]["C"],
        opt_d=parsed["options"]["D"],
        letter=parsed["letter"],
        correct_text=parsed["correct_text"],
    )
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _inject_rationale(pair: dict, rationale: str, letter: str) -> dict:
    new_output = f"<think>\n{rationale}\n</think>\n{letter}"
    return {"instruction": pair["instruction"], "output": new_output}


def _load_done(output_path: Path) -> set[int]:
    """Return set of line indices already written (for resume)."""
    done: set[int] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if line.strip():
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
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Anthropic model for rationale generation")
    ap.add_argument("--max-workers", type=int, default=6,
                    help="Parallel API calls (default: 6)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip items already in output file")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_client()

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
            # NF conversation — pass through unchanged
            result = dict(pair)
            with write_lock:
                stats["conv"] += 1
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            return
        try:
            rationale = _generate_rationale(parsed, client, args.model)
            result = _inject_rationale(pair, rationale, parsed["letter"])
            with write_lock:
                stats["mcq"] += 1
                stats["done"] += 1
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                if stats["done"] % 100 == 0:
                    out_fh.flush()
                    log.info("Progress: %d MCQ rationales added, %d errors",
                             stats["done"], stats["errors"])
        except Exception as exc:
            log.warning("Error on pair %d: %s — keeping original", idx, exc)
            with write_lock:
                stats["errors"] += 1
                out_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

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
        "Done. MCQ with rationale: %d | Conversations: %d | Errors: %d | Total: %d → %s",
        stats["mcq"], stats["conv"], stats["errors"], total, output_path,
    )
    print(f"\nMCQ pairs upgraded: {stats['mcq']}")
    print(f"Conversation pairs (passthrough): {stats['conv']}")
    print(f"Errors (kept original): {stats['errors']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
