#!/usr/bin/env python3
"""Convert a flat MCQ JSONL (question/A/B/C/D/answer/topic/category) into
instruction/output training pairs with genuine CoT rationale, matching the
schema prepare_dataset_v3.py's _load_qmd_mcq() expects.

Uses `claude -p` subprocess (same pattern as scripts/add_rationale.py) — no
OpenRouter dependency, rides the existing Claude subscription.

Optionally wraps a fraction of the output in the canonical Natural Farming
Fertilizer Assistant system prompt (Phase 3b's bare/wrapped split — see
MCQ_DIAGNOSTIC_PLAN.md). Bare and wrapped variants are written to separate
output files so downstream tooling can treat them as distinct sources.

Usage:
    python scripts/format_mcq_to_training_pairs.py \\
        --input data/sources/qmd_mcq_generated.jsonl \\
        --output data/sources/qmd_mcq_generated.formatted.jsonl \\
        [--wrap-fraction 0.2 --wrap-output data/sources/qmd_mcq_generated.wrapped.jsonl] \\
        [--model haiku] [--max-workers 6] [--api-concurrency 2]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

_OPTION_LETTERS = ("A", "B", "C", "D")

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

_CANONICAL_SYSTEM_PROMPT = """\
You are a Natural Farming Fertilizer Assistant specializing in wildcrafted fertilizer production with Korean Natural Farming (KNF) expertise. Key traits:

- Provide precise, evidence-based guidance for DIY fertilizers using IMO, LABs, and local ingredients
- Always answer questions directly and comprehensively
- Focus on location-specific recommendations using only locally available materials
- For aquaponics systems, recommend only liquid inputs (FPJ, Liquid IMO, LABs, filtered ferments)

Required response format:

[Short Introduction]
[Direct Answer]
[Crop Nutritional Needs or Pest Life Cycle if pest-related]
[List of 5+ Local Bioaccumulator Plants]
[3 Recipe Sets with Local Ingredients]
[Data Tables]
| Crops | Nutritional Needs | Bioaccumulators |
| ----- | ---------------- | --------------- |
[Usage Instructions]
[Application Schedule & Rotation Plan]
[Natural Farming Philosophy Note]
[Alternative Local Recipes]
[Encouragement]

Key Guidelines:

- Use specific numbers, volumes, and metrics
- Recommend diverse vegetative materials over seeds
- Prefer IMO4 over IMO3
- Combine KNF and Jadam recipes
- For aquaponics: liquid inputs only
- Never recommend salt/synthetic fertilizers
- Match ingredients to user's location using bioaccumulator data
- Include nutrient ratios and exact quantities
- Base all recommendations on provided bioaccumulator data
- If discussing weather protection or specific procedures, adapt format accordingly

Tables must include:

- Crop names and nutrient requirements
- Ingredient names and quantities
- Local bioaccumulator options (min. 5)
- Nutrient values (NPK ratios)
- Exact application quantities

If missing information, respond: "I don't know, but with more details I could..." followed by best guidance based on available data.

Respond in user's language unless otherwise specified."""

_RETRY_DELAYS = [5, 30, 90]
_RATE_LIMIT_KEYWORDS = frozenset(["rate limit", "too many", "overloaded", "429", "529"])

_active_procs: set[subprocess.Popen] = set()
_proc_lock = threading.Lock()


def _register_shutdown() -> None:
    def _handler(signum: int, frame: object) -> None:
        log.warning("Signal %d received — terminating %d active claude subprocess(es)",
                    signum, len(_active_procs))
        with _proc_lock:
            for proc in list(_active_procs):
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        sys.exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _call_claude(prompt: str, model: str, api_sem: threading.Semaphore, timeout: int = 90) -> str:
    full_prompt = f"{_SYSTEM}\n\n{prompt}"
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(len(_RETRY_DELAYS) + 1):
        if attempt > 0:
            delay = _RETRY_DELAYS[attempt - 1]
            log.info("Retry %d — waiting %ds before next attempt", attempt, delay)
            time.sleep(delay)

        with api_sem:
            proc = subprocess.Popen(
                ["claude", "-p", "--model", model, "--tools", "",
                 "--output-format", "json", full_prompt],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
            with _proc_lock:
                _active_procs.add(proc)
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                with _proc_lock:
                    _active_procs.discard(proc)
                last_exc = TimeoutError(f"claude -p timed out after {timeout}s")
                log.warning("Timeout on attempt %d", attempt + 1)
                continue
            else:
                with _proc_lock:
                    _active_procs.discard(proc)

            if proc.returncode != 0:
                err_text = (stderr or stdout).strip()
                last_exc = subprocess.CalledProcessError(proc.returncode, "claude", stdout, stderr)
                if any(kw in err_text.lower() for kw in _RATE_LIMIT_KEYWORDS):
                    log.warning("Rate limit on attempt %d: %s", attempt + 1, err_text[:120])
                else:
                    log.warning("claude -p error (attempt %d, rc=%d): %s",
                                attempt + 1, proc.returncode, err_text[:120])
                continue

            try:
                return json.loads(stdout)["result"].strip()
            except (json.JSONDecodeError, KeyError) as exc:
                last_exc = exc
                log.warning("Parse error on attempt %d: %s | stdout: %s",
                            attempt + 1, exc, stdout[:120])
                continue

    raise last_exc


def _generate_rationale(rec: dict, model: str, api_sem: threading.Semaphore) -> str:
    letter = rec["answer"]
    prompt = _RATIONALE_PROMPT.format(
        question=rec["question"], opt_a=rec["A"], opt_b=rec["B"], opt_c=rec["C"], opt_d=rec["D"],
        letter=letter, correct_text=rec[letter],
    )
    return _call_claude(prompt, model, api_sem)


def _build_pair(rec: dict, rationale: str, wrapped: bool) -> dict:
    options_block = "\n".join(f"{l}. {rec[l]}" for l in _OPTION_LETTERS)
    question_block = f"{rec['question']}\n\n{options_block}"
    instruction = f"{_CANONICAL_SYSTEM_PROMPT}\n\n{question_block}" if wrapped else question_block
    output = f"<think>\n{rationale}\n</think>\n{rec['answer']}"
    return {
        "instruction": instruction,
        "output": output,
        "_source": "qmd_mcq_wrapped" if wrapped else "qmd_mcq",
        "_source_question_id": rec.get("question_id", ""),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--wrap-fraction", type=float, default=0.0,
                     help="Fraction of items to wrap in the system prompt (default: 0, all bare)")
    ap.add_argument("--wrap-output", type=Path, default=None,
                     help="Separate output file for wrapped items (required if --wrap-fraction > 0)")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--api-concurrency", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.wrap_fraction > 0 and not args.wrap_output:
        ap.error("--wrap-output is required when --wrap-fraction > 0")

    _register_shutdown()

    with open(args.input, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    log.info("Loaded %d records from %s", len(records), args.input)

    random.seed(args.seed)
    shuffled = records[:]
    random.shuffle(shuffled)
    n_wrapped = int(len(shuffled) * args.wrap_fraction)
    wrapped_ids = {id(r) for r in shuffled[:n_wrapped]}
    log.info("Split: %d wrapped, %d bare", n_wrapped, len(records) - n_wrapped)

    api_sem = threading.Semaphore(args.api_concurrency)
    write_lock = threading.Lock()
    bare_pairs: list[dict] = []
    wrapped_pairs: list[dict] = []
    stats = {"done": 0, "errors": 0}

    def process(rec: dict) -> None:
        try:
            rationale = _generate_rationale(rec, args.model, api_sem)
            wrapped = id(rec) in wrapped_ids
            pair = _build_pair(rec, rationale, wrapped)
            with write_lock:
                (wrapped_pairs if wrapped else bare_pairs).append(pair)
                stats["done"] += 1
                if stats["done"] % 25 == 0:
                    log.info("Progress: %d/%d", stats["done"], len(records))
        except Exception as exc:
            log.warning("Error on question_id=%s: %s", rec.get("question_id"), exc)
            with write_lock:
                stats["errors"] += 1

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process, r): r for r in records}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("Unexpected error: %s", exc)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for pair in bare_pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    log.info("Wrote %d bare pairs to %s", len(bare_pairs), args.output)

    if args.wrap_output:
        with open(args.wrap_output, "w", encoding="utf-8") as fh:
            for pair in wrapped_pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
        log.info("Wrote %d wrapped pairs to %s", len(wrapped_pairs), args.wrap_output)

    print(f"\nBare pairs: {len(bare_pairs)}")
    print(f"Wrapped pairs: {len(wrapped_pairs)}")
    print(f"Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
