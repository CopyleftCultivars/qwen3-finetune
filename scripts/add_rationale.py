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

Shutdown: Ctrl+C kills all in-flight claude subprocesses cleanly; no orphaned
processes remain after exit.

Resume behavior:
  - Successfully processed pairs are written with "_src_idx" field.
  - Failed MCQ pairs are NOT written (so --resume retries them).
  - Conversation pass-through pairs are written with "_src_idx" immediately.

*** CONTAMINATED DEFAULT INPUT WARNING (see MCQ_DIAGNOSTIC_PLAN.md, 2026-07-09) ***
The default --input (data/train.jsonl) is 100% derived from benchmark_combined.jsonl (the
eval benchmark) — prepare_dataset.py built it by loading the benchmark directly as an MCQ
source. This script itself doesn't reference the benchmark, but it inherits the contamination
from its input. Point --input at a clean, non-benchmark-derived MCQ source once one exists
(Phase 3 of the redo plan).

Usage:
    python scripts/add_rationale.py \\
        --input data/train.jsonl \\
        --output data/sources/mcq_with_rationale.jsonl \\
        [--model haiku] \\
        [--max-workers 6] \\
        [--api-concurrency 2] \\
        [--resume]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import signal
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

_RETRY_DELAYS = [5, 30, 90]
_RATE_LIMIT_KEYWORDS = frozenset(["rate limit", "too many", "overloaded", "429", "529"])

# Global registry of active Popen objects so the shutdown handler can kill them.
_active_procs: set[subprocess.Popen] = set()
_proc_lock = threading.Lock()


def _register_shutdown() -> None:
    """Register SIGINT/SIGTERM handlers that kill all active claude subprocesses."""
    def _handler(signum: int, frame: object) -> None:
        log.warning(
            "Signal %d received — terminating %d active claude subprocess(es)",
            signum, len(_active_procs),
        )
        with _proc_lock:
            for proc in list(_active_procs):
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        sys.exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _call_claude(
    prompt: str,
    model: str,
    api_sem: threading.Semaphore,
    timeout: int = 90,
) -> str:
    """Call claude -p in a tracked subprocess and return the result text.

    Each call acquires api_sem so the number of concurrent API requests is
    capped independently of the thread pool size. Uses start_new_session=True
    so the process gets its own process group — the shutdown handler can kill
    it via killpg even after the parent thread returns.
    """
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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
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
                last_exc = subprocess.CalledProcessError(
                    proc.returncode, "claude", stdout, stderr
                )
                if any(kw in err_text.lower() for kw in _RATE_LIMIT_KEYWORDS):
                    log.warning(
                        "Rate limit on attempt %d: %s", attempt + 1, err_text[:120]
                    )
                else:
                    log.warning(
                        "claude -p error (attempt %d, rc=%d): %s",
                        attempt + 1, proc.returncode, err_text[:120],
                    )
                continue

            try:
                return json.loads(stdout)["result"].strip()
            except (json.JSONDecodeError, KeyError) as exc:
                last_exc = exc
                log.warning(
                    "Parse error on attempt %d: %s | stdout: %s",
                    attempt + 1, exc, stdout[:120],
                )
                continue

    raise last_exc


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


def _generate_rationale(parsed: dict, model: str, api_sem: threading.Semaphore) -> str:
    prompt = _RATIONALE_PROMPT.format(
        question=parsed["question"],
        opt_a=parsed["options"]["A"],
        opt_b=parsed["options"]["B"],
        opt_c=parsed["options"]["C"],
        opt_d=parsed["options"]["D"],
        letter=parsed["letter"],
        correct_text=parsed["correct_text"],
    )
    return _call_claude(prompt, model, api_sem)


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
                    help="Thread pool size (default: 6)")
    ap.add_argument("--api-concurrency", type=int, default=1,
                    help="Max concurrent claude -p API calls (default: 1, sequential)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip items already successfully written to output file")
    args = ap.parse_args()

    _register_shutdown()

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

    api_sem = threading.Semaphore(args.api_concurrency)
    write_lock = threading.Lock()
    stats = {"mcq": 0, "conv": 0, "errors": 0, "done": 0}

    mode = "a" if args.resume else "w"
    out_fh = open(output_path, mode, encoding="utf-8")  # noqa: WPS515

    def process(idx: int, pair: dict) -> None:
        if idx in done_indices:
            return
        parsed = _parse_mcq(pair)
        if parsed is None:
            result = {**pair, "_src_idx": idx}
            with write_lock:
                stats["conv"] += 1
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            return
        try:
            rationale = _generate_rationale(parsed, args.model, api_sem)
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
        "Done. MCQ upgraded: %d | Conversations: %d | Errors: %d | Written: %d → %s",
        stats["mcq"], stats["conv"], stats["errors"], total, output_path,
    )
    print(f"\nMCQ pairs upgraded: {stats['mcq']}")
    print(f"Conversation pairs (passthrough): {stats['conv']}")
    print(f"Errors (will retry on --resume): {stats['errors']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
