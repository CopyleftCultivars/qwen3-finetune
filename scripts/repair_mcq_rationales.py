#!/usr/bin/env python3
"""Repair only bad or missing MCQ rationales without changing valid records.

This script is intentionally conservative:
  - Existing records that pass local validation are copied unchanged.
  - MCQs with refusal/source-request rationales are regenerated.
  - MCQs missing from the current output are regenerated from the input file.
  - Option text labels such as "B) ..." inside an "A. ..." option are stripped
    deterministically, without changing the answer letter.
  - Failed repairs keep the existing record when one exists, so a partial run does
    not destroy usable data.

Uses `claude -p` subprocess, matching scripts/add_rationale.py.

*** CONTAMINATED DEFAULT INPUT WARNING (see MCQ_DIAGNOSTIC_PLAN.md, 2026-07-09) ***
data/train.jsonl and data/sources/mcq_with_rationale.jsonl (the default --input/--current)
are both 100% derived from benchmark_combined.jsonl (the eval benchmark). This script's own
repair logic is sound, but repairing a contaminated source produces a contaminated output —
data/sources/mcq_with_rationale.repaired.jsonl inherited the same ~100% benchmark overlap.
Point --input/--current at clean, non-benchmark-derived sources once one exists (Phase 3 of
the redo plan) rather than reusing these defaults.

Usage:
    python scripts/repair_mcq_rationales.py \
        --input data/train.jsonl \
        --current data/sources/mcq_with_rationale.jsonl \
        --output data/sources/mcq_with_rationale.repaired.jsonl \
        --template-only
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
import tempfile
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
_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_EMBEDDED_OPTION_LABEL = re.compile(r"^\s*[A-D][\).]\s+")

_SUSPECT_RATIONALE_PHRASES = (
    "need the source",
    "need source",
    "source text",
    "source material",
    "hasn't been provided",
    "haven't been provided",
    "don't have access",
    "do not have access",
    "could you provide",
    "please provide",
    "once i see",
    "i need to see",
    "i'd be happy to write",
    "cannot determine",
    "can't determine",
    "not enough information",
    "without the text",
    "without access",
    "rather than making assumptions",
)

_SYSTEM = (
    "You are an expert in natural farming, agroecology, soil science, "
    "permaculture, and sustainable agriculture. Your task is to write a brief "
    "educational rationale for a multiple-choice training example."
)

_RATIONALE_PROMPT = """\
Write a concise 1-3 sentence rationale for this multiple-choice question.

Important constraints:
- Do not ask for source text, excerpts, documents, or additional context.
- Use the correct option and the contrast with the distractors to explain the answer.
- If the question references an excerpt or source, treat the provided correct answer
  as already verified and explain why that answer fits.
- Start directly with the explanation. Do not start with "The correct answer is".
- Write only the rationale, no preamble.

Question: {question}

A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Correct answer: {letter}. {correct_text}
"""

_RETRY_DELAYS = [5, 30, 90]
_RATE_LIMIT_KEYWORDS = frozenset(["rate limit", "too many", "overloaded", "429", "529"])

_active_procs: set[subprocess.Popen] = set()
_proc_lock = threading.Lock()


def _register_shutdown() -> None:
    def _handler(signum: int, frame: object) -> None:
        log.warning(
            "Signal %d received; terminating %d active claude subprocess(es)",
            signum,
            len(_active_procs),
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
    timeout: int,
) -> str:
    full_prompt = f"{_SYSTEM}\n\n{prompt}"
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(len(_RETRY_DELAYS) + 1):
        if attempt > 0:
            delay = _RETRY_DELAYS[attempt - 1]
            log.info("Retry %d; waiting %ds before next attempt", attempt, delay)
            time.sleep(delay)

        with api_sem:
            proc = subprocess.Popen(
                [
                    "claude",
                    "-p",
                    "--model",
                    model,
                    "--tools",
                    "",
                    "--output-format",
                    "json",
                    full_prompt,
                ],
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
                continue
            else:
                with _proc_lock:
                    _active_procs.discard(proc)

            if proc.returncode != 0:
                err_text = (stderr or stdout).strip()
                last_exc = subprocess.CalledProcessError(
                    proc.returncode,
                    "claude",
                    stdout,
                    stderr,
                )
                if any(kw in err_text.lower() for kw in _RATE_LIMIT_KEYWORDS):
                    log.warning("Rate limit on attempt %d: %s", attempt + 1, err_text[:120])
                else:
                    log.warning(
                        "claude -p error on attempt %d, rc=%d: %s",
                        attempt + 1,
                        proc.returncode,
                        err_text[:120],
                    )
                continue

            try:
                return json.loads(stdout)["result"].strip()
            except (json.JSONDecodeError, KeyError) as exc:
                last_exc = exc
                log.warning(
                    "Parse error on attempt %d: %s | stdout: %s",
                    attempt + 1,
                    exc,
                    stdout[:120],
                )

    raise last_exc


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_no}: {exc}") from exc
    return rows


def _parse_mcq(pair: dict) -> dict | None:
    m = _MCQ_PATTERN.match(pair.get("instruction", "").strip())
    if not m:
        return None
    am = _ANSWER_PATTERN.search(pair.get("output", ""))
    if not am:
        return None
    letter = am.group(1)
    opts = {
        "A": m.group(2).strip(),
        "B": m.group(3).strip(),
        "C": m.group(4).strip(),
        "D": m.group(5).strip(),
    }
    return {
        "question": m.group(1).strip(),
        "options": opts,
        "letter": letter,
        "correct_text": opts[letter],
    }


def _strip_embedded_option_labels(instruction: str) -> tuple[str, bool]:
    m = _MCQ_PATTERN.match(instruction.strip())
    if not m:
        return instruction, False

    opts = [m.group(i).strip() for i in range(2, 6)]
    cleaned = [_EMBEDDED_OPTION_LABEL.sub("", opt).strip() for opt in opts]
    if cleaned == opts:
        return instruction, False

    options_block = "\n".join(
        f"{letter}. {value}" for letter, value in zip("ABCD", cleaned)
    )
    return f"{m.group(1).strip()}\n\n{options_block}", True


def _rationale_text(output: str) -> str | None:
    m = _THINK_PATTERN.search(output)
    if not m:
        return None
    return " ".join(m.group(1).split())


def _is_suspect_rationale(output: str) -> bool:
    rationale = _rationale_text(output)
    if not rationale:
        return True
    lower = rationale.lower()
    if any(phrase in lower for phrase in _SUSPECT_RATIONALE_PHRASES):
        return True
    if len(rationale.split()) < 12:
        return True
    return False


def _has_valid_mcq_output(pair: dict) -> bool:
    if _parse_mcq(pair) is None:
        return False
    if not _ANSWER_PATTERN.search(pair.get("output", "")):
        return False
    return not _is_suspect_rationale(pair.get("output", ""))


def _generate_rationale(
    parsed: dict,
    model: str,
    api_sem: threading.Semaphore,
    timeout: int,
) -> str:
    prompt = _RATIONALE_PROMPT.format(
        question=parsed["question"],
        opt_a=parsed["options"]["A"],
        opt_b=parsed["options"]["B"],
        opt_c=parsed["options"]["C"],
        opt_d=parsed["options"]["D"],
        letter=parsed["letter"],
        correct_text=parsed["correct_text"],
    )
    rationale = _call_claude(prompt, model, api_sem, timeout)
    rationale = re.sub(r"^the correct answer is\s+[A-D]\.?\s*", "", rationale, flags=re.I)
    rationale = rationale.strip()
    if _is_suspect_rationale(f"<think>\n{rationale}\n</think>\n{parsed['letter']}"):
        raise ValueError(f"Generated suspect rationale: {rationale[:160]}")
    return rationale


def _template_rationale(parsed: dict) -> str:
    """Build a local-only rationale without sending data to an external model."""
    correct = parsed["correct_text"].rstrip(".")
    distractors = [
        f"{opt_letter} points to {text.rstrip('.')}"
        for opt_letter, text in parsed["options"].items()
        if opt_letter != parsed["letter"]
    ]
    if len(distractors) >= 3:
        contrast = (
            f"By contrast, {distractors[0]}, {distractors[1]}, and "
            f"{distractors[2]}, which do not match the relationship asked about."
        )
    else:
        contrast = "By contrast, the other options do not match the relationship asked about."
    return (
        f"{correct} is the option that directly matches the question's stated "
        f"agricultural concept or source-based detail. {contrast}"
    )


def _record_by_src_idx(records: list[dict]) -> dict[int, dict]:
    by_idx: dict[int, dict] = {}
    for rec in records:
        idx = rec.get("_src_idx")
        if isinstance(idx, int) and idx not in by_idx:
            by_idx[idx] = rec
    return by_idx


def _atomic_write_jsonl(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_name, output_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Original input JSONL")
    ap.add_argument("--current", required=True, help="Current generated JSONL")
    ap.add_argument("--output", required=True, help="Repaired output JSONL")
    ap.add_argument("--model", default="haiku", help="Claude model alias")
    ap.add_argument("--max-workers", type=int, default=4, help="Thread pool size")
    ap.add_argument(
        "--api-concurrency",
        type=int,
        default=1,
        help="Max concurrent claude -p API calls",
    )
    ap.add_argument("--timeout", type=int, default=90, help="Per-call timeout seconds")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and print counts without calling Claude or writing output",
    )
    ap.add_argument(
        "--template-only",
        action="store_true",
        help="Repair with local template rationales instead of calling Claude",
    )
    args = ap.parse_args()

    _register_shutdown()

    input_rows = _load_jsonl(Path(args.input))
    current_rows = _load_jsonl(Path(args.current))
    current_by_idx = _record_by_src_idx(current_rows)

    tasks: list[tuple[int, dict, str]] = []
    passthrough_indices: set[int] = set()
    label_fix_count = 0
    suspect_count = 0
    missing_count = 0

    for idx, source_pair in enumerate(input_rows):
        parsed_source = _parse_mcq(source_pair)
        current = current_by_idx.get(idx)
        if parsed_source is None:
            passthrough_indices.add(idx)
            continue
        if current is None:
            missing_count += 1
            tasks.append((idx, source_pair, "missing"))
            continue
        cleaned_instruction, changed = _strip_embedded_option_labels(current["instruction"])
        if changed:
            label_fix_count += 1
        candidate = {**current, "instruction": cleaned_instruction}
        if not _has_valid_mcq_output(candidate):
            suspect_count += 1
            tasks.append((idx, candidate, "suspect"))

    print(f"Input records:              {len(input_rows)}")
    print(f"Current records:            {len(current_rows)}")
    print(f"Repairs needed:             {len(tasks)}")
    print(f"  missing MCQs:             {missing_count}")
    print(f"  suspect MCQ rationales:   {suspect_count}")
    print(f"Option-label fixes:         {label_fix_count}")

    if args.dry_run:
        return

    api_sem = threading.Semaphore(args.api_concurrency)
    repaired: dict[int, dict] = {}
    failures: dict[int, str] = {}
    write_lock = threading.Lock()

    def repair_one(idx: int, pair: dict, reason: str) -> None:
        cleaned_instruction, _ = _strip_embedded_option_labels(pair["instruction"])
        clean_pair = {**pair, "instruction": cleaned_instruction}
        parsed = _parse_mcq(clean_pair)
        if parsed is None:
            failures[idx] = "not parseable as MCQ"
            return
        if args.template_only:
            rationale = _template_rationale(parsed)
        else:
            rationale = _generate_rationale(parsed, args.model, api_sem, args.timeout)
        new_output = f"<think>\n{rationale}\n</think>\n{parsed['letter']}"
        with write_lock:
            repaired[idx] = {
                "instruction": cleaned_instruction,
                "output": new_output,
                "_src_idx": idx,
            }
            log.info("Repaired idx=%d (%s)", idx, reason)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(repair_one, idx, pair, reason): idx
            for idx, pair, reason in tasks
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                with write_lock:
                    failures[idx] = str(exc)
                log.warning("Repair failed for idx=%d: %s", idx, exc)

    output_rows: list[dict] = []
    for idx, source_pair in enumerate(input_rows):
        current = current_by_idx.get(idx)
        if idx in repaired:
            output_rows.append(repaired[idx])
            continue
        if current is not None:
            cleaned_instruction, _ = _strip_embedded_option_labels(current["instruction"])
            output_rows.append({**current, "instruction": cleaned_instruction})
            continue
        output_rows.append({**source_pair, "_src_idx": idx})

    _atomic_write_jsonl(Path(args.output), output_rows)

    print(f"Repaired MCQs:              {len(repaired)}")
    print(f"Repair failures:            {len(failures)}")
    print(f"Output:                     {args.output}")
    if failures:
        first = sorted(failures.items())[:10]
        print("First failures:")
        for idx, error in first:
            print(f"  {idx}: {error[:160]}")


if __name__ == "__main__":
    main()
