#!/usr/bin/env python3
"""make_synthetic_traces.py - Teacher-generated reasoning traces for Plan B SFT.

Chainlink issue #111 (I6), Epic #105, Milestone #2 (M2: Plan B SFT). See
TRACE_EVAL_REASONING_SFT_RLM_PLAN.md Part 1 / Plan B step 1.

For each MCQ-shaped TRAIN row in data/train_v3.jsonl (parsed via the frozen
score_mcq_ll.py's normalize_records -- the same ~1107-row MCQ subset that
pipeline already recognizes out of 4400 total train rows; the other ~3293
rows are free-form conversational SFT data with no "gold letter", out of
scope for this trace-SFT pass), calls a teacher model to produce a reasoning
trace that justifies and converges on the gold letter.

Teacher backend is pluggable (--teacher-backend), so a real run doesn't have
to spend metered OpenRouter credits when a local subscription-based CLI
agent is available:
  - claude   (default): `claude -p` non-interactive print mode, --restricted
    (no Bash/code-exec tools needed for a pure text-completion task).
  - codex:   `codex exec` non-interactive mode.
  - opencode: `opencode run` non-interactive mode.
  - openrouter: the original path, scripts/judge_training_data.py's
    client/retry plumbing (metered, needs OPENROUTER_API_KEY) -- kept as a
    real option, just no longer the default. Was the previous default before
    this was pointed out to have an unnecessary-cost default: the OpenRouter
    teacher's own DEFAULT_MODEL is "anthropic/claude-haiku-4-5" -- i.e. the
    original design would have paid OpenRouter's per-token rate to call a
    Claude model that's already reachable via this machine's own Claude Code
    subscription for free.

All backends share the exact same question/options formatting
(score_mcq_trace_ll.py's build_question_messages, and its leakage-marker
guard, for free) and the same TEACHER_SYSTEM_PROMPT/TEACHER_USER_TEMPLATE, so
switching backends never changes what's being asked, only who answers it.

Contamination pre-filter (before spending any API call): reuses
prepare_dataset_v3.py's _instruction_hash to reject any train row whose
instruction text exact-matches a row in data/val_v3.jsonl or
eval_freeform/dev.jsonl.

DELIBERATE DEVIATION FROM THE PLAN DOC, FLAGGED FOR REVIEW (chainlink #111
comment has the full note): Plan B step 2 also lists eval_freeform/test.jsonl
as a contamination-check source. This script does NOT read that file. Doing
so would require opening the sealed test set's raw content, which this
epic's own assumption 3 ("Never read eval_freeform/test.jsonl during any loop
step") and program.md's policy ("touched exactly once at final evaluation")
both forbid -- and which jspace_annotator.py / finetune.py /
score_mcq_trace_ll.py all refuse to do unconditionally elsewhere in this repo.
A sealed-test contamination check, if wanted, belongs in a separate,
explicitly-gated, one-time manual audit at final evaluation, not in a script
meant to be rerun while iterating on teacher prompts.

Usage:
  # default: local claude -p subscription, no API key needed
  python data/make_synthetic_traces.py \
    --output data/synthetic_traces_v1.jsonl \
    --limit 20   # smoke test; omit for the full ~1107-row MCQ pass

  # OpenRouter instead (needs OPENROUTER_API_KEY in env or .env):
  python data/make_synthetic_traces.py --teacher-backend openrouter --limit 20 ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "autoresearch"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from score_mcq_ll import normalize_records  # noqa: E402 -- frozen module, imported not edited
from score_mcq_trace_ll import GoldLeakageError, build_question_messages  # noqa: E402

_CLI_RETRY_DELAYS = [2, 5, 15]


def _run_teacher_subprocess(cmd: list[str], timeout: int) -> str:
    for attempt, delay in enumerate([0] + _CLI_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if attempt == len(_CLI_RETRY_DELAYS):
                raise
            log.warning("teacher CLI error (attempt %d): %s -- retrying", attempt + 1, exc)
    raise RuntimeError("All retries exhausted")


def call_teacher_claude(system: str, prompt: str, model: str | None, max_tokens: int) -> str:
    """`claude -p`, non-interactive print mode. --restricted drops the
    Bash/code-exec tools since this is a pure text-completion task with no
    need for them; --output-format text keeps stdout clean (no JSON wrapper
    to parse, no session/session-id bookkeeping)."""
    cmd = ["claude", "-p", prompt, "--append-system-prompt", system, "--output-format", "text", "--restricted"]
    if model:
        cmd.extend(["--model", model])
    return _run_teacher_subprocess(cmd, timeout=120)


def call_teacher_codex(system: str, prompt: str, model: str | None, max_tokens: int) -> str:
    """`codex exec`, non-interactive mode."""
    cmd = ["codex", "exec", f"{system}\n\n{prompt}"]
    if model:
        cmd.extend(["-c", f'model="{model}"'])
    return _run_teacher_subprocess(cmd, timeout=120)


def call_teacher_opencode(system: str, prompt: str, model: str | None, max_tokens: int) -> str:
    """`opencode run`, non-interactive mode, plain-text format."""
    cmd = ["opencode", "run", f"{system}\n\n{prompt}", "--format", "default"]
    if model:
        cmd.extend(["--model", model])
    return _run_teacher_subprocess(cmd, timeout=120)


def make_teacher_caller(backend: str, model: str | None, max_tokens: int, openrouter_key: str | None) -> Callable[[str, str], str]:
    """Return a (system, prompt) -> trace_text callable for the chosen backend.
    All backends share the same TEACHER_SYSTEM_PROMPT/TEACHER_USER_TEMPLATE
    (see build_teacher_prompt) -- only who answers changes, never what's asked."""
    if backend == "claude":
        return lambda system, prompt: call_teacher_claude(system, prompt, model, max_tokens)
    if backend == "codex":
        return lambda system, prompt: call_teacher_codex(system, prompt, model, max_tokens)
    if backend == "opencode":
        return lambda system, prompt: call_teacher_opencode(system, prompt, model, max_tokens)
    if backend == "openrouter":
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from judge_training_data import DEFAULT_MODEL, _call_judge, _get_client, _load_api_key  # noqa: E402

        api_key = _load_api_key(openrouter_key)
        client, resolved_model = _get_client(api_key, model or DEFAULT_MODEL)
        return lambda system, prompt: _call_judge(system, prompt, client, resolved_model, max_tokens=max_tokens)
    raise ValueError(f"unknown --teacher-backend: {backend!r}")

# Same hashing primitive prepare_dataset_v3.py's contamination filtering already
# uses (sha256 of the raw instruction text) -- kept identical so a row that
# passes/fails here is judged the same way the rest of the pipeline judges it.
def instruction_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


TEACHER_SYSTEM_PROMPT = (
    "You are an expert tutor writing worked-solution reasoning for a natural-farming and "
    "agriculture multiple-choice question bank. You will be given a question, four options, and "
    "the correct option. Write a concise (3-5 sentence) chain of reasoning that a knowledgeable "
    "person would use to arrive at the correct option -- ground it in the specific content of "
    "the correct option and briefly explain why the strongest distractor(s) are wrong. "
    "Do not mention that you were told the answer, do not say phrases like 'the answer is' or "
    "'the correct option is', and do not end by stating a bare letter (A/B/C/D) -- converge on "
    "the *content* of the right option in your own words and stop there."
)

TEACHER_USER_TEMPLATE = """Question:
{question_text}

Correct option: {gold_letter}. {gold_text}

Write the reasoning trace."""


def build_teacher_prompt(item: dict[str, Any]) -> tuple[str, str]:
    messages = build_question_messages(item)  # raises GoldLeakageError if the question text is contaminated
    question_text = messages[0]["content"]
    gold_letter = item["answer"]
    return TEACHER_SYSTEM_PROMPT, TEACHER_USER_TEMPLATE.format(
        question_text=question_text, gold_letter=gold_letter, gold_text=item[gold_letter]
    )


def load_eval_hash_denylist(paths: list[Path], text_field_candidates: tuple[str, ...]) -> set[str]:
    """Hash whichever text field is present per row across the given (non-sealed) files."""
    denylist: set[str] = set()
    for path in paths:
        if not path.exists():
            log.warning("contamination-check source not found, skipping: %s", path)
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for field in text_field_candidates:
                    if field in row and row[field]:
                        denylist.add(instruction_hash(str(row[field])))
                        break
    return denylist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default="data/train_v3.jsonl")
    parser.add_argument("--val-data", default="data/val_v3.jsonl", help="Non-sealed contamination-check source")
    parser.add_argument(
        "--freeform-dev", default="eval_freeform/dev.jsonl", help="Non-sealed contamination-check source"
    )
    parser.add_argument("--output", default="data/synthetic_traces_v1.jsonl")
    parser.add_argument(
        "--teacher-backend",
        choices=("claude", "codex", "opencode", "openrouter"),
        default="claude",
        help="claude/codex/opencode use a local subscription-based CLI (no metered API cost); "
        "openrouter needs OPENROUTER_API_KEY (default: claude)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override passed to the chosen backend (backend-specific format, e.g. "
        "'sonnet' for claude, 'o3' for codex, 'anthropic/claude-haiku-4-5' for openrouter). "
        "Default: let the backend pick its own default.",
    )
    parser.add_argument("--openrouter-key", default=None, help="Only used by --teacher-backend openrouter")
    parser.add_argument("--temperature", type=float, default=0.7, help="Only used by --teacher-backend openrouter")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N MCQ train rows")
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of teacher calls to run in parallel (each is an independent subprocess/API "
        "call). At ~4-5s/call for the claude backend, 1107 rows sequentially is ~90 min; "
        "raise this to run faster.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.train_data, encoding="utf-8") as handle:
        train_rows = [json.loads(line) for line in handle if line.strip()]
    records, skipped = normalize_records(train_rows)
    log.info("train_data=%s: %d MCQ-shaped rows, %d non-MCQ (conversational, out of scope) rows",
              args.train_data, len(records), len(skipped))

    denylist = load_eval_hash_denylist(
        [Path(args.val_data), Path(args.freeform_dev)], text_field_candidates=("instruction", "question")
    )
    log.info("contamination denylist: %d hashes from %s, %s", len(denylist), args.val_data, args.freeform_dev)

    from score_mcq_ll import normalize_mcq_record, McqRecordError  # noqa: E402

    eligible = []
    contaminated_examples = []
    for row_number, row in enumerate(train_rows, start=1):
        if "instruction" not in row or "output" not in row:
            continue
        h = instruction_hash(row["instruction"])
        if h in denylist:
            contaminated_examples.append(row_number)
            continue
        try:
            item = normalize_mcq_record(row, row_number)
        except McqRecordError:
            continue
        eligible.append(item)

    log.info(
        "%d/%d MCQ train rows dropped for contamination (exact hash match against %s/%s); %d eligible",
        len(contaminated_examples), len(records), args.val_data, args.freeform_dev, len(eligible),
    )

    if args.limit is not None:
        eligible = eligible[: args.limit]
    if not eligible:
        parser.error("no eligible (non-contaminated, MCQ-shaped) train rows found")

    teacher_call = make_teacher_caller(args.teacher_backend, args.model, args.max_tokens, args.openrouter_key)
    log.info("teacher backend: %s (model=%s)", args.teacher_backend, args.model or "backend default")

    def process_one(item: dict[str, Any]) -> dict[str, Any] | None:
        try:
            system, prompt = build_teacher_prompt(item)
        except GoldLeakageError as exc:
            log.warning("skipping %s: %s", item.get("question_id"), exc)
            return None
        try:
            trace = teacher_call(system, prompt)
        except Exception as exc:  # noqa: BLE001 -- teacher call, log and continue past transient failures
            log.warning("teacher call failed for %s: %s", item.get("question_id"), exc)
            return None
        return {
            "question_id": item["question_id"],
            "question": item["question"],
            "A": item["A"],
            "B": item["B"],
            "C": item["C"],
            "D": item["D"],
            "gold": item["answer"],
            "teacher_trace": trace,
            "teacher_backend": args.teacher_backend,
            "teacher_model": args.model,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_failed = 0
    n_done = 0
    with output_path.open("w", encoding="utf-8") as out_handle:
        if args.concurrency <= 1:
            results = (process_one(item) for item in eligible)
        else:
            from concurrent.futures import ThreadPoolExecutor

            # Independent subprocess calls (one `claude -p`/`codex exec`/etc. per item) --
            # thread-based concurrency is safe here since each call owns its own process.
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = list(pool.map(process_one, eligible))
        for result in results:
            n_done += 1
            if result is None:
                n_failed += 1
                continue
            out_handle.write(json.dumps(result, sort_keys=True) + "\n")
            n_written += 1
            if args.verbose and n_done % 25 == 0:
                log.info("progress: %d/%d processed (%d written, %d failed)", n_done, len(eligible), n_written, n_failed)

    print(f"wrote {output_path} ({n_written} traces, {n_failed} failed/skipped)")
    print(f"mcq_train_rows={len(records)}")
    print(f"contaminated_dropped={len(contaminated_examples)}")
    print(f"eligible={len(eligible)}")
    print(f"traces_written={n_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
