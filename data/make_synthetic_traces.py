#!/usr/bin/env python3
"""make_synthetic_traces.py - Teacher-generated reasoning traces for Plan B SFT.

Chainlink issue #111 (I6), Epic #105, Milestone #2 (M2: Plan B SFT). See
TRACE_EVAL_REASONING_SFT_RLM_PLAN.md Part 1 / Plan B step 1.

For each MCQ-shaped TRAIN row in data/train_v3.jsonl (parsed via the frozen
score_mcq_ll.py's normalize_records -- the same ~1107-row MCQ subset that
pipeline already recognizes out of 4400 total train rows; the other ~3293
rows are free-form conversational SFT data with no "gold letter", out of
scope for this trace-SFT pass), calls an OpenRouter teacher model to produce
a reasoning trace that justifies and converges on the gold letter. Reuses
scripts/judge_training_data.py's client/retry plumbing rather than
reimplementing OpenRouter calling, and score_mcq_trace_ll.py's
build_question_messages for the exact same question/options text formatting
the eval pipeline uses (and its leakage-marker guard, for free).

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
  export OPENROUTER_API_KEY=...  # or set it in .env (already present here)
  python data/make_synthetic_traces.py \
    --train-data data/train_v3.jsonl \
    --val-data data/val_v3.jsonl \
    --freeform-dev eval_freeform/dev.jsonl \
    --output data/synthetic_traces_v1.jsonl \
    --limit 20   # smoke test; omit for the full ~1107-row MCQ pass
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "autoresearch"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from score_mcq_ll import normalize_records  # noqa: E402 -- frozen module, imported not edited
from score_mcq_trace_ll import GoldLeakageError, build_question_messages  # noqa: E402
from judge_training_data import DEFAULT_MODEL, _call_judge, _get_client, _load_api_key  # noqa: E402

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
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter teacher model (default: {DEFAULT_MODEL})")
    parser.add_argument("--openrouter-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N MCQ train rows")
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

    api_key = _load_api_key(args.openrouter_key)
    client, model = _get_client(api_key, args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_failed = 0
    with output_path.open("w", encoding="utf-8") as out_handle:
        for item in eligible:
            try:
                system, prompt = build_teacher_prompt(item)
            except GoldLeakageError as exc:
                log.warning("skipping %s: %s", item.get("question_id"), exc)
                n_failed += 1
                continue
            try:
                trace = _call_judge(system, prompt, client, model, max_tokens=args.max_tokens)
            except Exception as exc:  # noqa: BLE001 -- teacher API call, log and continue past transient failures
                log.warning("teacher call failed for %s: %s", item.get("question_id"), exc)
                n_failed += 1
                continue
            out_handle.write(
                json.dumps(
                    {
                        "question_id": item["question_id"],
                        "question": item["question"],
                        "A": item["A"],
                        "B": item["B"],
                        "C": item["C"],
                        "D": item["D"],
                        "gold": item["answer"],
                        "teacher_trace": trace,
                        "teacher_model": model,
                        "teacher_temperature": args.temperature,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            n_written += 1

    print(f"wrote {output_path} ({n_written} traces, {n_failed} failed/skipped)")
    print(f"mcq_train_rows={len(records)}")
    print(f"contaminated_dropped={len(contaminated_examples)}")
    print(f"eligible={len(eligible)}")
    print(f"traces_written={n_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
