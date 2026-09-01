#!/usr/bin/env python3
"""filter_synthetic_traces.py - Contamination audit + causal verification filter.

Chainlink issue #112 (I7), Epic #105, Milestone #2 (M2: Plan B SFT). See
TRACE_EVAL_REASONING_SFT_RLM_PLAN.md Part 1 / Plan B steps 2-3.

Two checks on a make_synthetic_traces.py (#111 / I6) output file:

1. Contamination audit (report, not just trust I6's pre-filter): recompute
   instruction/question-hash overlap against data/val_v3.jsonl and
   eval_freeform/dev.jsonl. Never eval_freeform/test.jsonl -- same sealed-test
   rationale as make_synthetic_traces.py's docstring; reading that file's raw
   content to hash it is exactly the read this repo's sealed-test wall exists
   to prevent.

2. Causal verification (the actual quality filter): score each teacher trace
   with --model (default unsloth/Qwen3-0.6B, "the same base model" Plan B
   step 5 SFTs -- this repo's phases always train fresh from base, they don't
   continue from the v5clean adapter, unlike the separate GRPO epic), using
   score_mcq_trace_ll.py's build_trace_prompt/score_symbol -- the identical
   Pass-2 scoring mechanism the eval pipeline uses. Keep a row only if that
   scoring's argmax equals the gold letter; a trace that doesn't cause the
   base model to actually pick the right answer isn't a useful SFT target no
   matter how well-written it reads. Drop everything else and report the
   drop rate.

Usage:
  python data/filter_synthetic_traces.py \
    --input data/synthetic_traces_v1.jsonl \
    --model unsloth/Qwen3-0.6B \
    --output data/synthetic_traces_v1_verified.jsonl \
    --report data/synthetic_traces_v1_verification_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "autoresearch"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from score_mcq_ll import DEFAULT_DIAGNOSTICS_DIR, LETTERS, _softmax_probs, import_diagnostics_helpers  # noqa: E402
from score_mcq_trace_ll import score_with_trace  # noqa: E402
from make_synthetic_traces import instruction_hash, load_eval_hash_denylist  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="make_synthetic_traces.py output JSONL")
    parser.add_argument("--val-data", default="data/val_v3.jsonl")
    parser.add_argument("--freeform-dev", default="eval_freeform/dev.jsonl")
    parser.add_argument("--model", default="unsloth/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostics-dir", default=str(DEFAULT_DIAGNOSTICS_DIR))
    parser.add_argument("--output", required=True, help="Verified (causally-passing) rows only")
    parser.add_argument("--report", default=None, help="Default: --output with _report.json suffix")
    parser.add_argument(
        "--full-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use diagnostics full-precision (bf16) model loading instead of 4-bit NF4.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.input, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    log.info("loaded %d synthetic-trace rows from %s", len(rows), args.input)

    # Check 1: contamination audit (report; I6 already pre-filtered at generation time).
    denylist = load_eval_hash_denylist(
        [Path(args.val_data), Path(args.freeform_dev)], text_field_candidates=("instruction", "question")
    )
    contamination_hits = [row["question_id"] for row in rows if instruction_hash(row["question"]) in denylist]
    log.info("contamination audit: %d/%d rows overlap %s/%s", len(contamination_hits), len(rows), args.val_data, args.freeform_dev)
    rows = [row for row in rows if row["question_id"] not in contamination_hits]

    # Check 2: causal verification -- does the trace actually cause the base model
    # to pick the gold letter when scored the same way score_mcq_trace_ll.py's
    # Pass 2 scores a self-generated trace?
    helpers = import_diagnostics_helpers(Path(args.diagnostics_dir))
    model, tokenizer = helpers["load_model_and_tokenizer"](args.model, full_precision=args.full_precision, device=args.device)
    letter_ids = helpers["get_letter_token_ids"](tokenizer)
    score_symbol = helpers["score_symbol"]

    verified: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        item = {"question_id": row["question_id"], "question": row["question"],
                "A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"], "answer": row["gold"]}
        logprobs = score_with_trace(model, tokenizer, item, row["teacher_trace"], letter_ids, score_symbol, args.device)
        probs = _softmax_probs(logprobs)
        top_letter = max(LETTERS, key=lambda letter: logprobs[letter])
        causally_correct = top_letter == row["gold"]
        enriched = {
            **row,
            "causal_top_letter": top_letter,
            "causal_gold_logprob": logprobs[row["gold"]],
            "causal_option_set_gold_logprob": math.log(probs[row["gold"]]),
            "causal_verified": causally_correct,
        }
        (verified if causally_correct else dropped).append(enriched)

    total_after_contamination = len(rows)
    drop_rate = len(dropped) / total_after_contamination if total_after_contamination else 0.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in verified:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    report_path = Path(args.report) if args.report else output_path.with_name(output_path.stem + "_report.json")
    report = {
        "format": "autoresearch_synthetic_trace_verification_report_v1",
        "chainlink_issue": 112,
        "input": args.input,
        "model": args.model,
        "input_rows": len(rows) + len(contamination_hits),
        "contamination_dropped": len(contamination_hits),
        "contamination_check_sources": [args.val_data, args.freeform_dev],
        "contamination_check_excludes_sealed_test": True,
        "post_contamination_rows": total_after_contamination,
        "causal_verified_count": len(verified),
        "causal_dropped_count": len(dropped),
        "causal_drop_rate": drop_rate,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {output_path} ({len(verified)} verified rows)")
    print(f"wrote {report_path}")
    print(f"contamination_dropped={len(contamination_hits)}")
    print(f"causal_drop_rate={drop_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
