#!/usr/bin/env python3
"""prepare_dataset_v6.py - Splice verified synthetic reasoning traces into the
v3 training mix, keeping think/answer separable.

Chainlink issue #113 (I8), Epic #105, Milestone #2 (M2: Plan B SFT). See
TRACE_EVAL_REASONING_SFT_RLM_PLAN.md Part 1 / Plan B step 4.

Does NOT reimplement prepare_dataset_v3.py's conversational-loading /
cyclic-shift-expansion / dedup machinery (494+ lines, non-trivial, already
correct) -- instead post-processes its already-built output
(data/train_v3.jsonl), substituting the reasoning content for exactly the
rows that have a causally-verified synthetic trace
(filter_synthetic_traces.py, #112 / I7), keyed by question_id. That id is
"row_{line_number:06d}" (make_synthetic_traces.py's fallback via
normalize_mcq_record when no explicit question_id exists), which is a stable,
deterministic 1:1 mapping back to data/train_v3.jsonl's exact line -- correct
even though prepare_dataset_v3.py's cyclic-shift expansion means the same QMD
source question appears multiple times with different option letterings; the
trace was generated against one specific shifted variant, so it must only
ever be spliced back into that same line.

Target format reuses prepare_dataset_v3.py's own _format_sft_output(bare-letter)
shape unchanged: "<think>\n{reasoning}\n</think>\n{answer_letter}" -- levers.md's
already-SETTLED SFT target format. Rows without a verified trace keep their
original v3 output (usually an empty-think bare-letter target) unchanged, so
MCQ coverage doesn't shrink just because a row lacked (or failed) a trace.
Conversational (non-MCQ) rows and data/val_v3.jsonl are passed through
unchanged -- Plan B doesn't ask to trace-augment the validation set (it's
scored, unmodified, as score_mcq_trace_ll.py's baseline already does).

Usage:
  python prepare_dataset_v6.py \
    --base-train data/train_v3.jsonl \
    --base-val data/val_v3.jsonl \
    --verified-traces data/synthetic_traces_v1_verified.jsonl \
    --output-train data/train_v6.jsonl \
    --output-val data/val_v6.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "autoresearch"))
sys.path.insert(0, str(REPO_ROOT))

from score_mcq_ll import _extract_gold_from_output  # noqa: E402 -- frozen module, imported not edited
from prepare_dataset_v3 import _format_sft_output  # noqa: E402 -- not frozen, reused not duplicated


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_trace_index(verified_path: Path) -> dict[str, str]:
    """question_id ("row_NNNNNN") -> causally-verified teacher_trace."""
    index: dict[str, str] = {}
    for row in load_jsonl(verified_path):
        index[row["question_id"]] = row["teacher_trace"]
    return index


def splice_traces(train_rows: list[dict[str, Any]], trace_index: dict[str, str]) -> tuple[list[dict[str, Any]], int]:
    spliced: list[dict[str, Any]] = []
    n_spliced = 0
    for row_number, row in enumerate(train_rows, start=1):
        question_id = f"row_{row_number:06d}"
        trace = trace_index.get(question_id)
        if trace is None or "instruction" not in row or "output" not in row:
            spliced.append(row)
            continue
        try:
            gold_letter = _extract_gold_from_output(row["output"])
        except Exception:  # noqa: BLE001 -- not an MCQ-shaped output, leave the row untouched
            spliced.append(row)
            continue
        new_row = dict(row)
        new_row["output"] = _format_sft_output(trace, gold_letter, sft_format="bare-letter")
        spliced.append(new_row)
        n_spliced += 1
    return spliced, n_spliced


def schema_check(rows: list[dict[str, Any]], label: str) -> None:
    """Same validation train.py::load_jsonl applies to each record, inlined
    rather than imported: importing train.py triggers its own top-level
    `from unsloth import FastLanguageModel` (gated on train.py's sys.argv
    seeing "-h"/"--help", which this script's argv never will), pulling in a
    heavy training-only dependency just to validate a JSONL schema. Keep this
    in sync with train.py::load_jsonl if that function's validation changes."""
    for lineno, row in enumerate(rows, start=1):
        if "messages" in row:
            messages = row["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{label}:{lineno}: messages must contain at least two turns")
            for turn in messages:
                if not isinstance(turn, dict) or turn.get("role") not in {"system", "user", "assistant"}:
                    raise ValueError(f"{label}:{lineno}: invalid message role")
                if not isinstance(turn.get("content"), str) or not turn["content"].strip():
                    raise ValueError(f"{label}:{lineno}: message content must be non-empty text")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{label}:{lineno}: messages must end with an assistant target")
        elif "instruction" not in row or "output" not in row:
            raise ValueError(f"{label}:{lineno}: record must have 'instruction'/'output' or 'messages' fields")
    log.info("schema check passed: %d %s records are valid train.py inputs", len(rows), label)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-train", default="data/train_v3.jsonl")
    parser.add_argument("--base-val", default="data/val_v3.jsonl")
    parser.add_argument("--verified-traces", required=True, help="filter_synthetic_traces.py output")
    parser.add_argument("--output-train", default="data/train_v6.jsonl")
    parser.add_argument("--output-val", default="data/val_v6.jsonl")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    train_rows = load_jsonl(Path(args.base_train))
    trace_index = build_trace_index(Path(args.verified_traces))
    spliced_train, n_spliced = splice_traces(train_rows, trace_index)
    log.info("spliced %d/%d verified traces into %s", n_spliced, len(trace_index), args.base_train)

    schema_check(spliced_train, "train")
    val_rows = load_jsonl(Path(args.base_val))
    schema_check(val_rows, "val")  # unchanged passthrough, but still worth proving valid

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_train, "w", encoding="utf-8") as handle:
        for row in spliced_train:
            handle.write(json.dumps(row) + "\n")
    with open(args.output_val, "w", encoding="utf-8") as handle:
        for row in val_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"wrote {args.output_train} ({len(spliced_train)} rows, {n_spliced} trace-spliced)")
    print(f"wrote {args.output_val} ({len(val_rows)} rows, unchanged passthrough)")
    print(f"traces_available={len(trace_index)}")
    print(f"traces_spliced={n_spliced}")
    print("schema_check=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
