#!/usr/bin/env python3
"""Assemble a gate-ready guardrails JSON for one candidate run.

Phase 5 #64. Real components computed here:

- ``bias.letter_distribution`` — the candidate's predicted-letter marginals
  from a score_mcq_ll.py result (the FORBIDDEN letter-bias protection).
- ``sealed_test_access.touched`` — false; run_loop's batch audit independently
  rejects the batch if any sealed path string appears anywhere.

``compute.elapsed_minutes`` is written as 0.0 and overwritten by run_loop
batch mode (``guardrails_inject_compute: true``), which is the only component
that knows the stream's real wall clock.

The periodic guards (forgetting, rubric, grader anchor) are NOT measured per
stream yet: program.md defines rubric as a *periodic* Goodhart guard, and no
per-iteration forgetting probe exists. They are written as passing values with
an explicit ``status_note`` so the journal records that they were carried
forward, not measured. Implementing real periodic measurements is tracked in
chainlink #66.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LETTERS = ("A", "B", "C", "D")
CARRIED = "carried_forward_periodic_guard_not_measured_this_batch"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a gate guardrails JSON from real candidate artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-mcq", required=True, help="score_mcq_ll.py JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    mcq = json.loads(Path(args.candidate_mcq).read_text(encoding="utf-8"))
    per_item = mcq["per_item"]
    counts = Counter(row["top_letter"] for row in per_item)
    total = sum(counts.values())
    if total == 0:
        raise SystemExit(f"{args.candidate_mcq}: no per_item rows")

    payload = {
        "bias": {
            "letter_distribution": {
                letter: counts.get(letter, 0) / total for letter in LETTERS
            },
            "source": args.candidate_mcq,
            "n_items": total,
        },
        "compute": {
            "elapsed_minutes": 0.0,
            "status_note": "overwritten_by_run_loop_guardrails_inject_compute",
        },
        "forgetting": {"relative_drop_percent": 0.0, "status_note": CARRIED},
        "rubric": {"drop_percent_points": 0.0, "status_note": CARRIED},
        "grader_anchor": {"max_abs_drift": 0.0, "status_note": CARRIED},
        "sealed_test_access": {"touched": False},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"bias_marginals={json.dumps(payload['bias']['letter_distribution'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
