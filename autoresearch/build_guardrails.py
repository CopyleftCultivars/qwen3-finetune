#!/usr/bin/env python3
"""Assemble a gate-ready guardrails JSON for one candidate run.

Phase 5 #64/#66. Real components computed here:

- ``bias.letter_distribution`` — the candidate's predicted-letter marginals
  from a score_mcq_ll.py result (the FORBIDDEN letter-bias protection).
- ``forgetting.relative_drop_percent`` — when ``--candidate-general-mcq`` and
  ``--baseline-general-mcq`` are both passed, the real relative drop in
  argmax accuracy on ``autoresearch/general_knowledge_probe.jsonl`` (a small
  frozen non-farming MCQ set, scored via the same frozen score_mcq_ll.py
  pipeline). Uses accuracy rather than log-likelihood because gate.py's
  ``relative_drop_percent`` requires a positive baseline.
- ``sealed_test_access.touched`` — false; run_loop's batch audit independently
  rejects the batch if any sealed path string appears anywhere.

``compute.elapsed_minutes`` is written as 0.0 and overwritten by run_loop
batch mode (``guardrails_inject_compute: true``), which is the only component
that knows the stream's real wall clock.

``rubric`` and ``grader_anchor`` remain carried-forward placeholders: both
require ``eval_freeform/score_rubric.py``'s LLM judge, which needs
``OPENROUTER_API_KEY``. That key is not configured for this project (chainlink
#66 comment, 2026-07-18) — implementing them is blocked on either providing
the key or a human decision to point the judge at a different backend.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LETTERS = ("A", "B", "C", "D")
CARRIED = "carried_forward_periodic_guard_not_measured_this_batch"
BLOCKED_NO_OPENROUTER_KEY = (
    "blocked_on_missing_OPENROUTER_API_KEY_see_chainlink_66"
)


def relative_drop_percent(baseline: float, candidate: float) -> float:
    """Same formula as gate.py's guard: positive means the candidate dropped."""
    if baseline <= 0:
        raise ValueError("baseline score must be > 0 for relative drop guard")
    return 100.0 * (baseline - candidate) / baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a gate guardrails JSON from real candidate artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-mcq", required=True, help="score_mcq_ll.py JSON")
    parser.add_argument(
        "--baseline-mcq",
        required=True,
        help="Champion score_mcq_ll.py JSON; enables the relative bias rule",
    )
    parser.add_argument(
        "--candidate-general-mcq",
        default=None,
        help=(
            "score_mcq_ll.py JSON for the candidate scored on "
            "general_knowledge_probe.jsonl; enables the real forgetting guard"
        ),
    )
    parser.add_argument(
        "--baseline-general-mcq",
        default=None,
        help="Same, for the champion; required together with --candidate-general-mcq",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    def letter_marginals(path: str) -> dict[str, float]:
        mcq = json.loads(Path(path).read_text(encoding="utf-8"))
        counts = Counter(row["top_letter"] for row in mcq["per_item"])
        total = sum(counts.values())
        if total == 0:
            raise SystemExit(f"{path}: no per_item rows")
        return {letter: counts.get(letter, 0) / total for letter in LETTERS}

    def accuracy(path: str) -> float:
        mcq = json.loads(Path(path).read_text(encoding="utf-8"))
        return float(mcq["summary"]["argmax_accuracy"])

    if bool(args.candidate_general_mcq) != bool(args.baseline_general_mcq):
        parser.error(
            "--candidate-general-mcq and --baseline-general-mcq must be passed together"
        )

    candidate_dist = letter_marginals(args.candidate_mcq)

    if args.candidate_general_mcq:
        baseline_acc = accuracy(args.baseline_general_mcq)
        candidate_acc = accuracy(args.candidate_general_mcq)
        forgetting = {
            "probe": "autoresearch/general_knowledge_probe.jsonl",
            "baseline_accuracy": baseline_acc,
            "candidate_accuracy": candidate_acc,
            "relative_drop_percent": relative_drop_percent(baseline_acc, candidate_acc),
            "source": args.candidate_general_mcq,
            "baseline_source": args.baseline_general_mcq,
        }
    else:
        forgetting = {"relative_drop_percent": 0.0, "status_note": CARRIED}

    payload = {
        "bias": {
            "letter_distribution": candidate_dist,
            "baseline_letter_distribution": letter_marginals(args.baseline_mcq),
            "source": args.candidate_mcq,
            "baseline_source": args.baseline_mcq,
        },
        "compute": {
            "elapsed_minutes": 0.0,
            "status_note": "overwritten_by_run_loop_guardrails_inject_compute",
        },
        "forgetting": forgetting,
        "rubric": {"drop_percent_points": 0.0, "status_note": BLOCKED_NO_OPENROUTER_KEY},
        "grader_anchor": {"max_abs_drift": 0.0, "status_note": BLOCKED_NO_OPENROUTER_KEY},
        "sealed_test_access": {"touched": False},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"bias_marginals={json.dumps(payload['bias']['letter_distribution'], sort_keys=True)}")
    print(
        "baseline_bias_marginals="
        + json.dumps(payload["bias"]["baseline_letter_distribution"], sort_keys=True)
    )
    if args.candidate_general_mcq:
        print(
            f"forgetting_relative_drop_percent={forgetting['relative_drop_percent']:.4f} "
            f"(baseline_acc={forgetting['baseline_accuracy']:.4f} "
            f"candidate_acc={forgetting['candidate_accuracy']:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
