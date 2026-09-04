#!/usr/bin/env python3
"""Assemble a gate-ready guardrails JSON for one candidate run.

Phase 5 #64/#66/#75. Real components computed here:

- ``bias.letter_distribution`` — the candidate's predicted-letter marginals
  from a score_mcq_ll.py result (the FORBIDDEN letter-bias protection).
- ``forgetting.relative_drop_percent`` — when ``--candidate-general-mcq`` and
  ``--baseline-general-mcq`` are both passed, the real relative drop in
  argmax accuracy on ``autoresearch/general_knowledge_probe.jsonl`` (a small
  frozen non-farming MCQ set, scored via the same frozen score_mcq_ll.py
  pipeline). Uses accuracy rather than log-likelihood because gate.py's
  ``relative_drop_percent`` requires a positive baseline.
- ``rubric`` / ``grader_anchor`` — when ``--periodic-guards`` is passed (the
  JSON written by ``measure_periodic_guards.py``), the real rubric drop and
  grader-anchor drift, so a stream's accept/reject decision is final the
  first time (chainlink #75) instead of needing a later out-of-band
  reconciliation like #66's reversal of #74.
- ``sealed_test_access.touched`` — false; run_loop's batch audit independently
  rejects the batch if any sealed path string appears anywhere.

``compute.elapsed_minutes`` is written as 0.0 and overwritten by run_loop
batch mode (``guardrails_inject_compute: true``), which is the only component
that knows the stream's real wall clock.

Without ``--periodic-guards``, ``rubric``/``grader_anchor`` remain
carried-forward placeholders (pre-#75 behavior, kept for backward
compatibility with specs that don't pass it).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LETTERS = ("A", "B", "C", "D")
CARRIED = "carried_forward_periodic_guard_not_measured_this_batch"


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
    parser.add_argument(
        "--periodic-guards",
        default=None,
        help=(
            "measure_periodic_guards.py JSON output; enables the real rubric "
            "and grader_anchor guards instead of carried-forward placeholders"
        ),
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

    def gold_marginals(path: str) -> dict[str, float]:
        """Gold answer-letter marginals of the scored eval.

        Guard-version change (2026-09-04): the null for letter-position bias is
        the eval's gold marginals, not a uniform reference. val_v3's 113 rows
        are skewed (A=15, C=43), so a candidate calibrated to gold deviates from
        uniform and was falsely flagged. The gold marginal is the same whichever
        model scored it, so this is computed from the candidate's per_item gold.
        """
        mcq = json.loads(Path(path).read_text(encoding="utf-8"))
        counts = Counter(row["gold"] for row in mcq["per_item"])
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

    if args.periodic_guards:
        periodic = json.loads(Path(args.periodic_guards).read_text(encoding="utf-8"))
        rubric = {
            "baseline_percent": periodic["rubric"]["baseline_percent"],
            "candidate_percent": periodic["rubric"]["candidate_percent"],
            "drop_percent_points": periodic["rubric"]["drop_percent_points"],
            "source": args.periodic_guards,
        }
        grader_anchor = {
            "max_abs_drift": periodic["grader_anchor"]["max_abs_drift"],
            "total_drift_alerts": periodic["grader_anchor"]["total_drift_alerts"],
            "source": args.periodic_guards,
        }
    else:
        rubric = {"drop_percent_points": 0.0, "status_note": CARRIED}
        grader_anchor = {"max_abs_drift": 0.0, "status_note": CARRIED}

    payload = {
        "bias": {
            "letter_distribution": candidate_dist,
            "baseline_letter_distribution": letter_marginals(args.baseline_mcq),
            "reference_distribution": gold_marginals(args.candidate_mcq),
            "reference_source": "gold_letter_marginals_of_scored_eval",
            "source": args.candidate_mcq,
            "baseline_source": args.baseline_mcq,
        },
        "compute": {
            "elapsed_minutes": 0.0,
            "status_note": "overwritten_by_run_loop_guardrails_inject_compute",
        },
        "forgetting": forgetting,
        "rubric": rubric,
        "grader_anchor": grader_anchor,
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
    if args.periodic_guards:
        print(f"rubric_drop_percent_points={rubric['drop_percent_points']:.4f}")
        print(f"grader_anchor_max_abs_drift={grader_anchor['max_abs_drift']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
