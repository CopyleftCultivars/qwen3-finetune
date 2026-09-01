#!/usr/bin/env python3
"""gate_with_trace_track.py - PARALLEL trace-conditioned track for gate.py.

Chainlink issue #110 (I5), Epic #105, Milestone #1 (M1: Plan A eval). See
TRACE_EVAL_REASONING_SFT_RLM_PLAN.md Part 1 / Plan A step 7-8 and
autoresearch/objective_weights_v3.json for the policy this implements.

`gate.py` is frozen (autoresearch/frozen_manifest.json) and this epic's own
constraint (TRACE_EVAL_REASONING_SFT_RLM_PLAN.md assumption 2) is to add new
modules rather than edit frozen ones. This wrapper follows the same pattern
`build_guardrails.py` already uses for assembling gate.py inputs: it imports
`gate.decide` UNCHANGED and calls it exactly as `gate.py`'s own CLI would --
the v2 primary accept/reject/pause decision is byte-identical to running
`gate.py` directly with the same --baseline/--candidate/--guardrails/--config.

The trace-conditioned track is attached as an separate, clearly-marked
`parallel_tracks.trace_conditioned` block and NEVER influences `decision` /
`recommended_action` -- it is informational only until objective_weights_v3.json's
`promotion_gate` is explicitly flipped (chainlink #110 does not do that; it
only wires the reporting path). Passing no --trace-* args makes this script a
pure pass-through to gate.decide().

Usage (byte-identical to gate.py when no --trace-* args given):
  python autoresearch/gate_with_trace_track.py \
    --baseline results/primary_objective_v5_clean.json \
    --candidate results/primary_objective_<candidate>.json \
    --guardrails results/autoresearch_guardrails_<candidate>.json \
    --output results/autoresearch_gate_<candidate>.json

  # with the trace-conditioned parallel track attached:
  python autoresearch/gate_with_trace_track.py \
    --baseline results/primary_objective_v5_clean.json \
    --candidate results/primary_objective_<candidate>.json \
    --guardrails results/autoresearch_guardrails_<candidate>.json \
    --trace-baseline results/mcq_trace_ll_v5_clean.json \
    --trace-candidate results/mcq_trace_ll_<candidate>.json \
    --output results/autoresearch_gate_<candidate>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate  # noqa: E402 -- frozen module, imported not edited


def load_trace_track(path: Path) -> dict[str, Any]:
    """Reduce a score_mcq_trace_ll.py result to the fields this track reports."""
    data = gate.load_json(path)
    summary = data.get("summary", data)
    return {
        "source": str(path),
        "model": summary.get("model"),
        "mean_trace_option_set_gold_logprob": summary.get("mean_trace_option_set_gold_logprob"),
        "mean_delta_trace": summary.get("mean_delta_trace"),
        "delta_trace_positive_fraction": summary.get("delta_trace_positive_fraction"),
        "argmax_accuracy": summary.get("argmax_accuracy"),
        "empty_trace_argmax_accuracy": summary.get("empty_trace_argmax_accuracy"),
        "scored_mcq_rows": summary.get("scored_mcq_rows"),
    }


def build_trace_parallel_track(baseline_path: Path | None, candidate_path: Path | None) -> dict[str, Any]:
    track: dict[str, Any] = {
        "format": "autoresearch_trace_parallel_track_v1",
        "role": "informational_only_not_gating",
        "policy": "autoresearch/objective_weights_v3.json",
        "note": (
            "This block never influences 'decision'/'recommended_action' above. "
            "objective_weights_v3.json's promotion_gate.status is 'parallel_track_not_yet_gating'; "
            "this script only reports the track, it does not decide promotion."
        ),
    }
    if baseline_path is not None:
        track["baseline"] = load_trace_track(baseline_path)
    if candidate_path is not None:
        track["candidate"] = load_trace_track(candidate_path)
    if baseline_path is not None and candidate_path is not None:
        base = track["baseline"]["mean_trace_option_set_gold_logprob"]
        cand = track["candidate"]["mean_trace_option_set_gold_logprob"]
        if base is not None and cand is not None:
            track["trace_primary_delta"] = cand - base
        track["candidate_delta_trace_positive_fraction"] = track["candidate"]["delta_trace_positive_fraction"]
        track["step8_majority_positive_clears"] = (
            track["candidate"]["delta_trace_positive_fraction"] is not None
            and track["candidate"]["delta_trace_positive_fraction"] > 0.5
        )
    return track


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="gate.py wrapper that attaches an informational trace-conditioned parallel track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=gate.DEFAULT_CONFIG)
    parser.add_argument("--baseline", default=None, help="Baseline score_primary.py JSON (v2 primary)")
    parser.add_argument("--candidate", default=None, help="Candidate score_primary.py JSON (v2 primary)")
    parser.add_argument("--guardrails", default=None, help="Guardrail report JSON")
    parser.add_argument("--output", default=None, help="Decision JSON output path")
    parser.add_argument(
        "--budget-key",
        default="compute_budget",
        help="Passed through to gate.decide() unchanged -- see gate.py --help.",
    )
    parser.add_argument("--trace-baseline", default=None, help="Baseline score_mcq_trace_ll.py JSON (I4/#109)")
    parser.add_argument("--trace-candidate", default=None, help="Candidate score_mcq_trace_ll.py JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = gate.load_json(Path(args.config))
    gate.validate_config(config, budget_key=args.budget_key)

    missing = [name for name in ("baseline", "candidate", "guardrails") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing required argument(s): {', '.join(missing)}")

    result = gate.decide(
        gate.load_json(Path(args.baseline)),
        gate.load_json(Path(args.candidate)),
        gate.load_json(Path(args.guardrails)),
        config,
        budget_key=args.budget_key,
    )

    if args.trace_baseline or args.trace_candidate:
        result["parallel_tracks"] = {
            "trace_conditioned": build_trace_parallel_track(
                Path(args.trace_baseline) if args.trace_baseline else None,
                Path(args.trace_candidate) if args.trace_candidate else None,
            )
        }

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"decision={result['decision']}")
    print(f"recommended_action={result['recommended_action']}")
    print(f"primary_delta={result['primary_delta']:.10f}")
    print(f"primary_delta_threshold={result['primary_delta_threshold']:.10f}")
    if "parallel_tracks" in result:
        tt = result["parallel_tracks"]["trace_conditioned"]
        if "candidate_delta_trace_positive_fraction" in tt:
            print(f"trace_track.delta_trace_positive_fraction={tt['candidate_delta_trace_positive_fraction']}")
            print(f"trace_track.step8_majority_positive_clears={tt['step8_majority_positive_clears']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
