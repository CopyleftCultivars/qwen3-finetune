#!/usr/bin/env python3
"""
Termination checker for the MCQ regeneration loop (Issue #67).

Reads per-round stats from regenerated_mcqs/{category}_round*/round_stats.json
and decides whether the loop should continue or stop.

Stopping conditions (first match wins):
  1. TARGET REACHED   — all topics have accepted >= needed (topics_remaining is empty)
  2. MAX ROUNDS       — completed rounds >= --max-rounds (default: 3)
  3. YIELD COLLAPSE   — acceptance_rate in the latest round < --min-yield (default: 0.20)
  4. NO PROGRESS      — latest round filled 0 new topics and generated > 0 items

Exit codes
----------
  0 — continue (next round number printed to stdout)
  1 — stop (reason printed to stdout)
  2 — error / missing data

Usage
-----
    # Check whether to run another round
    conda run -n nf-benchmark python quality/termination_checker.py \\
        --category natural_fertilizers

    # With custom limits
    conda run -n nf-benchmark python quality/termination_checker.py \\
        --category natural_fertilizers --max-rounds 4 --min-yield 0.15

    # Machine-readable JSON output
    conda run -n nf-benchmark python quality/termination_checker.py \\
        --category natural_fertilizers --json
"""

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
REGEN_DIR = WORKSPACE / "regenerated_mcqs"

DEFAULT_MAX_ROUNDS = 3
DEFAULT_MIN_YIELD = 0.20


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_round_stats(category: str) -> list[dict]:
    """
    Return a list of round_stats dicts sorted by round number ascending.
    Looks for regenerated_mcqs/{category}_round*/round_stats.json.
    """
    pattern = f"{category}_round*"
    stats: list[dict] = []
    for d in sorted(REGEN_DIR.glob(pattern)):
        path = d / "round_stats.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                stats.append(data)
            except (json.JSONDecodeError, OSError):
                continue
    stats.sort(key=lambda s: s.get("round", 0))
    return stats


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


def check_termination(
    stats: list[dict],
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    min_yield: float = DEFAULT_MIN_YIELD,
    post_judge_remaining: list[str] | None = None,
) -> dict:
    """
    Return a decision dict:
      {
        "continue": bool,
        "reason": str,
        "next_round": int | None,   # set when continue=True
        "rounds_completed": int,
        "latest_acceptance_rate": float | None,
        "topics_remaining": list[str],
      }

    post_judge_remaining: topics still needing items per regeneration_needed.json
    (after judging). This is the authoritative source for TARGET REACHED — it
    overrides the round-stats topics_remaining, which only reflects generation
    acceptance and does not account for post-judge rejections.
    """
    if not stats:
        return {
            "continue": False,
            "reason": "no round stats found — run at least one regeneration round first",
            "next_round": None,
            "rounds_completed": 0,
            "latest_acceptance_rate": None,
            "topics_remaining": [],
        }

    latest = stats[-1]
    rounds_completed = len(stats)
    acceptance_rate = latest.get("acceptance_rate")
    total_generated = latest.get("total_generated", 0)
    topics_filled_latest = latest.get("topics_filled", [])

    # Use post-judge truth when available; fall back to round-stats estimate
    if post_judge_remaining is not None:
        topics_remaining = post_judge_remaining
    else:
        topics_remaining = latest.get("topics_remaining", [])

    # Condition 1: target reached
    if not topics_remaining:
        return {
            "continue": False,
            "reason": "TARGET REACHED — all topics have sufficient accepted items",
            "next_round": None,
            "rounds_completed": rounds_completed,
            "latest_acceptance_rate": acceptance_rate,
            "topics_remaining": [],
        }

    # Condition 2: max rounds exceeded
    if rounds_completed >= max_rounds:
        return {
            "continue": False,
            "reason": (
                f"MAX ROUNDS REACHED — completed {rounds_completed}/{max_rounds} rounds. "
                f"{len(topics_remaining)} topic(s) remain unfilled."
            ),
            "next_round": None,
            "rounds_completed": rounds_completed,
            "latest_acceptance_rate": acceptance_rate,
            "topics_remaining": topics_remaining,
        }

    # Condition 3: yield collapse
    if acceptance_rate is not None and acceptance_rate < min_yield:
        return {
            "continue": False,
            "reason": (
                f"YIELD COLLAPSE — acceptance rate {acceptance_rate:.1%} is below "
                f"threshold {min_yield:.1%}. Remaining failures are likely systemic."
            ),
            "next_round": None,
            "rounds_completed": rounds_completed,
            "latest_acceptance_rate": acceptance_rate,
            "topics_remaining": topics_remaining,
        }

    # Condition 4: no progress (generated items but filled nothing new)
    if total_generated > 0 and not topics_filled_latest:
        return {
            "continue": False,
            "reason": (
                f"NO PROGRESS — round {latest.get('round')} generated {total_generated} items "
                "but filled 0 topics. Further rounds unlikely to help."
            ),
            "next_round": None,
            "rounds_completed": rounds_completed,
            "latest_acceptance_rate": acceptance_rate,
            "topics_remaining": topics_remaining,
        }

    # All clear — continue
    next_round = latest.get("round", rounds_completed) + 1
    return {
        "continue": True,
        "reason": (
            f"Round {latest.get('round')} completed: "
            f"{len(topics_filled_latest)} topic(s) filled, "
            f"{len(topics_remaining)} remaining, "
            f"acceptance rate {acceptance_rate:.1%}."
        ),
        "next_round": next_round,
        "rounds_completed": rounds_completed,
        "latest_acceptance_rate": acceptance_rate,
        "topics_remaining": topics_remaining,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether the MCQ regeneration loop should continue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--category", required=True, metavar="NAME",
                        help="Category slug (e.g. natural_fertilizers)")
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                        metavar="N",
                        help=f"Maximum regeneration rounds (default: {DEFAULT_MAX_ROUNDS})")
    parser.add_argument("--min-yield", type=float, default=DEFAULT_MIN_YIELD,
                        metavar="RATE",
                        help=f"Minimum acceptance rate before yield-collapse stop "
                             f"(default: {DEFAULT_MIN_YIELD})")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output decision as JSON instead of human-readable text")
    args = parser.parse_args()

    stats = _load_round_stats(args.category)

    # Load post-judge truth from regeneration_needed.json when available.
    # This overrides the round-stats topics_remaining, which only reflects
    # generation-time acceptance and does not account for post-judge rejections.
    post_judge_remaining: list[str] | None = None
    regen_report = WORKSPACE / "quality" / "final" / "regeneration_needed.json"
    if regen_report.exists():
        try:
            report = json.loads(regen_report.read_text(encoding="utf-8"))
            post_judge_remaining = [
                t["topic"]
                for t in report.get("topics", [])
                if t.get("category") == args.category
            ]
        except (json.JSONDecodeError, OSError):
            pass  # fall back to round-stats

    decision = check_termination(stats, args.max_rounds, args.min_yield, post_judge_remaining)

    if args.json_output:
        print(json.dumps(decision, indent=2))
        return 0 if decision["continue"] else 1

    # Human-readable output
    status = "CONTINUE" if decision["continue"] else "STOP"
    print(f"[{status}] {decision['reason']}")
    if decision["continue"]:
        print(f"  Next round: {decision['next_round']}")
    print(f"  Rounds completed:   {decision['rounds_completed']}")
    if decision["latest_acceptance_rate"] is not None:
        print(f"  Last acceptance:    {decision['latest_acceptance_rate']:.1%}")
    if decision["topics_remaining"]:
        print(f"  Topics remaining:   {', '.join(decision['topics_remaining'])}")

    return 0 if decision["continue"] else 1


if __name__ == "__main__":
    sys.exit(main())
