#!/usr/bin/env python3
"""
End-to-end quality pipeline runner for MCQ benchmarks (Issue #60).

Chains all quality pipeline layers for a single category, with:
  - Resume-safety: each step is skipped if its output already exists
  - Human pause points after LLM judge scoring (Layer 3) and after
    Layer 5 flag detection (before packaging)
  - Dry-run mode: prints the plan without executing anything

Pipeline steps
--------------
1. validate     Structural validation (validate_mcqs.py)
2. judge        LLM quality scoring (quality/llm_judge.py)
3. [PAUSE]      Optional human review of judge scores
4. irt-eval     12-model response collection (quality/irt_analysis.py eval)
5. irt-fit      Fit 2PL IRT model (quality/irt_analysis.py fit)
6. irt-filter   Apply IRT filter criteria (quality/irt_analysis.py filter)
7. cartography  Dataset cartography map (quality/cartography.py map)
8. cleanlab     Label error detection (quality/cleanlab_detection.py detect)
9. [PAUSE]      Optional human review of Layer 5 flags
10. package     Final packaging + regeneration report (quality/package_benchmark.py)

Usage
-----
    # Full pipeline for natural_fertilizers
    conda run -n nf-benchmark python quality/run_quality_pipeline.py \\
        --category natural_fertilizers

    # Dry run — show plan and step outputs without executing
    conda run -n nf-benchmark python quality/run_quality_pipeline.py \\
        --category natural_fertilizers --dry-run

    # Skip IRT eval (expensive — 2,520 API calls for NF); use existing responses
    conda run -n nf-benchmark python quality/run_quality_pipeline.py \\
        --category natural_fertilizers --skip-irt-eval

    # Skip human pause prompts (for CI / automated runs)
    conda run -n nf-benchmark python quality/run_quality_pipeline.py \\
        --category natural_fertilizers --no-pause

    # Run a specific step only
    conda run -n nf-benchmark python quality/run_quality_pipeline.py \\
        --category natural_fertilizers --only judge
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent.parent
PYTHON = sys.executable   # same conda env that launched this script
PYTHON_UNBUFFERED = [sys.executable, "-u"]  # unbuffered — stdout streams live

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------
# Each step: name, script + args template, output path (existence = done)
# {category} is substituted at runtime.

STEPS = [
    {
        "name": "validate",
        "label": "Structural validation",
        "cmd": [
            *PYTHON_UNBUFFERED, "validate_mcqs.py",
            "--file", "benchmarks/{category}/{category}_benchmark_v1.jsonl",
        ],
        # validate always re-runs (no persistent output to check)
        "output": None,
        "skip_if_exists": False,
    },
    {
        "name": "judge",
        "label": "LLM quality scoring (Layer 3)",
        "cmd": [
            *PYTHON_UNBUFFERED, "quality/llm_judge.py",
            "--input", "benchmarks/{category}/{category}_benchmark_v1.jsonl",
        ],
        "output": "quality/scores/{category}_judged.jsonl",
        "skip_if_exists": True,
    },
    {
        "name": "irt-eval",
        "label": "IRT model evaluation (12 models × items)",
        "cmd": [
            *PYTHON_UNBUFFERED, "quality/irt_analysis.py", "eval",
            "--category", "{category}",
        ],
        # Presence of any response file for this category → partially done; let
        # irt_analysis.py's own resume cache handle the rest.
        "output": "quality/irt/responses",
        "skip_if_exists": False,   # always run; internal cache handles resume
    },
    {
        "name": "irt-fit",
        "label": "Fit 2PL IRT model",
        "cmd": [
            *PYTHON_UNBUFFERED, "quality/irt_analysis.py", "fit",
        ],
        "output": "quality/irt/irt_params.json",
        "skip_if_exists": True,
    },
    {
        "name": "irt-filter",
        "label": "Apply IRT filter criteria",
        "cmd": [
            *PYTHON_UNBUFFERED, "quality/irt_analysis.py", "filter",
        ],
        "output": "quality/irt/filter_report.json",
        "skip_if_exists": True,
    },
    {
        "name": "cartography",
        "label": "Dataset cartography mapping",
        "cmd": [
            *PYTHON_UNBUFFERED, "quality/cartography.py", "map",
        ],
        "output": "quality/cartography/cartography_map.jsonl",
        "skip_if_exists": True,
    },
    {
        "name": "cleanlab",
        "label": "Cleanlab label error detection (Layer 5)",
        "cmd": [
            *PYTHON_UNBUFFERED, "quality/cleanlab_detection.py", "detect",
        ],
        "output": "quality/cleanlab/flagged_items.jsonl",
        "skip_if_exists": True,
    },
    {
        "name": "package",
        "label": "Final packaging + regeneration report",
        "cmd": [
            *PYTHON_UNBUFFERED, "quality/package_benchmark.py",
            "--category", "{category}",
        ],
        "output": "quality/final/{category}_final.jsonl",
        "skip_if_exists": True,
    },
]

# Steps where the pipeline pauses for human review
PAUSE_AFTER = {
    "judge": (
        "\nLayer 3 complete — LLM judge scores written to quality/scores/.\n"
        "You may want to review flagged items before continuing to IRT eval.\n"
        "Press Enter to continue, Ctrl-C to pause here: "
    ),
    "cleanlab": (
        "\nLayer 5 complete — all flags written to quality/cleanlab/flagged_items.jsonl.\n"
        "Review potential label errors before final packaging.\n"
        "Press Enter to package, Ctrl-C to pause here: "
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub(template: str | list, category: str) -> str | list:
    if isinstance(template, list):
        return [t.replace("{category}", category) for t in template]
    return template.replace("{category}", category)


def _output_exists(step: dict, category: str) -> bool:
    if not step.get("skip_if_exists") or not step["output"]:
        return False
    path = WORKSPACE / _sub(step["output"], category)
    return path.exists()


def _run_step(step: dict, category: str, dry_run: bool) -> bool:
    """Run one step. Returns True on success, False on failure."""
    cmd = [str(c) for c in _sub(step["cmd"], category)]

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd, cwd=str(WORKSPACE))
    if result.returncode != 0:
        print(
            f"\n  ERROR: step '{step['name']}' exited with code {result.returncode}",
            file=sys.stderr,
        )
        return False
    return True


def _pause(message: str, no_pause: bool) -> None:
    if no_pause:
        return
    try:
        input(message)
    except KeyboardInterrupt:
        print("\nPaused. Re-run with --only <step> to resume from a specific step.")
        sys.exit(0)
    except EOFError:
        # Non-interactive environment (CI, piped input) — continue automatically
        print("(non-interactive — continuing)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end quality pipeline runner for MCQ benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--category",
        required=True,
        metavar="NAME",
        help="Category directory name (e.g. natural_fertilizers)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without executing any steps",
    )
    parser.add_argument(
        "--skip-irt-eval",
        action="store_true",
        help="Skip the IRT eval step (use existing response files)",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Skip human review pause prompts (for automated runs)",
    )
    parser.add_argument(
        "--only",
        metavar="STEP",
        help="Run only this step (e.g. judge, irt-eval, package)",
    )
    args = parser.parse_args()

    category = args.category
    benchmark_path = WORKSPACE / "benchmarks" / category / f"{category}_benchmark_v1.jsonl"
    if not benchmark_path.exists():
        print(f"ERROR: Benchmark not found: {benchmark_path}", file=sys.stderr)
        return 1

    # Filter steps
    steps = STEPS
    if args.only:
        steps = [s for s in STEPS if s["name"] == args.only]
        if not steps:
            names = ", ".join(s["name"] for s in STEPS)
            print(f"ERROR: Unknown step '{args.only}'. Valid: {names}", file=sys.stderr)
            return 1
    if args.skip_irt_eval:
        steps = [s for s in steps if s["name"] != "irt-eval"]

    print(f"Quality pipeline — category: {category}")
    print(f"{'=' * 60}")

    for step in steps:
        name = step["name"]
        label = step["label"]

        if _output_exists(step, category):
            print(f"  [skip]  {label}  (output exists)")
            continue

        print(f"  [run]   {label}")
        if not _run_step(step, category, args.dry_run):
            print(f"\nPipeline aborted at step '{name}'.", file=sys.stderr)
            return 1

        if name in PAUSE_AFTER and not args.only and not args.dry_run:
            _pause(PAUSE_AFTER[name], args.no_pause)

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print("Dry run complete — no steps executed.")
    else:
        print(f"Pipeline complete for: {category}")
        final_path = WORKSPACE / "quality" / "final" / f"{category}_final.jsonl"
        if final_path.exists():
            print(f"  Final benchmark: {final_path.relative_to(WORKSPACE)}")
        regen_path = WORKSPACE / "quality" / "final" / "regeneration_needed.json"
        if regen_path.exists():
            print(f"  Regeneration report: {regen_path.relative_to(WORKSPACE)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
