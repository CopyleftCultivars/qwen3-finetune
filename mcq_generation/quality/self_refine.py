#!/usr/bin/env python3
"""
Per-item self-refinement for borderline MCQs (Issue #68).

Implements the three-step generate→critique→correct loop from MCQG-SRefine
(Yao et al., NAACL 2025) for items in the "borderline" composite score zone
(default 3.0–3.9).  Items below 2.5 should be discarded; items above 3.9
already pass quality review.

Pipeline
--------
For each borderline item in the judge cache:
  1. CRITIQUE  — LLM evaluates the MCQ on 4 dimensions and flags failures
  2. CORRECT   — LLM rewrites only the failing parts (if any failures exist)
  3. The revised item is written to quality/refined/{category}_refined.json

The source passage is retrieved from qmd if --collection is provided; otherwise
the factual_correctness dimension is skipped in the critique.

Usage
-----
    # Dry run: show which items qualify
    conda run -n nf-benchmark python quality/self_refine.py \\
        --category natural_fertilizers --dry-run

    # Full run with source passage lookup
    conda run -n nf-benchmark python quality/self_refine.py \\
        --category natural_fertilizers \\
        --collection Natural_Fertilizers \\
        --model anthropic/claude-haiku-4.5

    # Custom score window
    conda run -n nf-benchmark python quality/self_refine.py \\
        --category natural_fertilizers \\
        --score-min 2.5 --score-max 4.2

Output
------
    quality/refined/{category}_refined.json  — list of revised MCQ records,
    each with an added "refinement" field documenting what changed.
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent.parent
SCORES_DIR = WORKSPACE / "quality" / "scores"
REFINED_DIR = WORKSPACE / "quality" / "refined"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

DEFAULT_SCORE_MIN = 3.0
DEFAULT_SCORE_MAX = 3.9
DEFAULT_MAX_ITEMS = 50  # safety cap

# ---------------------------------------------------------------------------
# QMD helper
# ---------------------------------------------------------------------------


def _qmd_fetch_passage(topic: str, collection: str, n_results: int = 3) -> str:
    """
    Query qmd for the top chunks for a topic. Returns concatenated text or ''.
    """
    try:
        result = subprocess.run(
            ["qmd", "query", topic, "--collection", collection,
             "--limit", str(n_results), "--no-color"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    return ""


# ---------------------------------------------------------------------------
# Critique step
# ---------------------------------------------------------------------------

_CRITIQUE_SYSTEM = (
    "You are an expert MCQ reviewer for agricultural science. "
    "Evaluate the provided multiple-choice question strictly and concisely. "
    "Respond ONLY with the requested JSON object."
)


def _critique_prompt(item: dict, source_passage: str) -> str:
    q = item["question"]
    opts = "\n".join(
        f"  {k}) {item[k]}" for k in ("A", "B", "C", "D") if k in item
    )
    answer = item.get("answer", "?")
    passage_block = (
        f"\nSOURCE PASSAGE:\n{textwrap.shorten(source_passage, 1200)}"
        if source_passage
        else "\nSOURCE PASSAGE: (not available — skip factual_correctness)"
    )

    return f"""Evaluate this multiple-choice question about natural fertilizers.

QUESTION: {q}
OPTIONS:
{opts}
MARKED ANSWER: {answer}
{passage_block}

For each dimension below, output PASS or FAIL and a one-sentence explanation.
If FAIL, add a specific "suggestion" for how to fix it.
If SOURCE PASSAGE is not available, mark factual_correctness as SKIP.

Respond with ONLY this JSON (no markdown fences):
{{
  "factual_correctness": {{"verdict": "PASS|FAIL|SKIP", "explanation": "...", "suggestion": ""}},
  "question_clarity":    {{"verdict": "PASS|FAIL",      "explanation": "...", "suggestion": ""}},
  "distractor_quality":  {{"verdict": "PASS|FAIL",      "explanation": "...", "suggestion": ""}},
  "difficulty":          {{"verdict": "PASS|FAIL",      "explanation": "...", "suggestion": ""}}
}}"""


# ---------------------------------------------------------------------------
# Correction step
# ---------------------------------------------------------------------------

_CORRECTION_SYSTEM = (
    "You are an expert MCQ writer for agricultural science. "
    "Revise the provided MCQ based on the critique. "
    "Only change what the critique marks as FAIL — preserve everything else. "
    "Respond ONLY with the requested JSON object."
)


def _correction_prompt(item: dict, critique: dict, source_passage: str) -> str:
    original = {k: item[k] for k in ("question", "A", "B", "C", "D", "answer") if k in item}
    passage_block = (
        f"\nSOURCE PASSAGE:\n{textwrap.shorten(source_passage, 1200)}"
        if source_passage
        else ""
    )

    failing = {dim: v for dim, v in critique.items() if v.get("verdict") == "FAIL"}
    critique_summary = "\n".join(
        f"- {dim}: {v['explanation']} → {v.get('suggestion', '')}"
        for dim, v in failing.items()
    )

    return f"""Revise this MCQ to fix ONLY the issues listed in the critique.
Do NOT change dimensions that passed.

ORIGINAL MCQ:
{json.dumps(original, indent=2)}

CRITIQUE (failing dimensions only):
{critique_summary}
{passage_block}

Output ONLY this JSON (no markdown fences):
{{
  "question": "...",
  "A": "A. ...",
  "B": "B. ...",
  "C": "C. ...",
  "D": "D. ...",
  "answer": "<letter>"
}}"""


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------


def _call_llm(client: OpenAI, model: str, system: str, user: str) -> str | None:
    """Single LLM call. Returns response text or None on failure."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [LLM error] {exc}", file=sys.stderr)
        return None


def _parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    # Strip markdown fences if present
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Per-item refinement
# ---------------------------------------------------------------------------


def refine_item(
    item: dict,
    client: OpenAI,
    model: str,
    collection: str,
    logger_print=print,
) -> dict | None:
    """
    Run the critique→correct loop for one item.
    Returns a revised item dict (with 'refinement' metadata), or None if
    no correction was needed or the correction step failed.
    """
    topic = item.get("topic", "")
    source_passage = _qmd_fetch_passage(topic, collection) if collection else ""

    # Step 1: Critique
    critique_text = _call_llm(
        client, model, _CRITIQUE_SYSTEM, _critique_prompt(item, source_passage)
    )
    critique = _parse_json(critique_text)
    if not critique:
        logger_print(f"  [critique] JSON parse failed for {item.get('question_id', '?')}")
        return None

    failing_dims = [dim for dim, v in critique.items() if v.get("verdict") == "FAIL"]
    if not failing_dims:
        logger_print("  [critique] all pass — no correction needed")
        return None

    logger_print(f"  [critique] failing: {', '.join(failing_dims)}")

    # Step 2: Correct
    correction_text = _call_llm(
        client, model, _CORRECTION_SYSTEM,
        _correction_prompt(item, critique, source_passage)
    )
    correction = _parse_json(correction_text)
    if not correction:
        logger_print("  [correct] JSON parse failed")
        return None

    # Merge correction back into original item record
    revised = dict(item)
    for field in ("question", "A", "B", "C", "D", "answer"):
        if field in correction:
            revised[field] = correction[field]

    revised["refinement"] = {
        "original_composite": item.get("quality_scores", {}).get("composite"),
        "failing_dimensions": failing_dims,
        "critique": critique,
        "model": model,
    }
    return revised


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-item self-refinement for borderline MCQs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--category", required=True, metavar="NAME")
    parser.add_argument("--collection", default="", metavar="NAME",
                        help="qmd collection name for source passage lookup")
    parser.add_argument("--model", default=DEFAULT_MODEL, metavar="ID")
    parser.add_argument("--score-min", type=float, default=DEFAULT_SCORE_MIN,
                        metavar="N", help=f"Lower composite bound (default {DEFAULT_SCORE_MIN})")
    parser.add_argument("--score-max", type=float, default=DEFAULT_SCORE_MAX,
                        metavar="N", help=f"Upper composite bound (default {DEFAULT_SCORE_MAX})")
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS,
                        metavar="N", help=f"Safety cap on items to refine (default {DEFAULT_MAX_ITEMS})")
    parser.add_argument("--openrouter-key", metavar="KEY")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show qualifying items without calling LLM")
    args = parser.parse_args()

    # Load judge cache
    cache_path = SCORES_DIR / f"{args.category}_cache.json"
    if not cache_path.exists():
        print(f"ERROR: judge cache not found: {cache_path}", file=sys.stderr)
        return 1
    cache: dict = json.loads(cache_path.read_text(encoding="utf-8"))

    # Find borderline items
    borderline = [
        rec for rec in cache.values()
        if args.score_min
        <= (rec.get("quality_scores") or {}).get("composite", 0)
        <= args.score_max
    ]
    borderline.sort(key=lambda r: r.get("quality_scores", {}).get("composite", 0))

    print(f"Category:    {args.category}")
    print(f"Score range: {args.score_min}–{args.score_max}")
    print(f"Borderline:  {len(borderline)} items")

    if not borderline:
        print("Nothing to refine.")
        return 0

    if args.dry_run:
        print("\nBorderline items (dry run):")
        for rec in borderline[:args.max_items]:
            scores = rec.get("quality_scores", {})
            print(
                f"  {rec.get('question_id', '?'):12s}  "
                f"composite={scores.get('composite', '?'):.1f}  "
                f"topic={rec.get('topic', '?')}"
            )
        if len(borderline) > args.max_items:
            print(f"  ... and {len(borderline) - args.max_items} more (--max-items cap)")
        return 0

    # API key
    api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    REFINED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REFINED_DIR / f"{args.category}_refined.json"

    # Load existing refined items to append/update
    if out_path.exists():
        existing: list = json.loads(out_path.read_text(encoding="utf-8"))
        existing_ids = {r["question_id"] for r in existing}
    else:
        existing = []
        existing_ids = set()

    to_process = [r for r in borderline if r.get("question_id") not in existing_ids]
    to_process = to_process[:args.max_items]

    print(f"To process:  {len(to_process)} (skipping {len(borderline) - len(to_process)} already refined)")
    print()

    n_refined = 0
    n_skipped = 0

    for i, item in enumerate(to_process, 1):
        qid = item.get("question_id", f"item-{i}")
        composite = item.get("quality_scores", {}).get("composite", "?")
        print(f"[{i}/{len(to_process)}] {qid}  composite={composite}")

        revised = refine_item(item, client, args.model, args.collection)
        if revised:
            existing.append(revised)
            existing_ids.add(qid)
            n_refined += 1
            print(f"  → revised ({len(revised['refinement']['failing_dimensions'])} dim(s) fixed)")
        else:
            n_skipped += 1

    # Write output
    out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nRefined: {n_refined}  No-change: {n_skipped}")
    print(f"Output:  {out_path.relative_to(WORKSPACE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
