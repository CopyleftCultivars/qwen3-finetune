#!/usr/bin/env python3
"""
LLM-as-Judge quality scoring for MCQ benchmarks.

For each MCQ, an LLM judge scores 5 quality dimensions using the per-category
rubric from quality/rubrics/. Composite scores determine keep/review/reject.

Quality Dimensions (1–5 Likert):
  1. factual_correctness
  2. question_clarity
  3. distractor_quality
  4. domain_relevance
  5. difficulty_appropriateness

Composite score = mean of 5 dimensions.
Verdict thresholds are loaded from each category's rubric file.

Usage:
    # Score a single benchmark file (category inferred from parent directory)
    conda run -n nf-benchmark python quality/llm_judge.py \\
        --input benchmarks/agroecology/agroecology_benchmark_v1.jsonl

    # Score all 9 benchmark categories
    conda run -n nf-benchmark python quality/llm_judge.py --all-benchmarks

    # Override judge model
    conda run -n nf-benchmark python quality/llm_judge.py \\
        --all-benchmarks --model qwen/qwen3.6-plus

    # Dry run (no API calls, shows plan)
    conda run -n nf-benchmark python quality/llm_judge.py \\
        --all-benchmarks --dry-run

Output:
    quality/scores/{category}_judged.jsonl   — original items + quality_scores
    quality/scores/{category}_cache.json     — resume cache (keyed by question_id)
    quality/scores/judge_summary.json        — aggregate stats across all categories
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent.parent
RUBRICS_DIR = WORKSPACE / "quality" / "rubrics"
SCORES_DIR = WORKSPACE / "quality" / "scores"
BENCHMARKS_DIR = WORKSPACE / "benchmarks"

load_dotenv(WORKSPACE / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL = "anthropic/claude-haiku-4.5"
RETRY_DELAYS = [2, 5, 15]  # seconds between retries
API_TIMEOUT = 90            # seconds per call
MAX_WORKERS = 5

DIMENSIONS = [
    "factual_correctness",
    "question_clarity",
    "distractor_quality",
    "domain_relevance",
    "difficulty_appropriateness",
]

# Category directory name → rubric file stem
CATEGORY_RUBRICS: dict[str, str] = {
    "agroecology":                        "rubric_agroecology",
    "indigenous_and_traditional_knowledge": "rubric_indigenous_and_traditional_knowledge",
    "integrated_farming_systems":          "rubric_integrated_farming_systems",
    "natural_fertilizers":                 "rubric_natural_fertilizers",
    "organic_farming":                     "rubric_organic_farming",
    "permaculture":                        "rubric_permaculture",
    "regenerative_agriculture":            "rubric_regenerative_agriculture",
    "soil_science_and_microbiology":       "rubric_soil_science_and_microbiology",
    "sustainable_water_management":        "rubric_sustainable_water_management",
    # Phase 3b (2026-07-11): conversation-derived MCQs span multiple domains rather
    # than one benchmark category, so use the domain-agnostic base rubric directly.
    "conversations":                       "rubric_base",
}

# Default thresholds from rubric_base.md; overridden per-category below
DEFAULT_THRESHOLDS: dict[str, float] = {"keep": 4.0, "review": 3.0}

# ---------------------------------------------------------------------------
# Rubric helpers
# ---------------------------------------------------------------------------


def load_rubric(category: str) -> str:
    """Load the per-category rubric markdown string."""
    stem = CATEGORY_RUBRICS.get(category)
    if not stem:
        raise ValueError(f"No rubric mapping for category '{category}'")
    path = RUBRICS_DIR / f"{stem}.md"
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {path}")
    return path.read_text(encoding="utf-8")


def parse_thresholds(rubric_text: str) -> dict[str, float]:
    """
    Extract acceptance thresholds from rubric markdown.

    Looks for patterns like:
        - **Composite ≥ 4.0:** Accept / Keep / High
        - **3.5 ≤ Composite < 4.0:** Review

    Returns {"keep": float, "review": float}.
    Falls back to DEFAULT_THRESHOLDS for any value not found.
    """
    thresholds = dict(DEFAULT_THRESHOLDS)

    # "Composite ≥ X.X" or "Composite >= X.X" → keep threshold
    keep_m = re.search(
        r"Composite\s*[≥>=]+\s*([\d.]+)[^:\n]*:\s*\*{0,2}(Accept|Keep|High)",
        rubric_text,
        re.IGNORECASE,
    )
    if keep_m:
        thresholds["keep"] = float(keep_m.group(1))

    # "X.X ≤ Composite < Y.Y" → lower bound is the review threshold
    review_m = re.search(
        r"\*{0,2}([\d.]+)\s*[≤<=]+\s*Composite\s*<",
        rubric_text,
        re.IGNORECASE,
    )
    if review_m:
        thresholds["review"] = float(review_m.group(1))

    return thresholds


def apply_verdict(composite: float, thresholds: dict[str, float]) -> str:
    if composite >= thresholds["keep"]:
        return "keep"
    if composite >= thresholds["review"]:
        return "review"
    return "reject"


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert MCQ quality assessor for agricultural benchmark datasets. "
    "You evaluate multiple-choice questions on 5 quality dimensions using provided rubrics. "
    "Always respond with valid JSON only — no prose, no markdown code fences."
)


def build_judge_prompt(item: dict, rubric_text: str, category: str) -> str:
    category_title = category.replace("_", " ").title()
    return f"""## Category: {category_title}

## Quality Rubric
{rubric_text}

---

## MCQ to Evaluate

Question: {item["question"]}
A: {item["A"]}
B: {item["B"]}
C: {item["C"]}
D: {item["D"]}
Correct Answer: {item["answer"]}

---

## Scoring Instructions

Score this MCQ on each of the 5 quality dimensions defined in the rubric above.
Use the category-specific anchor examples to calibrate your scores.
Score each dimension independently on a 1–5 Likert scale.

Respond with ONLY this JSON (no other text, no code fences):

{{
  "scores": {{
    "factual_correctness": <integer 1-5>,
    "question_clarity": <integer 1-5>,
    "distractor_quality": <integer 1-5>,
    "domain_relevance": <integer 1-5>,
    "difficulty_appropriateness": <integer 1-5>
  }},
  "justifications": {{
    "factual_correctness": "<one concise sentence>",
    "question_clarity": "<one concise sentence>",
    "distractor_quality": "<one concise sentence>",
    "domain_relevance": "<one concise sentence>",
    "difficulty_appropriateness": "<one concise sentence>"
  }}
}}"""


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------

def call_judge(prompt: str, client: OpenAI, model: str) -> str:
    """Call the OpenRouter judge model. Returns raw response text."""
    last_error = ""
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
                timeout=API_TIMEOUT,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            last_error = f"empty response from {model}"
            print(f"\n  Empty response (attempt {attempt + 1})", file=sys.stderr)
        except Exception as e:
            last_error = str(e)
            if attempt == len(RETRY_DELAYS):
                raise RuntimeError(
                    f"Judge API failed after {attempt + 1} attempts: {last_error}"
                ) from e
            print(f"\n  API error (attempt {attempt + 1}): {e}", file=sys.stderr)
    raise RuntimeError(f"Judge API failed: {last_error}")


def parse_judge_response(text: str) -> dict:
    """
    Parse JSON response from the judge.

    Strips <think>...</think> blocks (emitted by reasoning models) and any
    markdown code fences before JSON parsing.
    """
    # Strip reasoning traces
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    data = json.loads(text)

    scores = data.get("scores", {})
    justifications = data.get("justifications", {})

    # Validate and coerce scores
    validated: dict[str, int] = {}
    for dim in DIMENSIONS:
        if dim not in scores:
            raise ValueError(f"Missing score dimension: {dim}")
        s = scores[dim]
        if not isinstance(s, (int, float)) or not (1 <= float(s) <= 5):
            raise ValueError(f"Score out of range for '{dim}': {s!r}")
        validated[dim] = int(round(float(s)))

    composite = round(sum(validated[d] for d in DIMENSIONS) / len(DIMENSIONS), 2)

    return {
        "scores": validated,
        "justifications": {d: str(justifications.get(d, "")) for d in DIMENSIONS},
        "composite_score": composite,
    }


# ---------------------------------------------------------------------------
# Score a single MCQ item
# ---------------------------------------------------------------------------

def score_item(
    item: dict,
    category: str,
    rubric_text: str,
    thresholds: dict[str, float],
    client: OpenAI,
    model: str,
) -> dict:
    """Score one MCQ item. Returns augmented item dict."""
    prompt = build_judge_prompt(item, rubric_text, category)
    raw = call_judge(prompt, client, model)
    parsed = parse_judge_response(raw)
    verdict = apply_verdict(parsed["composite_score"], thresholds)

    result = dict(item)
    result["quality_scores"] = {
        **parsed["scores"],
        "composite": parsed["composite_score"],
        "verdict": verdict,
    }
    result["quality_justifications"] = parsed["justifications"]
    result["judge_model"] = model
    result["judge_category"] = category
    return result


# ---------------------------------------------------------------------------
# JSONL / cache helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_cache(path: Path, cache: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Score a whole benchmark file
# ---------------------------------------------------------------------------

def score_file(
    input_path: Path,
    category: str,
    client: OpenAI,
    model: str,
    max_workers: int = MAX_WORKERS,
    dry_run: bool = False,
) -> dict:
    """
    Score all items in a JSONL file.

    Outputs:
        quality/scores/{category}_judged.jsonl
        quality/scores/{category}_cache.json

    Returns a summary dict.
    """
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORES_DIR / f"{category}_judged.jsonl"
    cache_path = SCORES_DIR / f"{category}_cache.json"

    rubric_text = load_rubric(category)
    thresholds = parse_thresholds(rubric_text)
    items = load_jsonl(input_path)
    cache = load_cache(cache_path)
    cache_lock = Lock()

    pending = [it for it in items if it.get("question_id") not in cache]
    n_cached = len(items) - len(pending)

    print(
        f"\n[{category}] {len(items)} items | "
        f"{n_cached} cached | {len(pending)} to score",
        flush=True,
    )
    print(f"  Thresholds: keep≥{thresholds['keep']} review≥{thresholds['review']}", flush=True)

    if dry_run:
        print("  DRY RUN — skipping API calls")
        return {
            "category": category,
            "input_file": str(input_path),
            "total_items": len(items),
            "cached": n_cached,
            "pending": len(pending),
            "dry_run": True,
        }

    errors = 0

    def _score_and_cache(item: dict) -> dict:
        result = score_item(item, category, rubric_text, thresholds, client, model)
        qid = item.get("question_id", "")
        with cache_lock:
            cache[qid] = result
            _write_cache(cache_path, cache)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_and_cache, item): item for item in pending}
        with tqdm(total=len(pending), desc=category, unit="item") as bar:
            for future in as_completed(futures):
                item = futures[future]
                qid = item.get("question_id", "?")
                try:
                    future.result()
                except Exception as e:
                    print(f"\n  ERROR scoring {qid}: {e}", file=sys.stderr)
                    errors += 1
                bar.update(1)

    # Write output in original item order
    all_results = []
    for item in items:
        qid = item.get("question_id", "")
        all_results.append(cache.get(qid, item))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute aggregate stats
    scored = [r for r in all_results if "quality_scores" in r]
    verdicts: dict[str, int] = {"keep": 0, "review": 0, "reject": 0}
    composites: list[float] = []
    dim_totals: dict[str, float] = {d: 0.0 for d in DIMENSIONS}

    for r in scored:
        qs = r["quality_scores"]
        v = qs.get("verdict", "unknown")
        verdicts[v] = verdicts.get(v, 0) + 1
        composites.append(qs["composite"])
        for d in DIMENSIONS:
            dim_totals[d] += qs.get(d, 0)

    n = len(scored)
    summary = {
        "category": category,
        "input_file": str(input_path),
        "output_file": str(out_path),
        "total_items": len(items),
        "scored_items": n,
        "errors": errors,
        "verdicts": verdicts,
        "mean_composite": round(sum(composites) / n, 3) if n else 0.0,
        "mean_scores": (
            {d: round(dim_totals[d] / n, 3) for d in DIMENSIONS} if n else {}
        ),
        "thresholds": thresholds,
        "judge_model": model,
    }

    print(
        f"  → {n} scored | keep={verdicts['keep']} "
        f"review={verdicts['review']} reject={verdicts['reject']} "
        f"| mean composite={summary['mean_composite']} | errors={errors}",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# Benchmark discovery
# ---------------------------------------------------------------------------

def discover_benchmarks() -> list[tuple[Path, str]]:
    """
    Find non-graded *_v1.jsonl files for all known categories.
    Prefers files without 'graded' in the name.
    """
    results = []
    for category in CATEGORY_RUBRICS:
        cat_dir = BENCHMARKS_DIR / category
        if not cat_dir.is_dir():
            print(f"  WARNING: category dir not found: {cat_dir}", file=sys.stderr)
            continue
        candidates = sorted(cat_dir.glob("*_v1.jsonl"))
        non_graded = [p for p in candidates if "graded" not in p.name]
        chosen = non_graded[0] if non_graded else (candidates[0] if candidates else None)
        if chosen:
            results.append((chosen, category))
        else:
            print(f"  WARNING: no JSONL found for category '{category}'", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge quality scoring for MCQ benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", metavar="FILE",
        help="Path to a single JSONL benchmark file",
    )
    group.add_argument(
        "--all-benchmarks", action="store_true",
        help="Score all 9 benchmark categories (auto-discovers files)",
    )
    parser.add_argument(
        "--category",
        help=(
            "Explicit category name (only needed with --input when the parent "
            "directory name doesn't match a known category)"
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help=f"OpenRouter model ID for judging (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Parallel API workers per category (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--openrouter-key", default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY in .env)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover files and show plan without making API calls",
    )
    args = parser.parse_args()

    api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print(
            "ERROR: OpenRouter API key required. "
            "Set OPENROUTER_API_KEY in .env or pass --openrouter-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key or "dry-run",
    )

    if args.all_benchmarks:
        targets = discover_benchmarks()
        if not targets:
            print("ERROR: No benchmark files found.", file=sys.stderr)
            sys.exit(1)
        print(f"Discovered {len(targets)} benchmark files:")
        for path, cat in targets:
            print(f"  [{cat}] {path.name}")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        category = args.category or input_path.parent.name
        if category not in CATEGORY_RUBRICS:
            print(
                f"ERROR: '{category}' is not a known category.\n"
                f"Known categories: {sorted(CATEGORY_RUBRICS)}",
                file=sys.stderr,
            )
            sys.exit(1)
        targets = [(input_path, category)]

    summaries: list[dict] = []
    for path, category in targets:
        try:
            summary = score_file(
                path,
                category,
                client,
                args.model,
                max_workers=args.workers,
                dry_run=args.dry_run,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"\nERROR processing '{category}': {e}", file=sys.stderr)
            summaries.append({"category": category, "error": str(e)})

    if not args.dry_run:
        SCORES_DIR.mkdir(parents=True, exist_ok=True)
        # Per-category summary files (never overwritten by other categories)
        for s in summaries:
            cat = s.get("category")
            if cat:
                cat_path = SCORES_DIR / f"{cat}_judge_summary.json"
                with open(cat_path, "w", encoding="utf-8") as f:
                    json.dump(s, f, indent=2, ensure_ascii=False)
                print(f"Summary written to {cat_path}")
        # Aggregate summary only when running --all-benchmarks
        if args.all_benchmarks:
            summary_path = SCORES_DIR / "judge_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
            print(f"Aggregate summary written to {summary_path}")

    # Overall totals
    total = sum(s.get("total_items", 0) for s in summaries)
    scored_n = sum(s.get("scored_items", 0) for s in summaries)
    keep = sum(s.get("verdicts", {}).get("keep", 0) for s in summaries)
    review = sum(s.get("verdicts", {}).get("review", 0) for s in summaries)
    reject = sum(s.get("verdicts", {}).get("reject", 0) for s in summaries)
    errors = sum(s.get("errors", 0) for s in summaries)

    status = "DRY RUN complete" if args.dry_run else "Done"
    print(
        f"\n{status}. {scored_n}/{total} items scored | "
        f"keep={keep} review={review} reject={reject} errors={errors}"
    )


if __name__ == "__main__":
    main()
