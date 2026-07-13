#!/usr/bin/env python3
"""
Judge calibration workflow and inter-annotator agreement (IAA) metrics.

Two subcommands:

  sample   Stratified sample of benchmark items, score them with the LLM
           judge, then export a CSV template for expert manual scoring.

  report   Given a filled-in expert CSV alongside the LLM judge scores,
           compute IAA metrics per dimension and overall.

IAA metrics computed:
  - Pearson r        (continuous, target ≥ 0.7)
  - Spearman ρ       (rank, target ≥ 0.7)
  - MAE              (mean absolute error, target ≤ 0.5)
  - Cohen's κ        (quadratic-weighted, target ≥ 0.7)
  - Krippendorff's α (ordinal, target ≥ 0.7)

Usage:
    # 1. Sample 63 items, run LLM judge, produce expert CSV template
    conda run -n nf-benchmark python quality/judge_calibration.py sample \\
        --n 63 --output-dir quality/calibration

    # Dry-run: sample only, skip API calls
    conda run -n nf-benchmark python quality/judge_calibration.py sample \\
        --n 63 --dry-run

    # 2. After filling quality/calibration/expert_scoring_template.csv:
    conda run -n nf-benchmark python quality/judge_calibration.py report \\
        --expert quality/calibration/expert_scoring_filled.csv \\
        --judged quality/calibration/calibration_items.jsonl

Output (sample):
    quality/calibration/calibration_items.jsonl   — items + LLM judge scores
    quality/calibration/expert_scoring_template.csv — blank template for experts

Output (report):
    quality/calibration/iaa_report.json           — full IAA metrics
    Prints per-dimension table + pass/fail summary to stdout
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from itertools import combinations
from pathlib import Path

import krippendorff
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Shared paths / imports
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent.parent
QUALITY_DIR = WORKSPACE / "quality"
BENCHMARKS_DIR = WORKSPACE / "benchmarks"

# Allow importing from sibling quality/ module
sys.path.insert(0, str(QUALITY_DIR))
from llm_judge import (  # noqa: E402
    CATEGORY_RUBRICS,
    DIMENSIONS,
    DEFAULT_JUDGE_MODEL,
    OPENROUTER_BASE_URL,
    load_jsonl,
    load_rubric,
    parse_thresholds,
    score_item,
)

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI       # noqa: E402
from tqdm import tqdm           # noqa: E402

load_dotenv(WORKSPACE / ".env")

# ---------------------------------------------------------------------------
# IAA targets
# ---------------------------------------------------------------------------
TARGETS = {
    "pearson_r":         0.70,
    "spearman_rho":      0.70,
    "mae":               0.50,   # lower is better — pass if ≤ target
    "cohens_kappa":      0.70,
    "krippendorff_alpha": 0.70,
}

CSV_EXPERT_COLS = [
    "question_id", "category", "question", "A", "B", "C", "D", "answer",
] + DIMENSIONS + ["notes"]


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(n_total: int, seed: int = 42) -> list[dict]:
    """
    Sample up to n_total items, distributed evenly across categories.
    Returns a list of dicts with original MCQ fields + 'category' key.
    """
    n_cats = len(CATEGORY_RUBRICS)
    per_cat = math.ceil(n_total / n_cats)

    rng = random.Random(seed)
    sampled = []

    for category in CATEGORY_RUBRICS:
        cat_dir = BENCHMARKS_DIR / category
        candidates = sorted(cat_dir.glob("*_v1.jsonl"))
        non_graded = [p for p in candidates if "graded" not in p.name]
        if not non_graded:
            print(f"  WARNING: no source JSONL for '{category}'", file=sys.stderr)
            continue

        items = load_jsonl(non_graded[0])
        k = min(per_cat, len(items))
        chosen = rng.sample(items, k)
        for item in chosen:
            item["category"] = category
        sampled.extend(chosen)

    # Trim to exactly n_total if we overshot
    rng.shuffle(sampled)
    return sampled[:n_total]


# ---------------------------------------------------------------------------
# sample subcommand
# ---------------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judged_path = out_dir / "calibration_items.jsonl"
    template_path = out_dir / "expert_scoring_template.csv"

    print(f"Sampling {args.n} items stratified across {len(CATEGORY_RUBRICS)} categories…")
    items = stratified_sample(args.n, seed=args.seed)
    print(f"  → {len(items)} items sampled")

    if args.dry_run:
        # Write template only (no LLM scoring)
        _write_expert_template(items, template_path, scored_items=None)
        print(f"DRY RUN — template written to {template_path} (no LLM scoring)")
        return

    # Run LLM judge on each sampled item
    api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY required. Set in .env or pass --openrouter-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    model = args.model

    # Load cached results to enable resume
    cache: dict[str, dict] = {}
    if judged_path.exists():
        for item in load_jsonl(judged_path):
            qid = item.get("question_id", "")
            if qid and "quality_scores" in item:
                cache[qid] = item

    pending = [it for it in items if it.get("question_id") not in cache]
    print(f"  {len(cache)} cached | {len(pending)} to judge (model: {model})")

    errors = 0
    with tqdm(total=len(pending), desc="judging", unit="item") as bar:
        for item in pending:
            category = item["category"]
            rubric_text = load_rubric(category)
            thresholds = parse_thresholds(rubric_text)
            try:
                result = score_item(item, category, rubric_text, thresholds, client, model)
                cache[item["question_id"]] = result
            except Exception as e:
                print(f"\n  ERROR {item.get('question_id', '?')}: {e}", file=sys.stderr)
                cache[item["question_id"]] = item  # keep unscored
                errors += 1
            bar.update(1)

    # Write judged JSONL in sample order
    scored_items = []
    for item in items:
        qid = item.get("question_id", "")
        scored_items.append(cache.get(qid, item))

    with open(judged_path, "w", encoding="utf-8") as f:
        for r in scored_items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_expert_template(items, template_path, scored_items=scored_items)

    judged_count = sum(1 for it in scored_items if "quality_scores" in it)
    print(f"\nDone. {judged_count}/{len(items)} items judged, {errors} errors.")
    print(f"  Judged JSONL : {judged_path}")
    print(f"  Expert template: {template_path}")
    print(
        "\nNext: fill in the score columns (1–5) in the CSV template, "
        "save as expert_scoring_filled.csv, then run the 'report' subcommand."
    )


def _write_expert_template(
    items: list[dict],
    path: Path,
    scored_items: list[dict] | None,
) -> None:
    """Write the blank CSV template for expert scoring."""
    # Build a quick lookup of judge scores for the preview columns
    judge_lookup: dict[str, dict] = {}
    if scored_items:
        for r in scored_items:
            qid = r.get("question_id", "")
            if qid and "quality_scores" in r:
                judge_lookup[qid] = r["quality_scores"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_EXPERT_COLS + [
            f"judge_{d}" for d in DIMENSIONS
        ] + ["judge_composite", "judge_verdict"])
        writer.writeheader()
        for item in items:
            qid = item.get("question_id", "")
            js = judge_lookup.get(qid, {})
            row = {
                "question_id":  qid,
                "category":     item.get("category", ""),
                "question":     item.get("question", ""),
                "A":            item.get("A", ""),
                "B":            item.get("B", ""),
                "C":            item.get("C", ""),
                "D":            item.get("D", ""),
                "answer":       item.get("answer", ""),
                "notes":        "",
                "judge_composite": js.get("composite", ""),
                "judge_verdict":   js.get("verdict", ""),
            }
            for d in DIMENSIONS:
                row[d] = ""                   # expert fills these in
                row[f"judge_{d}"] = js.get(d, "")  # LLM score for reference
            writer.writerow(row)


# ---------------------------------------------------------------------------
# IAA computation helpers
# ---------------------------------------------------------------------------

def _load_expert_scores(csv_path: Path) -> dict[str, dict[str, int]]:
    """
    Read expert CSV. Returns {question_id: {dimension: score}}.
    Skips rows with any missing dimension score.
    """
    scores: dict[str, dict[str, int]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("question_id", "").strip()
            if not qid:
                continue
            dim_scores: dict[str, int] = {}
            skip = False
            for d in DIMENSIONS:
                val = row.get(d, "").strip()
                if not val:
                    skip = True
                    break
                try:
                    s = int(val)
                    if not (1 <= s <= 5):
                        raise ValueError(f"score {s} out of range")
                    dim_scores[d] = s
                except ValueError:
                    skip = True
                    break
            if not skip:
                scores[qid] = dim_scores
    return scores


def _load_judge_scores(jsonl_path: Path) -> dict[str, dict[str, int]]:
    """
    Read judge JSONL. Returns {question_id: {dimension: score}}.
    Skips items without quality_scores.
    """
    scores: dict[str, dict[str, int]] = {}
    for item in load_jsonl(jsonl_path):
        qid = item.get("question_id", "")
        qs = item.get("quality_scores", {})
        if not qid or not qs:
            continue
        scores[qid] = {d: int(qs[d]) for d in DIMENSIONS if d in qs}
    return scores


def _iaa_for_dimension(
    expert_vec: list[int],
    judge_vec: list[int],
    dim: str,
) -> dict:
    """Compute all IAA metrics for one dimension."""
    e = np.array(expert_vec, dtype=float)
    j = np.array(judge_vec, dtype=float)

    pearson, _ = pearsonr(e, j)
    spearman, _ = spearmanr(e, j)
    mae = float(np.mean(np.abs(e - j)))

    # Cohen's κ with quadratic weights (handles ordinal scale)
    labels = list(range(1, 6))
    kappa = cohen_kappa_score(expert_vec, judge_vec, weights="quadratic", labels=labels)

    # Krippendorff's α (ordinal) — reliability_data shape: (raters, units)
    reliability_data = np.array([expert_vec, judge_vec], dtype=float)
    alpha = krippendorff.alpha(reliability_data, level_of_measurement="ordinal")

    return {
        "n": len(expert_vec),
        "pearson_r":          round(float(pearson), 3),
        "spearman_rho":       round(float(spearman), 3),
        "mae":                round(mae, 3),
        "cohens_kappa":       round(float(kappa), 3),
        "krippendorff_alpha": round(float(alpha), 3),
    }


def _print_iaa_table(
    dim_results: dict[str, dict],
    overall: dict,
    dim_pass: dict[str, dict[str, bool]],
    overall_pass: dict[str, bool],
    title: str = "",
) -> None:
    """Print a formatted IAA metrics table to stdout."""
    col_w = 28
    metric_keys = list(TARGETS.keys())
    header = f"{'Dimension':<{col_w}}" + "".join(f"{k:>20}" for k in metric_keys) + f"{'PASS':>8}"
    if title:
        print(f"\n{title}")
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    def _row(label: str, metrics: dict, passes: dict) -> str:
        vals = "".join(f"{metrics.get(k, '-'):>20}" for k in metric_keys)
        return f"{label:<{col_w}}{vals}{('✓' if all(passes.values()) else '✗'):>8}"

    for dim in DIMENSIONS:
        if dim in dim_results:
            print(_row(dim, dim_results[dim], dim_pass.get(dim, {})))

    print("-" * len(header))
    if overall:
        print(_row("OVERALL (composite)", overall, overall_pass))
    print("=" * len(header))
    print("\nTargets: r ≥ 0.70 | ρ ≥ 0.70 | MAE ≤ 0.50 | κ ≥ 0.70 | α ≥ 0.70\n")


def _passes_targets(metrics: dict) -> dict[str, bool]:
    results = {}
    for key, target in TARGETS.items():
        val = metrics.get(key)
        if val is None:
            results[key] = False
        elif key == "mae":
            results[key] = val <= target
        else:
            results[key] = val >= target
    return results


# ---------------------------------------------------------------------------
# report subcommand
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    expert_path = Path(args.expert)
    judged_path = Path(args.judged)

    if not expert_path.exists():
        print(f"ERROR: expert CSV not found: {expert_path}", file=sys.stderr)
        sys.exit(1)
    if not judged_path.exists():
        print(f"ERROR: judged JSONL not found: {judged_path}", file=sys.stderr)
        sys.exit(1)

    expert_scores = _load_expert_scores(expert_path)
    judge_scores = _load_judge_scores(judged_path)

    # Intersect on question_id
    shared_ids = sorted(set(expert_scores) & set(judge_scores))
    if not shared_ids:
        print("ERROR: no question_ids matched between expert CSV and judged JSONL.", file=sys.stderr)
        sys.exit(1)

    print(f"Matched {len(shared_ids)} items (expert: {len(expert_scores)}, judged: {len(judge_scores)})")

    # Per-dimension metrics
    dim_results: dict[str, dict] = {}
    for dim in DIMENSIONS:
        e_vec = [expert_scores[qid][dim] for qid in shared_ids if dim in expert_scores[qid]]
        j_vec = [judge_scores[qid][dim] for qid in shared_ids if dim in judge_scores[qid]]
        # Only use items where both have scores for this dimension
        pairs = [
            (expert_scores[qid][dim], judge_scores[qid][dim])
            for qid in shared_ids
            if dim in expert_scores[qid] and dim in judge_scores[qid]
        ]
        if len(pairs) < 5:
            print(f"  WARNING: only {len(pairs)} paired scores for '{dim}' — skipping", file=sys.stderr)
            continue
        e_vec = [p[0] for p in pairs]
        j_vec = [p[1] for p in pairs]
        dim_results[dim] = _iaa_for_dimension(e_vec, j_vec, dim)

    # Overall composite metrics (treat composite scores as continuous)
    composite_pairs = []
    for qid in shared_ids:
        e_comp = np.mean([expert_scores[qid][d] for d in DIMENSIONS if d in expert_scores[qid]])
        j_qs = judge_scores[qid]
        if all(d in j_qs for d in DIMENSIONS):
            j_comp = np.mean([j_qs[d] for d in DIMENSIONS])
            composite_pairs.append((e_comp, j_comp))

    overall: dict = {}
    if len(composite_pairs) >= 5:
        e_comp_vec = [p[0] for p in composite_pairs]
        j_comp_vec = [p[1] for p in composite_pairs]
        pearson, _ = pearsonr(e_comp_vec, j_comp_vec)
        spearman, _ = spearmanr(e_comp_vec, j_comp_vec)
        mae = float(np.mean(np.abs(np.array(e_comp_vec) - np.array(j_comp_vec))))

        # Round to nearest integer for κ/α on composites
        e_rounded = [round(v) for v in e_comp_vec]
        j_rounded = [round(v) for v in j_comp_vec]
        kappa = cohen_kappa_score(e_rounded, j_rounded, weights="quadratic", labels=list(range(1, 6)))
        alpha = krippendorff.alpha(
            np.array([e_comp_vec, j_comp_vec], dtype=float),
            level_of_measurement="interval",
        )
        overall = {
            "n": len(composite_pairs),
            "pearson_r":          round(float(pearson), 3),
            "spearman_rho":       round(float(spearman), 3),
            "mae":                round(mae, 3),
            "cohens_kappa":       round(float(kappa), 3),
            "krippendorff_alpha": round(float(alpha), 3),
        }

    # Pass/fail
    overall_pass = _passes_targets(overall) if overall else {}
    dim_pass = {dim: _passes_targets(m) for dim, m in dim_results.items()}

    _print_iaa_table(dim_results, overall, dim_pass, overall_pass)

    # Write report
    report = {
        "n_matched": len(shared_ids),
        "n_expert": len(expert_scores),
        "n_judged": len(judge_scores),
        "targets": TARGETS,
        "dimensions": dim_results,
        "dimension_pass": {
            dim: all(p.values()) for dim, p in dim_pass.items()
        },
        "overall": overall,
        "overall_pass": {k: v for k, v in overall_pass.items()},
        "all_dimensions_pass": all(
            all(p.values()) for p in dim_pass.values()
        ),
        "overall_all_pass": all(overall_pass.values()) if overall_pass else False,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "iaa_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"IAA report written to {report_path}")

    # Summary verdict
    all_pass = report["all_dimensions_pass"] and report["overall_all_pass"]
    if all_pass:
        print("CALIBRATION PASSED — LLM judge meets all IAA targets.")
    else:
        failing = [
            k for k, v in overall_pass.items() if not v
        ]
        print(f"CALIBRATION NEEDS REVIEW — failing metrics: {failing}")


# ---------------------------------------------------------------------------
# Multi-model automated IAA
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_MODELS = [
    "xiaomi/mimo-v2-pro",
    "qwen/qwen3.6-plus",
    "anthropic/claude-3.5-haiku",
]


def _model_slug(model_id: str) -> str:
    """Convert model ID to a filesystem-safe slug, e.g. xiaomi/mimo-v2-pro → xiaomi_mimo-v2-pro."""
    return model_id.replace("/", "_")


def _score_with_model(
    items: list[dict],
    model: str,
    client: OpenAI,
    out_dir: Path,
) -> dict[str, dict[str, int]]:
    """
    Score all items with one model. Resume-safe via per-model JSONL cache.
    Returns {question_id: {dimension: score}}.
    """
    slug = _model_slug(model)
    cache_path = out_dir / f"{slug}_scored.jsonl"

    # Load existing scored items
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for r in load_jsonl(cache_path):
            qid = r.get("question_id", "")
            if qid and "quality_scores" in r:
                cache[qid] = r

    pending = [it for it in items if it.get("question_id") not in cache]
    print(f"  [{model}] {len(cache)} cached | {len(pending)} to score", flush=True)

    errors = 0
    with tqdm(total=len(pending), desc=slug[:35], unit="item") as bar:
        for item in pending:
            category = item.get("category") or item.get("judge_category", "")
            qid = item.get("question_id", "?")
            try:
                rubric_text = load_rubric(category)
                thresholds = parse_thresholds(rubric_text)
                result = score_item(item, category, rubric_text, thresholds, client, model)
                cache[qid] = result
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"\n  ERROR {qid}: {e}", file=sys.stderr)
                errors += 1
            bar.update(1)

    if errors:
        print(f"  [{model}] {errors} scoring errors", file=sys.stderr)

    return {
        qid: {d: r["quality_scores"][d] for d in DIMENSIONS}
        for qid, r in cache.items()
        if "quality_scores" in r and all(d in r["quality_scores"] for d in DIMENSIONS)
    }


def _multi_model_iaa(
    model_scores: dict[str, dict[str, dict[str, int]]],
    shared_ids: list[str],
    dim: str,
) -> dict:
    """
    Compute IAA across N models for one dimension.

    model_scores: {model_id: {qid: {dim: score}}}
    Returns dict of metric values.
    """
    model_names = list(model_scores.keys())
    n_models = len(model_names)
    n_items = len(shared_ids)

    # Build (n_models, n_items) matrix; np.nan for missing
    matrix = np.full((n_models, n_items), np.nan)
    for i, model in enumerate(model_names):
        for j, qid in enumerate(shared_ids):
            s = model_scores[model].get(qid, {}).get(dim)
            if s is not None:
                matrix[i, j] = float(s)

    # Krippendorff's α — naturally supports N raters and ordinal scale
    alpha = krippendorff.alpha(matrix, level_of_measurement="ordinal")

    # Pairwise metrics averaged across all model pairs
    pearson_vals, spearman_vals, mae_vals, kappa_vals = [], [], [], []
    for i1, i2 in combinations(range(n_models), 2):
        pairs = [
            (matrix[i1, j], matrix[i2, j])
            for j in range(n_items)
            if not np.isnan(matrix[i1, j]) and not np.isnan(matrix[i2, j])
        ]
        if len(pairs) < 5:
            continue
        v1 = [p[0] for p in pairs]
        v2 = [p[1] for p in pairs]
        pr, _ = pearsonr(v1, v2)
        sr, _ = spearmanr(v1, v2)
        mae = float(np.mean(np.abs(np.array(v1) - np.array(v2))))
        kappa = cohen_kappa_score(
            [int(round(x)) for x in v1],
            [int(round(x)) for x in v2],
            weights="quadratic",
            labels=list(range(1, 6)),
        )
        pearson_vals.append(pr)
        spearman_vals.append(sr)
        mae_vals.append(mae)
        kappa_vals.append(kappa)

    n_complete = int(np.sum(~np.any(np.isnan(matrix), axis=0)))

    return {
        "n_items": n_complete,
        "n_models": n_models,
        "pearson_r":           round(float(np.mean(pearson_vals)), 3) if pearson_vals else None,
        "spearman_rho":        round(float(np.mean(spearman_vals)), 3) if spearman_vals else None,
        "mae":                 round(float(np.mean(mae_vals)), 3) if mae_vals else None,
        "cohens_kappa":        round(float(np.mean(kappa_vals)), 3) if kappa_vals else None,
        "krippendorff_alpha":  round(float(alpha), 3),
    }


def cmd_auto(args: argparse.Namespace) -> None:
    """Fully automated multi-model IAA — no human experts required."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = args.models
    print(f"Automated multi-model IAA with {len(models)} judges:")
    for m in models:
        print(f"  • {m}")

    # Step 1: calibration sample
    sample_path = out_dir / "calibration_items.jsonl"
    if args.reuse_sample and sample_path.exists():
        print(f"\nReusing existing sample: {sample_path}")
        items = load_jsonl(sample_path)
        for item in items:
            if "category" not in item and "judge_category" in item:
                item["category"] = item["judge_category"]
    else:
        print(f"\nSampling {args.n} items across {len(CATEGORY_RUBRICS)} categories…")
        items = stratified_sample(args.n, seed=args.seed)
        print(f"  → {len(items)} items sampled")
        with open(sample_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    if args.dry_run:
        print(f"\nDRY RUN — {len(items)} items, {len(models)} models, no API calls.")
        return

    # Step 2: score with each model
    api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY required. Set in .env or pass --openrouter-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    model_scores: dict[str, dict[str, dict[str, int]]] = {}
    for model in models:
        print(f"\nScoring with {model}…")
        model_scores[model] = _score_with_model(items, model, client, out_dir)

    # Step 3: items scored by ALL models
    shared_ids = sorted(
        set.intersection(*(set(s.keys()) for s in model_scores.values()))
    )
    print(f"\n{len(shared_ids)} items scored by all {len(models)} models")
    if len(shared_ids) < 10:
        print("ERROR: too few shared items for reliable IAA.", file=sys.stderr)
        sys.exit(1)

    # Step 4: per-dimension IAA
    print("Computing IAA metrics…")
    dim_results: dict[str, dict] = {
        dim: _multi_model_iaa(model_scores, shared_ids, dim)
        for dim in DIMENSIONS
    }

    # Step 5: overall composite IAA
    model_names = list(model_scores.keys())
    comp_matrix = np.array([
        [
            float(np.mean([model_scores[m][qid][d] for d in DIMENSIONS]))
            for qid in shared_ids
            if all(d in model_scores[m].get(qid, {}) for d in DIMENSIONS)
        ]
        for m in model_names
    ])

    overall: dict = {}
    n_comp = comp_matrix.shape[1] if comp_matrix.ndim == 2 else 0
    if n_comp >= 5:
        alpha = krippendorff.alpha(comp_matrix, level_of_measurement="interval")
        pr_vals, sr_vals, mae_vals, k_vals = [], [], [], []
        for i1, i2 in combinations(range(len(model_names)), 2):
            v1, v2 = comp_matrix[i1], comp_matrix[i2]
            pr, _ = pearsonr(v1, v2)
            sr, _ = spearmanr(v1, v2)
            mae = float(np.mean(np.abs(v1 - v2)))
            kappa = cohen_kappa_score(
                [int(round(x)) for x in v1],
                [int(round(x)) for x in v2],
                weights="quadratic", labels=list(range(1, 6)),
            )
            pr_vals.append(pr)
            sr_vals.append(sr)
            mae_vals.append(mae)
            k_vals.append(kappa)
        overall = {
            "n_items": n_comp,
            "n_models": len(model_names),
            "pearson_r":          round(float(np.mean(pr_vals)), 3),
            "spearman_rho":       round(float(np.mean(sr_vals)), 3),
            "mae":                round(float(np.mean(mae_vals)), 3),
            "cohens_kappa":       round(float(np.mean(k_vals)), 3),
            "krippendorff_alpha": round(float(alpha), 3),
        }

    # Step 6: print table + write report
    overall_pass = _passes_targets(overall) if overall else {}
    dim_pass = {dim: _passes_targets(m) for dim, m in dim_results.items()}
    _print_iaa_table(
        dim_results, overall, dim_pass, overall_pass,
        title=f"Multi-Model IAA ({', '.join(models)})",
    )

    report = {
        "models": models,
        "n_items": len(shared_ids),
        "targets": TARGETS,
        "dimensions": dim_results,
        "dimension_pass": {dim: all(p.values()) for dim, p in dim_pass.items()},
        "overall": overall,
        "overall_pass": overall_pass,
        "all_dimensions_pass": all(all(p.values()) for p in dim_pass.values()),
        "overall_all_pass": all(overall_pass.values()) if overall_pass else False,
    }

    report_path = out_dir / "multi_model_iaa_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report written to {report_path}")

    all_pass = report["all_dimensions_pass"] and report["overall_all_pass"]
    if all_pass:
        print("AUTOMATED CALIBRATION PASSED — judges show strong cross-model agreement.")
    else:
        failing = [k for k, v in overall_pass.items() if not v]
        print(f"REVIEW NEEDED — failing metrics: {failing}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge calibration and IAA workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # --- sample ---
    p_sample = sub.add_parser(
        "sample",
        help="Stratified sample + LLM judge + export expert CSV template",
    )
    p_sample.add_argument(
        "--n", type=int, default=63,
        help="Total items to sample (evenly distributed across 9 categories, default 63)",
    )
    p_sample.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default 42)",
    )
    p_sample.add_argument(
        "--output-dir", default="quality/calibration",
        help="Output directory (default: quality/calibration)",
    )
    p_sample.add_argument(
        "--model",
        default=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL})",
    )
    p_sample.add_argument(
        "--openrouter-key", default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY in .env)",
    )
    p_sample.add_argument(
        "--dry-run", action="store_true",
        help="Sample items and write CSV template without making API calls",
    )

    # --- report ---
    p_report = sub.add_parser(
        "report",
        help="Compute IAA metrics from expert CSV + LLM judge JSONL",
    )
    p_report.add_argument(
        "--expert", required=True, metavar="CSV",
        help="Expert-scored CSV (filled copy of expert_scoring_template.csv)",
    )
    p_report.add_argument(
        "--judged",
        default="quality/calibration/calibration_items.jsonl",
        help="LLM-judged JSONL from the 'sample' step (default: quality/calibration/calibration_items.jsonl)",
    )
    p_report.add_argument(
        "--output-dir", default="quality/calibration",
        help="Directory for iaa_report.json (default: quality/calibration)",
    )

    # --- auto ---
    p_auto = sub.add_parser(
        "auto",
        help="Automated multi-model IAA — no human experts required",
    )
    p_auto.add_argument(
        "--models", nargs="+", default=DEFAULT_JUDGE_MODELS,
        metavar="MODEL",
        help=(
            f"OpenRouter model IDs to use as judges "
            f"(default: {', '.join(DEFAULT_JUDGE_MODELS)})"
        ),
    )
    p_auto.add_argument(
        "--n", type=int, default=63,
        help="Items to sample if not reusing an existing sample (default 63)",
    )
    p_auto.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default 42)",
    )
    p_auto.add_argument(
        "--reuse-sample", action="store_true",
        help="Reuse quality/calibration/calibration_items.jsonl if it exists",
    )
    p_auto.add_argument(
        "--output-dir", default="quality/calibration",
        help="Output directory (default: quality/calibration)",
    )
    p_auto.add_argument(
        "--openrouter-key", default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY in .env)",
    )
    p_auto.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without making API calls",
    )

    args = parser.parse_args()

    if args.subcommand == "sample":
        cmd_sample(args)
    elif args.subcommand == "report":
        cmd_report(args)
    elif args.subcommand == "auto":
        cmd_auto(args)


if __name__ == "__main__":
    main()
