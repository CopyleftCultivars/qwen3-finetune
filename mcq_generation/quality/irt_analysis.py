#!/usr/bin/env python3
"""
IRT statistical refinement for MCQ benchmarks (Issue #56).

Fits a 2-Parameter Logistic (2PL) IRT model using responses from 12 evaluation
models and filters out items with poor psychometric properties.

Subcommands
-----------
eval
    Run N evaluation models on all benchmark items via OpenRouter API.
    Each model answers every MCQ; responses are stored per-model per-category.
    Resume-safe: already-answered items are skipped.

fit
    Build the response matrix from collected responses and fit a 2PL IRT model
    via py-irt.  Saves difficulty (b) and discrimination (a) per item.

filter
    Apply filter criteria to each benchmark:
      - Remove: discrimination a < MIN_DISC (default 0.3) — item has no signal
      - Remove: ceiling items (all models correct) — too easy / trivial
      - Flag:   floor items (all models wrong)  — potential label errors for Layer 5

report
    Print per-category IRT statistics and filter counts.

Usage
-----
    # Step 1 — collect model responses (expensive: 12 models × ~1500 items)
    conda run -n nf-benchmark python quality/irt_analysis.py eval

    # Step 1 dry-run (shows plan, no API calls)
    conda run -n nf-benchmark python quality/irt_analysis.py eval --dry-run

    # Step 2 — fit 2PL model (CPU, ~2–5 min)
    conda run -n nf-benchmark python quality/irt_analysis.py fit

    # Step 3 — apply filter criteria
    conda run -n nf-benchmark python quality/irt_analysis.py filter

    # Step 4 — report
    conda run -n nf-benchmark python quality/irt_analysis.py report

Output layout
-------------
    quality/irt/responses/{model_slug}_{category}.jsonl   per-model responses
    quality/irt/response_matrix.jsonl                     py-irt input format
    quality/irt/irt_params.json                           fitted 2PL parameters
    quality/irt/filtered_{category}.jsonl                 items passing all filters
    quality/irt/floor_items.jsonl                         floor items (→ Layer 5)
    quality/irt/removed_items.jsonl                       removed (ceiling + low disc)
    quality/irt/filter_report.json                        per-item filter decisions
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
BENCHMARKS_DIR = WORKSPACE / "benchmarks"
IRT_DIR = WORKSPACE / "quality" / "irt"

load_dotenv(WORKSPACE / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRY_DELAYS = [2, 5, 15]
API_TIMEOUT = 60
DEFAULT_WORKERS = 5
MIN_DISC_DEFAULT = 0.3
IRT_EPOCHS_DEFAULT = 2000

# 12-model evaluation pool from quality_assessment_plan.md
EVAL_MODELS: list[dict] = [
    # Tier, model_id
    {"tier": "frontier", "model": "anthropic/claude-opus-4.6"},
    {"tier": "frontier", "model": "google/gemini-2.5-pro"},
    {"tier": "strong",   "model": "anthropic/claude-sonnet-4.6"},
    {"tier": "strong",   "model": "google/gemini-2.5-flash"},
    {"tier": "strong",   "model": "qwen/qwen3-235b-a22b"},
    {"tier": "mid",      "model": "anthropic/claude-haiku-4.5"},
    {"tier": "mid",      "model": "mistralai/mistral-large-2512"},
    {"tier": "mid",      "model": "meta-llama/llama-3.3-70b-instruct"},
    {"tier": "mid",      "model": "qwen/qwen-turbo"},
    {"tier": "weak",     "model": "qwen/qwen3-32b"},
    {"tier": "weak",     "model": "meta-llama/llama-3.1-8b-instruct"},
    {"tier": "weak",     "model": "qwen/qwen3-8b"},
]

# Thinking models need room for <think> blocks; non-thinking only need 1 token
_MODEL_MAX_TOKENS: dict[str, int] = {
    "qwen/qwen3-235b-a22b": 2048,
    "qwen/qwen3-32b":       2048,
    "qwen/qwen3-8b":        2048,
}
_DEFAULT_MAX_TOKENS = 16

VALID_ANSWERS = frozenset({"A", "B", "C", "D"})

# ---------------------------------------------------------------------------
# Helpers: paths and slugs
# ---------------------------------------------------------------------------


def model_slug(model_id: str) -> str:
    """Convert 'vendor/model-name' to 'vendor__model_name' for filenames."""
    return model_id.replace("/", "__").replace("-", "_").replace(".", "_")


def response_path(model_id: str, category: str) -> Path:
    return IRT_DIR / "responses" / f"{model_slug(model_id)}_{category}.jsonl"


def discover_benchmarks() -> list[tuple[str, Path]]:
    """Return [(category, path), ...] for all benchmark JSONL files."""
    results = []
    for path in sorted(BENCHMARKS_DIR.rglob("*_benchmark_v1.jsonl")):
        category = path.parent.name
        results.append((category, path))
    return results


def load_items(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# MCQ evaluation prompt
# ---------------------------------------------------------------------------

MCQ_EVAL_SYSTEM = (
    "You are an expert answering multiple-choice questions. "
    "Read the question and options carefully, then reply with ONLY the letter "
    "of the correct answer: A, B, C, or D. No explanation. No punctuation."
)


def build_eval_prompt(item: dict) -> str:
    return (
        f"Question: {item['question']}\n"
        f"A: {item['A']}\n"
        f"B: {item['B']}\n"
        f"C: {item['C']}\n"
        f"D: {item['D']}\n\n"
        "Answer (A, B, C, or D):"
    )


def parse_answer(text: str) -> str | None:
    """Extract A/B/C/D from a model response. Returns None if unparseable."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # First, check if the very first non-whitespace char is a letter
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------


def call_model(prompt: str, client: OpenAI, model: str) -> str:
    """Call a model via OpenRouter. Returns raw response text."""
    max_tokens = _MODEL_MAX_TOKENS.get(model, _DEFAULT_MAX_TOKENS)
    last_error = ""
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MCQ_EVAL_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=API_TIMEOUT,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            last_error = f"empty response from {model}"
        except Exception as exc:
            last_error = str(exc)
            if attempt == len(RETRY_DELAYS):
                raise RuntimeError(
                    f"API failed after {attempt + 1} attempts: {last_error}"
                ) from exc
            print(f"\n  API error (attempt {attempt + 1}): {exc}", file=sys.stderr)
    raise RuntimeError(f"API failed: {last_error}")


# ---------------------------------------------------------------------------
# eval subcommand
# ---------------------------------------------------------------------------


def _load_existing_responses(path: Path) -> dict[str, dict]:
    """Load already-completed responses keyed by question_id."""
    cache: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[rec["question_id"]] = rec
    return cache


def _eval_model_on_category(
    model_id: str,
    category: str,
    items: list[dict],
    client: OpenAI,
    workers: int,
) -> None:
    """Score all items in a category for one model, writing to response JSONL."""
    out_path = response_path(model_id, category)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = _load_existing_responses(out_path)
    pending = [it for it in items if it["question_id"] not in cache]

    if not pending:
        return

    file_lock = Lock()

    def score_one(item: dict) -> dict:
        prompt = build_eval_prompt(item)
        try:
            raw = call_model(prompt, client, model_id)
            predicted = parse_answer(raw)
            correct = predicted == item.get("answer")
        except RuntimeError as exc:
            print(f"\n  [skip] {item['question_id']}: {exc}", file=sys.stderr)
            raw = None
            predicted = None
            correct = False
        return {
            "question_id": item["question_id"],
            "model": model_id,
            "predicted": predicted,
            "correct_answer": item.get("answer"),
            "correct": correct,
            "raw_response": raw,
        }

    with open(out_path, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(score_one, it): it for it in pending}
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"  {model_slug(model_id)[:30]} / {category}",
                leave=False,
            ):
                rec = fut.result()
                with file_lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()


def cmd_eval(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: OPENROUTER_API_KEY not set in environment or .env", file=sys.stderr)
        return 1

    models = args.models or [m["model"] for m in EVAL_MODELS]
    benchmarks = discover_benchmarks()

    if args.category:
        benchmarks = [(cat, path) for cat, path in benchmarks if cat == args.category]
        if not benchmarks:
            print(f"ERROR: No benchmark found for category '{args.category}'", file=sys.stderr)
            return 1

    if not benchmarks:
        print("No benchmark files found.", file=sys.stderr)
        return 1

    total_items = sum(len(load_items(p)) for _, p in benchmarks)
    print(
        f"Eval plan: {len(models)} models × "
        f"{len(benchmarks)} categories × ~{total_items} items"
    )

    if args.dry_run:
        print("\n[dry-run] Models that would be called:")
        for m in models:
            entry = next((e for e in EVAL_MODELS if e["model"] == m), {"tier": "custom"})
            print(f"  [{entry['tier']:8s}] {m}")
        print("\n[dry-run] Benchmarks:")
        for cat, path in benchmarks:
            n = len(load_items(path))
            print(f"  {cat:45s} {n:4d} items")
        return 0

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    for model_id in models:
        entry = next((e for e in EVAL_MODELS if e["model"] == model_id), {"tier": "custom"})
        print(f"\n[{entry['tier']:8s}] {model_id}")
        for category, path in benchmarks:
            items = load_items(path)
            _eval_model_on_category(
                model_id=model_id,
                category=category,
                items=items,
                client=client,
                workers=args.workers,
            )

    print("\nEval complete.")
    return 0


# ---------------------------------------------------------------------------
# fit subcommand
# ---------------------------------------------------------------------------


def _build_response_matrix(
    benchmarks: list[tuple[str, Path]],
    models: list[str],
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """
    Build IRT-format records and a {model → {qid: 0/1}} matrix.

    Returns:
        records  — list of py-irt subject rows
        matrix   — {model_id: {question_id: 0/1}}
    """
    # Collect all question_ids in order
    all_qids: list[str] = []
    for _cat, path in benchmarks:
        for item in load_items(path):
            all_qids.append(item["question_id"])

    matrix: dict[str, dict[str, int]] = {}
    for model_id in models:
        responses: dict[str, int] = {}
        for category, _ in benchmarks:
            rpath = response_path(model_id, category)
            if not rpath.exists():
                continue
            for line in rpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    responses[rec["question_id"]] = 1 if rec.get("correct") else 0
        if responses:
            matrix[model_id] = responses

    # Build py-irt subject rows: only items answered by this model
    records = []
    for model_id, responses in matrix.items():
        records.append({"subject_id": model_id, "responses": responses})

    return records, matrix


def cmd_fit(args: argparse.Namespace) -> int:
    try:
        from py_irt.training import IrtModelTrainer, IrtConfig
        from py_irt.dataset import Dataset
    except ImportError:
        print("ERROR: py-irt not installed. Run: pip install py-irt", file=sys.stderr)
        return 1

    benchmarks = discover_benchmarks()
    if not benchmarks:
        print("No benchmark files found.", file=sys.stderr)
        return 1

    models = [m["model"] for m in EVAL_MODELS]

    # Check we have at least some response data
    response_dir = IRT_DIR / "responses"
    if not response_dir.exists() or not list(response_dir.glob("*.jsonl")):
        print(
            "No response files found in quality/irt/responses/.\n"
            "Run `eval` subcommand first.",
            file=sys.stderr,
        )
        return 1

    print("Building response matrix...")
    records, matrix = _build_response_matrix(benchmarks, models)

    if not records:
        print("ERROR: No response data found for any model.", file=sys.stderr)
        return 1

    n_models = len(records)
    n_items_total = len(set(qid for r in records for qid in r["responses"]))
    print(f"  {n_models} models, {n_items_total} items with at least one response")

    # Write response matrix JSONL
    IRT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = IRT_DIR / "response_matrix.jsonl"
    with open(matrix_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"  Response matrix written: {matrix_path.relative_to(WORKSPACE)}")

    # Fit 2PL IRT model
    print(f"Fitting 2PL IRT model ({args.epochs} epochs, device=cpu)...")
    dataset = Dataset.from_jsonlines(matrix_path)
    config = IrtConfig(model_type="2pl", epochs=args.epochs)
    trainer = IrtModelTrainer(data_path=matrix_path, config=config, dataset=dataset)
    trainer.train(device="cpu")

    # Save parameters
    params_path = IRT_DIR / "irt_params.json"
    trainer.save(params_path)
    print(f"  IRT params saved: {params_path.relative_to(WORKSPACE)}")

    # Quick summary
    params = json.loads(params_path.read_text(encoding="utf-8"))
    disc_list = params.get("disc", [])
    diff_list = params.get("diff", [])
    if disc_list:
        import statistics
        print(
            f"  Discrimination (a): "
            f"min={min(disc_list):.3f}, "
            f"mean={statistics.mean(disc_list):.3f}, "
            f"max={max(disc_list):.3f}"
        )
    if diff_list:
        import statistics
        print(
            f"  Difficulty (b):     "
            f"min={min(diff_list):.3f}, "
            f"mean={statistics.mean(diff_list):.3f}, "
            f"max={max(diff_list):.3f}"
        )

    return 0


# ---------------------------------------------------------------------------
# filter subcommand
# ---------------------------------------------------------------------------


def _load_irt_params() -> dict:
    params_path = IRT_DIR / "irt_params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"IRT params not found at {params_path}. Run `fit` first."
        )
    return json.loads(params_path.read_text(encoding="utf-8"))


def _build_per_item_stats(
    params: dict,
    matrix: dict[str, dict[str, int]],
) -> dict[str, dict]:
    """
    Build per-item stats dict: {question_id: {disc, diff, n_correct, n_total}}.
    """
    ix_to_item = params.get("item_ids", {})
    disc_list = params.get("disc", [])
    diff_list = params.get("diff", [])

    per_item: dict[str, dict] = {}
    for ix_str, qid in ix_to_item.items():
        ix = int(ix_str)
        per_item[qid] = {
            "question_id": qid,
            "disc": disc_list[ix] if ix < len(disc_list) else None,
            "diff": diff_list[ix] if ix < len(diff_list) else None,
            "n_correct": 0,
            "n_total": 0,
        }

    # Accumulate response counts
    for _model, responses in matrix.items():
        for qid, correct in responses.items():
            if qid in per_item:
                per_item[qid]["n_total"] += 1
                per_item[qid]["n_correct"] += correct

    return per_item


def cmd_filter(args: argparse.Namespace) -> int:
    benchmarks = discover_benchmarks()
    if not benchmarks:
        print("No benchmark files found.", file=sys.stderr)
        return 1

    try:
        params = _load_irt_params()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    models = [m["model"] for m in EVAL_MODELS]
    _, matrix = _build_response_matrix(benchmarks, models)

    if not matrix:
        print(
            "ERROR: No response data. Run `eval` first.", file=sys.stderr
        )
        return 1

    per_item = _build_per_item_stats(params, matrix)

    IRT_DIR.mkdir(parents=True, exist_ok=True)
    floor_items: list[dict] = []
    removed_items: list[dict] = []
    filter_report: list[dict] = []

    n_models_total = len(matrix)

    for category, path in benchmarks:
        items = load_items(path)
        kept: list[dict] = []

        for item in items:
            qid = item["question_id"]
            stats = per_item.get(qid)

            if stats is None:
                # No IRT data: keep by default (not enough data to filter)
                kept.append(item)
                filter_report.append(
                    {"question_id": qid, "category": category, "decision": "keep",
                     "reason": "no_irt_data"}
                )
                continue

            disc = stats["disc"]
            n_correct = stats["n_correct"]
            n_total = stats["n_total"]

            # Ceiling: all evaluated models got it right
            if n_total > 0 and n_correct == n_total:
                removed_items.append({**item, "irt_reason": "ceiling",
                                      "disc": disc, "diff": stats["diff"],
                                      "n_correct": n_correct, "n_total": n_total})
                filter_report.append(
                    {"question_id": qid, "category": category, "decision": "remove",
                     "reason": "ceiling", "disc": disc, "n_correct": n_correct,
                     "n_total": n_total}
                )
                continue

            # Floor: all evaluated models got it wrong → flag for Layer 5
            if n_total > 0 and n_correct == 0:
                floor_items.append({**item, "irt_reason": "floor",
                                    "disc": disc, "diff": stats["diff"],
                                    "n_correct": n_correct, "n_total": n_total})
                filter_report.append(
                    {"question_id": qid, "category": category, "decision": "flag_floor",
                     "reason": "floor", "disc": disc, "n_correct": n_correct,
                     "n_total": n_total}
                )
                continue

            # Low discrimination
            if disc is not None and disc < args.min_disc:
                removed_items.append({**item, "irt_reason": "low_discrimination",
                                      "disc": disc, "diff": stats["diff"],
                                      "n_correct": n_correct, "n_total": n_total})
                filter_report.append(
                    {"question_id": qid, "category": category, "decision": "remove",
                     "reason": "low_discrimination", "disc": disc,
                     "n_correct": n_correct, "n_total": n_total}
                )
                continue

            kept.append({**item, "irt_disc": disc, "irt_diff": stats["diff"]})
            filter_report.append(
                {"question_id": qid, "category": category, "decision": "keep",
                 "reason": "pass", "disc": disc, "diff": stats["diff"],
                 "n_correct": n_correct, "n_total": n_total}
            )

        # Write filtered benchmark
        out_path = IRT_DIR / f"filtered_{category}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for item in kept:
                fh.write(json.dumps(item) + "\n")
        print(
            f"  {category:45s}  "
            f"{len(kept):4d} kept / {len(items):4d} total "
            f"({len(items) - len(kept)} removed)"
        )

    # Write floor items (→ Layer 5)
    floor_path = IRT_DIR / "floor_items.jsonl"
    with open(floor_path, "w", encoding="utf-8") as fh:
        for item in floor_items:
            fh.write(json.dumps(item) + "\n")

    # Write all removed items
    removed_path = IRT_DIR / "removed_items.jsonl"
    with open(removed_path, "w", encoding="utf-8") as fh:
        for item in removed_items:
            fh.write(json.dumps(item) + "\n")

    # Write full filter report
    report_path = IRT_DIR / "filter_report.json"
    report_path.write_text(
        json.dumps({"min_disc_threshold": args.min_disc,
                    "n_models_evaluated": n_models_total,
                    "items": filter_report},
                   indent=2),
        encoding="utf-8",
    )

    total_kept = sum(1 for r in filter_report if r["decision"] == "keep")
    total_removed = sum(1 for r in filter_report if r["decision"] == "remove")
    total_floor = len(floor_items)

    print(f"\n{'=' * 60}")
    print(f"Filter summary (min_disc={args.min_disc})")
    print(f"  Kept:             {total_kept}")
    print(f"  Removed:          {total_removed}  (ceiling + low discrimination)")
    print(f"  Flagged (floor):  {total_floor}  → quality/irt/floor_items.jsonl")
    print(f"  Report:           {report_path.relative_to(WORKSPACE)}")

    return 0


# ---------------------------------------------------------------------------
# report subcommand
# ---------------------------------------------------------------------------


def cmd_report(_args: argparse.Namespace) -> int:
    params_path = IRT_DIR / "irt_params.json"
    report_path = IRT_DIR / "filter_report.json"

    if not params_path.exists():
        print("IRT params not found. Run `fit` first.", file=sys.stderr)
        return 1

    params = json.loads(params_path.read_text(encoding="utf-8"))
    disc_list = params.get("disc", [])
    diff_list = params.get("diff", [])

    print("2PL IRT Model — Parameter Summary")
    print(f"  Items fitted:       {len(disc_list)}")

    if disc_list:
        import statistics
        low_disc = sum(1 for a in disc_list if a < MIN_DISC_DEFAULT)
        print("  Discrimination (a):")
        print(f"    min / mean / max: "
              f"{min(disc_list):.3f} / {statistics.mean(disc_list):.3f} / {max(disc_list):.3f}")
        print(f"    a < {MIN_DISC_DEFAULT}: {low_disc} items")

    if diff_list:
        import statistics
        print("  Difficulty (b):")
        print(f"    min / mean / max: "
              f"{min(diff_list):.3f} / {statistics.mean(diff_list):.3f} / {max(diff_list):.3f}")

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        items = report.get("items", [])
        by_category: dict[str, dict[str, int]] = {}
        for r in items:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = {"keep": 0, "remove": 0, "flag_floor": 0, "no_irt": 0}
            decision = r["decision"]
            if decision == "keep":
                reason = r.get("reason", "pass")
                if reason == "no_irt_data":
                    by_category[cat]["no_irt"] += 1
                else:
                    by_category[cat]["keep"] += 1
            elif decision == "remove":
                by_category[cat]["remove"] += 1
            elif decision == "flag_floor":
                by_category[cat]["flag_floor"] += 1

        print(f"\nFilter Report (min_disc={report.get('min_disc_threshold', MIN_DISC_DEFAULT)})")
        print(f"  {'Category':45s}  {'Keep':>5}  {'Remove':>6}  {'Floor':>5}  {'No IRT':>6}")
        print("  " + "-" * 70)
        for cat, counts in sorted(by_category.items()):
            print(
                f"  {cat:45s}  "
                f"{counts['keep']:5d}  "
                f"{counts['remove']:6d}  "
                f"{counts['flag_floor']:5d}  "
                f"{counts['no_irt']:6d}"
            )
        total_keep = sum(c["keep"] for c in by_category.values())
        total_rem = sum(c["remove"] for c in by_category.values())
        total_fl = sum(c["flag_floor"] for c in by_category.values())
        total_noirt = sum(c["no_irt"] for c in by_category.values())
        print("  " + "-" * 70)
        print(
            f"  {'TOTAL':45s}  "
            f"{total_keep:5d}  "
            f"{total_rem:6d}  "
            f"{total_fl:5d}  "
            f"{total_noirt:6d}"
        )

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="IRT statistical refinement for MCQ benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- eval ---
    eval_p = sub.add_parser("eval", help="Run evaluation models on benchmarks")
    eval_p.add_argument(
        "--category",
        metavar="NAME",
        default=None,
        help="Restrict eval to a single category (default: all categories)",
    )
    eval_p.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Override model list (default: all 12 from plan)",
    )
    eval_p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel API workers per model (default: {DEFAULT_WORKERS})",
    )
    eval_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without making API calls",
    )

    # --- fit ---
    fit_p = sub.add_parser("fit", help="Fit 2PL IRT model from collected responses")
    fit_p.add_argument(
        "--epochs",
        type=int,
        default=IRT_EPOCHS_DEFAULT,
        help=f"Training epochs (default: {IRT_EPOCHS_DEFAULT})",
    )

    # --- filter ---
    filt_p = sub.add_parser("filter", help="Apply IRT filter criteria to benchmarks")
    filt_p.add_argument(
        "--min-disc",
        type=float,
        default=MIN_DISC_DEFAULT,
        metavar="THRESHOLD",
        help=f"Minimum discrimination (a) to keep item (default: {MIN_DISC_DEFAULT})",
    )

    # --- report ---
    sub.add_parser("report", help="Print IRT statistics summary")

    args = parser.parse_args()

    dispatch = {
        "eval": cmd_eval,
        "fit": cmd_fit,
        "filter": cmd_filter,
        "report": cmd_report,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
