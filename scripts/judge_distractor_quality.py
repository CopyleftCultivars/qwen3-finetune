#!/usr/bin/env python3
"""Run the pinned OpenRouter MCQ-quality judge without overwriting old scores."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcq_generation.quality.llm_judge import (  # noqa: E402
    OPENROUTER_BASE_URL,
    apply_verdict,
    build_judge_prompt,
    load_rubric,
    parse_judge_response,
    parse_thresholds,
)


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="anthropic/claude-haiku-4.5")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--openrouter-key", default=None)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        parser.error("OPENROUTER_API_KEY is not configured")
    rows = load(args.input)
    rubric = load_rubric("conversations")
    thresholds = parse_thresholds(rubric)
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def score(row: dict) -> dict:
        prompt = build_judge_prompt(row, rubric, "conversations")
        from mcq_generation.quality.llm_judge import call_judge

        raw = call_judge(prompt, client, args.model)
        parsed = parse_judge_response(raw)
        result = dict(row)
        result["quality_scores"] = {
            **parsed["scores"],
            "composite": parsed["composite_score"],
            "verdict": apply_verdict(parsed["composite_score"], thresholds),
        }
        result["quality_justifications"] = parsed["justifications"]
        result["judge_model"] = args.model
        result["judge_category"] = "conversations"
        return result

    results: dict[str, dict] = {}
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(score, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - network failure path
                errors.append({"question_id": row.get("question_id"), "error": str(exc)})
                continue
            results[str(result.get("question_id"))] = result
            print(f"judged {len(results)}/{len(rows)} {result.get('question_id')}", flush=True)

    ordered = [results[str(row.get("question_id"))] for row in rows if str(row.get("question_id")) in results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    verdicts = {}
    for row in ordered:
        verdict = row.get("quality_scores", {}).get("verdict")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
    summary = {
        "format": "distractor_quality_judge_report_v1",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "judge_model": args.model,
        "judge_category": "conversations",
        "count": len(ordered),
        "errors": errors,
        "verdict_counts": verdicts,
        "thresholds": thresholds,
        "external_content_transmission": True,
        "sealed_test_paths_read": [],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
