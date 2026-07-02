#!/usr/bin/env python3
"""LLM-as-judge validation pipeline for Qwen3 training data.

Scores each instruction/output pair on 5 dimensions and assigns a verdict:
  keep   (composite ≥ 4.0) — include in training data
  review (composite ≥ 3.0) — inspect manually before including
  reject (composite < 3.0)  — exclude from training data

Scoring dimensions:
  1. think_block_quality   — is <think>...</think> a genuine rationale (not label restatement)?
  2. format_compliance     — does output exactly match <think>...</think>\\n{letter}?
  3. answer_matches_rationale — does the rationale logically support the stated answer letter?
  4. factual_accuracy      — is the agricultural content plausible and factually sound?
  5. domain_relevance      — is this natural farming / agroecology domain?

For NF conversation pairs (no MCQ structure), think_block_quality and
answer_matches_rationale are replaced by response_quality and helpfulness.

Uses OpenRouter → Claude (same pattern as benchmark-workspace/quality/llm_judge.py).
Requires OPENROUTER_API_KEY in environment or .env.

Usage:
    python scripts/judge_training_data.py \
        --input data/sources/mcq_with_rationale.jsonl \
        --output data/sources/mcq_with_rationale_judged.jsonl \
        [--model anthropic/claude-haiku-4-5] \
        [--keep-only] \
        [--max-workers 8] \
        [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_THRESHOLDS = {"keep": 4.0, "review": 3.0}
_RETRY_DELAYS = [2, 5, 15]

_MCQ_DETECT = re.compile(
    r"\nA\. .+\nB\. .+\nC\. .+\nD\. .+$",
    re.DOTALL,
)
_FORMAT_DETECT = re.compile(r"<think>.*?</think>\n([A-D])\s*$", re.DOTALL)

_MCQ_SYSTEM = """\
You are an expert evaluator of machine learning training data for agricultural AI models.
Score each training example on the given dimensions using a 1–5 Likert scale.
Return ONLY valid JSON — no prose, no markdown fences."""

_CONV_SYSTEM = """\
You are an expert evaluator of conversational AI training data about natural farming.
Score each training example on the given dimensions using a 1–5 Likert scale.
Return ONLY valid JSON — no prose, no markdown fences."""

_MCQ_JUDGE_PROMPT = """\
Evaluate this MCQ training pair for a natural farming AI model.

--- INSTRUCTION ---
{instruction}

--- OUTPUT ---
{output}

Score each dimension 1 (very poor) to 5 (excellent):

1. think_block_quality: Does the <think> block contain a genuine rationale explaining \
WHY the answer is correct, rather than just restating "The correct answer is X"?
   1=just restates label, 3=partial reasoning, 5=clear factual explanation of why X is right

2. format_compliance: Does the output exactly follow <think>...</think>\\n{single_letter}?
   1=wrong format entirely, 3=mostly correct with minor issues, 5=exactly correct

3. answer_matches_rationale: Does the rationale logically support the stated answer letter?
   1=rationale contradicts or is unrelated to stated answer, 3=partially consistent, \
5=rationale clearly and specifically supports that letter

4. factual_accuracy: Is the agricultural content in the rationale factually plausible?
   1=clearly wrong, 3=plausible but vague, 5=specific and factually accurate

5. domain_relevance: Is this question about natural farming, agroecology, soil science, \
permaculture, or sustainable agriculture?
   1=unrelated, 3=tangentially related, 5=core NF domain

Return JSON object only:
{{"think_block_quality": N, "format_compliance": N, "answer_matches_rationale": N, \
"factual_accuracy": N, "domain_relevance": N, "justification": "one sentence"}}"""

_CONV_JUDGE_PROMPT = """\
Evaluate this conversational training pair for a natural farming AI model.

--- INSTRUCTION (user message) ---
{instruction}

--- OUTPUT (assistant response) ---
{output}

Score each dimension 1 (very poor) to 5 (excellent):

1. response_quality: Is the response informative, well-structured, and helpful?
   1=unhelpful or confused, 3=acceptable, 5=clear and thorough

2. format_compliance: Is the response in natural prose (not MCQ format)?
   1=clearly wrong format, 3=minor issues, 5=perfect conversational format

3. factual_consistency: Is the content internally consistent and logically sound?
   1=contradictions or errors, 3=mostly consistent, 5=fully consistent

4. factual_accuracy: Is the natural farming content factually plausible?
   1=clearly wrong, 3=plausible but vague, 5=specific and accurate

5. domain_relevance: Is this about natural farming, agroecology, or sustainable agriculture?
   1=unrelated, 3=tangentially related, 5=core NF domain

Return JSON object only:
{{"response_quality": N, "format_compliance": N, "factual_consistency": N, \
"factual_accuracy": N, "domain_relevance": N, "justification": "one sentence"}}"""


def _get_client(api_key: str, model: str):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL), model


def _load_api_key(override: str | None) -> str:
    if override:
        return override
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError(
        "OPENROUTER_API_KEY not found. Set it in environment or .env"
    )


def _is_mcq_pair(pair: dict) -> bool:
    return bool(_MCQ_DETECT.search(pair.get("instruction", "")))


def _build_prompt(pair: dict) -> tuple[str, str]:
    if _is_mcq_pair(pair):
        return _MCQ_SYSTEM, _MCQ_JUDGE_PROMPT.format(
            instruction=pair["instruction"][:2000],
            output=pair["output"][:800],
        )
    return _CONV_SYSTEM, _CONV_JUDGE_PROMPT.format(
        instruction=pair["instruction"][:1500],
        output=pair["output"][:1500],
    )


def _call_judge(system: str, prompt: str, client, model: str) -> str:
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=256,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == len(_RETRY_DELAYS):
                raise
            log.warning("API error (attempt %d): %s — retrying", attempt + 1, exc)
    raise RuntimeError("All retries exhausted")


def _parse_response(raw: str) -> dict[str, Any]:
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON object in response: {raw[:200]}")
    parsed = json.loads(json_match.group())
    numeric = {k: v for k, v in parsed.items()
               if isinstance(v, (int, float)) and k != "composite_score"}
    if not numeric:
        raise ValueError("No numeric scores in parsed response")
    composite = round(sum(numeric.values()) / len(numeric), 3)
    parsed["composite_score"] = composite
    parsed["dimension_scores"] = {k: float(v) for k, v in numeric.items()}
    return parsed


def _apply_verdict(composite: float) -> str:
    if composite >= DEFAULT_THRESHOLDS["keep"]:
        return "keep"
    if composite >= DEFAULT_THRESHOLDS["review"]:
        return "review"
    return "reject"


def _score_pair(pair: dict, client, model: str) -> dict:
    system, prompt = _build_prompt(pair)
    raw = _call_judge(system, prompt, client, model)
    parsed = _parse_response(raw)
    result = dict(pair)
    result["judge_scores"] = parsed.get("dimension_scores", {})
    result["judge_composite"] = parsed.get("composite_score", 0.0)
    result["judge_justification"] = parsed.get("justification", "")
    result["judge_verdict"] = _apply_verdict(result["judge_composite"])
    result["judge_model"] = model
    return result


def _load_done_instructions(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                instr = rec.get("instruction", "")
                if instr:
                    done.add(instr[:200])
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line in resume file: %s", exc)
    return done


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input JSONL (instruction/output pairs)")
    ap.add_argument("--output", required=True, help="Output JSONL with judge_verdict added")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"OpenRouter model (default: {DEFAULT_MODEL})")
    ap.add_argument("--keep-only", action="store_true",
                    help="Write only keep-verdict pairs to output")
    ap.add_argument("--max-workers", type=int, default=8,
                    help="Parallel API workers (default: 8)")
    ap.add_argument("--openrouter-key", default=None,
                    help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip items already in output file")
    ap.add_argument("--thresholds", default="keep:4.0,review:3.0",
                    help="Verdict thresholds (default: keep:4.0,review:3.0)")
    args = ap.parse_args()

    # Parse custom thresholds
    for part in args.thresholds.split(","):
        k, v = part.strip().split(":")
        DEFAULT_THRESHOLDS[k.strip()] = float(v.strip())

    api_key = _load_api_key(args.openrouter_key)
    client, model = _get_client(api_key, args.model)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as fh:
        pairs = [json.loads(line) for line in fh if line.strip()]
    log.info("Loaded %d pairs from %s", len(pairs), input_path)

    done_keys: set[str] = set()
    if args.resume:
        done_keys = _load_done_instructions(output_path)
        log.info("Resuming: %d items already done", len(done_keys))

    write_lock = threading.Lock()
    verdicts: dict[str, int] = {"keep": 0, "review": 0, "reject": 0}
    stats = {"done": 0, "errors": 0}

    mode = "a" if args.resume else "w"
    out_fh = open(output_path, mode, encoding="utf-8")  # noqa: WPS515

    def process(pair: dict) -> None:
        key = pair.get("instruction", "")[:200]
        if key in done_keys:
            return
        try:
            result = _score_pair(pair, client, model)
            with write_lock:
                verdicts[result["judge_verdict"]] += 1
                stats["done"] += 1
                if not args.keep_only or result["judge_verdict"] == "keep":
                    out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                if stats["done"] % 50 == 0:
                    out_fh.flush()
                    log.info(
                        "Scored %d | keep=%d review=%d reject=%d errors=%d",
                        stats["done"],
                        verdicts["keep"], verdicts["review"],
                        verdicts["reject"], stats["errors"],
                    )
        except Exception as exc:
            log.warning("Error scoring pair: %s", exc)
            with write_lock:
                stats["errors"] += 1

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process, p): i for i, p in enumerate(pairs)}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("Unexpected error on future %d: %s", futures[fut], exc)

    out_fh.flush()
    out_fh.close()

    total = sum(verdicts.values())
    log.info(
        "Done. Scored: %d | keep=%d (%.1f%%) review=%d reject=%d errors=%d → %s",
        total,
        verdicts["keep"], 100 * verdicts["keep"] / max(total, 1),
        verdicts["review"], verdicts["reject"], stats["errors"],
        output_path,
    )
    print(f"\nTotal scored:  {total}")
    print(f"Keep:          {verdicts['keep']} ({100*verdicts['keep']/max(total,1):.1f}%)")
    print(f"Review:        {verdicts['review']}")
    print(f"Reject:        {verdicts['reject']}")
    print(f"Errors:        {stats['errors']}")
    print(f"Output:        {output_path}")
    print(f"Thresholds:    keep≥{DEFAULT_THRESHOLDS['keep']} review≥{DEFAULT_THRESHOLDS['review']}")


if __name__ == "__main__":
    main()
