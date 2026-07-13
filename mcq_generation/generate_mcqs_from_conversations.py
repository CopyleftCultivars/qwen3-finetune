#!/usr/bin/env python3
"""Phase 3b: generate MCQs from the trusted gemma4 conversation training data
(data/sources/gemma4/hf_dataset/train.jsonl) rather than the QMD corpus.

Zero benchmark-corpus overlap risk by construction — this source isn't derived
from the same documents benchmark_combined.jsonl was built from. Targets
v2-clean's actual diagnosed failure mode (MCQ-format unfamiliarity/letter bias)
by reinforcing facts the model already has conversationally, in MCQ form.

Reuses generate_mcqs_rag.py's API-calling, validation, and quality machinery —
only the retrieval layer differs (conversation excerpts instead of QMD chunks,
since there's no document corpus to search here).

Usage:
    python generate_mcqs_from_conversations.py \\
        --conversations ../data/sources/gemma4/hf_dataset/train.jsonl \\
        --output-dir output/conversations_batch \\
        --n-conversations 400 \\
        --batch-size 20 \\
        --questions-per-batch 12
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from generate_mcqs_rag import (
    DEFAULT_MODEL, FALLBACK_MODEL, OPENROUTER_BASE_URL,
    call_with_fallback, parse_response,
)

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert in natural farming, agroecology, soil science, permaculture, "
    "Korean Natural Farming (KNF), and sustainable agriculture. Your task is to generate "
    "high-quality multiple-choice questions (MCQs) from the provided conversation excerpts "
    "for use in a professional benchmark that evaluates language models on natural farming "
    "knowledge."
)

_USER_PROMPT_TEMPLATE = """\
The following excerpts are real answers from a natural farming assistant, responding to \
user questions about fertilizers, composting, soil health, and related topics.

<excerpts>
{excerpts}
</excerpts>

For each MCQ you generate, strictly follow these guidelines:

1. QUESTION REQUIREMENTS:
   - Questions must be based on factual information present in the excerpts above.
   - Generate {n} MCQs covering different aspects of the excerpts — do not generate multiple
     questions about the same fact.
   - Questions should range from factual recall to applied reasoning.
   - Avoid yes/no questions, trivially obvious questions, or questions answerable by common
     sense alone.
   - Questions must be SELF-CONTAINED: a reader with domain knowledge must be able to answer
     without access to the source excerpts.
   - Do NOT use fill-in-the-blank format. Every question must be a complete interrogative
     sentence.
   - Do NOT reference the source material anywhere in the question. Never say "according to
     the excerpt/assistant/conversation", "as mentioned above", "the response states", or any
     variant. A test-taker must be able to answer from domain knowledge alone.

2. ANSWER OPTIONS:
   - Exactly 4 options (A, B, C, D), one correct, three plausible distractors.
   - Distractors should test common misconceptions, not be obviously wrong.

3. OUTPUT FORMAT:
   - Respond with ONLY valid JSON matching the schema below. No markdown, no commentary.

{{
  "questions": [
    {{
      "question": "<question text>",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answer": "<A, B, C, or D>",
      "topic": "natural_farming_conversations",
      "source_passage": "<verbatim sentence from the excerpts supporting the answer>"
    }}
  ]
}}
"""


def load_gpt_excerpts(path: Path, min_chars: int = 400) -> list[str]:
    excerpts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            for turn in rec.get("conversations", []):
                if turn["from"] == "gpt" and len(turn["value"]) >= min_chars:
                    excerpts.append(turn["value"])
    return excerpts


def process_batch(
    batch: list[str],
    batch_idx: int,
    output_dir: Path,
    model: str,
    fallback_model: str,
    client: OpenAI,
    n_questions: int,
) -> dict:
    excerpts_text = "\n\n---\n\n".join(f"[Excerpt {i+1}]\n{ex}" for i, ex in enumerate(batch))
    user_prompt = _USER_PROMPT_TEMPLATE.format(excerpts=excerpts_text, n=n_questions)
    label = f"conv_batch_{batch_idx:03d}"

    try:
        raw, model_used = call_with_fallback(user_prompt, _SYSTEM, model, fallback_model, client, log)
    except RuntimeError as e:
        log.error("  Both models failed for %s: %s", label, e)
        (output_dir / f"{label}.error.txt").write_text(str(e), encoding="utf-8")
        return {"batch": label, "questions_generated": 0, "error": str(e)}

    try:
        data = parse_response(raw, f"conversations:{label}")
    except (json.JSONDecodeError, ValueError) as e:
        log.error("  Parse error for %s: %s", label, e)
        (output_dir / f"{label}.error.txt").write_text(
            f"PARSE ERROR: {e}\n\nRAW RESPONSE:\n{raw}", encoding="utf-8"
        )
        return {"batch": label, "questions_generated": 0, "error": str(e)}

    out = {
        "source_document": f"conversations:{label}",
        "category": "Conversation-Derived (Phase 3b)",
        "topic": "natural_farming_conversations",
        "generation_timestamp": datetime.now().isoformat(),
        "questions": data["questions"],
    }
    (output_dir / f"{label}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("  %s -> %d questions (%s)", label, len(data["questions"]), model_used)
    return {"batch": label, "questions_generated": len(data["questions"]), "error": None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conversations", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--n-conversations", type=int, default=400,
                     help="How many gpt-turn excerpts to sample (default: 400)")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--questions-per-batch", type=int, default=12)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fallback-model", default=FALLBACK_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set")
        sys.exit(1)
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    excerpts = load_gpt_excerpts(args.conversations)
    log.info("Loaded %d candidate excerpts (>=400 chars) from %s", len(excerpts), args.conversations)

    random.seed(args.seed)
    sample = random.sample(excerpts, min(args.n_conversations, len(excerpts)))
    log.info("Sampled %d excerpts", len(sample))

    batches = [sample[i:i + args.batch_size] for i in range(0, len(sample), args.batch_size)]
    log.info("%d batches of up to %d excerpts each", len(batches), args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total_questions": 0, "errors": 0}
    for i, batch in enumerate(batches):
        result = process_batch(
            batch, i, args.output_dir, args.model, args.fallback_model, client, args.questions_per_batch
        )
        if result["error"]:
            stats["errors"] += 1
        else:
            stats["total_questions"] += result["questions_generated"]
        if args.delay and i < len(batches) - 1:
            time.sleep(args.delay)

    log.info("=== Summary === batches=%d questions=%d errors=%d",
              len(batches), stats["total_questions"], stats["errors"])
    print(f"\nBatches processed: {len(batches)}")
    print(f"Total questions generated: {stats['total_questions']}")
    print(f"Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
