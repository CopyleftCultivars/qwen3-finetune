#!/usr/bin/env python3
"""Generate free-form candidate answers for the rubric guard (chainlink #66).

Loads a model, picks a deterministic dev-set subset (first N items per type,
in file order), and generates one candidate answer per item using the same
chat-template convention eval_freeform/score_refll.py uses for reference-LL
scoring (``enable_thinking=False``, ``prompt_mode=chat_assistant_reference_v1``)
so generation and scoring stay consistent. Output rows are ready for
eval_freeform/score_rubric.py's ``--input``.

Never reads eval_freeform/test.jsonl (sealed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_DIAGNOSTICS_DIR = Path("../benchmark-eval-workspace/diagnostics")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from eval_freeform.guards import assert_not_sealed_path, validate_dev_ids  # noqa: E402


def check_not_sealed(path: str | Path) -> Path:
    try:
        return assert_not_sealed_path(path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def import_diagnostics_helpers(diagnostics_dir: Path):
    resolved = diagnostics_dir.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"diagnostics directory not found: {resolved}")
    sys.path.insert(0, str(resolved))
    from utils import load_model_and_tokenizer  # type: ignore[import-not-found]

    return load_model_and_tokenizer


def select_subset(rows: list[dict[str, Any]], items_per_type: int) -> list[dict[str, Any]]:
    """First N rows per type, in file order — deterministic, no randomness."""
    counts: Counter = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        row_type = row.get("type", "unknown")
        if counts[row_type] < items_per_type:
            selected.append(row)
            counts[row_type] += 1
    return selected


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate free-form candidate answers for the rubric guard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dev-input", default="eval_freeform/dev.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--items-per-type", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import torch

    torch.manual_seed(0)

    dev_path = check_not_sealed(args.dev_input)
    load_model_and_tokenizer = import_diagnostics_helpers(DEFAULT_DIAGNOSTICS_DIR)

    rows = [json.loads(line) for line in dev_path.open(encoding="utf-8") if line.strip()]
    validate_dev_ids((row.get("id", "") for row in rows))
    subset = select_subset(rows, args.items_per_type)
    print(f"selected {len(subset)} items from {dev_path} ({args.items_per_type}/type)")

    model, tokenizer = load_model_and_tokenizer(
        args.model, full_precision=True, device=args.device
    )
    model.eval()

    results = []
    with torch.no_grad():
        for index, row in enumerate(subset):
            prompt = build_prompt(tokenizer, row["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append(
                {
                    "id": row["id"],
                    "type": row.get("type"),
                    "source_doc": row.get("source_doc"),
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "rubric": row["rubric"],
                    "candidate_answer": answer,
                    "generation_model": args.model,
                    "prompt_mode": "chat_assistant_reference_v1",
                    "enable_thinking": False,
                }
            )
            print(f"  [{index + 1}/{len(subset)}] {row['id']}: {len(answer)} chars")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"wrote {output_path}")
    print(f"n_items={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
