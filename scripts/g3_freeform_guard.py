#!/usr/bin/env python3
"""#239 G3: held-out free-form factuality / instruction-following guard.

Generates answers for the FIRST 30 rows of eval_freeform/dev.jsonl (fixed
subset, predeclared) with a given HF model and reports mechanical quality
signals: empty rate, degenerate-repetition rate, mean char length, and
truncation rate. No judge model, no OpenRouter key, no sealed test use:
eval_freeform/test.jsonl is never read here.

Compare the stock and adapted rows of the output JSON to apply the
predeclared free-form gate (results/QWEN3_5_G3_GATES.json).

Run with the isolated transformers-5.5 venv:
  V=localllm-qwen3.5/retry-evidence/q1-isolated/venv/bin/python
  $V scripts/g3_freeform_guard.py --model <hf-path-or-id> --label <name>

Two falsifiable signals, both local (no judge API, reproducible offline):
  1. mechanical: non-degenerate rate (not empty, no 4-gram repetition
     collapse) on the fixed first-30 dev rows;
  2. reference NLL: mean teacher-forced per-token NLL of the dev
     reference_answer given the question -- a collapsed or factually
     degraded model scores worse. A rubric-LLM judge (score_rubric.py,
     OpenRouter) is deliberately NOT used: it is non-deterministic across
     backends and would make the gate unreproducible.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEV = REPO / "eval_freeform/dev.jsonl"
N_ITEMS = 30
MAX_NEW_TOKENS = 128


def degenerate(text: str) -> bool:
    """Collapsing repetition: one 4-gram covers >40% of a long-enough text."""
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    if len(words) < 20:
        return False
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    top = max(grams.count(g) for g in set(grams))
    return top / len(grams) > 0.40


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="results/g3_freeform")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in open(DEV, encoding="utf-8")
            if line.strip()][:N_ITEMS]
    assert len(rows) == N_ITEMS, f"expected {N_ITEMS} dev rows, got {len(rows)}"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": 0}, dtype=torch.bfloat16,
        trust_remote_code=True)
    model.eval()
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    per_item = []
    ref_nlls = []
    out_path = Path(f"{args.out_prefix}_{args.label}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for i, row in enumerate(rows, start=1):
            messages = [{"role": "user", "content": row["question"]}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            # Reference NLL: teacher-forced NLL of reference_answer tokens
            # conditioned on the templated question.
            pref_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ref_ids = tokenizer(str(row["reference_answer"]),
                                add_special_tokens=False)["input_ids"]
            full = pref_ids + ref_ids
            labels = [-100] * len(pref_ids) + ref_ids
            with torch.no_grad():
                inp = torch.tensor([full], device=model.device)
                lab = torch.tensor([labels], device=model.device)
                nll = float(torch.nn.functional.cross_entropy(
                    model(input_ids=inp, attention_mask=torch.ones_like(inp))
                    .logits[0].float(),
                    lab[0], ignore_index=-100, reduction="mean"))
            ref_nlls.append(nll)
            torch.manual_seed(args.seed)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            started = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, pad_token_id=pad)
            elapsed = (time.perf_counter() - started) * 1000.0
            new = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new, skip_special_tokens=True)
            rec = {
                "id": row.get("id"),
                "text": text[:1000],
                "chars": len(text),
                "empty": len(text.strip()) == 0,
                "degenerate": degenerate(text),
                "truncated": int(new.shape[0]) >= MAX_NEW_TOKENS,
                "ref_nll": round(nll, 4),
                "latency_ms": round(elapsed, 1),
            }
            handle.write(json.dumps(rec) + "\n")
            per_item.append(rec)
            if i % 10 == 0:
                print(f"  [{args.label}] {i}/{len(rows)}", flush=True)

    n = len(per_item)
    summary = {
        "label": args.label,
        "model": args.model,
        "n": n,
        "empty": sum(r["empty"] for r in per_item),
        "degenerate": sum(r["degenerate"] for r in per_item),
        "non_degenerate_rate": round(
            sum(not r["empty"] and not r["degenerate"] for r in per_item) / n, 4),
        "mean_chars": round(sum(r["chars"] for r in per_item) / n, 1),
        "truncated": sum(r["truncated"] for r in per_item),
        "mean_ref_nll": round(sum(ref_nlls) / len(ref_nlls), 4),
        "mean_latency_ms": round(
            sum(r["latency_ms"] for r in per_item) / n, 1),
    }
    Path(f"{args.out_prefix}_{args.label}_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
