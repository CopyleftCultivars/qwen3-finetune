"""F5 (#142): generate qwen3:8b teacher data over the training-domain MCQ pool.

Plan B.1/B.2. Primary teacher qwen3:8b, scored locally at 4-bit through the
same harness as `ladder_score_local.py` (0.8571 on the 210-item ladder, and --
critically -- *worse than the student on hard items*). The corrected targeting
(see #142 comment and `results/LADDER_210Q_RESULTS.md`) is: generate over the
full pool, capture each teacher final answer AND its think-channel rationale,
and let F6 (#143) do the easy/mid weighting and the gold filter on hard items.
This script only produces raw teacher outputs plus a config manifest; it does
not build the training set.

Loading is done through Unsloth's `FastLanguageModel` (the repo's established
inference/training path, `train.py`), not raw bitsandbytes: Unsloth maps
`Qwen/Qwen3-8B` to its pre-quantized `unsloth/qwen3-8b-unsloth-bnb-4bit`
checkpoint, which loads in ~7.5 GiB and fits the 12 GiB card. The quantization
level is a named knob (--quant 4bit|8bit) so the exact teacher level is
recorded in the manifest rather than assumed.

Pool = the QMD-derived MCQ source (`data/sources/qmd_mcq_generated.clean.jsonl`,
the question-level-audited flat QMD pool) plus `mcq_generation/output/*.jsonl`
(the per-category generated MCQs). Deduplicated by normalized question+options,
and every item is checked against the 210 eval questions (the benchmark file)
so nothing that could leak into evaluation is generated.

Protocol: same `SYSTEM_PROMPT` and `extract_answer` parser as the ladder
(imported from `benchmark-workspace/evaluate.py`, never duplicated), greedy
temperature 0, fixed seed, think ON. The final letter is extracted with the
*same* parser the ladder uses, so `teacher_answer` is directly comparable to
the ladder rows; the `<think>...</think>` block is kept as the rationale for
Variant 2 (rationale-augmented SFT).

Run under the `qwen3-train` env (Unsloth)::
    ~/anaconda3/envs/qwen3-train/bin/python scripts/generate_teacher_data.py \
        --quant 4bit --limit 5 --out-prefix results/teacher_data_smoke
"""

from __future__ import annotations

import argparse
import os
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_WS = Path(os.environ.get("BENCHMARK_ROOT", "."))
DEFAULT_BENCH = BENCH_WS / "benchmarks/natural_fertilizers/natural_fertilizers_benchmark_v1_graded.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-8B"

POOL_FILES = [
    *sorted((REPO_ROOT / "mcq_generation" / "output").glob("*.jsonl")),
    REPO_ROOT / "data" / "sources" / "qmd_mcq_generated.clean.jsonl",
]

GENERATION_PROMPT_TEMPLATE = (
    "Answer the following multiple-choice question. Think through your reasoning "
    "step by step, then give only the single letter (A, B, C, or D) of the correct answer.\n\n"
    "Question: {question}\n"
    "A. {A}\n"
    "B. {B}\n"
    "C. {C}\n"
    "D. {D}\n\n"
    "Answer:"
)


def norm_key(rec: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Normalized (question text, sorted option texts) key for dedup."""
    q = re.sub(r"\s+", " ", str(rec.get("question", "")).strip().lower())
    opts = tuple(sorted(re.sub(r"\s+", " ", str(rec.get(k, "")).strip().lower()) for k in "ABCD"))
    return q, opts


def load_eval_keys(path: Path) -> set[tuple[str, tuple[str, ...]]]:
    return {norm_key(json.loads(line)) for line in open(path, encoding="utf-8") if line.strip()}


def load_pool() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the training-domain pool, deduplicated and eval-excluded."""
    eval_keys = load_eval_keys(DEFAULT_BENCH)
    seen: dict[tuple, dict] = {}
    stats = {"files": 0, "raw": 0, "dup": 0, "eval_excluded": 0, "kept": 0, "non_mcq": 0}
    for path in POOL_FILES:
        if not path.exists():
            continue
        stats["files"] += 1
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            stats["raw"] += 1
            if not all(k in rec for k in ("A", "B", "C", "D")):
                stats["non_mcq"] += 1
                continue
            key = norm_key(rec)
            if key in seen:
                stats["dup"] += 1
                continue
            if key in eval_keys:
                stats["eval_excluded"] += 1
                continue
            out = dict(rec)
            out["_source_file"] = path.name
            seen[key] = out
    stats["kept"] = len(seen)
    order = sorted(seen.values(), key=lambda r: (r["_source_file"], str(r.get("question_id", ""))))
    return order, stats


def extract_rationale(raw: str) -> str:
    """Pull the Qwen3 think-channel content out of a raw response."""
    if not raw:
        return ""
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    return m.group(1).strip() if m else ""


def load_teacher(model_name: str, quant: str, max_seq_length: int = 2048):
    """Load the teacher through Unsloth's quantization path.

    Returns (model, tokenizer) ready for greedy inference (for_inference).
    """
    from unsloth import FastLanguageModel  # noqa: E402 -- imported here so the script's
    # non-generation paths (pool/stats) run without the heavy Unsloth import.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=(quant == "4bit"),
        load_in_8bit=(quant == "8bit"),
        dtype=None,  # auto: bf16 if supported
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_chat(model, tokenizer, messages, max_new_tokens: int, enable_thinking: bool, seed: int) -> str:
    import torch  # noqa: E402

    torch.manual_seed(seed)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF hub id or path")
    parser.add_argument("--quant", default="4bit", choices=["4bit", "8bit"])
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Enough for the think block plus the letter")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="skip question_ids already present in the output file")
    parser.add_argument("--out-prefix", default="results/teacher_data_qwen3-8b")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(BENCH_WS))
    import evaluate as ev  # noqa: E402

    system_prompt = ev.SYSTEM_PROMPT
    extract_answer = ev.extract_answer

    records, stats = load_pool()
    if args.limit:
        records = records[: args.limit]
    print(f"pool: {stats}", flush=True)
    print(f"generating over {len(records)} items (quant={args.quant}, limit={args.limit})", flush=True)

    out_path = Path(f"{args.out_prefix}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.resume and out_path.exists():
        done = {json.loads(line)["question_id"] for line in open(out_path, encoding="utf-8") if line.strip()}
        print(f"resume: {len(done)} already done", flush=True)

    model, tokenizer = load_teacher(args.model, args.quant)
    per_item: list[dict[str, Any]] = []
    mode = "a" if args.resume else "w"
    with open(out_path, mode, encoding="utf-8") as handle:
        for i, row in enumerate(records, start=1):
            qid = str(row["question_id"])
            if qid in done:
                continue
            user = GENERATION_PROMPT_TEMPLATE.format(
                question=row["question"], A=row["A"], B=row["B"], C=row["C"], D=row["D"]
            )
            started = time.perf_counter()
            raw = generate_chat(
                model, tokenizer,
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user}],
                max_new_tokens=args.max_new_tokens, enable_thinking=True, seed=args.seed,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            answer = extract_answer(raw)
            gold = str(row["answer"]).strip().upper()
            rec = {
                "question_id": qid,
                "question": row["question"],
                "A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"],
                "gold": gold,
                "topic": row.get("topic"),
                "category": row.get("category"),
                "source_file": row.get("_source_file"),
                "teacher_raw": raw,
                "teacher_answer": answer,
                "teacher_rationale": extract_rationale(raw),
                "teacher_matches_gold": answer is not None and answer == gold,
                "latency_ms": round(elapsed, 1),
            }
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            per_item.append(rec)
            if i % 10 == 0 or i == 1:
                print(f"  {i}/{len(records)} (last {qid} -> {answer}, gold {gold}, "
                      f"{elapsed:.0f}ms)", flush=True)

    n = len(per_item)
    manifest = {
        "teacher": "qwen3:8b",
        "model": args.model,
        "loader": "unsloth FastLanguageModel (train.py path)",
        "quantization": "bnb-4bit (unsloth pre-quantized checkpoint)" if args.quant == "4bit"
                       else "8bit (unsloth)",
        "temperature": 0.0,
        "seed": args.seed,
        "enable_thinking": True,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt": system_prompt,
        "prompt_template": GENERATION_PROMPT_TEMPLATE,
        "answer_parser": "benchmark-workspace/evaluate.py::extract_answer (imported, not duplicated)",
        "benchmark_excluded": str(args.benchmark),
        "pool_files": [str(p) for p in POOL_FILES],
        "pool_stats": stats,
        "n_generated": n,
        "n_matches_gold": sum(r["teacher_matches_gold"] for r in per_item),
        "match_rate": round(sum(r["teacher_matches_gold"] for r in per_item) / n, 4) if n else 0.0,
        "n_rationale_captured": sum(1 for r in per_item if r["teacher_rationale"]),
        "n_ambiguous": sum(1 for r in per_item if r["teacher_answer"] is None),
        "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in per_item), 1) if n else 0.0,
    }
    Path(f"{args.out_prefix}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("quantization", "n_generated", "n_matches_gold", "match_rate",
                       "n_rationale_captured", "n_ambiguous", "mean_latency_ms")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
