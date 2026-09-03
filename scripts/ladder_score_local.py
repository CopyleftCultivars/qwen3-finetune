"""F3 (#140): score a local HF checkpoint on the 210-item ladder.

`benchmark-workspace/evaluate.py` can only drive the claude CLI, an Ollama HTTP
server, or a pre-existing JSONL field -- there is no local-HF path, and the
champion is a merged HF checkpoint. Rather than reimplement the protocol, this
imports the frozen pieces from evaluate.py itself:

    SYSTEM_PROMPT, PROMPT_TEMPLATE, extract_answer

so the prompt wording and the A/B/C/D parser (including its `</think>`-first
rule and the ambiguous=None outcome) are byte-identical to what produced the
reference rows. Only the transport changes.

Plan A.0.3 / A.3: greedy, temperature = 0, fixed seed. Ambiguous items are
scored incorrect, matching evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

BENCH_WS = Path("[LOCAL_PATH]")
DEFAULT_BENCH = BENCH_WS / "benchmarks/natural_fertilizers/natural_fertilizers_benchmark_v1_graded.jsonl"
REPO_ROOT = Path(__file__).resolve().parent.parent


def import_protocol():
    """Pull the prompt and parser out of evaluate.py -- do not duplicate them."""
    sys.path.insert(0, str(BENCH_WS))
    import evaluate as ev  # noqa: E402

    return ev.SYSTEM_PROMPT, ev.PROMPT_TEMPLATE, ev.extract_answer


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval. Plan A.4 wants this rather than a normal
    approximation, which misbehaves near 0 and 1 at this sample size."""
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _load_4bit_host(model_path: str, device: str, seed: int):
    """ModelHost equivalent for a model that will not fit at fp16.

    ModelHost delegates loading to the diagnostics helper, which has no
    quantization path, so this builds the same duck-typed surface directly.
    Only generate_chat is exercised by this script.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    sys.path.insert(0, str(REPO_ROOT))
    from rlm_slice.guards import set_determinism
    from rlm_slice.model_host import DecodeStats

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16
    )
    model.eval()
    set_determinism(seed)

    class _Quantized:
        """Minimal ModelHost-compatible shim."""

        def __init__(self):
            self.model, self.tokenizer = model, tokenizer
            self.model_path, self.device, self.seed = model_path, device, seed
            self.stats = DecodeStats()

        def generate_chat(self, messages, max_new_tokens=256, enable_thinking=True):
            torch.manual_seed(self.seed)
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            new = out[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new, skip_special_tokens=True), int(new.shape[0])

    return _Quantized()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF path or hub id")
    parser.add_argument("--label", required=True, help="Row label for the report")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="The prompt demands a bare letter; 8 is ample and keeps the sweep fast")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Qwen3 think mode. Off by default: the prompt says 'Do not explain'")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-4bit", action="store_true",
                        help="NF4 quantized load, for models too large for 12GB at fp16 "
                             "(Qwen3-8B fp16 needs ~16GB). Matches how Ollama served qwen3:8b, "
                             "which is Q4_K_M by default, so the reference row is also 4-bit.")
    parser.add_argument("--out-prefix", default="results/ladder210")
    args = parser.parse_args(argv)

    system_prompt, prompt_template, extract_answer = import_protocol()
    sys.path.insert(0, str(REPO_ROOT))
    from rlm_slice.model_host import ModelHost  # noqa: E402

    rows = [json.loads(l) for l in open(args.benchmark, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} ladder items from {Path(args.benchmark).name}", flush=True)

    if args.load_4bit:
        host = _load_4bit_host(args.model, args.device, args.seed)
    else:
        host = ModelHost(model_path=args.model, device=args.device, seed=args.seed)
    per_item: list[dict[str, Any]] = []
    out_path = Path(f"{args.out_prefix}_{args.label}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as handle:
        for i, row in enumerate(rows, start=1):
            user = prompt_template.format(
                question=row["question"], A=row["A"], B=row["B"], C=row["C"], D=row["D"]
            )
            started = time.perf_counter()
            raw, _n = host.generate_chat(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user}],
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            predicted = extract_answer(raw)
            gold = str(row["answer"]).strip().upper()
            rec = {
                "question_id": row["question_id"],
                "correct_answer": gold,
                "model_answer": predicted,
                "raw_response": raw[:400],
                # evaluate.py semantics: an unparseable answer is ambiguous AND wrong.
                "ambiguous": predicted is None,
                "correct": predicted is not None and predicted == gold,
                "difficulty": row.get("difficulty"),
                "topic": row.get("topic"),
                "latency_ms": round(elapsed, 1),
            }
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            per_item.append(rec)
            if i % 25 == 0:
                print(f"  [{args.label}] {i}/{len(rows)}", flush=True)

    n = len(per_item)
    correct = sum(r["correct"] for r in per_item)
    amb = sum(r["ambiguous"] for r in per_item)
    lo, hi = wilson(correct, n)
    summary = {
        "label": args.label,
        "model": args.model,
        "benchmark": args.benchmark,
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "ambiguous": amb,
        "ambiguous_rate": round(amb / n, 4) if n else 0.0,
        "enable_thinking": args.enable_thinking,
        "temperature": 0.0,
        "seed": args.seed,
        "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in per_item), 1) if n else 0.0,
        "by_difficulty": {},
    }
    for diff in sorted({str(r["difficulty"]) for r in per_item}):
        sub = [r for r in per_item if str(r["difficulty"]) == diff]
        summary["by_difficulty"][diff] = {
            "n": len(sub),
            "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 4),
        }
    Path(f"{args.out_prefix}_{args.label}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
