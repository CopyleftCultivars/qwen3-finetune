"""F5 step 1 (#142): pick the distillation teacher on evidence, not assumption.

Plan B.1 names qwen3:8b primary with opus optional for hard items. The
same-harness ladder run inverted that premise: qwen3-8b scores 0.4545 on the
hard tier against v5clean's 0.5455, so on exactly the items the student needs
help with, the planned teacher is *worse than the student*. Generating a large
teacher set before checking this would bake a bad policy into the dataset.

So this scores candidate teachers on the **hard tier first** (33 items, the tier
that carries the remaining gap) through the same protocol as every other ladder
row -- evaluate.py's SYSTEM_PROMPT / PROMPT_TEMPLATE / extract_answer, via its
own `query_claude_cli`, which runs on Claude Code auth and needs no API key.

Cheap by construction: 33 items per candidate rather than 210.
"""

from __future__ import annotations

import argparse
import os
import json
import math
import sys
import time
from pathlib import Path

BENCH_WS = Path(os.environ.get("BENCHMARK_ROOT", "."))
DEFAULT_BENCH = BENCH_WS / "benchmarks/natural_fertilizers/natural_fertilizers_benchmark_v1_graded.jsonl"


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="claude-opus-4-6,claude-opus-4-7,claude-opus-4-8,claude-opus-5")
    parser.add_argument("--difficulty", default="hard", help="'hard', 'easy' or 'all'")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCH))
    parser.add_argument("--out-prefix", default="results/teacher_select")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(BENCH_WS))
    import evaluate as ev  # noqa: E402

    rows = [json.loads(line) for line in open(args.benchmark, encoding="utf-8")]
    if args.difficulty != "all":
        rows = [r for r in rows if str(r.get("difficulty")) == args.difficulty]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} '{args.difficulty}' items", flush=True)

    Path("results").mkdir(exist_ok=True)
    summaries = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"=== {model} ===", flush=True)
        per_item = []
        out = Path(f"{args.out_prefix}_{model}_{args.difficulty}.jsonl")
        with open(out, "w", encoding="utf-8") as handle:
            for i, row in enumerate(rows, start=1):
                prompt = ev.PROMPT_TEMPLATE.format(
                    question=row["question"], A=row["A"], B=row["B"], C=row["C"], D=row["D"]
                )
                try:
                    raw = ev.query_claude_cli(model, prompt)
                except Exception as exc:  # noqa: BLE001 -- one bad item must not kill the sweep
                    raw = ""
                    print(f"  item {row['question_id']} failed: {type(exc).__name__}", flush=True)
                pred = ev.extract_answer(raw)
                gold = str(row["answer"]).strip().upper()
                rec = {
                    "question_id": row["question_id"],
                    "model": model,
                    "correct_answer": gold,
                    "model_answer": pred,
                    "raw_response": raw[:300],
                    "ambiguous": pred is None,
                    "correct": pred is not None and pred == gold,
                    "difficulty": row.get("difficulty"),
                    "topic": row.get("topic"),
                }
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
                per_item.append(rec)
                if i % 10 == 0:
                    print(f"  {i}/{len(rows)}", flush=True)
                time.sleep(args.delay)

        n = len(per_item)
        c = sum(r["correct"] for r in per_item)
        lo, hi = wilson(c, n)
        s = {
            "model": model, "difficulty": args.difficulty, "n": n, "correct": c,
            "accuracy": round(c / n, 4) if n else 0.0,
            "wilson95": [round(lo, 4), round(hi, 4)],
            "ambiguous": sum(r["ambiguous"] for r in per_item),
        }
        summaries.append(s)
        print(json.dumps(s), flush=True)

    Path(f"{args.out_prefix}_{args.difficulty}_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    print("\n=== teacher candidates ===", flush=True)
    for s in sorted(summaries, key=lambda x: -x["accuracy"]):
        print(f"  {s['model']:<22} {s['accuracy']:.4f}  {s['wilson95']}  amb={s['ambiguous']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
