"""F5 (#142) follow-up: restore the 22 teacher outputs lost to an id-collision bug.

`regenerate_truncated_teacher.py` keyed rows by `question_id`, which is not
unique across the pool: `permaculture.jsonl` and `qmd_mcq_generated.clean.jsonl`
both assign `PC-TRAIN-XXXX` ids to *different* questions. Rewriting the file
through a question_id dict collapsed 1117 rows to 1095. This script finds the
pool items missing from the output (by normalized question+options, the same
key the pool loader dedups on), generates teacher outputs for them, appends,
and refreshes the manifest. It is idempotent.
"""

from __future__ import annotations

import argparse
import os
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_WS = Path(os.environ.get("BENCHMARK_ROOT", "."))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/teacher_data_qwen3-8b.jsonl")
    parser.add_argument("--manifest", default="results/teacher_data_qwen3-8b_manifest.json")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--quant", default="4bit", choices=["4bit", "8bit"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(BENCH_WS))
    sys.path.insert(0, str(REPO_ROOT))
    import evaluate as ev  # noqa: E402
    from scripts.generate_teacher_data import (  # noqa: E402
        GENERATION_PROMPT_TEMPLATE,
        extract_rationale,
        generate_chat,
        load_pool,
        load_teacher,
        norm_key,
    )

    pool, _stats = load_pool()
    out = [json.loads(line) for line in open(args.out, encoding="utf-8") if line.strip()]
    have = {norm_key(r) for r in out}
    missing = [r for r in pool if norm_key(r) not in have]
    print(f"pool {len(pool)}; output {len(out)}; missing {len(missing)}", flush=True)
    if not missing:
        print("nothing to do", flush=True)
        return 0

    model, tokenizer = load_teacher(args.model, args.quant)
    latencies: list[float] = []
    for i, r in enumerate(missing, start=1):
        user = GENERATION_PROMPT_TEMPLATE.format(
            question=r["question"], A=r["A"], B=r["B"], C=r["C"], D=r["D"]
        )
        started = time.perf_counter()
        raw = generate_chat(
            model, tokenizer,
            [{"role": "system", "content": ev.SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            max_new_tokens=args.max_new_tokens, enable_thinking=True, seed=args.seed,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed)
        answer = ev.extract_answer(raw)
        gold = str(r["answer"]).strip().upper()
        rec = {
            "question_id": str(r["question_id"]),
            "question": r["question"],
            "A": r["A"], "B": r["B"], "C": r["C"], "D": r["D"],
            "gold": gold,
            "topic": r.get("topic"),
            "category": r.get("category"),
            "source_file": r.get("_source_file"),
            "teacher_raw": raw,
            "teacher_answer": answer,
            "teacher_rationale": extract_rationale(raw),
            "teacher_matches_gold": answer is not None and answer == gold,
            "latency_ms": round(elapsed, 1),
        }
        out.append(rec)
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 5 == 0 or i == 1:
            print(f"  {i}/{len(missing)} ({rec['question_id']} -> {answer}, gold {gold}, "
                  f"{elapsed:.0f}ms)", flush=True)

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    manifest["n_generated"] = len(out)
    manifest["n_matches_gold"] = sum(r["teacher_matches_gold"] for r in out)
    manifest["match_rate"] = round(manifest["n_matches_gold"] / len(out), 4)
    manifest["n_rationale_captured"] = sum(1 for r in out if r["teacher_rationale"])
    manifest["n_ambiguous"] = sum(1 for r in out if r["teacher_answer"] is None)
    manifest["mean_latency_ms"] = round(statistics.mean(r["latency_ms"] for r in out), 1)
    manifest["id_collision_restore"] = {"n_restored": len(missing), "max_new_tokens": args.max_new_tokens}
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps({"n_restored": len(missing), "total": len(out),
                      "match_rate": manifest["match_rate"],
                      "n_rationale_captured": manifest["n_rationale_captured"],
                      "n_ambiguous": manifest["n_ambiguous"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
