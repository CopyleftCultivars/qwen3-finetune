"""F5 (#142) follow-up: regenerate teacher outputs whose think block was truncated.

The first pass ran at max_new_tokens=1024; 126 of 1117 items exceeded that cap,
cutting off the `<think>...</think>` block before its closing tag (and, in two
cases, before any answer letter). Those rows' `teacher_answer` values were
extracted from mid-reasoning text and are therefore unreliable. This script
re-runs exactly those items at max_new_tokens=2048, updates the rows in place,
and rewrites the output + manifest so the F5 artifact is clean for F6 (#143).

Reuses the generator's loader/protocol verbatim (Unsloth 4-bit, SYSTEM_PROMPT +
extract_answer from evaluate.py, greedy temperature 0, seed 42, think ON).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_WS = Path("[LOCAL_PATH]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-out", default="results/teacher_data_qwen3-8b.jsonl")
    parser.add_argument("--manifest", default="results/teacher_data_qwen3-8b_manifest.json")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--quant", default="4bit", choices=["4bit", "8bit"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(BENCH_WS))
    sys.path.insert(0, str(REPO_ROOT))
    import evaluate as ev  # noqa: E402
    from scripts.generate_teacher_data import (  # noqa: E402
        GENERATION_PROMPT_TEMPLATE,
        extract_rationale,
        generate_chat,
        load_teacher,
        norm_key,
    )

    rows = [json.loads(line) for line in open(args.in_out, encoding="utf-8") if line.strip()]
    truncated = [r for r in rows if "</think>" not in r["teacher_raw"]]
    if args.limit:
        truncated = truncated[: args.limit]
    print(f"{len(rows)} rows loaded; {len(truncated)} truncated -> regenerate", flush=True)

    model, tokenizer = load_teacher(args.model, args.quant)
    # question_id is NOT unique across pool files (permaculture.jsonl and
    # qmd_mcq_generated.clean.jsonl both use PC-TRAIN-XXXX ids for different
    # questions); key by normalized question+options instead.
    by_key = {norm_key(r): r for r in rows}
    latencies: list[float] = []
    for i, r in enumerate(truncated, start=1):
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
        r["teacher_raw"] = raw
        r["teacher_answer"] = ev.extract_answer(raw)
        r["teacher_rationale"] = extract_rationale(raw)
        r["teacher_matches_gold"] = r["teacher_answer"] is not None and r["teacher_answer"] == r["gold"]
        r["latency_ms"] = round(elapsed, 1)
        by_key[norm_key(r)] = r
        if i % 10 == 0 or i == 1:
            print(f"  {i}/{len(truncated)} ({r['question_id']} -> {r['teacher_answer']}, "
                  f"gold {r['gold']}, rat {'y' if r['teacher_rationale'] else 'n'}, {elapsed:.0f}ms)",
                  flush=True)

    new_rows = sorted(by_key.values(), key=lambda r: (r.get("source_file", ""), str(r["question_id"])))
    with open(args.in_out, "w", encoding="utf-8") as handle:
        for r in new_rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    still_truncated = [r for r in new_rows if "</think>" not in r["teacher_raw"]]
    manifest = json.load(open(args.manifest, encoding="utf-8")) if Path(args.manifest).exists() else {}
    manifest["regenerate_truncated"] = {
        "n_regenerated": len(truncated),
        "max_new_tokens": args.max_new_tokens,
        "still_truncated_after": len(still_truncated),
    }
    manifest["n_generated"] = len(new_rows)
    manifest["n_matches_gold"] = sum(r["teacher_matches_gold"] for r in new_rows)
    manifest["match_rate"] = round(manifest["n_matches_gold"] / len(new_rows), 4)
    manifest["n_rationale_captured"] = sum(1 for r in new_rows if r["teacher_rationale"])
    manifest["n_ambiguous"] = sum(1 for r in new_rows if r["teacher_answer"] is None)
    manifest["mean_latency_ms"] = round(statistics.mean(r["latency_ms"] for r in new_rows), 1)
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps({
        "n_regenerated": len(truncated),
        "still_truncated_after": len(still_truncated),
        "match_rate": manifest["match_rate"],
        "n_rationale_captured": manifest["n_rationale_captured"],
        "n_ambiguous": manifest["n_ambiguous"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
