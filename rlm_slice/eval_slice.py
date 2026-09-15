"""Arm sweep over the MCQ track (L6 / #129, part of I12 / #117).

LOCAL_RLM_SLICE_PLAN.md sec.3.4.1/3.4.2/3.4.4: run an arm over the scorable MCQ
rows in data/val_v3.jsonl, score with the frozen trace objective so numbers sit
directly beside results/mcq_trace_ll_v5_clean.json, and record per-item results,
latency and decode calls.

Scoring is not reimplemented here. Every arm ends at
`score_mcq_trace_ll.score_with_trace` via `ModelHost`, and this module only
sequences items, writes rows and aggregates. The one number this file computes
itself is `delta_trace` -- the sec.3 ablation -- and it does so with the same
empty-trace Pass-2 call `score_mcq_trace_ll.main` uses, so the comparison to the
v5clean baseline is like-for-like.

Usage:
  python -m rlm_slice.eval_slice --arms A,R --limit 113 --out-prefix results/slice
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

from rlm_slice.controller import ARMS, run_arm
from rlm_slice.guards import DEFAULT_SEED, check_not_sealed
from rlm_slice.model_host import DEFAULT_MODEL, ModelHost

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_records(dataset: str, limit: int | None) -> list[dict[str, Any]]:
    check_not_sealed(dataset)  # sec.2.2, before any file IO
    sys.path.insert(0, str(REPO_ROOT / "autoresearch"))
    from score_mcq_ll import normalize_records  # noqa: E402

    rows = [json.loads(line) for line in open(dataset, encoding="utf-8")]
    records, _skipped = normalize_records(rows)
    return records[:limit] if limit else records


def summarize(arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"arm": arm, "n_items": 0}

    def mean(field: str) -> float:
        values = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
        return round(statistics.mean(values), 6) if values else 0.0

    deltas = [r["delta_trace"] for r in rows if r.get("delta_trace") is not None]
    note_counts: dict[str, int] = {}
    for row in rows:
        for note in row.get("notes", []):
            key = note.split(":")[0]
            note_counts[key] = note_counts.get(key, 0) + 1

    return {
        "arm": arm,
        "n_items": len(rows),
        "accuracy": round(sum(bool(r["correct"]) for r in rows) / len(rows), 6),
        "mean_trace_option_set_gold_logprob": mean("option_set_gold_logprob"),
        "mean_delta_trace": round(statistics.mean(deltas), 6) if deltas else 0.0,
        "delta_trace_positive_fraction": (
            round(sum(1 for d in deltas if d > 0) / len(deltas), 6) if deltas else 0.0
        ),
        "mean_decode_calls": mean("decode_calls"),
        "mean_latency_ms": mean("latency_ms"),
        "total_latency_s": round(sum(r.get("latency_ms", 0.0) for r in rows) / 1000.0, 1),
        "note_counts": note_counts,
    }


def run_sweep(
    host: ModelHost,
    arm: str,
    records: list[dict[str, Any]],
    arm_kwargs: dict[str, Any],
    per_item_path: Path,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(per_item_path, "w", encoding="utf-8") as handle:
        for index, item in enumerate(records, start=1):
            started = time.perf_counter()
            row = run_arm(arm, host, item, **arm_kwargs).as_dict()

            # sec.3 ablation: the same Pass-2 call with an empty trace. One extra
            # forward pass, no extra generation -- identical to how the trace
            # scorer computes delta_trace.
            empty = host.score_with_trace(item, "")
            empty_probs = host.scorer._softmax_probs(empty)
            row["empty_trace_option_set_gold_logprob"] = math.log(empty_probs[item["answer"]])
            row["delta_trace"] = (
                row["option_set_gold_logprob"] - row["empty_trace_option_set_gold_logprob"]
            )
            row["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 1)

            handle.write(json.dumps(row) + "\n")
            handle.flush()  # a 45-minute sweep should survive being interrupted
            rows.append(row)
            if verbose and index % 10 == 0:
                print(f"  [{arm}] {index}/{len(records)}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/val_v3.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arms", default="A,C,R")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-trace-tokens", type=int, default=256)
    parser.add_argument("--query-mode", default="question", choices=["question", "model"])
    parser.add_argument("--base-url", default=None, help="Arm B only; requires rlm_slice.server")
    parser.add_argument("--out-prefix", default="results/slice")
    args = parser.parse_args(argv)

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arms {unknown}; expected from {list(ARMS)}")

    records = load_records(args.dataset, args.limit)
    print(f"{len(records)} scorable MCQ rows from {args.dataset}", flush=True)

    host = ModelHost(model_path=args.model, device=args.device, seed=args.seed)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    for arm in arms:
        arm_kwargs: dict[str, Any] = {"max_trace_tokens": args.max_trace_tokens}
        if arm in ("R", "M", "N"):
            arm_kwargs["query_mode"] = args.query_mode
        if arm == "B":
            from rlm_slice.rlm_arm import DEFAULT_BASE_URL, build_rlm

            base_url = args.base_url or DEFAULT_BASE_URL
            arm_kwargs["rlm_client"] = build_rlm(host.model_path, base_url=base_url)
            arm_kwargs["base_url"] = base_url

        print(f"=== arm {arm} ===", flush=True)
        started = time.perf_counter()
        rows = run_sweep(host, arm, records, arm_kwargs, Path(f"{out_prefix}_{arm}.jsonl"))
        summary = summarize(arm, rows)
        summary["sweep_wall_s"] = round(time.perf_counter() - started, 1)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "max_trace_tokens": args.max_trace_tokens,
        "query_mode": args.query_mode,
        "n_items": len(records),
        "arms": summaries,
    }
    summary_path = Path(f"{out_prefix}_summary.json")
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
