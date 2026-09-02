"""Free-form dev rubric track (L7 / #130, part of I12 / #117).

LOCAL_RLM_SLICE_PLAN.md sec.3.4.3: the deploy target is free-form question/answer,
not MCQ, so this generates one candidate answer per arm over the same dev-set
subset the codexv5 rubric baseline was graded on, ready for
eval_freeform/score_rubric.py's --input contract. This module only generates
and records telemetry; rubric grading is a separate, explicit step (grading
sends candidate answers to an external judge -- see eval_freeform/score_rubric.py
--judge-backend).

Never reads eval_freeform/test.jsonl (sealed).

Usage:
  python -m rlm_slice.eval_slice_freeform --arms A,B,C,R --out-prefix results/slice_v1_freeform
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rlm_slice.freeform_arm import ARMS, run_arm
from rlm_slice.guards import DEFAULT_SEED, check_not_sealed
from rlm_slice.model_host import DEFAULT_MODEL, ModelHost

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = "eval_freeform/dev.jsonl"
DEFAULT_BASELINE = "results/periodic_rubric_baseline_codexv5.jsonl"


def load_paired_dev_rows(dataset: str, baseline: str) -> list[dict[str, Any]]:
    """The same dev-item subset the codexv5 baseline was graded on.

    L7's accept criterion is "same-judge paired against the codexv5 baseline"
    (#130) -- pairing requires identical item ids, not just the same dev file,
    since the historical baseline covers 75 of the dev set's 90 rows.
    """
    check_not_sealed(dataset)  # sec.2.2, before any file IO
    baseline_ids = {json.loads(line)["id"] for line in open(baseline, encoding="utf-8") if line.strip()}
    rows = [json.loads(line) for line in open(dataset, encoding="utf-8") if line.strip()]
    paired = [row for row in rows if row["id"] in baseline_ids]
    missing = baseline_ids - {row["id"] for row in paired}
    if missing:
        raise SystemExit(
            f"{len(missing)} baseline ids not found in {dataset}: {sorted(missing)[:5]}"
        )
    return paired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arms", default="A,B,C,R")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--base-url", default=None, help="Arm B only; requires rlm_slice.server")
    parser.add_argument("--out-prefix", default="results/slice_v1_freeform")
    args = parser.parse_args(argv)

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arms {unknown}; expected from {list(ARMS)}")

    records = load_paired_dev_rows(args.dataset, args.baseline)
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} dev rows paired with {args.baseline}", flush=True)

    host = ModelHost(model_path=args.model, device=args.device, seed=args.seed)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        arm_kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
        if arm == "B":
            from rlm_slice.rlm_arm import DEFAULT_BASE_URL

            arm_kwargs["base_url"] = args.base_url or DEFAULT_BASE_URL

        print(f"=== arm {arm} ===", flush=True)
        started = time.perf_counter()
        out_path = Path(f"{out_prefix}_{arm}.jsonl")
        rows_out: list[dict[str, Any]] = []
        with open(out_path, "w", encoding="utf-8") as handle:
            for index, item in enumerate(records, start=1):
                item_started = time.perf_counter()
                result = run_arm(arm, host, item, **arm_kwargs)
                wall_ms = round((time.perf_counter() - item_started) * 1000.0, 1)
                row = {
                    "id": item["id"],
                    "type": item.get("type"),
                    "source_doc": item.get("source_doc"),
                    "question": item["question"],
                    "reference_answer": item["reference_answer"],
                    "rubric": item["rubric"],
                    "candidate_answer": result.pop("candidate_answer"),
                    "generation_model": args.model,
                    "prompt_mode": "rlm_slice_freeform_v1",
                    "enable_thinking": False,
                    "arm": arm,
                    "wall_ms": wall_ms,
                    **result,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()  # a long sweep should survive being interrupted
                rows_out.append(row)
                if index % 10 == 0:
                    print(f"  [{arm}] {index}/{len(records)}", flush=True)

        n = len(rows_out)
        mean_decode = round(sum(r.get("decode_calls", 0) for r in rows_out) / n, 2) if n else 0.0
        mean_wall_ms = round(sum(r.get("wall_ms", 0.0) for r in rows_out) / n, 1) if n else 0.0
        sweep_s = round(time.perf_counter() - started, 1)
        print(
            f"[{arm}] wrote {out_path}: n={n} mean_decode_calls={mean_decode} "
            f"mean_wall_ms={mean_wall_ms} sweep_s={sweep_s}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
