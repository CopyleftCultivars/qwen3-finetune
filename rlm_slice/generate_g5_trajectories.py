"""G5 (#152): collect RLM trajectories from Qwen3-8B on domain items.

Reuses Epic F's contamination wall (make_synthetic_traces.instruction_hash /
load_eval_hash_denylist, checked against data/val_v3.jsonl and
eval_freeform/dev.jsonl -- never eval_freeform/test.jsonl, the sealed set) and
the same causal-verification quality filter idea as
data/filter_synthetic_traces.py: keep a trajectory only if Pass-2 scoring
(the frozen score_with_trace objective) actually lands on the gold letter.

Domain pool: data/train_v3.jsonl's MCQ subset (the same ~1107-row pool
data/make_synthetic_traces.py draws from for the analogous plain-teacher-trace
pipeline), sampled deterministically.

Unlike controller.arm_b (which only keeps summary telemetry -- repl_blocks,
rlm_iterations counts), this records the FULL per-iteration trajectory
(prompt, response, REPL code + stdout/stderr, final_answer) from the rlm
package's RLMLogger, via result.metadata -- that raw transcript is the actual
SFT-training material G6 needs, not just a pass/fail score.

Shares the Qwen3-8B in-process singleton with rlm_slice.server the same way
run_g3_qwen3_8b.py does, for the same VRAM reason (two 4-bit 8B copies do not
fit on a 12GB card).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
QWEN3_8B = "unsloth/qwen3-8b-unsloth-bnb-4bit"


def load_domain_pool(train_path: Path, val_path: Path, freeform_dev_path: Path) -> tuple[list[dict], int, int]:
    """Load train_v3's MCQ subset, minus any hash overlap with eval sources.

    Returns (clean_records, total_mcq_records, denylist_size).
    """
    sys.path.insert(0, str(REPO_ROOT / "autoresearch"))
    sys.path.insert(0, str(REPO_ROOT / "data"))
    from score_mcq_ll import normalize_records  # noqa: E402
    from make_synthetic_traces import instruction_hash, load_eval_hash_denylist  # noqa: E402

    rows = [json.loads(l) for l in train_path.open(encoding="utf-8")]
    records, _skipped = normalize_records(rows)

    denylist = load_eval_hash_denylist(
        [val_path, freeform_dev_path], text_field_candidates=("instruction", "question")
    )
    clean = [r for r in records if instruction_hash(str(r["question"])) not in denylist]
    return clean, len(records), len(denylist)


def start_server_thread(host_obj, bind_host: str = "127.0.0.1", port: int = 8000):
    from rlm_slice import server as rlm_server
    import threading
    import urllib.request

    assert rlm_server.get_host() is host_obj, "singleton mismatch before server start"
    thread = threading.Thread(
        target=lambda: rlm_server.app.run(host=bind_host, port=port, threaded=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://{bind_host}:{port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    else:
        raise RuntimeError("rlm_slice.server did not come up within 10s")
    return thread


def run_trajectory(host, client, item: dict[str, Any], base_url: str) -> dict[str, Any]:
    """One item through the real RLM LocalREPL, with the full transcript kept.

    Mirrors controller.arm_b's telemetry/scoring path (same _finalize tail via
    host.score_with_trace) but additionally stashes result.metadata verbatim.
    """
    from rlm_slice.controller import LETTERS, _question_text, _ARM_B_INSTRUCTION
    from rlm_slice.rlm_arm import repl_telemetry, reset_server_stats, fetch_server_stats
    import math

    host.take_stats()
    notes: list[str] = []
    try:
        reset_server_stats(base_url)
    except Exception:
        notes.append("server_stats_unavailable")

    trace = ""
    telemetry: dict[str, Any] = {}
    full_metadata: dict | None = None
    try:
        result = client.completion(
            _ARM_B_INSTRUCTION + _question_text(item),
            root_prompt=_question_text(item),
        )
        trace = (getattr(result, "response", "") or "").strip()
        full_metadata = getattr(result, "metadata", None)
        telemetry = repl_telemetry(result)
    except Exception as exc:
        notes.append(f"rlm_error:{type(exc).__name__}")

    if telemetry and telemetry.get("repl_blocks", 0) == 0:
        notes.append("no_repl_blocks:degenerate_arm_b")
    if telemetry and not telemetry.get("submitted_answer"):
        notes.append("no_submitted_answer:iterations_exhausted")

    server_stats: dict[str, Any] = {}
    try:
        server_stats = fetch_server_stats(base_url)
    except Exception:
        notes.append("server_stats_unavailable")

    if not trace:
        notes.append("empty_rlm_response")

    # Pass-2 scoring, frozen objective -- also the causal-verification filter signal.
    logprobs = host.score_with_trace(item, trace)
    probs = host.scorer._softmax_probs(logprobs)
    gold = item["answer"]
    top_letter = max(LETTERS, key=lambda letter: logprobs[letter])

    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "A": item["A"],
        "B": item["B"],
        "C": item["C"],
        "D": item["D"],
        "gold": gold,
        "top_letter": top_letter,
        "correct": top_letter == gold,
        "option_set_gold_logprob": math.log(probs[gold]),
        "final_response": trace,
        "trajectory": full_metadata,
        "repl_blocks": telemetry.get("repl_blocks", 0),
        "rlm_iterations": telemetry.get("rlm_iterations", 0),
        "submitted_answer": telemetry.get("submitted_answer", False),
        "server_stats": server_stats,
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="data/train_v3.jsonl")
    parser.add_argument("--val-data", default="data/val_v3.jsonl")
    parser.add_argument("--freeform-dev", default="eval_freeform/dev.jsonl")
    parser.add_argument("--model", default=QWEN3_8B)
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--out", default="results/g5_trajectories_qwen3-8b.jsonl")
    parser.add_argument("--out-filtered", default="results/g5_trajectories_qwen3-8b_filtered.jsonl")
    parser.add_argument("--config-out", default="results/g5_generation_config.json")
    args = parser.parse_args(argv)

    clean_pool, total_mcq, denylist_size = load_domain_pool(
        Path(args.train_data), Path(args.val_data), Path(args.freeform_dev)
    )
    print(
        f"domain pool: {total_mcq} MCQ records in {args.train_data}, "
        f"{denylist_size} eval hashes denylisted, {len(clean_pool)} clean after exclusion",
        flush=True,
    )

    rng = random.Random(args.seed)
    sample = clean_pool[:]
    rng.shuffle(sample)
    sample = sample[: args.n_items]
    print(f"sampled {len(sample)} items (seed={args.seed})", flush=True)

    from rlm_slice.model_host import get_host
    from rlm_slice.rlm_arm import build_rlm, DEFAULT_BASE_URL

    host = get_host(model_path=args.model, device=args.device, seed=args.seed, full_precision=False)
    base_url = f"http://127.0.0.1:{args.server_port}/v1"
    start_server_thread(host, port=args.server_port)
    print(f"rlm_slice.server up in-thread at {base_url}, sharing the singleton", flush=True)

    client = build_rlm(host.model_path, base_url=base_url)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(sample, start=1):
            row = run_trajectory(host, client, item, base_url)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            if index % 10 == 0:
                elapsed = time.perf_counter() - started_all
                print(f"  {index}/{len(sample)} ({elapsed:.0f}s elapsed)", flush=True)

    n_correct = sum(1 for r in rows if r["correct"])
    n_repl = sum(1 for r in rows if r["repl_blocks"] > 0)
    n_submitted = sum(1 for r in rows if r["submitted_answer"])

    # Quality filter (causal verification, per filter_synthetic_traces.py's
    # convention): keep only trajectories that both used the REPL and reached
    # the gold answer -- a correct-but-scaffold-free row is not evidence the
    # RLM behaviour itself is trainable, and a REPL-using-but-wrong row is not
    # a target we want the fine-tune to imitate.
    filtered = [r for r in rows if r["correct"] and r["repl_blocks"] > 0]
    filtered_path = Path(args.out_filtered)
    with filtered_path.open("w", encoding="utf-8") as handle:
        for row in filtered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "model": args.model,
        "backend": "rlm LocalREPL (rlm_slice.rlm_arm.build_rlm), max_depth=1, temperature=0",
        "seed": args.seed,
        "domain_source": args.train_data,
        "total_mcq_records_in_source": total_mcq,
        "eval_denylist_sources": [args.val_data, args.freeform_dev],
        "eval_denylist_size": denylist_size,
        "clean_pool_size": len(clean_pool),
        "n_sampled": len(sample),
        "n_generated": len(rows),
        "n_correct": n_correct,
        "n_repl_engaged": n_repl,
        "n_submitted_answer": n_submitted,
        "n_kept_after_quality_filter": len(filtered),
        "quality_filter": "correct AND repl_blocks > 0 (causal verification + REPL-engagement)",
        "eval_overlap_excluded": True,
        "eval_overlap_count": 0,
        "raw_output": str(out_path),
        "filtered_output": str(filtered_path),
        "wall_s": round(time.perf_counter() - started_all, 1),
    }
    Path(args.config_out).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
