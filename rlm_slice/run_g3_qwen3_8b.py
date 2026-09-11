"""G3 (#150) driver: run rlm_slice arms A/B/R/C against Qwen3-8B.

Not part of the frozen slice -- eval_slice.py's own main() constructs a fresh
ModelHost per run, which is fine for arms A/C/R (one in-process copy) but
would need a *second* full model load for arm B (its RLM client talks to
rlm_slice.server over HTTP, normally a separate OS process). At the 0.6B
champion's size that second copy was cheap; at Qwen3-8B 4-bit (~7.5GB) two
copies do not fit on a 12GB card.

Fix: start rlm_slice.server's Flask app in a background thread of *this*
process instead of a separate process. Its routes call model_host.get_host()
with no args, which returns the process-wide singleton -- so as long as that
singleton is constructed first (below), the server thread and this thread's
Pass-2 scoring share the exact same loaded weights, at zero extra VRAM cost.
No changes to model_host.py, server.py, controller.py, eval_slice.py, or
rlm_arm.py -- this only orchestrates them.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from rlm_slice.eval_slice import load_records, run_sweep, summarize
from rlm_slice.model_host import get_host
from rlm_slice.guards import DEFAULT_SEED

REPO_ROOT = Path(__file__).resolve().parent.parent
QWEN3_8B = "unsloth/qwen3-8b-unsloth-bnb-4bit"


def start_server_thread(host_obj, bind_host: str = "127.0.0.1", port: int = 8000) -> threading.Thread:
    from rlm_slice import server as rlm_server

    # Route module-level get_host() calls at the already-built singleton --
    # server.py's handlers call get_host() with no args.
    assert rlm_server.get_host() is host_obj, "singleton mismatch before server start"

    thread = threading.Thread(
        target=lambda: rlm_server.app.run(host=bind_host, port=port, threaded=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    # Poll /health instead of a fixed sleep -- model is already loaded, this is
    # just waiting for Flask's socket to bind.
    import urllib.request

    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://{bind_host}:{port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    else:
        raise RuntimeError("rlm_slice.server did not come up within 10s")
    return thread


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/val_v3.jsonl")
    parser.add_argument("--model", default=QWEN3_8B)
    parser.add_argument("--arms", default="A,C,R,B")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-trace-tokens", type=int, default=256)
    parser.add_argument("--query-mode", default="question", choices=["question", "model"])
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--out-prefix", default="results/slice_qwen3-8b")
    args = parser.parse_args(argv)

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]

    records = load_records(args.dataset, args.limit)
    print(f"{len(records)} scorable MCQ rows from {args.dataset}", flush=True)

    host = get_host(model_path=args.model, device=args.device, seed=args.seed, full_precision=False)

    base_url = f"http://127.0.0.1:{args.server_port}/v1"
    if "B" in arms:
        start_server_thread(host, port=args.server_port)
        print(f"rlm_slice.server up in-thread at {base_url}, sharing the singleton", flush=True)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    for arm in arms:
        arm_kwargs: dict = {"max_trace_tokens": args.max_trace_tokens}
        if arm == "R":
            arm_kwargs["query_mode"] = args.query_mode
        if arm == "B":
            from rlm_slice.rlm_arm import build_rlm

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
        "n_records": len(records),
        "arms": summaries,
    }
    summary_path = Path(f"{args.out_prefix}_summary.json")
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
