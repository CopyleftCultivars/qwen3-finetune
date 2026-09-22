#!/usr/bin/env python3
"""#206 engine-side runner for the S3 leak A/B and S4 argmax-parity tests.

Runs inside the M1 qemu arm64 harness against whichever libcactus_engine.so the
container mounts (CACTUS_LIB_PATH), on whichever bundle is mounted at /work/bundle.

Modes:
  bench   -- time a few generations.
  leak    -- N generations of `max_tokens` greedy tokens over a prompt x
             memory-variant set; count emissions in the reserved pool
             [151670,151935]. Supports --offset/--stride for sharded parallel runs.
  parity  -- emit the greedy token-id sequence per prompt so two engine/bundle
             conditions can be diffed on the host.

Kept dependency-free (cactus binding only) so it runs in the arm64 container.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cactus.bindings import cactus as c

RESERVED_LO, RESERVED_HI = 151670, 151935

PROMPTS = [
    "The best fertilizer for tomatoes is",
    "One fact about soil:",
    "How do I increase potassium in my garden?",
    "My tomato leaves are turning yellow, what should I do",
    "What is the best cover crop for a small farm in Tennessee",
    "I want to make fermented plant juice at home",
    "How much water does a young apple tree need",
    "Tell me about Korean natural farming inputs",
    "What can I use to add nitrogen naturally to my soil",
    "The capital of France is",
    "Summarize the benefits of compost tea",
    "My pepper plants have aphids, how do I handle it",
    "What is the ideal pH for growing blueberries",
    "Explain crop rotation in one paragraph",
    "Describe how to make biochar",
    "What are good companion plants for corn",
]

# Reserved IDs currently registered as <|memK_H|> in the p193 bundle: K in 0..23, H in 0..7.
MEM_VARIANTS = {
    "none": "",
    "1mem": " <|mem0_0|>",
    "8mem": " " + " ".join(f"<|mem{k}_0|>" for k in range(8)),
}


def generate(model, prompt: str, max_tokens: int) -> tuple[list[int], str, float]:
    ids: list[int] = []

    def cb(text, token_id):
        ids.append(int(token_id))

    t = time.time()
    res = c.cactus_complete(
        model,
        [{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "max_tokens": max_tokens},
        callback=cb,
    )
    dt = time.time() - t
    text = ""
    if isinstance(res, dict):
        text = res.get("response") or res.get("completion") or ""
    return ids, text, dt


def run_bench(model, args) -> dict:
    out = []
    for p in PROMPTS[:3]:
        ids, text, dt = generate(model, p, 16)
        out.append({"prompt": p, "seconds": round(dt, 2), "n": len(ids), "s_per_tok": round(dt / max(len(ids), 1), 3), "ids": ids})
    return {"mode": "bench", "runs": out}


def run_leak(model, args) -> dict:
    n = args.generations
    per_seq = []
    reserved_total = 0
    token_total = 0
    by_variant: dict[str, dict] = {v: {"generations": 0, "tokens": 0, "reserved": 0} for v in MEM_VARIANTS}
    jsonl = None
    if args.jsonl:
        jsonl = open(args.jsonl, "a", encoding="utf-8")
    for i in range(args.start, args.start + n):
        if args.offset is not None and i % args.stride != args.offset:
            continue
        variant_names = list(MEM_VARIANTS)
        variant = variant_names[i % len(variant_names)]
        prompt = PROMPTS[(i // len(variant_names)) % len(PROMPTS)] + MEM_VARIANTS[variant]
        ids, text, dt = generate(model, prompt, args.max_tokens)
        reserved = [t for t in ids if RESERVED_LO <= t <= RESERVED_HI]
        reserved_total += len(reserved)
        token_total += len(ids)
        by_variant[variant]["generations"] += 1
        by_variant[variant]["tokens"] += len(ids)
        by_variant[variant]["reserved"] += len(reserved)
        rec = {"i": i, "variant": variant, "prompt": prompt, "ids": ids, "reserved": len(reserved), "seconds": round(dt, 2)}
        per_seq.append(rec)
        if jsonl is not None:
            jsonl.write(json.dumps(rec) + "\n")
            jsonl.flush()
    if jsonl is not None:
        jsonl.close()
    return {
        "mode": "leak",
        "generations": len(per_seq),
        "max_tokens": args.max_tokens,
        "token_total": token_total,
        "reserved_total": reserved_total,
        "reserved_rate": round(reserved_total / max(token_total, 1), 6),
        "by_variant": by_variant,
        "sequences": per_seq,
    }


def run_parity(model, args) -> dict:
    per_prompt = []
    for p in PROMPTS:
        ids, text, dt = generate(model, p, args.max_tokens)
        per_prompt.append({"prompt": p, "ids": ids})
    return {"mode": "parity", "max_tokens": args.max_tokens, "per_prompt": per_prompt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("bench", "leak", "parity"))
    parser.add_argument("--bundle", default="/work/bundle")
    parser.add_argument("--out", default=None)
    parser.add_argument("--label", default="engine")
    parser.add_argument("--start", type=int, default=0, help="Global index to start at (for extending a run).")
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--jsonl", default=None, help="Append each generation as a JSON line (crash/timeout resilience).")
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t = time.time()
    model = c.cactus_init(args.bundle)
    init_s = time.time() - t
    if not model:
        raise SystemExit("cactus_init returned null")
    fn = {"bench": run_bench, "leak": run_leak, "parity": run_parity}[args.mode]
    result = fn(model, args)
    c.cactus_destroy(model)
    result["label"] = args.label
    result["bundle"] = args.bundle
    result["init_seconds"] = round(init_s, 2)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{args.label}] mode={args.mode} " + json.dumps({k: v for k, v in result.items() if k not in ("sequences", "per_prompt", "runs")}))
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
