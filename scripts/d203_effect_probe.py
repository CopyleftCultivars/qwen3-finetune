#!/usr/bin/env python3
"""#214 real-engine effect probe: selected vs random vs none.

For each prompt, three greedy conditions on the SAME bundle:
  none     -- base prompt, no reserved tokens
  selected -- + <|mem0_0|>..<|mem0_7|>  (slot 0: S0-selected vectors)
  random   -- + <|mem1_0|>..<|mem1_7|>  (slot 1: random slot vectors)

Reports per prompt: reserved-ID emissions per condition (must be 0 on the
patched engine) and whether the continuation differs from none / from each
other (#207's check). Container-side, cactus binding + stdlib only.
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
    "Describe how to make biochar",
    "What are good companion plants for corn",
    "Explain crop rotation in one paragraph",
]

MEM_SELECTED = " " + " ".join(f"<|mem0_{h}|>" for h in range(8))
MEM_RANDOM = " " + " ".join(f"<|mem1_{h}|>" for h in range(8))


def generate(model, prompt: str, max_tokens: int) -> dict:
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
    text = res.get("response") if isinstance(res, dict) else ""
    return {
        "ids": ids,
        "text": text or "",
        "reserved": [x for x in ids if RESERVED_LO <= x <= RESERVED_HI],
        "seconds": round(dt, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", default="/work/bundle")
    parser.add_argument("--out", required=True)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--label", default="effect")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = c.cactus_init(args.bundle)
    if not model:
        raise SystemExit("cactus_init returned null")
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    jsonl = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    records: list[dict] = []
    for index, base in enumerate(prompts):
        if args.offset is not None and index % args.stride != args.offset:
            continue
        cond = {
            "none": generate(model, base, args.max_tokens),
            "selected": generate(model, base + MEM_SELECTED, args.max_tokens),
            "random": generate(model, base + MEM_RANDOM, args.max_tokens),
        }
        rec = {
            "index": index,
            "prompt": base,
            "none": cond["none"],
            "selected": cond["selected"],
            "random": cond["random"],
            "selected_differs_none": cond["selected"]["ids"] != cond["none"]["ids"],
            "random_differs_none": cond["random"]["ids"] != cond["none"]["ids"],
            "selected_differs_random": cond["selected"]["ids"] != cond["random"]["ids"],
        }
        records.append(rec)
        if jsonl:
            jsonl.write(json.dumps(rec) + "\n")
            jsonl.flush()
    if jsonl:
        jsonl.close()
    c.cactus_destroy(model)

    def tot(cond_name: str, field: str) -> int:
        return sum(len(r[cond_name][field]) for r in records)

    summary = {
        "issue": 214,
        "label": args.label,
        "prompts": len(records),
        "max_tokens": args.max_tokens,
        "reserved_emitted": {k: tot(k, "reserved") for k in ("none", "selected", "random")},
        "tokens": {k: tot(k, "ids") for k in ("none", "selected", "random")},
        "selected_differs_none": sum(r["selected_differs_none"] for r in records),
        "random_differs_none": sum(r["random_differs_none"] for r in records),
        "selected_differs_random": sum(r["selected_differs_random"] for r in records),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_prompt": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
