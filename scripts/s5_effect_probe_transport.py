#!/usr/bin/env python3
"""#228 (S5/G6): three-condition effect-probe transport on the real engine.

Drives #214's 12 natural-farming prompts through the real ARM engine, greedy,
resetting the session before every generation so each is an independent fresh
context. Conditions (all built from p193-suppress-deploy, differing only in the
memory-prefix manifest):

  selected -- outputs/r179-m1/s2a/gate_bundles/armed  (#216 S0 post-transform A1)
  random   -- outputs/r179-m1/s2a/gate_bundles/random (same pipeline/post-transform,
              per-head random slot excluding S0; seed 20260924)
  none     -- outputs/r179-m1/p193-suppress-deploy    (never armed)
  matched  -- optional robustness control: this-seed fit's S0 post-transform
              (same fit as random) -- only run when --matched is passed

Writes raw per-prompt records (ids + text + reserved emissions) to --out. The
numeric verdict plus the qualitative review table are produced by the separate
scripts/s5_effect_probe_verdict.py comparator, so it can be independently
re-run against this output. Container-side (cactus + stdlib only).

Usage (inside the arm64 engine container):
  python3 s5_effect_probe_transport.py \
    --selected /work/.../gate_bundles/armed \
    --random   /work/.../gate_bundles/random \
    --none     /work/.../p193-suppress-deploy \
    --matched  /work/.../gate_bundles/armed_matched \
    --out /work/out/s5_effect_probe.json --max-tokens 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from cactus.bindings import cactus as c

# reuse #214's prompt list (single source of truth)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d203_effect_probe import PROMPTS  # noqa: E402

RESERVED_LO, RESERVED_HI = 151670, 151935
VOCAB = 151936
# standalone nan/inf words only -- a plain substring check false-positives on
# "information", "influence", etc. (same fix as s4_g5_sanity.py).
NAN_RE = re.compile(r"\b(-?nan|-?inf(inity)?)\b", re.IGNORECASE)


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
    text = text or ""
    bad_id = any(t < 0 or t >= VOCAB for t in ids)
    return {
        "ids": ids,
        "text": text,
        "reserved": [x for x in ids if RESERVED_LO <= x <= RESERVED_HI],
        "nan": bool(NAN_RE.search(text)) or bad_id or ("\ufffd" in text),
        "empty": (len(ids) == 0 or text.strip() == ""),
        "seconds": round(dt, 2),
    }


def _init(label: str, path: str):
    if not path:
        return None
    model = c.cactus_init(path)
    if not model:
        raise SystemExit(f"cactus_init({label}) null: {path}")
    return model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected", required=True)
    p.add_argument("--random", required=True)
    p.add_argument("--none", required=True)
    p.add_argument("--matched", default="", help="optional 4th corroborating condition")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS

    selected = _init("selected", args.selected)
    random_m = _init("random", args.random)
    none = _init("none", args.none)
    matched = _init("matched", args.matched)
    conditions = [("selected", selected), ("random", random_m), ("none", none)]
    if matched is not None:
        conditions.append(("matched", matched))

    records = []
    for i, prompt in enumerate(prompts):
        cond = {}
        for name, model in conditions:
            c.cactus_reset(model)
            cond[name] = generate(model, prompt, args.max_tokens)
        rec = {
            "index": i,
            "prompt": prompt,
            "selected": cond["selected"],
            "random": cond["random"],
            "none": cond["none"],
            "selected_differs_none": cond["selected"]["ids"] != cond["none"]["ids"],
            "random_differs_none": cond["random"]["ids"] != cond["none"]["ids"],
            "selected_differs_random": cond["selected"]["ids"] != cond["random"]["ids"],
        }
        if "matched" in cond:
            rec["matched"] = cond["matched"]
            rec["matched_differs_none"] = cond["matched"]["ids"] != cond["none"]["ids"]
            rec["matched_differs_random"] = cond["matched"]["ids"] != cond["random"]["ids"]
            rec["matched_equals_selected"] = cond["matched"]["ids"] == cond["selected"]["ids"]
        records.append(rec)
        print(
            f"[{i}] sel!=rand={rec['selected_differs_random']} "
            f"sel!=none={rec['selected_differs_none']} rand!=none={rec['random_differs_none']} "
            f"res(sel/rand/none)={len(cond['selected']['reserved'])}/{len(cond['random']['reserved'])}/{len(cond['none']['reserved'])}",
            flush=True,
        )

    for _, model in conditions:
        c.cactus_destroy(model)

    out = {
        "issue": 228,
        "max_tokens": args.max_tokens,
        "n_prompts": len(prompts),
        "reserved_range": [RESERVED_LO, RESERVED_HI],
        "conditions": [name for name, _ in conditions],
        "bundles": {
            "selected": args.selected,
            "random": args.random,
            "none": args.none,
            "matched": args.matched,
        },
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
