#!/usr/bin/env python3
"""#227 (S4): G4-secondary runtime-disarm check + G5 armed sanity.

Drives #214's 12-prompt set through the real engine in three conditions on the
same p193-suppress-deploy base, greedy, resetting the session before every
generation so each is an independent fresh context:

  armed          -- outputs/r179-m1/s2a/gate_bundles/armed (A1 prefix armed via manifest)
  armed_cleared  -- same bundle, cactus_clear_memory_prefix() called immediately after init
  disarmed       -- outputs/r179-m1/p193-suppress-deploy (never armed)

Writes raw per-prompt records (ids + text + reserved emissions) to --out. Verdicts
are computed by the separate scripts/s4_g5_verdict.py comparator so it can be
independently re-run against this output. Container-side (cactus + stdlib only).

  G4-secondary: armed_cleared ids == disarmed ids on ALL prompts
  G5: 0 reserved emissions in armed and disarmed; no NaN; no empty completion;
      armed ids != disarmed ids on >= 10/12
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from cactus.bindings import cactus as c

# reuse #214's prompt list (single source of truth)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d203_effect_probe import PROMPTS  # noqa: E402

RESERVED_LO, RESERVED_HI = 151670, 151935
VOCAB = 151936
# standalone nan/inf words only -- a plain substring check false-positives on
# "information", "influence", etc.
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--armed", required=True)
    p.add_argument("--disarmed", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS

    armed = c.cactus_init(args.armed)
    if not armed:
        raise SystemExit(f"cactus_init(armed) null: {args.armed}")
    armed_cleared = c.cactus_init(args.armed)
    if not armed_cleared:
        raise SystemExit("cactus_init(armed_cleared) null")
    if c.cactus_clear_memory_prefix(armed_cleared) != 0:
        raise SystemExit("cactus_clear_memory_prefix failed")
    disarmed = c.cactus_init(args.disarmed)
    if not disarmed:
        raise SystemExit(f"cactus_init(disarmed) null: {args.disarmed}")

    records = []
    for i, prompt in enumerate(prompts):
        c.cactus_reset(armed)
        a = generate(armed, prompt, args.max_tokens)
        c.cactus_reset(armed_cleared)
        ac = generate(armed_cleared, prompt, args.max_tokens)
        c.cactus_reset(disarmed)
        d = generate(disarmed, prompt, args.max_tokens)
        rec = {
            "index": i,
            "prompt": prompt,
            "armed": a,
            "armed_cleared": ac,
            "disarmed": d,
            "armed_differs_disarmed": a["ids"] != d["ids"],
            "armed_cleared_equals_disarmed": ac["ids"] == d["ids"],
        }
        records.append(rec)
        print(f"[{i}] differs={rec['armed_differs_disarmed']} cleared_eq={rec['armed_cleared_equals_disarmed']} "
              f"res(armed/dis)={len(a['reserved'])}/{len(d['reserved'])}", flush=True)

    c.cactus_destroy(armed)
    c.cactus_destroy(armed_cleared)
    c.cactus_destroy(disarmed)

    out = {
        "issue": 227,
        "max_tokens": args.max_tokens,
        "n_prompts": len(prompts),
        "reserved_range": [RESERVED_LO, RESERVED_HI],
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
