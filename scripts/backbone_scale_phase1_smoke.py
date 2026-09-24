#!/usr/bin/env python3
"""#231 Phase 1 smoke test: does a Qwen3-1.7B bundle load + generate, and does the
memory-prefix FFI accept the 2048-wide object, on the existing patched engine?

Run inside the arm64 conversion container (or any env with the patched
libcactus_engine.so). Container-side: cactus bindings + numpy/stdlib only.

Checks:
  1. cactus_init(bundle) succeeds.
  2. greedy cactus_complete on a few natural-farming prompts -> coherent text,
     0 reserved emissions, no NaN/empty.
  3. hidden_dim probe: cactus_set_memory_prefix with dim=2048 (should succeed on
     Qwen3-1.7B) and dim=1024 (should fail) -- empirically confirms the engine's
     config_.hidden_dim is 2048.
  4. arbitrary 2048-wide prefix (8 random rows) armed via the FFI, generation
     again -> no crash, 0 NaN/empty. Plumbing check only; the vectors are random.

Usage:
  python3 backbone_scale_phase1_smoke.py --bundle /path/to/qwen3-1.7b-cq4-conv \
      --out /work/out/phase1_smoke.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from cactus.bindings import cactus as c

RESERVED_LO, RESERVED_HI = 151670, 151935
VOCAB = 151936
NAN_RE = re.compile(r"\b(-?nan|-?inf(inity)?)\b", re.IGNORECASE)

PROMPTS = [
    "The best fertilizer for tomatoes is",
    "Tell me about Korean natural farming inputs",
    "Describe how to make biochar",
]


def generate(model, prompt, max_tokens):
    ids = []

    def cb(text, token_id):
        ids.append(int(token_id))

    res = c.cactus_complete(
        model,
        [{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "max_tokens": max_tokens},
        callback=cb,
    )
    text = (res.get("response") if isinstance(res, dict) else "") or ""
    bad_id = any(t < 0 or t >= VOCAB for t in ids)
    return {
        "ids": ids,
        "text": text,
        "reserved": [x for x in ids if RESERVED_LO <= x <= RESERVED_HI],
        "nan": bool(NAN_RE.search(text)) or bad_id or ("\ufffd" in text),
        "empty": (len(ids) == 0 or text.strip() == ""),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    out = {"issue": 231, "phase": 1, "bundle": args.bundle, "checks": {}}

    model = c.cactus_init(args.bundle)
    out["checks"]["init_ok"] = bool(model)
    if not model:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print("cactus_init FAILED", file=sys.stderr)
        return 1
    print("cactus_init OK")

    # 2. ordinary generation
    gens = []
    for prompt in PROMPTS:
        c.cactus_reset(model)
        g = generate(model, prompt, args.max_tokens)
        g["prompt"] = prompt
        gens.append(g)
        print(f"gen: {prompt!r} -> res={len(g['reserved'])} nan={g['nan']} empty={g['empty']} :: {g['text']!r}")
    out["checks"]["generation"] = {
        "n": len(gens),
        "reserved_total": sum(len(g["reserved"]) for g in gens),
        "nan": sum(int(g["nan"]) for g in gens),
        "empty": sum(int(g["empty"]) for g in gens),
        "records": gens,
    }

    # 3. hidden_dim probe via set_memory_prefix width validation
    rng = np.random.default_rng(231)
    probe = {}
    for dim in (2048, 1024):
        vec = rng.standard_normal((8, dim)).astype(np.float32).ravel().tolist()
        try:
            rc = c.cactus_set_memory_prefix(model, vec, 8, dim)
            probe[str(dim)] = {"rc": int(rc), "ok": True}
            print(f"set_memory_prefix dim={dim}: rc={rc}")
        except Exception as e:  # noqa: BLE001
            probe[str(dim)] = {"error": f"{type(e).__name__}: {e}", "ok": False}
            print(f"set_memory_prefix dim={dim}: ERROR {e}")
        finally:
            try:
                c.cactus_clear_memory_prefix(model)
            except Exception:  # noqa: BLE001
                pass
    out["checks"]["hidden_dim_probe"] = {
        "dim_2048_succeeds": probe.get("2048", {}).get("ok", False),
        "dim_1024_fails": not probe.get("1024", {}).get("ok", True),
        "detail": probe,
    }

    # 4. arbitrary 2048-wide prefix armed -> generate again
    vec = rng.standard_normal((8, 2048)).astype(np.float32).ravel().tolist()
    c.cactus_set_memory_prefix(model, vec, 8, 2048)
    print("armed arbitrary 8x2048 prefix")
    armed = []
    for prompt in PROMPTS:
        c.cactus_reset(model)
        g = generate(model, prompt, args.max_tokens)
        g["prompt"] = prompt
        armed.append(g)
        print(f"armed gen: {prompt!r} -> res={len(g['reserved'])} nan={g['nan']} empty={g['empty']} :: {g['text']!r}")
    out["checks"]["armed_arbitrary_prefix"] = {
        "n": len(armed),
        "reserved_total": sum(len(g["reserved"]) for g in armed),
        "nan": sum(int(g["nan"]) for g in armed),
        "empty": sum(int(g["empty"]) for g in armed),
        "records": armed,
    }
    c.cactus_clear_memory_prefix(model)
    c.cactus_destroy(model)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
