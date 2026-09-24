#!/usr/bin/env python3
"""#230: oracle-selector effect-probe transport (new conditions only).

Runs the oracle-selector condition (and, optionally, the supplementary soft
attention-mixture control) against #214's exact 12 prompts, greedy, session reset
before every generation. #228's existing selected/random/none/matched completions
are NOT re-run -- they are merged in by the comparator, so this script produces
only the fresh records.

  oracle -- hard argmax over the persona's own held-out probe attention
            (oracle_pack), post-transformed. The issue's specified condition.
  soft   -- SUPPLEMENTARY: the soft attention mixture over all N slots for the
            same probe (out_ln[h](mix+mlp(mix))), the #216 soft-oracle upper bound
            (recall 0.25-0.31), included to control for the hard-argmax oracle's
            flat per-head attention. Deployable only via the A2 runtime setter.

Reuses ``s5_effect_probe_transport.generate`` and #214's ``PROMPTS`` so the
transport is byte-for-byte the same protocol as #228. Container-side (cactus +
stdlib only).

Usage (inside the arm64 engine container):
  python3 oracle_effect_probe_transport.py \
    --oracle /work/.../gate_bundles/oracle \
    --soft   /work/.../gate_bundles/oracle_soft \
    --out /work/out/oracle_effect_probe.json --max-tokens 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cactus.bindings import cactus as c

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d203_effect_probe import PROMPTS  # noqa: E402
from s5_effect_probe_transport import RESERVED_HI, RESERVED_LO, generate  # noqa: E402


def _init(label: str, path: str):
    if not path:
        return None
    model = c.cactus_init(path)
    if not model:
        raise SystemExit(f"cactus_init({label}) null: {path}")
    return model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oracle", required=True)
    p.add_argument("--soft", default="", help="optional supplementary soft-mixture condition")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS

    conditions = [("oracle", _init("oracle", args.oracle))]
    if args.soft:
        conditions.append(("soft", _init("soft", args.soft)))

    records = []
    for i, prompt in enumerate(prompts):
        cond = {}
        for name, model in conditions:
            c.cactus_reset(model)
            cond[name] = generate(model, prompt, args.max_tokens)
        rec = {"index": i, "prompt": prompt}
        rec.update(cond)
        records.append(rec)
        print(
            f"[{i}] " + " ".join(
                f"res({name})={len(cond[name]['reserved'])} nan={cond[name]['nan']}" for name, _ in conditions
            ),
            flush=True,
        )

    for _, model in conditions:
        c.cactus_destroy(model)

    out = {
        "issue": 230,
        "max_tokens": args.max_tokens,
        "n_prompts": len(prompts),
        "reserved_range": [RESERVED_LO, RESERVED_HI],
        "conditions": [name for name, _ in conditions],
        "bundles": {name: (args.oracle if name == "oracle" else args.soft) for name, _ in conditions},
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
