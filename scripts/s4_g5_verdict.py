#!/usr/bin/env python3
"""#227 (S4) independent verdict comparator for the G4-secondary + G5 sanity run.

Reads the raw records written by scripts/s4_g5_sanity.py and recomputes the
verdicts WITHOUT running the engine. Re-running this against the same output must
produce byte-identical results (the epic's review requirement).

G4-secondary ACCEPT: armed_cleared == disarmed ids on ALL prompts.
G5 ACCEPT: 0 reserved emissions in armed and disarmed; no NaN; no empty
completion; armed != disarmed on >= 10/12 prompts.

Usage: python scripts/s4_g5_verdict.py --sanity results/... sanity json --out ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RESERVED_LO, RESERVED_HI = 151670, 151935
VOCAB = 151936
NAN_RE = re.compile(r"\b(-?nan|-?inf(inity)?)\b", re.IGNORECASE)


def is_nan(rec: dict) -> bool:
    ids = rec.get("ids", [])
    text = rec.get("text", "") or ""
    return bool(NAN_RE.search(text)) or any(t < 0 or t >= VOCAB for t in ids) or ("\ufffd" in text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sanity", required=True)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    data = json.loads(Path(args.sanity).read_text())
    records = data["records"]
    n = len(records)

    rows = []
    reserved_armed = reserved_disarmed = 0
    nan = empty = 0
    differs = 0
    cleared_eq = 0
    for rec in records:
        a, ac, d = rec["armed"], rec["armed_cleared"], rec["disarmed"]
        reserved_armed += len(a["reserved"])
        reserved_disarmed += len(d["reserved"])
        if is_nan(a) or is_nan(d):
            nan += 1
        if a["empty"] or d["empty"]:
            empty += 1
        if rec["armed_differs_disarmed"]:
            differs += 1
        if rec["armed_cleared_equals_disarmed"]:
            cleared_eq += 1
        rows.append({
            "index": rec["index"],
            "prompt": rec["prompt"],
            "armed_reserved": len(a["reserved"]),
            "disarmed_reserved": len(d["reserved"]),
            "armed_ids": a["ids"],
            "armed_cleared_ids": ac["ids"],
            "disarmed_ids": d["ids"],
            "armed_differs_disarmed": rec["armed_differs_disarmed"],
            "armed_cleared_equals_disarmed": rec["armed_cleared_equals_disarmed"],
            "armed_nan": is_nan(a),
            "armed_empty": a["empty"],
        })

    g4_secondary_pass = cleared_eq == n
    g5_pass = (reserved_armed == 0 and reserved_disarmed == 0 and nan == 0
               and empty == 0 and differs >= 10)

    verdict = {
        "issue": 227,
        "n_prompts": n,
        "max_tokens": data.get("max_tokens"),
        "G4_secondary": {
            "armed_cleared_equals_disarmed": f"{cleared_eq}/{n}",
            "pass": g4_secondary_pass,
        },
        "G5": {
            "reserved_emitted_armed": reserved_armed,
            "reserved_emitted_disarmed": reserved_disarmed,
            "nan_prompts": nan,
            "empty_prompts": empty,
            "armed_differs_disarmed": f"{differs}/{n}",
            "pass": g5_pass,
        },
        "rows": rows,
    }
    out_text = json.dumps(verdict, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_text)
    print(out_text)
    print(f"G4-secondary {'PASS' if g4_secondary_pass else 'FAIL'} ({cleared_eq}/{n}); "
          f"G5 {'PASS' if g5_pass else 'FAIL'} (reserved {reserved_armed}/{reserved_disarmed}, "
          f"differs {differs}/{n}, nan {nan}, empty {empty})")
    return 0 if (g4_secondary_pass and g5_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
