#!/usr/bin/env python3
"""#228 (S5/G6) independent verdict comparator for the effect-probe transport.

Reads the raw records written by scripts/s5_effect_probe_transport.py and
recomputes the numeric verdicts WITHOUT running the engine. Re-running this
against the same output must produce byte-identical results (the epic's review
requirement), so the qualitative verdict, when supplied, is read from an
externally authored JSON (--qualitative) and embedded verbatim rather than
hardcoded here.

G6a (numeric, requested): selected != random ids on >= 9/12 prompts.
G6b (qualitative, decisive): content plausibly tied to the specific selected
persona/memory, not a generic register shift regardless of which vectors fired.
This script CANNOT judge G6b; it emits the full per-prompt text table so a human
can, and embeds whatever verdict the human authored.

Sanity: 0 reserved-ID emissions, no NaN, no empty completions in any condition.

Usage:
  python scripts/s5_effect_probe_verdict.py \
    --probe results/F179_M3_S5_EFFECT_PROBE.json \
    --qualitative results/F179_M3_S5_QUALITATIVE.json \
    --out results/F179_M3_S5_EFFECT_PROBE_VERDICT.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RESERVED_LO, RESERVED_HI = 151670, 151935
VOCAB = 151936
NAN_RE = re.compile(r"\b(-?nan|-?inf(inity)?)\b", re.IGNORECASE)
NUMERIC_BAR = 9  # selected != random on >= NUMERIC_BAR / n prompts


def is_nan(rec: dict) -> bool:
    ids = rec.get("ids", [])
    text = rec.get("text", "") or ""
    return bool(NAN_RE.search(text)) or any(t < 0 or t >= VOCAB for t in ids) or ("\ufffd" in text)


def cond_stats(rec: dict) -> dict:
    return {
        "reserved": len(rec.get("reserved", [])),
        "nan": is_nan(rec),
        "empty": bool(rec.get("empty", False)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--qualitative", default="", help="human-authored qualitative verdict JSON to embed")
    args = p.parse_args(argv)

    data = json.loads(Path(args.probe).read_text())
    records = data["records"]
    n = len(records)
    has_matched = "matched" in records[0]

    conditions = ["selected", "random", "none"] + (["matched"] if has_matched else [])

    totals = {c: {"reserved": 0, "nan": 0, "empty": 0} for c in conditions}
    differs = {
        "selected_differs_random": 0,
        "selected_differs_none": 0,
        "random_differs_none": 0,
        "matched_differs_random": 0,
        "matched_differs_none": 0,
        "matched_equals_selected": 0,
    }
    rows = []
    for rec in records:
        c = {name: rec[name] for name in conditions}
        for name in conditions:
            st = cond_stats(c[name])
            totals[name]["reserved"] += st["reserved"]
            totals[name]["nan"] += int(st["nan"])
            totals[name]["empty"] += int(st["empty"])

        # recompute differences independently from raw ids
        sel_rand = c["selected"]["ids"] != c["random"]["ids"]
        sel_none = c["selected"]["ids"] != c["none"]["ids"]
        rand_none = c["random"]["ids"] != c["none"]["ids"]
        differs["selected_differs_random"] += int(sel_rand)
        differs["selected_differs_none"] += int(sel_none)
        differs["random_differs_none"] += int(rand_none)

        row = {
            "index": rec["index"],
            "prompt": rec["prompt"],
            "none_text": c["none"]["text"],
            "selected_text": c["selected"]["text"],
            "random_text": c["random"]["text"],
            "selected_differs_random": sel_rand,
            "selected_differs_none": sel_none,
            "random_differs_none": rand_none,
            "reserved": {name: len(c[name].get("reserved", [])) for name in conditions},
        }
        if has_matched:
            m = c["matched"]
            m_rand = m["ids"] != c["random"]["ids"]
            m_none = m["ids"] != c["none"]["ids"]
            m_sel = m["ids"] == c["selected"]["ids"]
            differs["matched_differs_random"] += int(m_rand)
            differs["matched_differs_none"] += int(m_none)
            differs["matched_equals_selected"] += int(m_sel)
            row["matched_text"] = m["text"]
            row["matched_differs_random"] = m_rand
            row["matched_differs_none"] = m_none
            row["matched_equals_selected"] = m_sel
        rows.append(row)

    sanity_ok = all(t["reserved"] == 0 and t["nan"] == 0 and t["empty"] == 0 for t in totals.values())
    g6a_pass = differs["selected_differs_random"] >= NUMERIC_BAR

    qualitative = None
    if args.qualitative:
        qualitative = json.loads(Path(args.qualitative).read_text())

    verdict = {
        "issue": 228,
        "n_prompts": n,
        "max_tokens": data.get("max_tokens"),
        "conditions": conditions,
        "bundles": data.get("bundles", {}),
        "G6a_numeric": {
            "bar": f"selected != random on >= {NUMERIC_BAR}/{n}",
            "selected_differs_random": f"{differs['selected_differs_random']}/{n}",
            "selected_differs_none": f"{differs['selected_differs_none']}/{n}",
            "random_differs_none": f"{differs['random_differs_none']}/{n}",
            "pass": g6a_pass,
        },
        "sanity": {
            "per_condition": totals,
            "pass": sanity_ok,
        },
        "G6b_qualitative": qualitative,
        "rows": rows,
    }
    if has_matched:
        verdict["supplementary_matched"] = {
            "matched_differs_random": f"{differs['matched_differs_random']}/{n}",
            "matched_differs_none": f"{differs['matched_differs_none']}/{n}",
            "matched_equals_selected": f"{differs['matched_equals_selected']}/{n}",
        }

    out_text = json.dumps(verdict, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_text)
    print(out_text)
    print(f"G6a numeric {'PASS' if g6a_pass else 'FAIL'} "
          f"(selected!=random {differs['selected_differs_random']}/{n}); "
          f"sanity {'PASS' if sanity_ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
