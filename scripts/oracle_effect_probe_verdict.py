#!/usr/bin/env python3
"""#230 independent combined verdict comparator for the oracle-selector probe.

Reads BOTH raw transports -- #228's already-verified
``results/F179_M3_S5_EFFECT_PROBE.json`` (selected/random/none/matched) and this
issue's ``results/F179_M3_ORACLE_EFFECT_PROBE.json`` (oracle + supplementary
soft) -- merges them by prompt index, and recomputes every numeric verdict and
sanity flag from raw token ids WITHOUT running the engine. Re-running this against
the same inputs must produce byte-identical output (the epic's review
requirement), so the qualitative verdict is read from an externally authored JSON
and embedded verbatim.

This is a disambiguation, not a pass/fail gate: the comparator reports the
selected/oracle/soft vs none/random overlap structure and the deterministic
persona-marker scan, but the interpretation is authored in the qualitative JSON.

Usage:
  python scripts/oracle_effect_probe_verdict.py \
    --base results/F179_M3_S5_EFFECT_PROBE.json \
    --oracle results/F179_M3_ORACLE_EFFECT_PROBE.json \
    --qualitative results/F179_M3_ORACLE_QUALITATIVE.json \
    --out results/F179_M3_ORACLE_EFFECT_PROBE_VERDICT.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RESERVED_LO, RESERVED_HI = 151670, 151935
VOCAB = 151936
NAN_RE = re.compile(r"\b(-?nan|-?inf(inity)?)\b", re.IGNORECASE)
# fallback marker list if the qualitative JSON does not carry one
DEFAULT_MARKERS = ["tulsa", "oklahoma", "growbed", "invasive", "pepper", "lettuce",
                   "kitchen herb", "[short introduction]", "greetings"]


def is_nan(rec: dict) -> bool:
    ids = rec.get("ids", [])
    text = rec.get("text", "") or ""
    return bool(NAN_RE.search(text)) or any(t < 0 or t >= VOCAB for t in ids) or ("\ufffd" in text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--qualitative", default="")
    p.add_argument("--issue", type=int, default=230,
                   help="Issue id recorded in the verdict (e.g. 231 for the Phase-4 1.7B rerun).")
    args = p.parse_args(argv)

    base = json.loads(Path(args.base).read_text())
    orq = json.loads(Path(args.oracle).read_text())

    base_by_idx = {r["index"]: r for r in base["records"]}
    new_by_idx = {r["index"]: r for r in orq["records"]}
    n = len(base_by_idx)

    base_conds = ["none", "selected", "random", "matched"]
    new_conds = [c for c in orq["conditions"] if c in ("oracle", "soft")]
    conditions = base_conds + new_conds
    # every condition's record must be reachable uniformly
    def rec_of(i: int, c: str) -> dict:
        if c in new_conds:
            return new_by_idx[i][c]
        return base_by_idx[i][c]

    qualitative = json.loads(Path(args.qualitative).read_text()) if args.qualitative else None
    markers = DEFAULT_MARKERS
    if qualitative:
        markers = (qualitative.get("persona_under_test", {}).get("distinctive_lexical_markers")
                   or qualitative.get("markers") or DEFAULT_MARKERS)

    totals = {c: {"reserved": 0, "nan": 0, "empty": 0} for c in conditions}
    marker_hits = {c: 0 for c in conditions}
    differ = {}
    for new in new_conds:
        for c in conditions:
            differ[f"{new}_differs_{c}"] = 0
    differ["oracle_differs_random"] = 0  # explicit
    differ["soft_differs_random"] = 0
    rows = []

    for i in range(n):
        c = {name: rec_of(i, name) for name in conditions}
        for name in conditions:
            totals[name]["reserved"] += len(c[name].get("reserved", []))
            totals[name]["nan"] += int(is_nan(c[name]))
            totals[name]["empty"] += int(bool(c[name].get("empty", False)))
            tl = (c[name].get("text") or "").lower()
            marker_hits[name] += sum(1 for mk in markers if mk in tl)

        row = {"index": i, "prompt": base_by_idx[i]["prompt"]}
        for name in conditions:
            row[f"{name}_text"] = c[name]["text"]
        for new in new_conds:
            for other in conditions:
                if other == new:
                    continue
                key = f"{new}_differs_{other}"
                val = c[new]["ids"] != c[other]["ids"]
                differ[key] += int(val)
                row[key] = val
        rows.append(row)

    sanity_ok = all(t["reserved"] == 0 and t["nan"] == 0 and t["empty"] == 0 for t in totals.values())
    marker_total = sum(marker_hits.values())

    verdict = {
        "issue": args.issue,
        "n_prompts": n,
        "max_tokens": base.get("max_tokens"),
        "conditions": conditions,
        "persona": (qualitative or {}).get("persona_under_test", {}).get("persona_id"),
        "markers_scanned": markers,
        "numeric": {
            **{f"{new}_differs_{other}": f"{differ[f'{new}_differs_{other}']}/{n}"
               for new in new_conds for other in conditions if other != new},
            "note": "disambiguation, not a pass/fail bar; see qualitative verdict",
        },
        "marker_scan": {
            "per_condition": marker_hits,
            "total_hits_all_conditions": marker_total,
        },
        "sanity": {"per_condition": totals, "pass": sanity_ok},
        "qualitative": qualitative,
        "rows": rows,
    }

    # compact oracle/soft vs S0/random comparison, the crux of the disambiguation
    verdict["crux"] = {
        "oracle_equals_random": f"{sum(1 for i in range(n) if rec_of(i,'oracle')['ids']==rec_of(i,'random')['ids'])}/{n}",
        "oracle_equals_selected": f"{sum(1 for i in range(n) if rec_of(i,'oracle')['ids']==rec_of(i,'selected')['ids'])}/{n}",
    }
    if "soft" in new_conds:
        verdict["crux"].update({
            "soft_equals_random": f"{sum(1 for i in range(n) if rec_of(i,'soft')['ids']==rec_of(i,'random')['ids'])}/{n}",
            "soft_equals_selected": f"{sum(1 for i in range(n) if rec_of(i,'soft')['ids']==rec_of(i,'selected')['ids'])}/{n}",
            "soft_equals_oracle": f"{sum(1 for i in range(n) if rec_of(i,'soft')['ids']==rec_of(i,'oracle')['ids'])}/{n}",
        })

    out_text = json.dumps(verdict, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_text)
    print(out_text)
    print(f"sanity {'PASS' if sanity_ok else 'FAIL'}; marker_hits={marker_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
