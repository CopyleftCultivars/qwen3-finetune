#!/usr/bin/env python3
"""#208: host-side comparator for the #206 S3-S4 raw engine output.

scripts/cactus_engine_s3s4.py runs inside the arm64 container and emits raw
parity_*.json (per-prompt greedy token-id sequences) and leak_*.json (per-run
token/reserved counts). This script turns those into the committed verdict files:

    results/F179_M3_S3S4_PARITY_VERDICT.json
    results/F179_M3_S3S4_LEAK_VERDICT.json
    results/F179_M3_S3S4_PATCHED_LEAK.json

It is the reproducibility half of #206 that was originally run ad hoc; running it
against outputs/r179-m1/s3s4/ reproduces the committed verdicts field-for-field.
Host-side only, stdlib only.

Parity bucketing rules (reverse-engineered from the committed verdicts):
  * mode "full"     : per prompt, compared += len(control ids); mismatch += 1 iff
                      the control and test id lists differ (per-prompt, not
                      per-position); control_reserved_positions stays 0.
  * mode "reserved" : per prompt, walk positions in lockstep. A non-reserved
                      control token -> compared += 1, and mismatch += 1 if the
                      test token differs. The FIRST reserved control token ->
                      control_reserved_positions += 1 and stop that prompt
                      (prefixes diverge past a reserved token). So
                      control_reserved_positions is a per-prompt count (at most
                      1 per prompt, by the break), not a per-position count.

Usage:
    python scripts/cactus_engine_s3s4_verdict.py all
    python scripts/cactus_engine_s3s4_verdict.py parity --out /tmp/p.json
    python scripts/cactus_engine_s3s4_verdict.py leak
    python scripts/cactus_engine_s3s4_verdict.py pair \
        --control outputs/r179-m1/s3s4/vanilla_baseline.json \
        --test    outputs/r179-m1/s3s4/patched_baselinesuppress.json --mode reserved
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cactus_engine_s3s4 import RESERVED_LO, RESERVED_HI  # noqa: E402  (single source of truth)

DEFAULT_DIR = REPO_ROOT / "outputs" / "r179-m1" / "s3s4"

# (bucket name, control file stem, test file stem, mode)
PARITY_COMPARISONS = [
    ("A_inertness_full_equality", "vanilla_p193deploy", "patched_p193deploy", "full"),
    ("B_denominator_baseline", "vanilla_baseline", "patched_baselinesuppress", "reserved"),
    ("C_active_token1", "vanilla_p193suppress", "patched_p193suppress", "reserved"),
    ("D_literal", "vanilla_p193deploy", "patched_baselinesuppress", "reserved"),
]

ACCEPTANCE = "0 reserved over 512x16=8192 tokens"


def is_reserved(token_id: int) -> bool:
    return RESERVED_LO <= token_id <= RESERVED_HI


def load_parity(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "per_prompt" not in data:
        raise SystemExit(f"not a parity file (no per_prompt): {path}")
    return data


def compare_full(control: dict, test: dict) -> dict:
    compared = mismatch = 0
    details: list = []
    for cp, tp in zip(control["per_prompt"], test["per_prompt"]):
        cids, tids = cp["ids"], tp["ids"]
        compared += len(cids)
        if cids != tids:
            mismatch += 1
            details.append({"prompt": cp["prompt"], "control": cids, "test": tids})
    return {"compared": compared, "mismatch": mismatch, "control_reserved_positions": 0, "details": details[:5]}


def compare_reserved(control: dict, test: dict) -> dict:
    compared = mismatch = control_reserved = 0
    details: list = []
    for cp, tp in zip(control["per_prompt"], test["per_prompt"]):
        for pos, (a, b) in enumerate(zip(cp["ids"], tp["ids"])):
            if is_reserved(a):
                control_reserved += 1
                break  # prefixes diverge past a reserved control token
            compared += 1
            if a != b:
                mismatch += 1
                details.append({"prompt": cp["prompt"], "pos": pos, "control": a, "test": b})
    return {
        "compared": compared,
        "mismatch": mismatch,
        "control_reserved_positions": control_reserved,
        "details": details[:5],
    }


def compare(control: dict, test: dict, mode: str) -> dict:
    if mode == "full":
        return compare_full(control, test)
    if mode == "reserved":
        return compare_reserved(control, test)
    raise SystemExit(f"unknown mode: {mode!r}")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Match the committed verdict files: indent=2, no trailing newline.
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def build_parity_verdict(data_dir: Path) -> dict:
    verdict: dict = {}
    for bucket, control_stem, test_stem, mode in PARITY_COMPARISONS:
        control = load_parity(data_dir / f"parity_{control_stem}.json")
        test = load_parity(data_dir / f"parity_{test_stem}.json")
        verdict[bucket] = compare(control, test, mode)
    return verdict


def _acc(agg: dict, field: str, value: int) -> None:
    agg[field] = agg.get(field, 0) + value


def sum_shards(paths: list[Path]) -> tuple[dict, dict]:
    """Sum per-shard leak aggregates: (totals, by_variant)."""
    totals = {"generations": 0, "tokens": 0, "reserved": 0}
    by_variant: dict[str, dict] = {}
    for path in sorted(paths):
        d = json.loads(path.read_text(encoding="utf-8"))
        _acc(totals, "generations", d["generations"])
        _acc(totals, "tokens", d["token_total"])
        _acc(totals, "reserved", d["reserved_total"])
        for variant, row in d.get("by_variant", {}).items():
            slot = by_variant.setdefault(variant, {"generations": 0, "tokens": 0, "reserved": 0})
            slot["generations"] += row["generations"]
            slot["tokens"] += row["tokens"]
            slot["reserved"] += row["reserved"]
    return totals, by_variant


def build_leak_verdicts(data_dir: Path) -> tuple[dict, dict]:
    c1 = json.loads((data_dir / "leak_c1_48.json").read_text(encoding="utf-8"))
    vanilla = json.loads((data_dir / "leak_vanilla_48.json").read_text(encoding="utf-8"))

    base_totals, _ = sum_shards(sorted(data_dir.glob("leak_patched_s*.json")))
    if not sum(base_totals.values()):
        raise SystemExit(f"no leak_patched_s*.json shards found in {data_dir}")

    combined_path = data_dir / "leak_patched_combined.json"
    if combined_path.exists():
        patched_combined = json.loads(combined_path.read_text(encoding="utf-8"))
    else:
        # Reduce base + extension shards ourselves (per #208: either is acceptable).
        ext_totals, ext_by_variant = sum_shards(sorted(data_dir.glob("leak_ext_*.json")))
        totals = {k: base_totals[k] + ext_totals[k] for k in base_totals}
        _, base_by_variant = sum_shards(sorted(data_dir.glob("leak_patched_s*.json")))
        merged: dict[str, dict] = {}
        for source in (base_by_variant, ext_by_variant):
            for variant, row in source.items():
                slot = merged.setdefault(variant, {"generations": 0, "tokens": 0, "reserved": 0})
                for key in ("generations", "tokens", "reserved"):
                    slot[key] += row[key]
        patched_combined = {
            "generations": totals["generations"],
            "tokens": totals["tokens"],
            "reserved": totals["reserved"],
            "by_variant": {k: merged[k] for k in sorted(merged)},
            "token_target_met": totals["tokens"] >= 512 * 16,
        }

    leak_verdict = {
        "patched": {
            "generations": base_totals["generations"],
            "tokens": base_totals["tokens"],
            "reserved": base_totals["reserved"],
        },
        "c1_48": {
            "generations": c1["generations"],
            "tokens": c1["token_total"],
            "reserved": c1["reserved_total"],
        },
        "vanilla_48": {
            "generations": vanilla["generations"],
            "tokens": vanilla["token_total"],
            "reserved": vanilla["reserved_total"],
        },
        "acceptance": ACCEPTANCE,
    }
    patched_leak = {
        "generations": patched_combined["generations"],
        "tokens": patched_combined["tokens"],
        "reserved": patched_combined["reserved"],
        "by_variant": patched_combined["by_variant"],
        "token_target_met": patched_combined["token_target_met"],
    }
    return leak_verdict, patched_leak


def cmd_parity(args: argparse.Namespace) -> None:
    verdict = build_parity_verdict(args.dir)
    out = args.out or (args.dir / "parity_verdict.json")
    write_json(out, verdict)
    print(json.dumps(verdict, indent=2))
    print(f"wrote {out}")


def cmd_pair(args: argparse.Namespace) -> None:
    control = load_parity(args.control)
    test = load_parity(args.test)
    result = compare(control, test, args.mode)
    if args.out:
        write_json(args.out, result)
        print(f"wrote {args.out}")
    print(json.dumps(result, indent=2))


def cmd_leak(args: argparse.Namespace) -> None:
    leak_verdict, patched_leak = build_leak_verdicts(args.dir)
    leak_out = args.leak_out or (args.dir / "leak_verdict.json")
    patched_out = args.patched_out or (args.dir / "patched_leak.json")
    write_json(leak_out, leak_verdict)
    write_json(patched_out, patched_leak)
    print(json.dumps(leak_verdict, indent=2))
    print(json.dumps(patched_leak, indent=2))
    print(f"wrote {leak_out} and {patched_out}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_parity(args)
    cmd_leak(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("parity", help="Reproduce PARITY_VERDICT.json from parity_*.json.")
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_parity)

    pair = sub.add_parser("pair", help="Generic single comparison of two parity files.")
    pair.add_argument("--control", type=Path, required=True)
    pair.add_argument("--test", type=Path, required=True)
    pair.add_argument("--mode", choices=("full", "reserved"), default="reserved")
    pair.add_argument("--out", type=Path, default=None)
    pair.set_defaults(func=cmd_pair)

    lk = sub.add_parser("leak", help="Reproduce LEAK_VERDICT.json + PATCHED_LEAK.json.")
    lk.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    lk.add_argument("--leak-out", type=Path, default=None)
    lk.add_argument("--patched-out", type=Path, default=None)
    lk.set_defaults(func=cmd_leak)

    allp = sub.add_parser("all", help="parity + leak with default paths.")
    allp.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    allp.add_argument("--out", type=Path, default=None)
    allp.add_argument("--leak-out", type=Path, default=None)
    allp.add_argument("--patched-out", type=Path, default=None)
    allp.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
