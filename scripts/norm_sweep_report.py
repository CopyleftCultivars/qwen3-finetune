#!/usr/bin/env python3
"""#222 independent comparator/report for the L1 post-transform norm sweep.

Reads the ``D203_NORM_SWEEP_*.json`` artifacts produced by
``scripts/lpm_quant_gap.py l1_sweep`` and independently re-derives the
aggregate table and the verdict, with structural invariants asserted. It does
NOT import the sweep code and does NOT re-fit anything: it only re-reads the
committed JSON output, so re-running it is a genuine check that the reported
numbers follow from the artifacts.

Usage:
    python scripts/norm_sweep_report.py

Exit code 0 if every invariant holds, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# (budget, run) -> the two disjoint K=12 slice artifacts (offset 0, offset 12)
ARTIFACTS: dict[tuple[str, str], list[str]] = {
    ("default", "run1"): ["D203_NORM_SWEEP_offset0.json", "D203_NORM_SWEEP_offset12.json"],
    ("default", "rerun"): ["D203_NORM_SWEEP_offset0_rerun.json", "D203_NORM_SWEEP_offset12_rerun.json"],
    ("m193", "run1"): ["D203_NORM_SWEEP_offset0_m193fit.json", "D203_NORM_SWEEP_offset12_m193fit.json"],
    ("m193", "rerun"): ["D203_NORM_SWEEP_offset0_m193fit_rerun.json", "D203_NORM_SWEEP_offset12_m193fit_rerun.json"],
}

CHANCE = 1.0 / 12.0            # per K=12 slice
SAFE_ZONE = 4.0               # M1 #186 boundary
NOMEM_FLOOR = 4.16            # reserved-arm no_memory_loss when rows ~ 0 (cap<=1)


def load(byoff: dict[int, dict]) -> dict[int, dict]:
    return byoff


def read() -> dict[tuple[str, str], dict[int, dict]]:
    out: dict[tuple[str, str], dict[int, dict]] = {}
    for key, names in ARTIFACTS.items():
        byoff: dict[int, dict] = {}
        for name in names:
            path = RESULTS / name
            if not path.exists():
                raise SystemExit(f"missing artifact: {path}")
            d = json.loads(path.read_text())
            byoff[d["persona_offset"]] = d
        out[key] = byoff
    return out


def aggregate(byoff: dict[int, dict], cap: float | None, path: str) -> tuple[int, float, float] | None:
    """Return (correct_count over 24, mean own_ntp, mean no_memory) for a cap."""
    rec, ntp, nom, found = 0, 0.0, 0.0, False
    for d in byoff.values():
        for e in d["sweep"]:
            if e["cap"] == cap:
                rec += round(e[path]["recall_at1"] * 12)
                ntp += e[path]["own_ntp_loss_mean"]
                nom += e[path]["no_memory_loss_mean"]
                found = True
    if not found:
        return None
    return rec, ntp / 2, nom / 2


def main() -> int:
    runs = read()
    failures: list[str] = []
    lines: list[str] = []

    # ---- structural invariants -------------------------------------------------
    for (budget, run), byoff in runs.items():
        for off, d in byoff.items():
            for e in d["sweep"]:
                cap, nmean, nmax = e["cap"], e["norm_mean"], e["norm_max"]
                if cap is not None:
                    if nmax > cap + 1e-3 or nmean > cap + 1e-3:
                        failures.append(f"{budget}/{run}/off{off} cap={cap}: norm {nmean}/{nmax} exceeds cap")
                else:
                    if not (30.0 <= nmean <= 45.0):
                        failures.append(f"{budget}/{run}/off{off}: unclamped norm_mean {nmean} outside #216 range")
            # contamination must fall monotonically as the cap shrinks
            ordered = [e for e in d["sweep"] if e["cap"] is not None]
            ordered.sort(key=lambda e: e["cap"], reverse=True)
            noms = [e["reserved"]["no_memory_loss_mean"] for e in ordered]
            for a, b in zip(noms, noms[1:]):
                if b > a + 1.0:
                    failures.append(f"{budget}/{run}/off{off}: reserved no_memory rose {a}->{b} as cap shrank")
            # unclamped reserved must reproduce #216's catastrophic band
            unc = next(e for e in d["sweep"] if e["cap"] is None)
            if not (350.0 <= unc["reserved"]["own_ntp_loss_mean"] <= 450.0):
                failures.append(
                    f"{budget}/{run}/off{off}: unclamped reserved own_ntp {unc['reserved']['own_ntp_loss_mean']} "
                    "not in #216's 350-450 catastrophic band")
            if not (0.08 <= unc["dynamic"]["recall_at1"] <= 0.45):
                failures.append(
                    f"{budget}/{run}/off{off}: unclamped dynamic recall {unc['dynamic']['recall_at1']} outside #216 band")

    # ---- aggregate table -------------------------------------------------------
    for budget in ("default", "m193"):
        lines.append("")
        lines.append(f"### budget {budget} (aggregate over offset 0 + 12; 24 evals; chance 0.0833)")
        lines.append("")
        lines.append("| cap | dynamic recall@1 | dynamic own NTP | reserved recall@1 | reserved own NTP | reserved no-mem |")
        lines.append("|---|---|---|---|---|---|")
        caps = [e["cap"] for e in runs[(budget, "run1")][0]["sweep"]]
        for cap in caps:
            cells = []
            for run in ("run1", "rerun"):
                a = aggregate(runs[(budget, run)], cap, "dynamic")
                b = aggregate(runs[(budget, run)], cap, "reserved")
                cells.append((a, b))
            r1a, r1b = cells[0]
            r2a, r2b = cells[1]
            lines.append(
                f"| {cap} | {r1a[0]/24:.4f} ({r1a[0]}/24) / {r2a[0]/24:.4f} ({r2a[0]}/24) "
                f"| {r1a[1]:.3f} / {r2a[1]:.3f} "
                f"| {r1b[0]/24:.4f} ({r1b[0]}/24) / {r2b[0]/24:.4f} ({r2b[0]}/24) "
                f"| {r1b[1]:.2f} / {r2b[1]:.2f} "
                f"| {r1b[2]:.2f} / {r2b[2]:.2f} |")

    # ---- viability scan --------------------------------------------------------
    # A cap is viable only if, in ALL four runs, reserved recall is clearly above
    # chance (>= chance + 1/12) AND reserved no-memory is not contaminated
    # (<= 2x the no-memory floor). Both must hold on the same cap.
    viable: list[float | None] = []
    caps = [e["cap"] for e in runs[("default", "run1")][0]["sweep"]]
    for cap in caps:
        ok = True
        for key in runs:
            a = aggregate(runs[key], cap, "dynamic")
            b = aggregate(runs[key], cap, "reserved")
            if a is None or b is None:
                ok = False
                break
            if b[0] / 24 < CHANCE + 1.0 / 12.0 or b[2] > 2 * NOMEM_FLOOR:
                ok = False
                break
        if ok:
            viable.append(cap)

    lines.append("")
    lines.append(f"viable caps (reserved recall >= chance+1/12 in all 4 runs AND reserved no-mem <= 2x floor): {viable or 'NONE'}")

    # decisive numbers at the M1 boundary
    lines.append("")
    lines.append("M1 boundary (cap=4.0):")
    for key in runs:
        a = aggregate(runs[key], SAFE_ZONE, "dynamic")
        b = aggregate(runs[key], SAFE_ZONE, "reserved")
        lines.append(
            f"  {key[0]}/{key[1]}: dynamic recall {a[0]}/24 own_ntp {a[1]:.3f} | "
            f"reserved recall {b[0]}/24 own_ntp {b[1]:.2f} no_mem {b[2]:.2f} (floor ~{NOMEM_FLOOR})")
        if b[0] / 24 > CHANCE + 1.0 / 12.0:
            failures.append(f"{key}: reserved recall at cap=4.0 above chance+1/12 ({b[0]}/24)")
        if b[2] < 3 * NOMEM_FLOOR:
            failures.append(f"{key}: reserved no_mem at cap=4.0 only {b[2]:.2f}, not clearly contaminated")

    if viable:
        failures.append(f"a viable cap was found: {viable} -- verdict must be re-examined")

    print("\n".join(lines))
    print()
    if failures:
        print("INVARIANTS FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL INVARIANTS HOLD: no viable reserved-row cap; Design A (inputs_embeds) mandatory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
