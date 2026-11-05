#!/usr/bin/env python3
"""#239 G3: paired comparison of two ladder runs (McNemar + paired bootstrap).

Pure functions (unit-tested in tests/test_g3_compare.py) plus a CLI that
compares two per-item ladder JSONL files on their shared question_ids.

  python3 scripts/g3_compare.py --a results/g3_ladder210_X.jsonl \
      --b results/g3_ladder210_Y.jsonl --label-a X --label-b Y
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load_run(path: str | Path) -> dict[str, bool]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["question_id"]] = bool(rec["correct"])
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. Returns 1.0 when b + c == 0."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(
    a: list[bool], b: list[bool], reps: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """95% percentile CI for mean(b) - mean(a) over paired resamples."""
    rng = random.Random(seed)
    n = len(a)
    diffs = [int(x) - int(y) for x, y in zip(b, a)]
    stats = []
    for _ in range(reps):
        stats.append(sum(rng.choice(diffs) for _ in range(n)) / n)
    stats.sort()
    return (stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1])


def compare(a: dict[str, bool], b: dict[str, bool]) -> dict:
    shared = sorted(set(a) & set(b))
    av = [a[q] for q in shared]
    bv = [b[q] for q in shared]
    b_only = sum(1 for x, y in zip(av, bv) if y and not x)
    a_only = sum(1 for x, y in zip(av, bv) if x and not y)
    acc_a = sum(av) / len(av)
    acc_b = sum(bv) / len(bv)
    lo, hi = paired_bootstrap_ci(av, bv)
    return {
        "n": len(shared),
        "acc_a": round(acc_a, 4),
        "acc_b": round(acc_b, 4),
        "delta_pp": round((acc_b - acc_a) * 100, 2),
        "b_only": b_only,
        "a_only": a_only,
        "mcnemar_p": mcnemar_exact(b_only, a_only),
        "bootstrap95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args(argv)
    res = compare(load_run(args.a), load_run(args.b))
    res["label_a"] = args.label_a
    res["label_b"] = args.label_b
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
