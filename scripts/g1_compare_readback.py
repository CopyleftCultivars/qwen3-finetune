#!/usr/bin/env python3
"""#226 (S3) G1 independent comparator.

Re-derives the G1 storage verdict from the persisted readback dumps written by
``cactus-engine/tests/test_memory_prefix_g1.cpp`` (CACTUS_TEST_G1_DUMP_DIR) and
the source FP32 vector file, WITHOUT importing the engine test. This is the
independent re-run of the G1 comparator required by the epic's review process.

For each prompt<i>.readback / prompt<i>.meta pair:
  * decode the buffer bytes at the buffer's element size (2 = FP16, 4 = FP32)
  * compare to the source FP32 file (same shape)
  * assert cos >= 0.99999 and max_abs_delta <= an FP16 half-ULP bound

Usage:
  python scripts/g1_compare_readback.py --dump-dir /tmp/g1 \
      --prefix-file outputs/r179-m1/s2a/a1_memory_prefix_p0.f32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def decode(buf: bytes, elem_bytes: int) -> np.ndarray:
    if elem_bytes == 2:
        return np.frombuffer(buf, dtype="<f2").astype(np.float64)
    if elem_bytes == 4:
        return np.frombuffer(buf, dtype="<f4").astype(np.float64)
    raise SystemExit(f"unsupported elem_bytes {elem_bytes}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-dir", required=True)
    p.add_argument("--prefix-file", required=True)
    args = p.parse_args()

    src = np.fromfile(args.prefix_file, dtype="<f4").astype(np.float64)
    dumps = sorted(Path(args.dump_dir).glob("prompt*.readback"))
    if not dumps:
        raise SystemExit(f"no readback dumps in {args.dump_dir}")

    results = []
    ok_all = True
    for rb_path in dumps:
        meta = {}
        for line in Path(str(rb_path).replace(".readback", ".meta")).read_text().splitlines():
            k, v = line.split()
            meta[k] = int(v)
        buf = rb_path.read_bytes()
        got = decode(buf, meta["elem_bytes"])
        if got.size != src.size:
            raise SystemExit(f"{rb_path}: {got.size} elems vs source {src.size}")
        cos = float(np.dot(src, got) / (np.linalg.norm(src) * np.linalg.norm(got)))
        max_delta = float(np.max(np.abs(src - got)))
        max_abs = float(np.max(np.abs(src)))
        bound = max_abs * (2.0 ** -10.0) + 1e-9 if meta["elem_bytes"] == 2 else 1e-6
        passed = cos >= 0.99999 and max_delta <= bound
        ok_all = ok_all and passed
        results.append({
            "prompt": rb_path.name,
            "rows": meta["rows"],
            "dim": meta["dim"],
            "elem_bytes": meta["elem_bytes"],
            "cos": cos,
            "max_abs_delta": max_delta,
            "fp16_bound": bound,
            "pass": passed,
        })

    print(json.dumps({"issue": 226, "gate": "G1", "n_prompts": len(results),
                      "all_pass": ok_all, "results": results}, indent=2))
    print(f"\nG1 comparator: {'PASS' if ok_all else 'FAIL'} ({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
