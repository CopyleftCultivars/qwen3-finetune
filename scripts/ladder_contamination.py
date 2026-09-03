"""F2 (#139): contamination check, 210-item ladder vs the training/val splits.

Plan A.2, amended to the 210-item benchmark. Runs BEFORE any score is trusted:
if the champion trained on ladder questions, its ladder number is not a
measurement of anything.

Three independent signals, because each misses a different failure:
  * exact  -- normalized question text identical (catches verbatim reuse)
  * ngram  -- Jaccard over 8-gram shingles >= threshold (catches light edits)
  * source -- the ladder row's source_document appearing in a training row
              (catches questions written from the same passage)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BENCH = Path("[LOCAL_PATH]"
             "natural_fertilizers/natural_fertilizers_benchmark_v1_graded.jsonl")
SPLITS = ["data/train_v3.jsonl", "data/train_v6.jsonl", "data/val_v3.jsonl"]
NGRAM_N = 8
JACCARD_ALARM = 0.6


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower())


def shingles(text: str, n: int = NGRAM_N) -> set[str]:
    words = norm(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def split_text(row: dict) -> str:
    """Splits are instruction/output; MCQ rows carry the stem in `instruction`."""
    return " ".join(str(row.get(k, "")) for k in ("instruction", "question", "output"))


def main() -> int:
    bench = [json.loads(l) for l in open(BENCH, encoding="utf-8")]
    print(f"ladder items: {len(bench)}")

    train_rows: list[tuple[str, dict]] = []
    for path in SPLITS:
        p = Path(path)
        if not p.exists():
            print(f"  MISSING split: {path}")
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        train_rows += [(path, r) for r in rows]
        print(f"  {path}: {len(rows)} rows")

    train_norm = {}
    train_shingles = []
    for path, r in train_rows:
        t = split_text(r)
        train_norm.setdefault(" ".join(norm(t).split()), (path, r))
        train_shingles.append((path, shingles(t)))

    train_blob = " ".join(" ".join(norm(split_text(r)).split()) for _, r in train_rows)

    exact, near, source_hits = [], [], []
    for item in bench:
        qid = item.get("question_id")
        q = str(item.get("question", ""))
        qn = " ".join(norm(q).split())
        if qn and qn in train_norm:
            exact.append((qid, train_norm[qn][0]))
            continue
        if qn and qn in train_blob:
            exact.append((qid, "substring"))
            continue
        qs = shingles(q)
        best = 0.0
        best_path = ""
        for path, ts in train_shingles:
            j = jaccard(qs, ts)
            if j > best:
                best, best_path = j, path
        if best >= JACCARD_ALARM:
            near.append((qid, round(best, 3), best_path))
        src = str(item.get("source_document", "")).strip()
        if src and src.lower() in train_blob:
            source_hits.append((qid, src))

    print(f"\nexact matches:        {len(exact)}")
    print(f"near matches (J>={JACCARD_ALARM}): {len(near)}")
    print(f"source_document hits: {len(source_hits)}")
    for label, rows in (("exact", exact), ("near", near), ("source", source_hits)):
        for r in rows[:5]:
            print(f"   {label}: {r}")

    contaminated = sorted({r[0] for r in exact} | {r[0] for r in near})
    out = {
        "benchmark": str(BENCH),
        "n_items": len(bench),
        "splits": SPLITS,
        "ngram_n": NGRAM_N,
        "jaccard_alarm": JACCARD_ALARM,
        "n_exact": len(exact),
        "n_near": len(near),
        "n_source_document_hits": len(source_hits),
        "contaminated_question_ids": contaminated,
        "exact": exact, "near": near, "source_document_hits": source_hits,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/ladder_210_contamination.json").write_text(json.dumps(out, indent=2))
    print(f"\ncontaminated ids: {len(contaminated)}")
    print("wrote results/ladder_210_contamination.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
