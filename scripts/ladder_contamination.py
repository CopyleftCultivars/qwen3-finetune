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

import argparse
import os
import json
import re
from pathlib import Path

BENCH = Path(os.environ.get("LADDER_210_PATH", "data/ladder210.jsonl"))
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


def ladder_topic(item: dict) -> str | None:
    """Topic from a ladder source_document label.

    Ladder rows carry source_document like
    ``corpus:Natural Fertilizers Benchmark/animal_manure`` — a synthetic corpus
    label, not a document path. The topic is the trailing segment, which is the
    same taxonomy the training pool uses in its ``topic`` / ``_source_topic``
    field.
    """
    src = str(item.get("source_document", "")).strip()
    if not src:
        return None
    if "corpus:" in src and "/" in src:
        return src.rsplit("/", 1)[-1].strip() or None
    return None


def _joinable_topic_field(rows: list[tuple[str, dict]]) -> str | None:
    """Return the training row field that can join to a ladder topic, or None.

    A source check that joins on a field the rows do not carry is a silent no-op
    and must not read as a pass. This inspects the first row with any candidate
    field so the mismatch is detected instead of assumed.
    """
    for _path, row in rows:
        for field in ("topic", "_source_topic", "source_document", "source_doc"):
            if field in row:
                return field
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pool",
        action="append",
        default=None,
        metavar="JSONL",
        help="Training-pool JSONL(s) carrying a topic/source field (mcq_generation/"
             "output/*.jsonl, data/sources/qmd_mcq_generated.clean.jsonl). The source "
             "signal joins ladder source_document topics to these; without it the "
             "signal cannot run and the script fails loudly instead of reporting 0.",
    )
    args = parser.parse_args(argv)

    bench = [json.loads(line) for line in open(BENCH, encoding="utf-8")]
    print(f"ladder items: {len(bench)}")

    train_rows: list[tuple[str, dict]] = []
    for path in SPLITS:
        p = Path(path)
        if not p.exists():
            print(f"  MISSING split: {path}")
            continue
        rows = [json.loads(line) for line in open(p, encoding="utf-8")]
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

    # Source signal: join ladder source_document topic -> training-pool topic.
    # This must FAIL LOUDLY when it cannot run (the historical no-op returned
    # zero because the split rows are instruction/output with no source field
    # and the ladder label is a corpus path, so a silent zero read as a pass).
    pool_topics: dict[str, int] = {}
    if args.source_pool:
        for pool_path in args.source_pool:
            for line in open(pool_path, encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                topic = (
                    row.get("topic")
                    or row.get("_source_topic")
                    or (row.get("source_document", "").rsplit("/", 1)[-1] if row.get("source_document") else None)
                )
                if topic:
                    pool_topics[str(topic)] = pool_topics.get(str(topic), 0) + 1
        print(f"source pool: {len(args.source_pool)} file(s), {sum(pool_topics.values())} rows, "
              f"{len(pool_topics)} topics", flush=True)
    else:
        # No pool supplied: the source signal can only run if the SPLIT rows
        # themselves carry a joinable topic/source field. They do not.
        join_field = _joinable_topic_field(train_rows)
        if join_field is None:
            raise SystemExit(
                "FAILED LOUDLY: the source-document signal cannot run against the "
                f"training splits ({', '.join(SPLITS)}) because they are instruction/"
                "output rows with no topic/source field, and no --source-pool was "
                "passed. A source check that returns zero on a missing join key is a "
                "silent no-op. Pass --source-pool <pool.jsonl> (repeatable) or confirm "
                "you intend to skip the source signal."
            )
        pool_topics = {r.get(join_field) for _p, r in train_rows if r.get(join_field)}

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
        topic = ladder_topic(item)
        if topic and topic in pool_topics:
            source_hits.append((qid, topic))

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
