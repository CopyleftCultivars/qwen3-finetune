"""Corpus assessment for the RAG arm (L9 / #132).

LOCAL_RLM_SLICE_PLAN.md sec.3.3.1 asks for "a small agronomy corpus (source to
be confirmed)" before the RAG arm is buildable. The QMD collections are that
corpus. This script decides whether they are *usable*, which is a different
question, and answers three things the sec.3.3.5 measurement depends on:

1. Does retrieval surface answer-bearing text at all? Measured as the lexical
   overlap between retrieved snippets and the GOLD option, against the same
   overlap for the three distractors. Gold >> distractors means retrieval is
   pulling text that discriminates the answer; gold ~= distractors means it is
   pulling topical filler that will not move the Pass-2 score.

2. How badly do the collections duplicate each other? The same source document
   is indexed under several collections (edible-forest-gardens appears in
   Permaculture, Agroecology and Integrated_Farming_Systems with an identical
   QMD docid), so an undeduplicated top-k can be three copies of one passage.

3. Which retrieval mode is affordable. `qmd query` runs LLM expansion and
   reranking; using it would inject a *larger* model's reasoning into an arm
   whose entire premise is a 0.6B running on a phone, and would confound the
   comparison against v5clean. `qmd search` is BM25 with no LLM.

Provenance caveat this script cannot resolve on its own: the collections are
rooted at benchmark-workspace/organized/, the same tree the MCQ benchmark was
generated from. Retrieval hitting the passage a question was written from is
optimistic relative to an unseen question, though it is also the real deployment
configuration for the app. Section 3.3.5's "measure the RAG arm separately" is
what keeps that honest -- and the gold-vs-distractor margin below is the number
that says how much of any gain is retrieval doing real work.

Usage:
  python -m rlm_slice.corpus_probe --limit 30 --mode search
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rlm_slice.guards import check_not_sealed

REPO_ROOT = Path(__file__).resolve().parent.parent

# Words too common in agronomy prose to carry discriminative signal.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on",
    "with", "as", "by", "that", "this", "it", "be", "from", "at", "which",
    "can", "will", "not", "all", "more", "than", "into", "their", "its", "has",
    "have", "was", "were", "but", "they", "you", "we", "when", "what", "how",
    "soil", "plant", "plants", "system", "systems", "used", "use",
}


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]{3,}", str(text).lower())
    return {w for w in words if w not in _STOPWORDS}


def qmd(mode: str, query: str, n: int = 5, timeout: int = 120) -> list[dict[str, Any]]:
    """Run one QMD retrieval. Returns [] on any failure -- a probe must not crash a sweep."""
    try:
        proc = subprocess.run(
            ["qmd", mode, query, "--json", "-n", str(n)],
            capture_output=True, text=True, timeout=timeout,
        )
        return json.loads(proc.stdout or "[]")
    except Exception:  # noqa: BLE001
        return []


def overlap(snippets: str, option_text: str) -> float:
    """Fraction of an option's content words that appear in the retrieved text."""
    option_words = content_words(option_text)
    if not option_words:
        return 0.0
    return len(option_words & content_words(snippets)) / len(option_words)


def dedupe_hits(hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Keep the best-scoring hit per QMD docid, then take k.

    The same source document is indexed under several collections, so an
    undeduplicated top-5 averages only ~2 distinct documents. Deduping by
    docid is what makes k mean what it says.
    """
    best: dict[str, dict[str, Any]] = {}
    for hit in hits:
        docid = hit.get("docid") or hit.get("file", "")
        if docid not in best:
            best[docid] = hit
    return list(best.values())[:k]


def probe_item(item: dict[str, Any], mode: str, k: int, dedupe: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    if dedupe:
        # Over-fetch, then collapse duplicates, so k distinct documents survive.
        hits = dedupe_hits(qmd(mode, item["question"], n=k * 4), k)
    else:
        hits = qmd(mode, item["question"], n=k)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    snippets = "\n".join(h.get("snippet", "") for h in hits)
    docids = [h.get("docid") for h in hits if h.get("docid")]
    gold = item["answer"]
    gold_overlap = overlap(snippets, item[gold])
    distractors = [overlap(snippets, item[L]) for L in "ABCD" if L != gold]

    return {
        "question_id": item["question_id"],
        "n_hits": len(hits),
        "n_unique_docs": len(set(docids)),
        "duplicate_hits": len(docids) - len(set(docids)),
        "top_score": hits[0].get("score") if hits else None,
        "gold_overlap": round(gold_overlap, 4),
        "mean_distractor_overlap": round(statistics.mean(distractors), 4) if distractors else 0.0,
        "gold_margin": round(gold_overlap - (statistics.mean(distractors) if distractors else 0.0), 4),
        "latency_ms": round(elapsed_ms, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/val_v3.jsonl")
    parser.add_argument("--mode", default="search", choices=["search", "vsearch", "query"])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    check_not_sealed(args.dataset)  # sec.2.2, before any file IO
    sys.path.insert(0, str(REPO_ROOT / "autoresearch"))
    from score_mcq_ll import normalize_records  # noqa: E402

    rows = [json.loads(line) for line in open(args.dataset, encoding="utf-8")]
    records, _ = normalize_records(rows)
    records = records[: args.limit]

    results = [probe_item(item, args.mode, args.k, dedupe=args.dedupe) for item in records]
    scored = [r for r in results if r["n_hits"]]

    def mean(field: str) -> float:
        return round(statistics.mean([r[field] for r in scored]), 4) if scored else 0.0

    margins = [r["gold_margin"] for r in scored]
    summary = {
        "mode": args.mode,
        "k": args.k,
        "dedupe": args.dedupe,
        "n_items": len(results),
        "n_with_hits": len(scored),
        "mean_gold_overlap": mean("gold_overlap"),
        "mean_distractor_overlap": mean("mean_distractor_overlap"),
        "mean_gold_margin": mean("gold_margin"),
        "gold_margin_positive_fraction": (
            round(sum(1 for m in margins if m > 0) / len(margins), 4) if margins else 0.0
        ),
        "mean_duplicate_hits": mean("duplicate_hits"),
        "mean_latency_ms": mean("latency_ms"),
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        Path(args.output).write_text(
            json.dumps({"summary": summary, "per_item": results}, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
