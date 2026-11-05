#!/usr/bin/env python3
"""#191 contamination check: persona-memory corpus vs the sealed/held-out evals.

This project's #139 convention (scripts/ladder_contamination.py,
scripts/domain_tool_data_contamination.py, data/make_synthetic_traces.py's
``load_eval_hash_denylist``) is: before trusting any new training data, prove it
does not echo the benchmark. The Gemma4-era logs are older and from a different
pipeline than the Qwen3 training data, so contamination is unlikely -- but
unverified, so it is measured, not assumed.

Three signals, because each misses a different failure:
  * exact      -- normalized candidate text equals an eval item's text
                  (sha256 instruction_hash also reported for parity with the
                  pipeline-wide denylist).
  * containment-- every 8-gram of a (short) eval item appears in the (long)
                  candidate. This is what catches a ladder question or val_v3
                  instruction quoted verbatim inside a multi-turn conversation,
                  which symmetric Jaccard dilutes to near zero.
  * jaccard    -- 8-gram Jaccard >= 0.6 (light edits / near-duplicates).

Sources checked: the 210-item natural-fertilizers ladder, ``data/val_v3.jsonl``,
and ``eval_freeform/dev.jsonl``. The sealed ``eval_freeform/test.jsonl`` is
deliberately NOT opened -- same documented policy as data/make_synthetic_traces.py
and the eval_freeform package (touched once at final evaluation only).

Usage:
  # audit the real single-session seed material and held-out conversations
  python scripts/persona_memory_contamination.py \
      --candidates data/persona_memory/seeds_single_session.jsonl \
      --clean-output data/persona_memory/seeds_single_session.clean.jsonl

  # audit generated dialogues before chunking them
  python scripts/persona_memory_contamination.py \
      --candidates data/persona_memory/synthetic_multisession.jsonl \
      --clean-output data/persona_memory/synthetic_multisession.clean.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from persona_memory_common import (
    JACCARD_ALARM,
    LADDER_210,
    NGRAM_N,
    REPO_ROOT,
    SEALED_EVAL_FREEFORM_TEST,
    containment,
    flatten_text,
    instruction_hash,
    jaccard,
    normalized_flat,
    read_jsonl,
    shingles,
    write_jsonl,
)

DEFAULT_VAL_V3 = REPO_ROOT / "data" / "val_v3.jsonl"
DEFAULT_FREEFORM_DEV = REPO_ROOT / "eval_freeform" / "dev.jsonl"

MIN_SUBSTRING_CHARS = 20


@dataclass
class EvalItem:
    source: str
    item_id: str
    text: str
    normalized: str = field(init=False)
    shingle_set: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.normalized = normalized_flat(self.text)
        self.shingle_set = shingles(self.text)


def _item(source: str, item_id: str, text: str) -> EvalItem | None:
    if not text or not text.strip():
        return None
    return EvalItem(source=source, item_id=str(item_id), text=text)


def load_eval_sources(
    *,
    ladder_path: Path = LADDER_210,
    val_v3_path: Path = DEFAULT_VAL_V3,
    freeform_dev_path: Path = DEFAULT_FREEFORM_DEV,
) -> list[EvalItem]:
    items: list[EvalItem] = []

    if ladder_path.exists():
        for row in read_jsonl(ladder_path):
            text = " ".join(
                str(row.get(k, "")) for k in ("question", "A", "B", "C", "D")
            )
            item = _item("ladder_210", row.get("question_id", "?"), text)
            if item:
                items.append(item)

    if val_v3_path.exists():
        for index, row in enumerate(read_jsonl(val_v3_path)):
            text = " ".join(str(row.get(k, "")) for k in ("instruction", "output"))
            item = _item("val_v3", row.get("id", f"row{index}"), text)
            if item:
                items.append(item)

    if freeform_dev_path.exists():
        for row in read_jsonl(freeform_dev_path):
            text = " ".join(
                str(row.get(k, "")) for k in ("question", "reference_answer")
            )
            item = _item("eval_freeform_dev", row.get("id", "?"), text)
            if item:
                items.append(item)

    return items


class ContaminationIndex:
    """Inverted 8-gram index over the eval items.

    One pass over a candidate's shingles yields |candidate & eval| for every
    eval item, which is enough for exact/containment/Jaccard without an O(N*M)
    pairwise scan.
    """

    def __init__(self, items: Iterable[EvalItem], *, jaccard_alarm: float = JACCARD_ALARM):
        self.items: dict[str, EvalItem] = {}
        self.postings: dict[str, list[str]] = {}
        self.jaccard_alarm = jaccard_alarm
        self.normalized_to_key: dict[str, str] = {}
        for item in items:
            key = f"{item.source}:{item.item_id}"
            self.items[key] = item
            self.normalized_to_key.setdefault(item.normalized, key)
            for gram in item.shingle_set:
                self.postings.setdefault(gram, []).append(key)

    def classify(self, text: str) -> list[dict[str, Any]]:
        """All eval items this text contaminates (empty list = clean)."""
        candidate_norm = normalized_flat(text)
        candidate_shingles = shingles(text)
        if not candidate_norm:
            return []

        hits: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        exact_key = self.normalized_to_key.get(candidate_norm)
        if exact_key:
            item = self.items[exact_key]
            hits.append(
                {
                    "signal": "exact",
                    "eval_source": item.source,
                    "eval_id": item.item_id,
                    "jaccard": 1.0,
                    "containment_eval_in_candidate": 1.0,
                    "instruction_hash": instruction_hash(text),
                }
            )
            seen_keys.add(exact_key)

        matches: dict[str, int] = {}
        for gram in candidate_shingles:
            for key in self.postings.get(gram, ()):
                matches[key] = matches.get(key, 0) + 1

        for key, _overlap in matches.items():
            if key in seen_keys:
                continue
            item = self.items[key]
            jac = jaccard(candidate_shingles, item.shingle_set)
            cont_eval = containment(item.shingle_set, candidate_shingles)
            if jac >= self.jaccard_alarm or cont_eval >= self.jaccard_alarm:
                hits.append(
                    {
                        "signal": "ngram",
                        "eval_source": item.source,
                        "eval_id": item.item_id,
                        "jaccard": round(jac, 4),
                        "containment_eval_in_candidate": round(cont_eval, 4),
                        "instruction_hash": instruction_hash(text),
                    }
                )
                seen_keys.add(key)

        # Verbatim substring containment. This is what catches an eval item
        # quoted inside a long turn when the item is shorter than NGRAM_N words
        # (shingles() then yields one phrase that no candidate N-gram matches).
        candidate_is_substantial = len(candidate_norm) >= MIN_SUBSTRING_CHARS
        for key, item in self.items.items():
            if key in seen_keys:
                continue
            eval_in_candidate = (
                len(item.normalized) >= MIN_SUBSTRING_CHARS
                and item.normalized in candidate_norm
            )
            candidate_in_eval = (
                candidate_is_substantial and candidate_norm in item.normalized
            )
            if eval_in_candidate or candidate_in_eval:
                hits.append(
                    {
                        "signal": "substring",
                        "eval_source": item.source,
                        "eval_id": item.item_id,
                        "jaccard": round(jaccard(candidate_shingles, item.shingle_set), 4),
                        "containment_eval_in_candidate": round(
                            containment(item.shingle_set, candidate_shingles), 4
                        ),
                        "instruction_hash": instruction_hash(text),
                    }
                )
                seen_keys.add(key)

        hits.sort(key=lambda h: (-h["jaccard"], h["eval_source"]))
        return hits


def candidate_text(row: dict[str, Any]) -> str:
    """Best-effort text blob for any corpus row shape this pipeline emits."""
    if "sessions" in row and isinstance(row["sessions"], list):
        parts: list[str] = []
        for session in row["sessions"]:
            if isinstance(session, dict) and "turns" in session:
                for turn in session["turns"]:
                    parts.append(str(turn.get("text", "")))
            elif isinstance(session, str):
                parts.append(session)
        return " ".join(parts)
    if "turns" in row and isinstance(row["turns"], list):
        return " ".join(str(turn.get("text", "")) for turn in row["turns"])
    for key in ("target_text", "text", "prompt_text", "question", "instruction"):
        if row.get(key):
            return str(row[key])
    return flatten_text(row)


def check_rows(
    rows: list[dict[str, Any]],
    index: ContaminationIndex,
    *,
    text_fn: Callable[[dict[str, Any]], str] = candidate_text,
    label_fn: Callable[[dict[str, Any]], str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (clean_rows, contaminated_rows). Contaminated rows carry a
    ``contamination`` field with the offending eval items."""
    clean: list[dict[str, Any]] = []
    contaminated: list[dict[str, Any]] = []
    for row in rows:
        hits = index.classify(text_fn(row))
        if hits:
            annotated = dict(row)
            label = label_fn(row) if label_fn else row.get("conversation_id") or row.get("persona_id")
            annotated["contamination"] = {"label": label, "hits": hits}
            contaminated.append(annotated)
        else:
            clean.append(row)
    return clean, contaminated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--candidates",
        action="append",
        required=True,
        metavar="JSONL",
        help="Corpus JSONL(s) to check (repeatable).",
    )
    parser.add_argument(
        "--report",
        default="results/persona_memory_contamination.json",
        help="Report JSON output path.",
    )
    parser.add_argument(
        "--clean-output",
        default=None,
        help="Optional path; when set, writes only non-contaminated rows.",
    )
    parser.add_argument(
        "--jaccard-alarm",
        type=float,
        default=JACCARD_ALARM,
        help="Overlap alarm threshold (8-gram Jaccard / eval containment).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if SEALED_EVAL_FREEFORM_TEST.exists():
        print(f"note: sealed {SEALED_EVAL_FREEFORM_TEST.name} deliberately NOT opened")

    items = load_eval_sources()
    index = ContaminationIndex(items, jaccard_alarm=args.jaccard_alarm)
    by_source: dict[str, int] = {}
    for item in items:
        by_source[item.source] = by_source.get(item.source, 0) + 1
    print(f"eval items indexed: {len(items)} {by_source}")

    all_clean: list[dict[str, Any]] = []
    all_contaminated: list[dict[str, Any]] = []
    per_candidate: list[dict[str, Any]] = []
    for candidate_path in args.candidates:
        rows = read_jsonl(Path(candidate_path))
        clean, contaminated = check_rows(rows, index)
        all_clean.extend(clean)
        all_contaminated.extend(contaminated)
        per_candidate.append(
            {
                "path": candidate_path,
                "rows": len(rows),
                "clean": len(clean),
                "contaminated": len(contaminated),
            }
        )
        print(
            f"  {candidate_path}: {len(rows)} rows, "
            f"{len(clean)} clean, {len(contaminated)} contaminated"
        )
        if args.verbose:
            for row in contaminated[:5]:
                print(f"    hit: {row['contamination']}")

    report = {
        "format": "persona_memory_contamination_v1",
        "issue": 191,
        "ngram_n": NGRAM_N,
        "jaccard_alarm": args.jaccard_alarm,
        "sealed_eval_freeform_test_opened": False,
        "eval_sources": by_source,
        "eval_items_total": len(items),
        "candidates": per_candidate,
        "n_clean": len(all_clean),
        "n_contaminated": len(all_contaminated),
        "contaminated": all_contaminated,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {report_path}")
    if args.clean_output:
        n = write_jsonl(Path(args.clean_output), all_clean)
        print(f"wrote {args.clean_output} ({n} clean rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
