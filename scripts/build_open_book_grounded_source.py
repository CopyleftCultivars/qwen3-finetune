#!/usr/bin/env python3
"""Build an isolated context-grounded open-book training source.

This is deliberately different from the earlier CROP-style ``open_book_qa``
auxiliary stream.  Each item contains three same-domain records: one record
that answers the question and two hard-negative records.  The target must
return both the answer and the source-record ID that supports it.  Documents
are assigned to train or validation before candidates and distractors are
selected, so no source record can cross the split boundary.

The source is the repository's English natural-farming RAG/structured corpus;
the generic CROP dataset and the Spanish benchmark are never inputs.  The
benchmark argument is an open English MCQ dev proxy used only for the textual
contamination audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_crop_auxiliary_datasets import (
    SOURCE_FILES,
    build_documents,
    canonical_json,
    digest_file,
    normalize,
    open_book_variants,
    render_record,
)


TASK = "open_book_grounded"
LETTERS = ("A", "B", "C", "D")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def stable_digest(*values: str) -> str:
    payload = "\0".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def digest_source_root(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path / filename for filename in SOURCE_FILES.values()):
        digest.update(child.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(digest_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def reject_eval_path(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    if "spanish-benchmark-workspace" in parts:
        raise ValueError("Spanish benchmark paths are evaluation-only")
    if "eval_freeform" in parts and path.name.casefold() == "test.jsonl":
        raise ValueError("sealed free-form test paths are forbidden")
    if "sealed" in path.name.casefold():
        raise ValueError("sealed benchmark paths are forbidden")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def benchmark_fields(path: Path) -> list[set[str]]:
    fields: list[set[str]] = []
    for index, row in enumerate(load_jsonl(path)):
        for field in ("question", *LETTERS):
            value = normalize(str(row.get(field, "")))
            if value:
                fields.append(set(value.split()))
    if not fields:
        raise ValueError(f"{path}: benchmark has no question/option text")
    return fields


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def assign_splits(
    documents: list[dict[str, Any]], *, seed: int, validation_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        by_kind[str(document["source_kind"])].append(document)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for kind, kind_documents in sorted(by_kind.items()):
        ordered = sorted(
            kind_documents,
            key=lambda document: stable_digest(str(seed), str(document["source_id"])),
        )
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation_ids = {
            document["source_id"] for document in ordered[:validation_count]
        }
        validation.extend(
            document for document in ordered if document["source_id"] in validation_ids
        )
        train.extend(
            document for document in ordered if document["source_id"] not in validation_ids
        )
    if not train or not validation:
        raise ValueError("source split produced an empty partition")
    return train, validation


def candidate_for(document: dict[str, Any], *, seed: int) -> dict[str, Any]:
    variants = open_book_variants(document)
    if not variants:
        raise ValueError(f"no grounded question variants for {document['source_id']}")
    index = int(stable_digest(str(seed), str(document["source_id"]))[:8], 16) % len(variants)
    variant, question, answer = variants[index]
    return {
        "document": document,
        "variant": variant,
        "question": question,
        "answer": answer,
    }


def select_documents(
    documents: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("selection count must be positive")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        groups[str(document["source_kind"])].append(document)
    for kind in groups:
        groups[kind].sort(
            key=lambda document: stable_digest(str(seed), kind, str(document["source_id"]))
        )

    selected: list[dict[str, Any]] = []
    kinds = sorted(groups)
    cursors = {kind: 0 for kind in kinds}
    while len(selected) < count:
        progressed = False
        for kind in kinds:
            if cursors[kind] >= len(groups[kind]):
                continue
            selected.append(groups[kind][cursors[kind]])
            cursors[kind] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"only {len(selected)} source documents available; need {count}")
    return selected


def negative_documents(
    positive: dict[str, Any], pool: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    candidates = [
        document
        for document in pool
        if document["source_id"] != positive["source_id"]
        and document["source_kind"] == positive["source_kind"]
    ]
    if len(candidates) < 2:
        candidates = [document for document in pool if document["source_id"] != positive["source_id"]]
    if len(candidates) < 2:
        raise ValueError("grounded context requires at least two negative source records")
    ordered = sorted(
        candidates,
        key=lambda document: stable_digest(str(seed), str(positive["source_id"]), str(document["source_id"])),
    )
    return ordered[:2]


def make_row(
    candidate: dict[str, Any], context_documents: list[dict[str, Any]], *, split: str
) -> dict[str, Any]:
    positive = candidate["document"]
    blocks = []
    for document in context_documents:
        blocks.append(
            f"SOURCE_ID: {document['source_id']}\n"
            f"SOURCE_KIND: {document['source_kind']}\n"
            f"REFERENCE RECORD:\n{render_record(document)}"
        )
    instruction = (
        "Answer the natural-farming question using only the reference records. "
        "One record contains the answer and the other records are distractors. "
        "Return exactly two lines: `Answer: <answer>` and "
        "`Evidence: <SOURCE_ID>`; if the answer is not stated, use `Answer: not stated` "
        "and cite the record that establishes that absence only when appropriate.\n\n"
        "REFERENCE RECORDS:\n\n"
        + "\n\n---\n\n".join(blocks)
        + f"\n\nQUESTION: {candidate['question']}"
    )
    output = f"Answer: {candidate['answer']}\nEvidence: {positive['source_id']}"
    return {
        "id": f"{TASK}:{positive['source_id']}:{candidate['variant']}",
        "instruction": instruction,
        "output": output,
        "language": "en",
        "grounded_task": TASK,
        "split": split,
        "source_id": positive["source_id"],
        "source_kind": positive["source_kind"],
        "evidence_source_id": positive["source_id"],
        "context_source_ids": [document["source_id"] for document in context_documents],
        "provenance": {
            "source_paths": [document["source_path"] for document in context_documents],
            "source_sha256": [document["source_sha256"] for document in context_documents],
            "source_record_sha256": [document["source_record_sha256"] for document in context_documents],
            "authoring_method": "deterministic_context_bundle_v1",
            "source_document_split": split,
            "source_document_independence": "not_claimed",
        },
    }


def audit_rows(
    rows: list[dict[str, Any]],
    benchmark_token_sets: list[set[str]],
    *,
    source_path: Path,
    benchmark_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    max_overlap = 0.0
    for index, row in enumerate(rows):
        reason: str | None = None
        row_id = str(row.get("id", ""))
        question = str(row.get("instruction", ""))
        question_tokens = set(normalize(question).split())
        if not row_id or row_id in seen_ids:
            reason = "duplicate_or_missing_id"
        elif row.get("language") != "en":
            reason = "unexpected_language"
        elif row.get("grounded_task") != TASK:
            reason = "wrong_task"
        elif row.get("split") not in {"train", "validation"}:
            reason = "invalid_split"
        elif len(row.get("context_source_ids", [])) != 3:
            reason = "context_must_have_three_sources"
        elif row.get("evidence_source_id") not in row.get("context_source_ids", []):
            reason = "evidence_not_in_context"
        else:
            for benchmark_tokens in benchmark_token_sets:
                overlap = jaccard(question_tokens, benchmark_tokens)
                max_overlap = max(max_overlap, overlap)
                if overlap >= 0.85:
                    reason = "high_lexical_benchmark_overlap"
                    break
        if reason:
            rejected.append({"index": index, "id": row_id, "reason": reason})
            continue
        clean_row = dict(row)
        clean_row.update(
            {
                "_contamination_status": "passed",
                "_contamination_audit": "open_book_grounded_benchmark_audit_v1",
                "_benchmark_question_text_overlap": "not_detected",
            }
        )
        clean.append(clean_row)
        seen_ids.add(row_id)
    report = {
        "format": "open_book_grounded_source_audit_v1",
        "source": str(source_path),
        "source_sha256": digest_source_root(source_path),
        "benchmark": str(benchmark_path),
        "benchmark_sha256": digest_file(benchmark_path),
        "candidate_count": len(rows),
        "clean_count": len(clean),
        "rejected_count": len(rejected),
        "maximum_instruction_benchmark_token_jaccard": round(max_overlap, 6),
        "rejected": rejected,
        "passed": not rejected,
        "sealed_test_paths_read": [],
        "notes": [
            "The audit checks the instruction/context surface against the open English MCQ dev proxy.",
            "Semantic independence from the benchmark's source documents is not claimed.",
            "The Spanish benchmark and all Spanish transfer files are evaluation-only and were not read.",
        ],
    }
    return clean, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return digest_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (args.source_root, args.benchmark):
        if not path.exists():
            parser.error(f"path not found: {path}")
        reject_eval_path(path)
    if args.train_count < 1 or args.validation_count < 1:
        parser.error("train and validation counts must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation fraction must be in (0, 1)")

    documents = build_documents(args.source_root.resolve())
    train_documents, validation_documents = assign_splits(
        documents, seed=args.seed, validation_fraction=args.validation_fraction
    )
    selected_train = select_documents(train_documents, count=args.train_count, seed=args.seed)
    selected_validation = select_documents(
        validation_documents, count=args.validation_count, seed=args.seed + 1
    )
    train_ids = {document["source_id"] for document in train_documents}
    validation_ids = {document["source_id"] for document in validation_documents}
    if train_ids & validation_ids:
        raise AssertionError("source document split overlap")

    train_rows = []
    for document in selected_train:
        candidate = candidate_for(document, seed=args.seed)
        context = [document, *negative_documents(document, train_documents, seed=args.seed)]
        context.sort(key=lambda value: stable_digest(str(args.seed), candidate["question"], str(value["source_id"])))
        train_rows.append(make_row(candidate, context, split="train"))
    validation_rows = []
    for document in selected_validation:
        candidate = candidate_for(document, seed=args.seed + 1)
        context = [document, *negative_documents(document, validation_documents, seed=args.seed + 1)]
        context.sort(key=lambda value: stable_digest(str(args.seed + 1), candidate["question"], str(value["source_id"])))
        validation_rows.append(make_row(candidate, context, split="validation"))

    benchmark_token_sets = benchmark_fields(args.benchmark.resolve())
    all_rows, audit = audit_rows(
        train_rows + validation_rows,
        benchmark_token_sets,
        source_path=args.source_root.resolve(),
        benchmark_path=args.benchmark.resolve(),
    )
    if audit["rejected_count"]:
        raise ValueError(f"contamination audit rejected {audit['rejected_count']} grounded rows")
    train_rows = [row for row in all_rows if row["split"] == "train"]
    validation_rows = [row for row in all_rows if row["split"] == "validation"]
    if len(train_rows) != args.train_count or len(validation_rows) != args.validation_count:
        raise AssertionError("audit changed the requested source counts")
    if {row["source_id"] for row in train_rows} & {row["source_id"] for row in validation_rows}:
        raise AssertionError("positive source IDs overlap across splits")
    if any(set(row["context_source_ids"]) & validation_ids for row in train_rows):
        raise AssertionError("training context contains a validation source document")
    if any(set(row["context_source_ids"]) & train_ids for row in validation_rows):
        raise AssertionError("validation context contains a training source document")

    output_root = args.output_root.resolve()
    train_path = output_root / "train.jsonl"
    validation_path = output_root / "validation.jsonl"
    train_sha = write_jsonl(train_path, train_rows)
    validation_sha = write_jsonl(validation_path, validation_rows)
    audit_path = output_root / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "format": "open_book_grounded_source_manifest_v1",
        "task": TASK,
        "language": "en",
        "training_allowed": True,
        "authoring_method": "deterministic_context_bundle_v1",
        "source_root": str(args.source_root.resolve()),
        "source_files": {
            kind: {
                "path": str((args.source_root / filename).resolve()),
                "sha256": digest_file((args.source_root / filename).resolve()),
            }
            for kind, filename in SOURCE_FILES.items()
        },
        "benchmark_audit": audit,
        "split": {
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "train_source_document_count": len(train_documents),
            "validation_source_document_count": len(validation_documents),
            "train_validation_source_overlap": [],
        },
        "train": {
            "path": str(train_path),
            "sha256": train_sha,
            "count": len(train_rows),
            "source_kind_counts": dict(sorted(Counter(row["source_kind"] for row in train_rows).items())),
        },
        "validation": {
            "path": str(validation_path),
            "sha256": validation_sha,
            "count": len(validation_rows),
            "source_kind_counts": dict(sorted(Counter(row["source_kind"] for row in validation_rows).items())),
        },
        "sealed_test_paths_read": [],
        "notes": [
            "Every context bundle has one answer-bearing record and two same-split hard negatives.",
            "No previous auxiliary, reflection, verification, distractor-quality, general-replay, or Spanish stream is included.",
            "This source tests context-conditioned answering; it is not a replacement for the closed-book MCQ primary.",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "task": TASK,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "train_source_documents": len(train_documents),
        "validation_source_documents": len(validation_documents),
        "source_kind_counts_train": manifest["train"]["source_kind_counts"],
        "source_kind_counts_validation": manifest["validation"]["source_kind_counts"],
        "maximum_instruction_benchmark_token_jaccard": audit["maximum_instruction_benchmark_token_jaccard"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
