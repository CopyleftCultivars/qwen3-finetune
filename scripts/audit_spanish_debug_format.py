#!/usr/bin/env python3
"""Audit Spanish training records and exact train.py formatting for truncation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
LETTER_RE = re.compile(r"(?:</think>\s*)?([A-D])\s*$", re.DOTALL)


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def quantile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return float(ordered[index])


def length_stats(rows: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> dict[str, Any]:
    lengths: list[int] = []
    over_limit = 0
    for row in rows:
        text = CHAT_TEMPLATE.format(
            instruction=str(row["instruction"]).strip(),
            output=str(row["output"]).strip(),
        )
        token_count = len(tokenizer(text, truncation=False, padding=False)["input_ids"])
        lengths.append(token_count)
        over_limit += token_count > max_seq_len
    return {
        "count": len(rows),
        "mean_tokens": mean(lengths) if lengths else 0.0,
        "p50_tokens": quantile(lengths, 0.50),
        "p95_tokens": quantile(lengths, 0.95),
        "p99_tokens": quantile(lengths, 0.99),
        "max_tokens": max(lengths, default=0),
        "over_max_seq_len": over_limit,
        "over_max_seq_len_fraction": over_limit / len(rows) if rows else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mcq-train", required=True, type=Path)
    parser.add_argument("--mcq-validation", required=True, type=Path)
    parser.add_argument("--freeform-train", required=True, type=Path)
    parser.add_argument("--freeform-validation", required=True, type=Path)
    parser.add_argument("--variant", action="append", nargs=3, metavar=("NAME", "TRAIN", "VALIDATION"), default=[])
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    canonical = {
        "mcq_train": jsonl(args.mcq_train),
        "mcq_validation": jsonl(args.mcq_validation),
        "freeform_train": jsonl(args.freeform_train),
        "freeform_validation": jsonl(args.freeform_validation),
    }
    canonical_checks: dict[str, Any] = {}
    canonical_token_stats: dict[str, Any] = {}
    for name, rows in canonical.items():
        source_ids = [str(row.get("source_question_id", "")) for row in rows]
        messages_ok = all(
            isinstance(row.get("messages"), list)
            and len(row["messages"]) == 2
            and all(isinstance(message.get("content"), str) and message.get("content", "").strip() for message in row["messages"])
            for row in rows
        )
        spanish_labels = sum(row.get("language") == "es" for row in rows)
        mcq_answer_surface = None
        if "mcq" in name:
            mcq_answer_surface = sum(bool(LETTER_RE.search(row["messages"][1]["content"])) for row in rows)
        canonical_checks[name] = {
            "count": len(rows),
            "language_es_count": spanish_labels,
            "messages_shape_valid": messages_ok,
            "unique_source_ids": len(set(source_ids)),
            "answer_surface_valid_count": mcq_answer_surface,
        }
        canonical_token_stats[name] = length_stats(
            [
                {"instruction": row["messages"][0]["content"], "output": row["messages"][1]["content"]}
                for row in rows
                if isinstance(row.get("messages"), list) and len(row["messages"]) == 2
            ],
            tokenizer,
            args.max_seq_len,
        )
    train_ids = {str(row.get("source_question_id")) for key, rows in canonical.items() if key.endswith("train") for row in rows}
    validation_ids = {str(row.get("source_question_id")) for key, rows in canonical.items() if key.endswith("validation") for row in rows}

    variants: dict[str, Any] = {}
    for name, train_path, validation_path in args.variant:
        train_rows = jsonl(Path(train_path))
        validation_rows = jsonl(Path(validation_path))
        variants[name] = {
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "train": length_stats(train_rows, tokenizer, args.max_seq_len),
            "validation": length_stats(validation_rows, tokenizer, args.max_seq_len),
        }

    report = {
        "format": "spanish_training_format_audit_v1",
        "model": str(args.model),
        "max_seq_len": args.max_seq_len,
        "canonical": canonical_checks,
        "canonical_token_stats": canonical_token_stats,
        "canonical_train_validation_source_id_overlap": sorted(train_ids & validation_ids),
        "variants": variants,
        "conclusion": {
            "spanish_canonical_structure_pass": all(item["messages_shape_valid"] and item["language_es_count"] == item["count"] for item in canonical_checks.values()),
            "spanish_mcq_answer_surface_pass": all(
                item["answer_surface_valid_count"] == item["count"]
                for name, item in canonical_checks.items()
                if "mcq" in name
            ),
            "source_split_disjoint": not bool(train_ids & validation_ids),
            "truncation_requires_followup": any(
                split["over_max_seq_len"] > 0
                for variant in variants.values()
                for split in (variant["train"], variant["validation"])
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["conclusion"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
