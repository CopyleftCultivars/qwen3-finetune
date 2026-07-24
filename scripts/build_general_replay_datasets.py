#!/usr/bin/env python3
"""Compose v5clean with a token-matched general-instruction replay dose."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CHAT_TEMPLATE = (
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n{output}<|im_end|>"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render(row: dict[str, Any]) -> str:
    if "messages" in row:
        return "\n".join(
            f"<|im_start|>{turn['role']}\n{turn['content'].strip()}<|im_end|>"
            for turn in row["messages"]
        )
    return CHAT_TEMPLATE.format(instruction=str(row["instruction"]).strip(), output=str(row["output"]).strip())


def token_lengths(rows: list[dict[str, Any]], tokenizer) -> list[int]:
    return [len(tokenizer(render(row), add_special_tokens=False)["input_ids"]) for row in rows]


def choose(rows: list[dict[str, Any]], lengths: list[int], target: int) -> tuple[list[dict[str, Any]], int]:
    if target <= 0:
        return [], 0
    order = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(str(rows[index]["_source_id"]).encode()).hexdigest(),
    )
    selected: list[int] = []
    total = 0
    for index in order:
        length = lengths[index]
        if total + length <= target:
            selected.append(index)
            total += length
    remaining = [index for index in order if index not in set(selected)]
    if remaining:
        best = min(remaining, key=lambda index: abs(target - (total + lengths[index])))
        if abs(target - (total + lengths[best])) < abs(target - total):
            selected.append(best)
            total += lengths[best]
    selected.sort()
    return [rows[index] for index in selected], total


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-validation", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="unsloth/Qwen3-0.6B")
    args = parser.parse_args()
    if not 0.0 <= args.ratio <= 1.0:
        parser.error("--ratio must be between 0 and 1")

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("training_allowed") is not True:
        parser.error("source manifest is not training-allowed")
    base_train = load_jsonl(args.base_train)
    base_validation = load_jsonl(args.base_validation)
    source_train = load_jsonl(args.source_train)
    source_validation = load_jsonl(args.source_validation)
    if not source_train or not source_validation:
        parser.error("general source train/validation must be non-empty")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    base_train_lengths = token_lengths(base_train, tokenizer)
    base_validation_lengths = token_lengths(base_validation, tokenizer)
    source_train_lengths = token_lengths(source_train, tokenizer)
    source_validation_lengths = token_lengths(source_validation, tokenizer)
    base_train_tokens = sum(base_train_lengths)
    base_validation_tokens = sum(base_validation_lengths)
    train_target = round(base_train_tokens * args.ratio)
    validation_target = round(base_validation_tokens * args.ratio)
    selected_train, selected_train_tokens = choose(source_train, source_train_lengths, train_target)
    selected_validation, selected_validation_tokens = choose(source_validation, source_validation_lengths, validation_target)

    for row in selected_train + selected_validation:
        row["_replay_source"] = "general_instruction"
        row["_replay_ratio_target"] = args.ratio
    composed_train = base_train + selected_train
    composed_validation = base_validation + selected_validation
    write_jsonl(args.output_dir / "train.jsonl", composed_train)
    write_jsonl(args.output_dir / "validation.jsonl", composed_validation)
    manifest = {
        "format": "general_replay_composed_manifest_v1",
        "training_allowed": True,
        "ratio": args.ratio,
        "ratio_definition": "general_instruction_tokens / v5clean_base_train_tokens",
        "tokenizer": args.tokenizer,
        "base_train": {"path": str(args.base_train.resolve()), "count": len(base_train), "tokens": base_train_tokens},
        "base_validation": {"path": str(args.base_validation.resolve()), "count": len(base_validation), "tokens": base_validation_tokens},
        "general_train": {"path": str(args.source_train.resolve()), "available_count": len(source_train), "selected_count": len(selected_train), "target_tokens": train_target, "selected_tokens": selected_train_tokens},
        "general_validation": {"path": str(args.source_validation.resolve()), "available_count": len(source_validation), "selected_count": len(selected_validation), "target_tokens": validation_target, "selected_tokens": selected_validation_tokens},
        "composed_train_count": len(composed_train),
        "composed_validation_count": len(composed_validation),
        "source_manifest": str(args.source_manifest.resolve()),
        "seed": "sha256_source_id_v1",
        "notes": [
            "The complete v5clean base is retained; general instruction is added at a token-matched dose.",
            "The 0.0 ratio is a reproducible control composition of the unchanged base.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ratio": args.ratio, "train_count": len(composed_train), "general_train_tokens": selected_train_tokens, "base_train_tokens": base_train_tokens}))


if __name__ == "__main__":
    main()
