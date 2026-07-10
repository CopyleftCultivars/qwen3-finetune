#!/usr/bin/env python3
"""Rebuild data/sources/gemma4/hf_dataset/train.jsonl from ALL available gemma4 sources.

The existing hf_dataset/train.jsonl (2865 conversations, ShareGPT format) was already built
from data/sources/gemma4/conversations/ (real chat log exports), KNF-Methods-QnA/, and
Semisynthetic_Data_Natural_Farming_Fundamentals/ (confirmed by exact-text match, 2026-07-09).

Checked two more candidate sources (2026-07-09):
  - Natural-Farming-Real-QandA-Conversations-Q1-2024-Update/ — 1126 real user Q&A pairs
    (1090 usable after filtering ~36 non-answer rows). Turned out ALL 1090 are already present
    in hf_dataset/train.jsonl (verified by this script's own dedup pass, not a spot check) —
    an earlier single-string grep check incorrectly concluded this source was unused; it was a
    false negative from a formatting mismatch, not a real gap. This script's dedup logic is the
    correct way to check, not a spot-check grep.
  - syntheticdata-distiset-farming-chemistry/ — 30 synthetic prompt/completion pairs, genuinely
    new, added.

data/sources/gemma4/datasets_for_rag/ and data/sources/gemma4/spaces/data/ were NOT checked by
this script and remain the real gap — see the module docstring's note on them below.

data/sources/gemma4/datasets_for_rag/ and data/sources/gemma4/spaces/data/ are NOT included
here — they're structured reference tables (nutrient/mineral data, RAG corpus + precomputed
embeddings), not conversation-shaped, and need a separate templated-or-synthesized conversion
approach. See MCQ_DIAGNOSTIC_PLAN.md.

Output: overwrites data/sources/gemma4/hf_dataset/train.jsonl (original backed up alongside
with a .pre_expansion.jsonl suffix on first run).

Usage:
    python scripts/collect_gemma4_conversations.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

GEMMA4_DIR = Path("data/sources/gemma4")
EXISTING_TRAIN = GEMMA4_DIR / "hf_dataset" / "train.jsonl"
BACKUP_TRAIN = GEMMA4_DIR / "hf_dataset" / "train.pre_expansion.jsonl"

_LOW_QUALITY_MARKERS = (
    "i don't know",
    "i do not know",
    "perhaps i can offer more guidance",
    "please provide additional",
    "could you provide more",
    "can you clarify",
)


def _sharegpt_pair(instruction: str, output: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": instruction.strip()},
            {"from": "gpt", "value": output.strip()},
        ]
    }


def _load_natural_farming_real(path: Path) -> list[dict]:
    pairs: list[dict] = []
    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            question = (row.get("Question") or "").strip()
            answer = (row.get("Answer") or "").strip()
            if not question or not answer:
                skipped += 1
                continue
            lower = answer.lower()
            if len(answer) < 20 or any(m in lower for m in _LOW_QUALITY_MARKERS):
                skipped += 1
                continue
            # Source CSV wraps some answers in markdown bold markers ("**Comfrey**...")
            answer = answer.strip("*").strip()
            pairs.append(_sharegpt_pair(question, answer))
    print(f"  Natural-Farming-Real: {len(pairs)} usable, {skipped} skipped", file=sys.stderr)
    return pairs


def _load_distiset(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    pairs = [
        _sharegpt_pair(row["prompt"], row["completion"])
        for row in table.to_pylist()
        if row.get("prompt") and row.get("completion")
    ]
    print(f"  syntheticdata-distiset: {len(pairs)} pairs", file=sys.stderr)
    return pairs


def main() -> None:
    if not EXISTING_TRAIN.exists():
        print(f"error: {EXISTING_TRAIN} not found", file=sys.stderr)
        sys.exit(1)

    if not BACKUP_TRAIN.exists():
        shutil.copy(EXISTING_TRAIN, BACKUP_TRAIN)
        print(f"Backed up original to {BACKUP_TRAIN}", file=sys.stderr)

    existing = [json.loads(line) for line in open(BACKUP_TRAIN, encoding="utf-8") if line.strip()]
    print(f"Existing hf_dataset/train.jsonl: {len(existing)} conversations", file=sys.stderr)

    print("Loading new sources...", file=sys.stderr)
    new_pairs: list[dict] = []
    new_pairs.extend(_load_natural_farming_real(
        GEMMA4_DIR / "Natural-Farming-Real-QandA-Conversations-Q1-2024-Update"
        / "Trainable_NF_conversations_Q1_2024_PLUS.csv"
    ))
    new_pairs.extend(_load_distiset(
        GEMMA4_DIR / "syntheticdata-distiset-farming-chemistry" / "data"
        / "train-00000-of-00001.parquet"
    ))

    # Dedup new pairs against existing content by first-turn text, in case of overlap.
    existing_human_turns = {
        conv["conversations"][0]["value"]
        for conv in existing
        if conv.get("conversations")
    }
    deduped = [
        p for p in new_pairs
        if p["conversations"][0]["value"] not in existing_human_turns
    ]
    dropped_dupes = len(new_pairs) - len(deduped)
    if dropped_dupes:
        print(f"  Dropped {dropped_dupes} duplicates of existing conversations", file=sys.stderr)

    combined = existing + deduped
    with open(EXISTING_TRAIN, "w", encoding="utf-8") as fh:
        for conv in combined:
            fh.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(combined)} conversations to {EXISTING_TRAIN}", file=sys.stderr)
    print(f"  ({len(existing)} existing + {len(deduped)} newly added)", file=sys.stderr)


if __name__ == "__main__":
    main()
