#!/usr/bin/env python3
"""#191 step 3: chunk the persona corpus for LPM next-token prediction.

The LPM training objective (arXiv 2606.20911, Algorithm 1) walks a user's
history chunk by chunk and, for each chunk C_t, retrieves the soft prompt from
the PREVIOUS chunk C_{t-1}. That ordering is the leakage-prevention requirement:
if the prompt were retrieved from C_t it could read the target tokens. This
script emits one record per (persona, chunk) with:

  prompt_chunk_id  -- always the previous chunk (its own id only for t=0, which
                      is what LPM does for the first chunk of a history)
  prompt_text      -- the previous chunk's text, fed to the (frozen) encoder to
                      retrieve the soft prompt
  target_chunk_id  -- the current chunk
  target_text      -- the current chunk, the NTP loss target

Invariant asserted on every record: ``prompt_source_index <= chunk_index``.
For t>0 the prompt text is physically distinct from the target text; a test
locks this down.

Inputs are the real single-session seeds and the synthetic multi-session
dialogues. The real multi-session held-out conversations are never read here.

Usage:
  python scripts/chunk_persona_memory_corpus.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from persona_memory_common import (  # noqa: E402
    REPO_ROOT,
    Turn,
    chunk_session,
    read_jsonl,
    write_jsonl,
)

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "persona_memory"


def session_turns(session: dict[str, Any]) -> list[Turn]:
    """Rehydrate a serialized session into Turn objects for chunk_session."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    turns: list[Turn] = []
    for index, turn in enumerate(session.get("turns", [])):
        turns.append(
            Turn(
                message_id=f"{session.get('session_index', 0)}:{index}",
                role=str(turn.get("from", "human")),
                text=str(turn.get("text", "")).strip(),
                created_at=base,
            )
        )
    return turns


def persona_chunks(row: dict[str, Any], *, chunk_turns: int, max_chunk_chars: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for session in row.get("sessions", []):
        for text in chunk_session(
            session_turns(session), chunk_turns=chunk_turns, max_chunk_chars=max_chunk_chars
        ):
            chunks.append(
                {
                    "text": text,
                    "session_index": session.get("session_index"),
                    "date": session.get("start") or session.get("days_later"),
                }
            )
    return chunks


def build_records(
    rows: list[dict[str, Any]],
    *,
    source_label: str,
    chunk_turns: int,
    max_chunk_chars: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        persona_id = str(row.get("persona_id") or row.get("conversation_id"))
        chunks = persona_chunks(row, chunk_turns=chunk_turns, max_chunk_chars=max_chunk_chars)
        if not chunks:
            continue
        for index, chunk in enumerate(chunks):
            prompt_index = max(0, index - 1)
            assert prompt_index <= index, "leakage guard: prompt must not be a future chunk"
            records.append(
                {
                    "example_id": f"{persona_id}#{index}",
                    "persona_id": persona_id,
                    "source": source_label,
                    "chunk_index": index,
                    "n_chunks": len(chunks),
                    "prompt_source_index": prompt_index,
                    "prompt_chunk_id": f"{persona_id}#{prompt_index}",
                    "prompt_text": chunks[prompt_index]["text"],
                    "target_chunk_id": f"{persona_id}#{index}",
                    "target_text": chunk["text"],
                    "session_index": chunk["session_index"],
                    "date": chunk["date"],
                    "is_first_chunk": index == 0,
                    "cross_chunk": index > 0,
                }
            )
    return records


def split_train_val(records: list[dict[str, Any]], val_fraction: float) -> tuple[list, list]:
    """Deterministic, persona-level split (no persona appears in both splits)."""
    val: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    if val_fraction <= 0:
        return records, val
    threshold = int(round(val_fraction * 10000))
    for record in records:
        bucket = int(hashlib.sha1(record["persona_id"].encode()).hexdigest()[:8], 16) % 10000
        (val if bucket < threshold else train).append(record)
    return train, val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seeds", type=Path, default=None, help="Default: <data-dir>/seeds_single_session.jsonl")
    parser.add_argument(
        "--synthetic", type=Path, default=None, help="Default: <data-dir>/synthetic_multisession.jsonl"
    )
    parser.add_argument("--output", type=Path, default=None, help="Default: <data-dir>/lpm_corpus.jsonl")
    parser.add_argument("--chunk-turns", type=int, default=2)
    parser.add_argument("--max-chunk-chars", type=int, default=1200)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir
    seeds_path = args.seeds or data_dir / "seeds_single_session.jsonl"
    synthetic_path = args.synthetic or data_dir / "synthetic_multisession.jsonl"
    output_path = args.output or data_dir / "lpm_corpus.jsonl"

    records: list[dict[str, Any]] = []
    inputs: dict[str, int] = {}
    if seeds_path.exists():
        rows = read_jsonl(seeds_path)
        inputs[str(seeds_path)] = len(rows)
        records += build_records(
            rows,
            source_label="real_single_session",
            chunk_turns=args.chunk_turns,
            max_chunk_chars=args.max_chunk_chars,
        )
    else:
        print(f"warning: {seeds_path} not found; skipping real single-session material")
    if synthetic_path.exists():
        rows = read_jsonl(synthetic_path)
        inputs[str(synthetic_path)] = len(rows)
        records += build_records(
            rows,
            source_label="synthetic_multisession",
            chunk_turns=args.chunk_turns,
            max_chunk_chars=args.max_chunk_chars,
        )
    else:
        print(f"warning: {synthetic_path} not found; skipping synthetic material")

    if not records:
        raise SystemExit("no chunks produced (both inputs missing or empty)")

    train, val = split_train_val(records, args.val_fraction)
    n = write_jsonl(output_path, records)
    train_path = output_path.with_name(output_path.stem + "_train.jsonl")
    val_path = output_path.with_name(output_path.stem + "_val.jsonl")
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    by_source: dict[str, int] = {}
    for record in records:
        by_source[record["source"]] = by_source.get(record["source"], 0) + 1
    manifest = {
        "format": "persona_memory_lpm_corpus_v1",
        "issue": 191,
        "inputs": inputs,
        "outputs": {"all": str(output_path), "train": str(train_path), "val": str(val_path)},
        "chunk_turns": args.chunk_turns,
        "max_chunk_chars": args.max_chunk_chars,
        "val_fraction": args.val_fraction,
        "records": n,
        "train_records": len(train),
        "val_records": len(val),
        "by_source": by_source,
        "personas": len({r["persona_id"] for r in records}),
        "first_chunks": sum(1 for r in records if r["is_first_chunk"]),
        "cross_chunks": sum(1 for r in records if r["cross_chunk"]),
        "leakage_guard": "prompt_source_index <= chunk_index for every record",
    }
    (data_dir / "lpm_corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
