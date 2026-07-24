#!/usr/bin/env python3
"""Build audited multi-turn and reflection training streams.

The source is the repository's English natural-farming ShareGPT corpus.  The
builder keeps one source thread per item, preserves a bounded recent turn
window, and emits both the full ``messages`` training surface and a prompt
surface for local native validation.  Reflection targets are deterministic
answer-audit tasks over existing source answers; they are not the rejected
QMD rationale replay.

No Spanish benchmark or sealed free-form test path is accepted.  The English
MCQ benchmark is read only as a contamination boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

# Keep direct-script execution and importlib-based tests on the same path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_crop_auxiliary_datasets import (
    benchmark_texts,
    digest_file,
    ensure_open_benchmark_path,
    jaccard,
    normalize,
)


TASKS = (
    "multiturn",
    "flattened_control",
    "interrupted_counterfactual",
    "answer_audit",
)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MIN_EXACT_CHARS = 48
MIN_OVERLAP_TOKENS = 6
LEXICAL_THRESHOLD = 0.85
MAX_WINDOW_PAIRS = 4
SYSTEM_PREFIXES = (
    "You are a Natural Farming",
    "You are an AI assistant",
    "You are a helpful",
)


def reject_training_path(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    if "spanish-benchmark-workspace" in parts:
        raise ValueError("Spanish benchmark paths are evaluation-only")
    if "eval_freeform" in parts and path.name.casefold() == "test.jsonl":
        raise ValueError("sealed free-form test paths are forbidden")
    if "sealed" in path.name.casefold():
        raise ValueError("sealed benchmark paths are forbidden")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def is_system_prefix(text: str) -> bool:
    return any(text.lstrip().startswith(prefix) for prefix in SYSTEM_PREFIXES)


def load_threads(path: Path) -> list[dict[str, Any]]:
    threads: list[dict[str, Any]] = []
    file_sha = digest_file(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            raw_turns = record.get("conversations", [])
            if not isinstance(raw_turns, list):
                continue
            turns: list[dict[str, str]] = []
            for turn in raw_turns:
                if not isinstance(turn, dict):
                    continue
                source_role = str(turn.get("from", "")).casefold()
                role = "user" if source_role == "human" else "assistant" if source_role in {"gpt", "assistant"} else None
                content = str(turn.get("value", "")).strip()
                if role and content:
                    turns.append({"role": role, "content": content})
            if len(turns) < 4 or turns[0]["role"] != "user":
                continue
            if is_system_prefix(turns[0]["content"]):
                turns = turns[2:]
            if len(turns) < 4 or turns[0]["role"] != "user":
                continue
            if any(turns[index]["role"] == turns[index - 1]["role"] for index in range(1, len(turns))):
                continue
            if turns[-1]["role"] != "assistant":
                continue
            pairs = len(turns) // 2
            if pairs < 2:
                continue
            # Keep the final local context within the model's 2048-token
            # surface. This still preserves multiple turns and makes the
            # structure-vs-flattening comparison deterministic.
            window = turns[-(MAX_WINDOW_PAIRS * 2):]
            source_record_sha = sha(record)
            source_id = f"conversation:{line_number:06d}:{source_record_sha[:12]}"
            threads.append(
                {
                    "source_id": source_id,
                    "source_line": line_number,
                    "source_file": str(path.resolve()),
                    "source_sha256": file_sha,
                    "source_record_sha256": source_record_sha,
                    "messages": window,
                    "all_pair_count": pairs,
                    "window_pair_count": len(window) // 2,
                }
            )
    return threads


def truncate_draft(answer: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    if len(sentences) > 1:
        return " ".join(sentences[: max(1, len(sentences) // 2)]).strip()
    words = answer.split()
    return " ".join(words[: max(8, len(words) // 2)]).strip()


def target_row(task: str, thread: dict[str, Any]) -> dict[str, Any]:
    messages = thread["messages"]
    prompt_messages = messages[:-1]
    final_user = prompt_messages[-1]["content"]
    answer = messages[-1]["content"]
    training_messages: list[dict[str, str]]
    output = answer
    authoring_method = "preserved_sharegpt_thread_v1"
    error_type = None

    if task == "multiturn":
        training_messages = messages
    elif task == "flattened_control":
        training_messages = [messages[-2], messages[-1]]
        authoring_method = "same_source_flattened_control_v1"
    elif task == "interrupted_counterfactual":
        draft = truncate_draft(answer)
        reflection_prompt = (
            "Reflection task: audit an interrupted answer and correct a counterfactual overgeneralization.\n\n"
            f"Draft answer:\n{draft}\n\n"
            "Counterfactual reviewer claim: this recommendation is universally safe, so site conditions, crop stage, "
            "and practical safeguards do not matter. Identify why that claim is not justified, then provide the "
            "complete grounded answer to the user's question."
        )
        output = (
            "Audit: The reviewer claim overgeneralizes. Recommendations depend on the stated context, site conditions, "
            "crop stage, and applicable safety or soil-test constraints.\n"
            f"Corrected answer:\n{answer}"
        )
        training_messages = prompt_messages + [{"role": "user", "content": reflection_prompt}, {"role": "assistant", "content": output}]
        authoring_method = "deterministic_interrupted_counterfactual_audit_v1"
        error_type = "unsupported_universal_claim"
    elif task == "answer_audit":
        draft = truncate_draft(answer)
        reflection_prompt = (
            "Reflection task: audit the draft answer for omissions or unsupported certainty, then provide a corrected "
            "final answer.\n\n"
            f"User question:\n{final_user}\n\nDraft answer:\n{draft}"
        )
        output = (
            "Audit: The draft is incomplete because it stops before the full answer and should not be treated as a "
            "complete recommendation.\n"
            f"Corrected answer:\n{answer}"
        )
        training_messages = [{"role": "user", "content": reflection_prompt}, {"role": "assistant", "content": output}]
        authoring_method = "deterministic_answer_audit_target_v1"
        error_type = "incomplete_draft"
    else:
        raise ValueError(f"unknown task: {task}")

    return {
        "id": f"{task}:{thread['source_id']}",
        "instruction": training_messages[-2]["content"],
        "output": output,
        "messages": training_messages,
        "prompt_messages": training_messages[:-1],
        "target_output": output,
        "language": "en",
        "structure_task": task,
        "source_id": thread["source_id"],
        "provenance": {
            "source_path": thread["source_file"],
            "source_sha256": thread["source_sha256"],
            "source_line": thread["source_line"],
            "source_record_sha256": thread["source_record_sha256"],
            "original_pair_count": thread["all_pair_count"],
            "window_pair_count": thread["window_pair_count"],
            "authoring_method": authoring_method,
            "reflection_error_type": error_type,
            "source_document_independence": "not_claimed",
        },
    }


def audit_rows(rows: list[dict[str, Any]], benchmark: list[tuple[str, str, str, set[str]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    maximum = {"score": 0.0}
    for index, row in enumerate(rows):
        reason = None
        match: dict[str, Any] = {}
        texts = [str(row.get("output", ""))]
        texts.extend(str(turn.get("content", "")) for turn in row.get("messages", []))
        for source_field, value in enumerate(texts):
            normalized = normalize(value)
            if len(normalized) < MIN_EXACT_CHARS:
                continue
            source_tokens = set(normalized.split())
            for question_id, benchmark_field, benchmark_text, benchmark_tokens in benchmark:
                if len(benchmark_text) >= MIN_EXACT_CHARS and (benchmark_text in normalized or normalized in benchmark_text):
                    reason = "exact_normalized_benchmark_substring"
                    match = {"benchmark_question_id": question_id, "benchmark_field": benchmark_field, "source_field_index": source_field}
                    break
                score = jaccard(source_tokens, benchmark_tokens)
                if score > maximum.get("score", 0.0):
                    maximum = {"score": round(score, 6), "id": row["id"], "source_field_index": source_field, "benchmark_question_id": question_id, "benchmark_field": benchmark_field}
                if len(source_tokens & benchmark_tokens) >= MIN_OVERLAP_TOKENS and score >= LEXICAL_THRESHOLD:
                    reason = "high_lexical_benchmark_overlap"
                    match = {"benchmark_question_id": question_id, "benchmark_field": benchmark_field, "source_field_index": source_field, "score": round(score, 6)}
                    break
            if reason:
                break
        if row["id"] in seen and reason is None:
            reason = "duplicate_item_id"
        seen.add(row["id"])
        if reason:
            rejected.append({"index": index, "id": row.get("id"), "source_id": row.get("source_id"), "reason": reason, **match})
        else:
            clean_row = dict(row)
            clean_row["_contamination_status"] = "passed"
            clean_row["_contamination_audit"] = "multiturn_reflection_benchmark_audit_v1"
            clean_row["_benchmark_question_text_overlap"] = "not_detected"
            clean.append(clean_row)
    report = {
        "format": "multiturn_reflection_benchmark_audit_v1",
        "candidate_count": len(rows),
        "clean_count": len(clean),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "maximum_token_jaccard": maximum,
        "lexical_overlap_threshold": LEXICAL_THRESHOLD,
        "minimum_overlap_tokens": MIN_OVERLAP_TOKENS,
        "passed": not rejected,
        "sealed_test_paths_read": [],
        "source_document_independence": "not_claimed",
    }
    return clean, report


def split_rows(rows: list[dict[str, Any]], train_count: int, validation_count: int, split: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_ids = sorted({str(row["source_id"]) for row in rows})
    ranked = sorted(source_ids, key=lambda value: stable_rank(value, seed))
    validation_count_sources = max(1, int(round(len(source_ids) * (1.0 - split))))
    validation_ids = set(ranked[:validation_count_sources])
    train_ids = set(source_ids) - validation_ids
    train_rows = sorted((row for row in rows if row["source_id"] in train_ids), key=lambda row: stable_rank(row["id"], seed))
    validation_rows = sorted((row for row in rows if row["source_id"] in validation_ids), key=lambda row: stable_rank(row["id"], seed))
    if len(train_rows) < train_count or len(validation_rows) < validation_count:
        raise ValueError(f"pilot dose unavailable: train={len(train_rows)} validation={len(validation_rows)}")
    train = train_rows[:train_count]
    validation = validation_rows[:validation_count]
    if {row["source_id"] for row in train} & {row["source_id"] for row in validation}:
        raise AssertionError("source-document split overlap")
    return train, validation, {
        "source_count": len(source_ids),
        "train_source_count": len(train_ids),
        "validation_source_count": len(validation_ids),
        "train_validation_source_overlap": sorted({row["source_id"] for row in train} & {row["source_id"] for row in validation}),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return digest_file(path)


def build_task(task: str, threads: list[dict[str, Any]], benchmark: list[tuple[str, str, str, set[str]]], output_root: Path, benchmark_path: Path, train_count: int, validation_count: int, split: float, seed: int) -> dict[str, Any]:
    candidates = [target_row(task, thread) for thread in threads]
    clean, audit = audit_rows(candidates, benchmark)
    if not clean:
        raise ValueError(f"{task}: contamination audit removed every candidate")
    train, validation, split_report = split_rows(clean, train_count, validation_count, split, seed)
    task_root = output_root / task
    train_path = task_root / "train.jsonl"
    validation_path = task_root / "validation.jsonl"
    train_sha = write_jsonl(train_path, train)
    validation_sha = write_jsonl(validation_path, validation)
    manifest = {
        "format": "multiturn_reflection_dataset_manifest_v1",
        "task": task,
        "language": "en",
        "pilot_dose": {"train_count": train_count, "validation_count": validation_count, "split_fraction": split, "seed": seed},
        "source": {"path": str(Path(threads[0]["source_file"]).resolve()), "sha256": threads[0]["source_sha256"], "thread_count": len(threads), "max_window_pairs": MAX_WINDOW_PAIRS},
        "benchmark": {"path": str(benchmark_path.resolve()), "sha256": digest_file(benchmark_path)},
        "candidate_count": len(candidates),
        "clean_candidate_count": len(clean),
        "contamination": audit,
        "split": split_report,
        "train": {"path": str(train_path.resolve()), "sha256": train_sha, "count": len(train)},
        "validation": {"path": str(validation_path.resolve()), "sha256": validation_sha, "count": len(validation)},
        "train_validation_source_overlap": split_report["train_validation_source_overlap"],
        "training_allowed": not audit["rejected"],
        "sealed_test_paths_read": [],
        "notes": [
            "This stream is a single autoresearch lever and must not be combined before its result is recorded.",
            "The flattened control uses the same source threads and final answers as the preserved multi-turn stream.",
            "No explicit step-back labels were found in the source; this phase tests genuine multi-turn preservation first.",
        ],
    }
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"task": task, "candidate_count": len(candidates), "clean_count": len(clean), "train_count": len(train), "validation_count": len(validation), "rejected_count": audit["rejected_count"], "source_overlap": split_report["train_validation_source_overlap"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=Path, default=Path("data/sources/gemma4/hf_dataset/train.jsonl"))
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/multiturn_reflection"))
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--validation-count", type=int, default=50)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=TASKS, action="append", dest="tasks")
    args = parser.parse_args()
    for path in (args.conversations, args.benchmark):
        if not path.exists():
            parser.error(f"file not found: {path}")
    reject_training_path(args.conversations)
    ensure_open_benchmark_path(args.benchmark)
    if args.train_count < 1 or args.validation_count < 1 or not 0.0 < args.split < 1.0:
        parser.error("invalid dose or split")
    threads = load_threads(args.conversations.resolve())
    if len(threads) < args.train_count + args.validation_count:
        parser.error(f"only {len(threads)} eligible multi-turn threads; requested {args.train_count + args.validation_count}")
    benchmark = benchmark_texts(args.benchmark.resolve())
    tasks = tuple(args.tasks or TASKS)
    summaries = [build_task(task, threads, benchmark, args.output_root.resolve(), args.benchmark.resolve(), args.train_count, args.validation_count, args.split, args.seed) for task in tasks]
    root = {
        "format": "multiturn_reflection_root_manifest_v1",
        "tasks": list(tasks),
        "source": str(args.conversations.resolve()),
        "source_sha256": digest_file(args.conversations.resolve()),
        "eligible_thread_count": len(threads),
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": digest_file(args.benchmark.resolve()),
        "pilot_dose": {"train_count": args.train_count, "validation_count": args.validation_count, "split_fraction": args.split, "seed": args.seed},
        "sealed_test_paths_read": [],
        "summaries": summaries,
    }
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    (args.output_root.resolve() / "manifest.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
