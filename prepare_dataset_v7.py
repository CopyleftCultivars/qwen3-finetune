#!/usr/bin/env python3
"""prepare_dataset_v7.py -- F6 (#143): build the qwen3:8b distillation dataset.

Plan B.4/B.5/B.2, AMENDED. Three stages, then a handoff to prepare_dataset_v3.py:

  1. Contamination wall (B.4). Drop any teacher item whose normalized question
     stem matches the 210-item eval benchmark, the frozen val split (val_v3 /
     val_v6), or the sealed free-form dev/test sets -- exact match plus 8-gram
     Jaccard >= 0.6, the same three-signal family F2 (#139) used. The teacher
     pool and the val split share the mcq_generation source, so exact stem
     overlap is expected and must be excluded or F8's frozen MCQ primary is
     contaminated.

  2. Quality filter (B.5). Keep only teacher outputs that match gold AND carry
     a complete  thinking rationale. This is the gold filter, and it also drops
     the 17 pathological rambles F5 flagged (empty rationale). Filter drop-rate
     is reported.

  3. Emit data/sources/teacher_distill_qwen3-8b.jsonl in the settled
     instruction/output shape (bare-letter: " thinking\\n{rationale}\\n
     response\\n{letter}"), then build data/train_distill.jsonl and
     data/val_distill.jsonl by reusing prepare_dataset_v3.py verbatim
     (conversations + distill MCQ, cyclic shift, 90/10, seed 42).

Accept criteria for #143: zero eval overlap demonstrated; filter drop-rate
reported; train/val splits built.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BENCH = Path(os.environ.get("LADDER_210_PATH", "data/ladder210.jsonl"))

NGRAM_N = 8
JACCARD_ALARM = 0.6

SEALED_SPLITS = [
    "data/val_v3.jsonl",
    "data/val_v6.jsonl",
]
SEALED_FREEFORM = [
    "eval_freeform/dev.jsonl",
    "eval_freeform/test.jsonl",
]

CONVERSATIONS = "data/sources/gemma4/hf_dataset/train.jsonl"
DISTILL_SOURCE = "data/sources/teacher_distill_qwen3-8b.jsonl"


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower())


def question_stem(text: str) -> str:
    """Split the question off an instruction (MCQ rows carry options after a blank line)."""
    return str(text).split("\n\nA.")[0].strip()


def shingles(text: str, n: int = NGRAM_N) -> set[str]:
    words = norm(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_sealed() -> dict[str, dict]:
    """Return {source: {"exact": set[norm-stem], "shingles": list[set]}} for the wall."""
    sealed: dict[str, dict] = {}
    # eval benchmark: question field
    exact: set[str] = set()
    shing: list[set[str]] = []
    for line in open(BENCH, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        stem = " ".join(norm(rec.get("question", "")).split())
        if stem:
            exact.add(stem)
            shing.append(shingles(stem))
    sealed["eval_benchmark"] = {"exact": exact, "shingles": shing}

    # frozen val splits: instruction field, stem stripped
    for path in SEALED_SPLITS:
        p = REPO_ROOT / path
        if not p.exists():
            continue
        exact = set()
        shing = []
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            stem = " ".join(norm(question_stem(rec.get("instruction", ""))).split())
            if stem:
                exact.add(stem)
                shing.append(shingles(stem))
        sealed[path] = {"exact": exact, "shingles": shing}

    # sealed free-form dev/test: question field
    for path in SEALED_FREEFORM:
        p = REPO_ROOT / path
        if not p.exists():
            continue
        exact = set()
        shing = []
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            stem = " ".join(norm(rec.get("question", "")).split())
            if stem:
                exact.add(stem)
                shing.append(shingles(stem))
        sealed[path] = {"exact": exact, "shingles": shing}
    return sealed


def overlap_source(stem: str, sealed: dict[str, dict]) -> str | None:
    """Return the sealed source this stem overlaps, or None."""
    stem_norm = " ".join(norm(stem).split())
    stem_sh = shingles(stem)
    for source, data in sealed.items():
        if stem_norm in data["exact"]:
            return source
        for other in data["shingles"]:
            if jaccard(stem_sh, other) >= JACCARD_ALARM:
                return source
    return None


def build_source(teacher_path: Path) -> tuple[list[dict], dict]:
    sealed = load_sealed()
    overlap_counts: dict[str, int] = {}
    stats = {"input": 0, "eval_overlap": 0, "val_overlap": 0, "freeform_overlap": 0,
             "gold_dropped": 0, "pathological_dropped": 0, "kept": 0}
    kept: list[dict] = []
    for line in open(teacher_path, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        stats["input"] += 1
        src = overlap_source(rec["question"], sealed)
        if src is not None:
            overlap_counts[src] = overlap_counts.get(src, 0) + 1
            if src == "eval_benchmark":
                stats["eval_overlap"] += 1
            elif src in SEALED_SPLITS:
                stats["val_overlap"] += 1
            else:
                stats["freeform_overlap"] += 1
            continue
        if not rec.get("teacher_matches_gold"):
            stats["gold_dropped"] += 1
            continue
        if not rec.get("teacher_rationale"):
            stats["pathological_dropped"] += 1
            continue
        instruction = f"{rec['question']}\n\nA. {rec['A']}\nB. {rec['B']}\nC. {rec['C']}\nD. {rec['D']}"
        output = f" thinking\n{rec['teacher_rationale']}\n response\n{rec['teacher_answer']}"
        kept.append({
            "instruction": instruction,
            "output": output,
            # "qmd_mcq" so prepare_dataset_v3's letter-distribution / balancing
            # machinery treats this like the QMD MCQ stream it replaces.
            "_source": "qmd_mcq",
            "_distill_teacher": "qwen3-8b",
            "_source_question_id": str(rec["question_id"]),
            "_source_category": rec.get("category", ""),
            "_source_topic": rec.get("topic", ""),
        })
    stats["kept"] = len(kept)
    stats["overlap_counts"] = overlap_counts
    return kept, stats


def build_splits(suffix: str = "_distill") -> tuple[int, int]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "prepare_dataset_v3.py"),
        "--output-dir", "data",
        "--suffix", suffix,
        "--split", "0.9",
        "--seed", "42",
        "--min-output-chars", "10",
        "--max-total-chars", "8192",
        "--sft-format", "bare-letter",
        "--conversations", CONVERSATIONS,
        "--qmd-mcq", DISTILL_SOURCE,
    ]
    print("$ " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=REPO_ROOT, check=True)
    train = REPO_ROOT / f"data/train{suffix}.jsonl"
    val = REPO_ROOT / f"data/val{suffix}.jsonl"
    return _count(train), _count(val)


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in open(path, encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="results/teacher_data_qwen3-8b.jsonl")
    parser.add_argument("--source-out", default=DISTILL_SOURCE)
    parser.add_argument("--suffix", default="_distill")
    parser.add_argument("--skip-split-build", action="store_true",
                        help="only write the distill source, do not build train/val")
    args = parser.parse_args(argv)

    kept, stats = build_source(Path(args.teacher))
    drop = stats["input"] - stats["kept"]
    print(f"contamination + filter: input={stats['input']} kept={stats['kept']} "
          f"dropped={drop} ({drop / stats['input']:.3%} drop-rate)", flush=True)
    print(f"  eval_overlap={stats['eval_overlap']} val_overlap={stats['val_overlap']} "
          f"freeform_overlap={stats['freeform_overlap']}", flush=True)
    print(f"  gold_dropped={stats['gold_dropped']} pathological_dropped={stats['pathological_dropped']}",
          flush=True)
    print(f"  overlap_counts={stats['overlap_counts']}", flush=True)

    out = Path(args.source_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for rec in kept:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(kept)} records)", flush=True)

    # deterministic source hash for the zero-overlap demonstration
    digest = hashlib.sha256()
    for rec in sorted(kept, key=lambda r: r["instruction"]):
        digest.update(rec["instruction"].encode())
    print(f"source_instruction_sha256={digest.hexdigest()[:16]}...", flush=True)

    report = {
        "input": stats["input"],
        "kept": stats["kept"],
        "dropped": drop,
        "drop_rate": round(drop / stats["input"], 4),
        "eval_overlap": stats["eval_overlap"],
        "val_overlap": stats["val_overlap"],
        "freeform_overlap": stats["freeform_overlap"],
        "gold_dropped": stats["gold_dropped"],
        "pathological_dropped": stats["pathological_dropped"],
        "overlap_counts": stats["overlap_counts"],
        "source": str(out),
    }
    if not args.skip_split_build:
        n_train, n_val = build_splits(args.suffix)
        report["train_rows"] = n_train
        report["val_rows"] = n_val
        print(f"built data/train{args.suffix}.jsonl ({n_train}) / "
              f"data/val{args.suffix}.jsonl ({n_val})", flush=True)

    Path("results/F6_distill_dataset_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
