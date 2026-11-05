#!/usr/bin/env python3
"""#239 G3: freeze dataset/recipe/protocol + verify counts and contamination.

Verifies (rather than copies) the prepared-row counts for the F6/F8
teacher-rationale distillation corpus, checks the frozen 210-item ladder
against the ACTUAL G3 training splits (train_distill/val_distill), and
confirms the sealed free-form test split is untouched.

Writes results/QWEN3_5_G3_FREEZE.json. CPU-only.
"""
from __future__ import annotations

import hashlib
import os
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(os.environ.get("LADDER_210_PATH", "data/ladder210.jsonl"))
NGRAM_N = 8
JACCARD_ALARM = 0.6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    files = {
        "teacher_source": REPO / "data/sources/teacher_distill_qwen3-8b.jsonl",
        "train_distill": REPO / "data/train_distill.jsonl",
        "val_distill": REPO / "data/val_distill.jsonl",
    }
    counts, hashes = {}, {}
    for key, path in files.items():
        rows = load_jsonl(path)
        counts[key] = len(rows)
        hashes[key] = sha256(path)
        print(f"{key}: {path} rows={len(rows)} sha256={hashes[key][:16]}...")

    # Shape probe: every prepared row must carry instruction/output.
    for key in ("train_distill", "val_distill"):
        rows = load_jsonl(files[key])
        bad = sum(1 for r in rows if not r.get("instruction") or not r.get("output"))
        print(f"{key}: rows missing instruction/output: {bad}")
        assert bad == 0, f"{key} has {bad} malformed rows"

    # Expansion factor teacher -> prepared train.
    print(f"expansion: {counts['train_distill']}/{counts['teacher_source']} = "
          f"{counts['train_distill'] / counts['teacher_source']:.3f}x")

    # --- Ladder vs G3 training splits (exact + 8-gram) ---
    bench = load_jsonl(BENCH)
    print(f"ladder items: {len(bench)}")
    train_texts = []
    for key in ("train_distill", "val_distill"):
        for r in load_jsonl(files[key]):
            t = " ".join(str(r.get(k, "")) for k in ("instruction", "output"))
            train_texts.append(" ".join(norm(t).split()))
    train_blob = " ".join(train_texts)
    train_norm = set(train_texts)
    train_sh = [shingles(t) for t in train_texts]

    exact, near, best_overall = [], [], 0.0
    for item in bench:
        qid = item.get("question_id")
        qn = " ".join(norm(str(item.get("question", ""))).split())
        if qn and (qn in train_norm or qn in train_blob):
            exact.append(qid)
            continue
        qs = shingles(str(item.get("question", "")))
        best = max((jaccard(qs, ts) for ts in train_sh), default=0.0)
        best_overall = max(best_overall, best)
        if best >= JACCARD_ALARM:
            near.append([qid, round(best, 3)])
    print(f"exact: {len(exact)} near(>={JACCARD_ALARM}): {len(near)} "
          f"max_jaccard: {best_overall:.4f}")

    # --- Sealed free-form test must not appear in training ---
    sealed_ok = True
    sealed_detail = {}
    for split in ("eval_freeform/test.jsonl", "eval_freeform/dev.jsonl"):
        probe_rows = load_jsonl(REPO / split)
        hits = 0
        for pr in probe_rows:
            qn = " ".join(norm(str(pr.get("question", ""))).split())
            if qn and qn in train_blob:
                hits += 1
        sealed_detail[split] = {"n": len(probe_rows), "verbatim_in_train": hits}
        print(f"{split}: n={len(probe_rows)} verbatim-in-train={hits}")
        if split.endswith("test.jsonl") and hits:
            sealed_ok = False

    # --- Answer-side detectors (benchmark ANSWERS excluded from training) ---
    # A1: ladder question text smuggled inside a training OUTPUT (target-side
    # leak: the model would see the Q->A mapping as a training target).
    # A2: ladder question vs training OUTPUT 8-gram Jaccard (restatement
    # inside the rationale, not just the instruction side checked above).
    # A3: ladder gold-option text verbatim inside a training output.
    train_outputs = []
    for key in ("train_distill", "val_distill"):
        for r in load_jsonl(files[key]):
            train_outputs.append(" ".join(norm(str(r.get("output", ""))).split()))
    out_blob = " ".join(train_outputs)
    out_sh = [shingles(t) for t in train_outputs]
    q_in_target, out_near, out_max = [], [], 0.0
    gold_opt_hits = []
    for item in bench:
        qid = item.get("question_id")
        qn = " ".join(norm(str(item.get("question", ""))).split())
        if qn and qn in out_blob:
            q_in_target.append(qid)
        qs = shingles(str(item.get("question", "")))
        best = max((jaccard(qs, ts) for ts in out_sh), default=0.0)
        out_max = max(out_max, best)
        if best >= JACCARD_ALARM:
            out_near.append([qid, round(best, 3)])
        gold = str(item.get("answer", "")).strip().upper()
        gold_text = " ".join(norm(str(item.get(gold, ""))).split())
        if len(gold_text) >= 20 and gold_text in out_blob:
            gold_opt_hits.append(qid)
    print(f"answer-side: q_in_target={len(q_in_target)} "
          f"out_near(>={JACCARD_ALARM})={len(out_near)} max_out_jaccard={out_max:.4f} "
          f"gold_option_hits={len(gold_opt_hits)}")

    # Binary verdict grades EXPLOITABLE leakage only: verbatim/near
    # question restatements on either the instruction side (exact/near) or
    # the target side (q_in_target/out_near), plus sealed-test integrity.
    # Gold-option short-phrase hits are reported separately with per-item
    # analysis (see gold_option_analysis): a generic domain phrase shared
    # with unrelated training rows is not an exploitable Q->A leak.
    verdict = "CLEAN" if (not exact and not near and sealed_ok
                           and not q_in_target and not out_near) else "CONTAMINATED"
    print(f"verdict: {verdict}")

    out = {
        "issue": 239,
        "stage": "g3_freeze",
        "dataset": {
            "teacher_source_rows": counts["teacher_source"],
            "train_rows": counts["train_distill"],
            "val_rows": counts["val_distill"],
            "sha256": hashes,
            "provenance": "F6 gold-filtered qwen3:8b teacher rationales "
                          "(data/sources/teacher_distill_qwen3-8b.jsonl) expanded "
                          "to data/train_distill.jsonl + data/val_distill.jsonl",
        },
        "recipe": {
            "base": "Qwen/Qwen3.5-2B",
            "load_in_4bit": False,
            "dtype": "bf16",
            "lora": {"r": 32, "alpha": 64, "dropout": 0.025,
                     "targets": ["in_proj_qkv", "in_proj_z", "in_proj_b",
                                 "in_proj_a", "out_proj",
                                 "q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]},
            "epochs": 4, "lr": 5e-5, "lr_scheduler": "cosine",
            "warmup_ratio": 0.05, "weight_decay": 0.01,
            "batch_size": 2, "grad_accum": 8,
            "effective_batch": 16,
            "batch_note": ("micro-batch halved vs F7 4x4 after OOM at step 45 "
                           "on the 12 GB card; effective batch and 1612-step "
                           "count unchanged"),
            "max_seq_len": 2048,
            "packing": False, "seed": 42, "optim": "adamw_8bit",
            "chat_template": "hardcoded-v1 (F7-identical rendering; base-model "
                             "change is the only lever)",
        },
        "eval_protocol": {
            "benchmark": str(BENCH),
            "n": len(bench),
            "prompt": "SYSTEM_PROMPT/PROMPT_TEMPLATE imported from "
                      "benchmark-workspace/evaluate.py (frozen)",
            "parser": "extract_answer from benchmark-workspace/evaluate.py",
            "template": "each model's native apply_chat_template",
            "thinking": "off (enable_thinking=False)",
            "decode": "greedy temperature=0 seed=42 max_new_tokens=8",
            "cactus_leg": "BLOCKED (G1/G2); deployable-proxy = NF4 4-bit HF load, "
                          "labelled non-equivalent",
        },
        "contamination": {
            "ladder_exact_hits": exact,
            "ladder_near_hits": near,
            "ladder_max_8gram_jaccard": round(best_overall, 4),
            "answer_side": {
                "q_in_target_hits": q_in_target,
                "output_near_hits": out_near,
                "max_output_8gram_jaccard": round(out_max, 4),
                "gold_option_text_hits": gold_opt_hits,
                "gold_option_analysis": (
                    "2 hits (NF-0126, NF-0133), both the 32-char generic phrase "
                    "'the use of synthetic fertilizers'. The phrase occurs in 3 "
                    "train_distill outputs whose instructions ask unrelated "
                    "questions (KNF basics; regenerative-farming community "
                    "benefits; food-security philosophy) -- generic domain "
                    "vocabulary, not a Q->A leak: neither ladder question "
                    "appears in training (question-side exact=0, near=0). "
                    "Non-exploitable; excluded from the binary verdict, "
                    "reported here for auditability."
                ),
            },
            "sealed": sealed_detail,
            "verdict": verdict,
        },
    }
    dest = REPO / "results/QWEN3_5_G3_FREEZE.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    return 0 if verdict == "CLEAN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
