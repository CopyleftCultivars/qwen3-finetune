#!/usr/bin/env python3
"""prepare_dataset_v3.py — Assemble v3 Qwen3 fine-tune training data.

Three source streams:
  1. finetune-gemma4 NF conversations (ShareGPT → instruction/output)
  2. repaired MCQ pairs with CoT rationale (output of scripts/repair_mcq_rationales.py)
  3. QMD-generated MCQ pairs (output of scripts/generate_mcq_from_qmd.py)
     — cyclic shifts applied here, with 2x weight on D-position shift

Optional streams:
  4. audited cloze pairs generated from clean chat answers
  5. reviewed adversarial-refusal pairs with explicit provenance markers

Optionally filters any source through LLM judge verdicts (judge_verdict="keep").

Stage B findings driving this v3 design:
  - Present-not-surfaced: fill think blocks with genuine rationale (not label restatement)
  - Generation accuracy 56.3% vs logprob 78.3%: format-matched training targets
  - D marginal 16.9% (target 25%): heavier D-position cyclic weight
  - Lower LR intended for train.py (--lr 5e-5 vs prior 1e-4)

*** CONTAMINATED SOURCE WARNING (see MCQ_DIAGNOSTIC_PLAN.md, 2026-07-09) ***
The default --mcq-rationale path below (data/sources/mcq_with_rationale.repaired.jsonl) is
100% derived from benchmark_combined.jsonl (the eval benchmark) — it was built by
scripts/repair_mcq_rationales.py from data/train.jsonl, which prepare_dataset.py built by
loading benchmark_combined.jsonl directly as its --mcq source. Every v3 checkpoint trained
with this default was trained on ~99.9% of the eval benchmark. Do not use this default until
a clean (non-benchmark-derived, deduped) replacement exists — see MCQ_DIAGNOSTIC_PLAN.md
Phase 3. Source stream 3 (QMD-generated MCQs) carries the same risk transitively: both it and
the original benchmark draw from [LOCAL_PATH]

Usage:
    python prepare_dataset_v3.py \\
        --conversations data/sources/gemma4/hf_dataset/train.jsonl \\
        --mcq-rationale <path to a verified non-benchmark-overlapping MCQ+rationale source> \\
        --qmd-mcq data/sources/qmd_mcq_generated.jsonl \\
        [--filter-system-prompts] \\
        [--adversarial-refusals data/sources/adversarial_refusals.jsonl] \\
        [--judged-filter data/sources/mcq_with_rationale_judged.jsonl] \\
        [--output-dir data] \\
        [--split 0.9] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Generator

from autoresearch.curriculum import order_pairs

_OPTION_LETTERS = ("A", "B", "C", "D")

# Cyclic shift weights: index = shift amount (0-3).
# shift=3 puts gold answer at D position — gets 2x weight per Stage A findings.
_SHIFT_WEIGHTS = [1, 1, 1, 2]

_MCQ_PATTERN = re.compile(
    r"^(.+?)\n\nA\. (.+)\nB\. (.+)\nC\. (.+)\nD\. (.+)$",
    re.DOTALL,
)
_THINK_CONTENT = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_LETTER = re.compile(r"\n([A-D])\s*$")

_SYSTEM_PROMPT_MARKERS = (
    "You are a Natural Farming",
    "You are an AI assistant",
    "You are a helpful",
)
_SYSTEM_PROMPT_END_MARKER = "Respond in user's language unless otherwise specified."


# ---------------------------------------------------------------------------
# Source 1 — ShareGPT conversations (finetune-gemma4)
# ---------------------------------------------------------------------------

def _is_system_prompt(text: str) -> bool:
    return any(text.lstrip().startswith(m) for m in _SYSTEM_PROMPT_MARKERS)


def _strip_embedded_system_prompt(text: str) -> tuple[str | None, bool]:
    """Remove a known persona prefix, or drop a prompt-only system row.

    QMD records sometimes concatenate the old assistant persona and the actual
    MCQ into one instruction. Conversation records usually contain the persona
    as its own human turn. We preserve a real question after the known
    boundary, but never keep the persona as user content.
    """
    if not _is_system_prompt(text):
        return text, False
    boundary = text.find(_SYSTEM_PROMPT_END_MARKER)
    if boundary >= 0:
        suffix = text[boundary + len(_SYSTEM_PROMPT_END_MARKER):].strip()
        if suffix:
            return suffix, True
    return None, False


def _filter_system_prompt_pairs(
    pairs: list[dict[str, str]],
    *,
    source: str,
    strip_composite: bool = True,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Filter embedded system/persona text at the source boundary."""
    kept: list[dict[str, str]] = []
    stats = {"stripped": 0, "dropped": 0}
    for pair in pairs:
        source_instruction = pair.get("_source_instruction", pair.get("instruction", ""))
        cleaned, stripped = _strip_embedded_system_prompt(source_instruction)
        if not strip_composite and _is_system_prompt(source_instruction):
            cleaned, stripped = None, False
        if cleaned is None:
            stats["dropped"] += 1
            continue
        updated = dict(pair)
        if stripped:
            instruction = pair.get("instruction", "")
            if instruction.startswith("Complete the missing phrase"):
                # Cloze prompts retain their generated wrapper; only the
                # original question metadata is used for this filter.
                updated["instruction"] = instruction.replace(
                    source_instruction, cleaned, 1
                )
                updated["_source_instruction"] = cleaned
            else:
                updated["instruction"] = cleaned
            stats["stripped"] += 1
        kept.append(updated)
    print(
        f"  System/preamble filter ({source}): stripped {stats['stripped']}, "
        f"dropped {stats['dropped']}",
        file=sys.stderr,
    )
    return kept, stats


def _load_conversations(path: Path) -> Generator[dict[str, str], None, None]:
    """Flatten ShareGPT multi-turn conversations to instruction/output pairs."""
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc

            turns = rec.get("conversations", [])
            pairs: list[tuple[str, str]] = []
            i = 0
            while i < len(turns) - 1:
                a, b = turns[i], turns[i + 1]
                if a.get("from") == "human" and b.get("from") in ("gpt", "assistant"):
                    pairs.append((a["value"].strip(), b["value"].strip()))
                    i += 2
                else:
                    i += 1

            if not pairs:
                continue

            start = 1 if _is_system_prompt(pairs[0][0]) else 0
            for instruction, output in pairs[start:]:
                if instruction and output:
                    yield {"instruction": instruction, "output": output,
                           "_source": "gemma4_conversations"}


# ---------------------------------------------------------------------------
# Source 2 — MCQ pairs already carrying rationale (from add_rationale.py)
# ---------------------------------------------------------------------------

def _load_mcq_with_rationale(path: Path) -> Generator[dict[str, str], None, None]:
    """Pass through instruction/output pairs that already carry rationale.

    These came from the existing benchmark MCQ cyclic-shift variants, so
    position balancing is already done. No re-shifting needed.
    """
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if "instruction" in rec and "output" in rec:
                yield {
                    "instruction": rec["instruction"],
                    "output": rec["output"],
                    "_source": rec.get("_source", "mcq_rationale"),
                }


# ---------------------------------------------------------------------------
# Source 2b — audited cloze pairs from clean chat data
# ---------------------------------------------------------------------------

def _load_cloze_pairs(path: Path) -> Generator[dict[str, str], None, None]:
    """Load only cloze records that passed the explicit contamination audit."""
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if rec.get("_contamination_status") != "passed":
                raise ValueError(
                    f"{path}:{lineno}: cloze record is not audit-approved; run "
                    "scripts/check_cloze_contamination.py first"
                )
            if "instruction" not in rec or "output" not in rec:
                continue
            yield {
                "instruction": str(rec["instruction"]),
                "output": str(rec["output"]),
                "_source": "chat_cloze",
                "_source_instruction": str(rec.get("_source_instruction", "")),
            }


# Source 2c — reviewed adversarial refusal examples

def _load_adversarial_refusals(path: Path) -> Generator[dict[str, str], None, None]:
    """Load only explicitly reviewed, provenance-labelled refusal examples."""
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if rec.get("_source") != "adversarial_refusal":
                raise ValueError(
                    f"{path}:{lineno}: refusal row must have _source="
                    "adversarial_refusal"
                )
            if rec.get("_review_status") != "approved":
                raise ValueError(
                    f"{path}:{lineno}: refusal row is not review-approved"
                )
            if "instruction" not in rec or "output" not in rec:
                raise ValueError(f"{path}:{lineno}: refusal row needs instruction/output")
            yield {
                "instruction": str(rec["instruction"]),
                "output": str(rec["output"]),
                "_source": "adversarial_refusal",
                "_refusal_id": str(rec.get("_refusal_id", f"row-{lineno}")),
            }


# ---------------------------------------------------------------------------
# Source 3 — QMD-generated MCQ (from generate_mcq_from_qmd.py)
# Apply cyclic shifts here with heavier D-position weight.
# ---------------------------------------------------------------------------

def _cyclic_shift_pair(
    pair: dict,
    shift: int,
    sft_format: str = "bare-letter",
) -> dict | None:
    """Apply cyclic option shift to a QMD MCQ pair.

    Parses instruction for question+options, rotates options by `shift`,
    updates the answer letter, keeps rationale content (which explains the
    concept, not the letter label).
    """
    m = _MCQ_PATTERN.match(pair["instruction"].strip())
    if not m:
        return None
    am = _ANSWER_LETTER.search(pair["output"])
    if not am:
        return None

    question = m.group(1).strip()
    orig_opts = [m.group(i).strip() for i in range(2, 6)]
    orig_letter = am.group(1)
    orig_idx = _OPTION_LETTERS.index(orig_letter)

    if shift == 0:
        shifted_opts = orig_opts[:]
        new_idx = orig_idx
    else:
        shifted_opts = orig_opts[-shift:] + orig_opts[:-shift]
        new_idx = (orig_idx + shift) % 4

    new_letter = _OPTION_LETTERS[new_idx]
    options_block = "\n".join(f"{l}. {shifted_opts[i]}"
                              for i, l in enumerate(_OPTION_LETTERS))

    think_m = _THINK_CONTENT.search(pair["output"])
    rationale = think_m.group(1).strip() if think_m else ""
    new_output = _format_sft_output(rationale, new_letter, sft_format)

    return {
        "instruction": f"{question}\n\n{options_block}",
        "output": new_output,
        "_source": pair.get("_source", "qmd_mcq"),
        "_source_collection": pair.get("source_collection", ""),
    }


def _format_sft_output(rationale: str, answer_letter: str, sft_format: str) -> str:
    prefix = f"<think>\n{rationale}\n</think>\n"
    if sft_format == "bare-letter":
        return f"{prefix}{answer_letter}"
    if sft_format == "answer-prefix":
        return f"{prefix}Answer: {answer_letter}"
    if sft_format == "sentence":
        return f"{prefix}The answer is {answer_letter}."
    raise ValueError(f"unknown SFT format: {sft_format}")


def _qmd_question_id(rec: dict) -> str | None:
    """Return the best available QMD question ID.

    The diagnostic augmentation seed set is keyed by benchmark question_id
    (for example, AG-0047). Current QMD-generated records do not carry that
    field; they carry _source_question_id values in a separate training ID
    namespace (for example, AG-TRAIN-0027). We support both so compatible
    future QMD sources work, but callers should treat a zero-overlap warning as
    a real scoping issue rather than assuming the flag had an effect.
    """
    value = rec.get("question_id", rec.get("_source_question_id"))
    return str(value) if value is not None else None


def _load_target_permutation_ids(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    seed_items = data.get("augmentation_seed_set")
    if not isinstance(seed_items, list):
        raise ValueError(f"{path}: expected augmentation_seed_set list")

    ids: set[str] = set()
    for index, item in enumerate(seed_items):
        if not isinstance(item, dict) or "question_id" not in item:
            raise ValueError(
                f"{path}: augmentation_seed_set[{index}] must contain question_id"
            )
        ids.add(str(item["question_id"]))
    return ids


def _load_qmd_mcq(
    path: Path,
    sft_format: str = "bare-letter",
    target_permutation_ids: set[str] | None = None,
    target_permutation_multiplier: int = 2,
    synth_item_count: int | None = None,
    seed: int = 42,
    excluded_source_docs: set[str] | None = None,
    filter_system_prompts: bool = False,
) -> Generator[dict[str, str], None, None]:
    """Load QMD MCQ pairs and emit cyclic-shifted variants with D-position 2x."""
    matched_target_ids: set[str] = set()
    qmd_ids_seen: set[str] = set()
    records: list[dict] = []

    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if "instruction" not in rec or "output" not in rec:
                continue
            if excluded_source_docs and _qmd_source_doc(rec) in excluded_source_docs:
                continue
            records.append(rec)

    if filter_system_prompts:
        records, _ = _filter_system_prompt_pairs(records, source="qmd_mcq")

    if synth_item_count is not None:
        if synth_item_count > len(records):
            raise ValueError(
                f"--synth-item-count {synth_item_count} exceeds available QMD "
                f"records ({len(records)}); this flag subsamples the existing pool "
                "and does not generate new items"
            )
        if synth_item_count < len(records):
            rng = random.Random(seed)
            selected = sorted(rng.sample(range(len(records)), synth_item_count))
            records = [records[i] for i in selected]

    for rec in records:

        question_id = _qmd_question_id(rec)
        if question_id is not None:
            qmd_ids_seen.add(question_id)
        is_target_permutation = (
            target_permutation_ids is not None
            and question_id in target_permutation_ids
        )
        if is_target_permutation and question_id is not None:
            matched_target_ids.add(question_id)

        for shift, weight in enumerate(_SHIFT_WEIGHTS):
            shifted = _cyclic_shift_pair(rec, shift, sft_format=sft_format)
            if shifted is None:
                continue
            effective_weight = weight
            if is_target_permutation:
                effective_weight *= target_permutation_multiplier
            for _ in range(effective_weight):
                yield shifted

    if target_permutation_ids is not None:
        print(
            f"  Target-permutation seed overlap: {len(matched_target_ids)} / "
            f"{len(target_permutation_ids)} seed IDs matched QMD IDs",
            file=sys.stderr,
        )
        if not matched_target_ids:
            qmd_sample = ", ".join(sorted(qmd_ids_seen)[:5]) if qmd_ids_seen else "none"
            seed_sample = ", ".join(sorted(target_permutation_ids)[:5])
            print(
                "  WARNING: --target-permutation-ids had zero overlap with QMD IDs. "
                "Current diagnostics are keyed by benchmark question_id values "
                f"(sample: {seed_sample}), while this QMD source uses IDs like "
                f"{qmd_sample}. The flag was applied literally but did not add "
                "extra weighted coverage for this input.",
                file=sys.stderr,
            )


def _qmd_source_doc(rec: dict) -> str | None:
    """Return source document metadata when a QMD record carries it."""
    for key in ("source_doc", "source_path", "doc_path", "_source_doc", "_source_path"):
        value = rec.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _load_freeform_source_exclusions(path: Path) -> set[str]:
    """Load eval_freeform/source_doc_exclusions.json."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_docs = payload.get("source_docs")
    if not isinstance(source_docs, list) or not all(isinstance(v, str) for v in source_docs):
        raise ValueError(f"{path}: expected source_docs list of strings")
    return set(source_docs)


# ---------------------------------------------------------------------------
# Judge filter (optional)
# ---------------------------------------------------------------------------

def _load_keep_instructions(judged_path: Path) -> set[str]:
    """Return set of instruction hashes that received judge verdict 'keep'."""
    keep_hashes: set[str] = set()
    with open(judged_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("judge_verdict") == "keep":
                    h = hashlib.sha256(
                        rec["instruction"].encode()
                    ).hexdigest()
                    keep_hashes.add(h)
            except (json.JSONDecodeError, KeyError):
                continue
    return keep_hashes


# ---------------------------------------------------------------------------
# Deduplication, split, write
# ---------------------------------------------------------------------------

def _instruction_hash(pair: dict[str, str]) -> str:
    return hashlib.sha256(pair["instruction"].encode()).hexdigest()


def filter_and_dedup(
    pairs: list[dict[str, str]],
    min_output_chars: int,
    max_total_chars: int,
    keep_hashes: set[str] | None,
) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    skipped = {"short": 0, "long": 0, "dup": 0, "judge": 0}

    for pair in pairs:
        if len(pair["output"].strip()) < min_output_chars:
            skipped["short"] += 1
            continue
        if len(pair["instruction"]) + len(pair["output"]) > max_total_chars:
            skipped["long"] += 1
            continue
        h = _instruction_hash(pair)
        if h in seen:
            skipped["dup"] += 1
            continue
        if keep_hashes is not None and h not in keep_hashes:
            skipped["judge"] += 1
            continue
        seen.add(h)
        result.append(pair)

    print(
        f"  Filtered: {skipped['short']} too-short, {skipped['long']} too-long, "
        f"{skipped['dup']} duplicates, {skipped['judge']} judge-rejected",
        file=sys.stderr,
    )
    return result


def _letter_distribution(pairs: list[dict[str, str]]) -> dict[str, int]:
    dist: dict[str, int] = {l: 0 for l in _OPTION_LETTERS}
    for pair in pairs:
        if pair.get("_source") not in {"qmd_mcq", "qmd_mcq_wrapped", "mcq_rationale"}:
            continue
        am = _ANSWER_LETTER.search(pair["output"])
        if am:
            dist[am.group(1)] += 1
    return dist


# ---------------------------------------------------------------------------
# Replay-ratio control (conversation : MCQ mix) — chainlink #26
# ---------------------------------------------------------------------------

def _stratified_subsample_by_letter(
    mcq_pairs: list[dict[str, str]],
    target_n: int,
    seed: int,
) -> list[dict[str, str]]:
    """Subsample MCQ pairs down to target_n, preserving the answer-letter distribution.

    Splits by answer letter first so the existing balance guarantee (see
    _SHIFT_WEIGHTS / gold-letter-balance SETTLED CONSTANT in autoresearch/levers.md)
    survives any ratio setting, instead of drifting under plain random sampling.
    """
    if target_n >= len(mcq_pairs):
        return mcq_pairs

    rng = random.Random(seed)
    by_letter: dict[str, list[dict]] = {l: [] for l in _OPTION_LETTERS}
    unmatched: list[dict] = []
    for pair in mcq_pairs:
        am = _ANSWER_LETTER.search(pair["output"])
        (by_letter[am.group(1)] if am else unmatched).append(pair)
    for bucket in by_letter.values():
        rng.shuffle(bucket)

    per_letter = target_n // 4
    remainder = target_n - per_letter * 4
    result: list[dict] = []
    for i, letter in enumerate(_OPTION_LETTERS):
        take = per_letter + (1 if i < remainder else 0)
        result.extend(by_letter[letter][:take])
    rng.shuffle(result)
    return result


def apply_conversation_ratio(
    conv_pairs: list[dict[str, str]],
    mcq_pairs: list[dict[str, str]],
    ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Subsample the larger pool so conversation:MCQ counts hit `ratio` (conv fraction).

    Subsample-only, by design: never upsamples/duplicates either pool. Duplicating the
    MCQ pool (currently only 311 raw questions before shift-expansion) risks
    memorization; subsampling the larger pool avoids that at the cost of discarding
    some data. If `ratio` can't be reached without upsampling one side (i.e. even
    fully keeping the smaller side and maximally trimming the larger side still
    overshoots or undershoots), the closer side is left at its natural size and the
    other is trimmed as far as it can go.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"conversation ratio must be in (0, 1), got {ratio}")
    n_conv, n_mcq = len(conv_pairs), len(mcq_pairs)
    if n_conv == 0 or n_mcq == 0:
        return conv_pairs, mcq_pairs

    # Target conv count if the full MCQ pool is kept: conv' / (conv' + n_mcq) == ratio.
    target_conv = round(ratio / (1 - ratio) * n_mcq)
    if target_conv <= n_conv:
        conv_pairs = rng_sample(conv_pairs, target_conv, seed)
        return conv_pairs, mcq_pairs

    # Conversation pool is too small to hit the ratio even keeping all of it —
    # trim the MCQ pool instead, keeping the full conversation pool.
    target_mcq = round((1 - ratio) / ratio * n_conv)
    mcq_pairs = _stratified_subsample_by_letter(mcq_pairs, min(target_mcq, n_mcq), seed)
    return conv_pairs, mcq_pairs


def rng_sample(pairs: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    if n >= len(pairs):
        return pairs
    return rng.sample(pairs, n)


def split_and_write(
    pairs: list[dict[str, str]],
    output_dir: Path,
    train_fraction: float,
    seed: int,
    suffix: str = "_v3",
    curriculum_order: str = "none",
    curriculum_report: Path | None = None,
) -> tuple[int, int]:
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * train_fraction))
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    if curriculum_order != "none":
        train, report = order_pairs(train, curriculum_order)
        if curriculum_report is not None:
            curriculum_report.parent.mkdir(parents=True, exist_ok=True)
            curriculum_report.write_text(
                json.dumps(
                    {
                        **report,
                        "seed": seed,
                        "train_fraction": train_fraction,
                        "train_count": len(train),
                        "validation_count": len(val),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    clean_fields = {"instruction", "output"}
    for dest_name, records in (
        (f"train{suffix}.jsonl", train),
        (f"val{suffix}.jsonl", val),
    ):
        dest = output_dir / dest_name
        with open(dest, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(
                    {k: v for k, v in rec.items() if k in clean_fields},
                    ensure_ascii=False,
                ) + "\n")

    return len(train), len(val)


# ---------------------------------------------------------------------------
# Contamination guard — see MCQ_DIAGNOSTIC_PLAN.md
# ---------------------------------------------------------------------------

_KNOWN_CONTAMINATED_MCQ_FILENAMES = frozenset({
    "benchmark_combined.jsonl",
    "comprehensive_agri_benchmark.jsonl",
    "mcq_with_rationale.jsonl",
    "mcq_with_rationale.repaired.jsonl",
    "mcq_with_rationale.repaired.pre_cleanup_backup.jsonl",
})


def _refuse_contaminated_mcq(path: Path, ap: argparse.ArgumentParser) -> None:
    """Refuse known-contaminated MCQ sources. Filename match only — not a substitute
    for an actual question-text dedup check against benchmark_combined.jsonl.
    See MCQ_DIAGNOSTIC_PLAN.md.
    """
    if path.name in _KNOWN_CONTAMINATED_MCQ_FILENAMES:
        ap.error(
            f"'{path.name}' is a known benchmark-contaminated MCQ source (99.9-100% "
            "question-text overlap with benchmark_combined.jsonl — see "
            "MCQ_DIAGNOSTIC_PLAN.md). Build a clean replacement (Phase 3) before "
            "training. If this file genuinely has zero overlap, rename it to bypass "
            "this filename check."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--conversations", metavar="PATH",
                    help="gemma4 ShareGPT JSONL (hf_dataset/train.jsonl)")
    ap.add_argument(
        "--mcq-rationale",
        metavar="PATH",
        default=None,
        help=(
            "MCQ pairs with rationale, verified to have zero question-text overlap "
            "with benchmark_combined.jsonl. No default — see MCQ_DIAGNOSTIC_PLAN.md "
            "Phase 3; the old default (data/sources/mcq_with_rationale.repaired.jsonl) "
            "is 99.9%% derived from the eval benchmark and must not be used."
        ),
    )
    ap.add_argument("--qmd-mcq", metavar="PATH",
                    help="QMD-generated MCQ (data/sources/qmd_mcq_generated.jsonl)")
    ap.add_argument(
        "--adversarial-refusals",
        metavar="PATH",
        default=None,
        help="Reviewed adversarial-refusal JSONL to add as a separate source",
    )
    ap.add_argument(
        "--adversarial-refusal-count",
        type=int,
        default=None,
        metavar="N",
        help="Deterministically subsample the reviewed refusal pool",
    )
    ap.add_argument(
        "--filter-system-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Strip known persona prefixes from composite MCQs and drop persona-only "
            "conversation rows"
        ),
    )
    ap.add_argument(
        "--cloze-source",
        metavar="PATH",
        help="Audit-approved cloze JSONL from generate/check_cloze_contamination.py",
    )
    ap.add_argument(
        "--cloze-count",
        type=int,
        default=None,
        metavar="N",
        help="Deterministically subsample the audited cloze pool to N items",
    )
    ap.add_argument(
        "--sft-format",
        choices=("bare-letter", "answer-prefix", "sentence"),
        default="bare-letter",
        help="QMD MCQ target format (default: bare-letter)",
    )
    ap.add_argument(
        "--target-permutation-ids",
        metavar="PATH",
        default=None,
        help=(
            "JSON report containing augmentation_seed_set question_id objects. "
            "Matching QMD question IDs receive extra cyclic-shift weighting. "
            "Default: disabled."
        ),
    )
    ap.add_argument(
        "--target-permutation-multiplier",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Extra multiplier for QMD records whose question ID appears in "
            "--target-permutation-ids (default: 2)"
        ),
    )
    ap.add_argument(
        "--synth-item-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Deterministically subsample N raw QMD MCQ records before cyclic "
            "shift expansion. Default: use the full existing QMD pool."
        ),
    )
    ap.add_argument(
        "--freeform-source-exclusions",
        metavar="PATH",
        default=None,
        help=(
            "JSON source_doc exclusion list emitted by eval_freeform/check_contamination.py. "
            "QMD records carrying matching source_doc/source_path metadata are skipped."
        ),
    )
    ap.add_argument("--judged-filter", metavar="PATH", default=None,
                    help="Judged JSONL; only keep judge_verdict='keep' instructions")
    ap.add_argument(
        "--conversation-ratio",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Target fraction of the combined conversation+MCQ pool that is "
            "conversation pairs, in (0, 1). Subsample-only (never upsamples/"
            "duplicates either pool); MCQ-side subsampling is stratified by "
            "answer letter to preserve the balance invariant. Default: no ratio "
            "control (current behavior — whatever the source files contain)."
        ),
    )
    ap.add_argument("--output-dir", default="data", metavar="DIR")
    ap.add_argument("--split", type=float, default=0.9, metavar="F")
    ap.add_argument("--min-output-chars", type=int, default=10)
    ap.add_argument("--max-total-chars", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="_v3",
                    help="Suffix for output filenames (default: _v3)")
    ap.add_argument(
        "--curriculum-order",
        choices=("none", "easy-to-hard", "hard-to-easy"),
        default="none",
        help=(
            "Order the training split with the deterministic text-only "
            "difficulty proxy; validation membership remains seed-shuffled"
        ),
    )
    ap.add_argument(
        "--curriculum-report",
        default=None,
        metavar="PATH",
        help="Optional JSON audit report for the curriculum signal and ordering",
    )
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if not any([
        args.conversations,
        args.mcq_rationale,
        args.qmd_mcq,
        args.cloze_source,
        args.adversarial_refusals,
    ]):
        ap.error(
            "At least one data source is required: conversations, MCQ rationale, "
            "QMD MCQ, cloze, or adversarial refusals"
        )
    if not 0.0 < args.split < 1.0:
        ap.error("--split must be between 0 and 1 (exclusive)")
    if args.conversation_ratio is not None and not 0.0 < args.conversation_ratio < 1.0:
        ap.error("--conversation-ratio must be between 0 and 1 (exclusive)")
    if args.target_permutation_multiplier < 1:
        ap.error("--target-permutation-multiplier must be >= 1")
    if args.synth_item_count is not None and args.synth_item_count < 1:
        ap.error("--synth-item-count must be >= 1")
    if args.adversarial_refusal_count is not None and args.adversarial_refusal_count < 1:
        ap.error("--adversarial-refusal-count must be >= 1")
    if args.cloze_count is not None and args.cloze_count < 1:
        ap.error("--cloze-count must be >= 1")
    if args.curriculum_order != "none" and args.curriculum_report is None:
        args.curriculum_report = str(
            Path("results") / f"curriculum_{args.suffix.lstrip('_')}.json"
        )

    source_counts: dict[str, int] = {}
    conv_pairs: list[dict[str, str]] = []
    mcq_pairs: list[dict[str, str]] = []
    cloze_pairs: list[dict[str, str]] = []
    refusal_pairs: list[dict[str, str]] = []
    target_permutation_ids: set[str] | None = None
    freeform_source_exclusions: set[str] | None = None

    if args.target_permutation_ids:
        seed_path = Path(args.target_permutation_ids)
        if not seed_path.exists():
            ap.error(f"Target permutation IDs file not found: {seed_path}")
        try:
            target_permutation_ids = _load_target_permutation_ids(seed_path)
        except ValueError as exc:
            ap.error(str(exc))
        print(
            f"Loaded {len(target_permutation_ids)} target-permutation seed IDs "
            f"from {seed_path}",
            file=sys.stderr,
        )

    if args.freeform_source_exclusions:
        exclusion_path = Path(args.freeform_source_exclusions)
        if not exclusion_path.exists():
            ap.error(f"Free-form source exclusions file not found: {exclusion_path}")
        try:
            freeform_source_exclusions = _load_freeform_source_exclusions(exclusion_path)
        except ValueError as exc:
            ap.error(str(exc))
        print(
            f"Loaded {len(freeform_source_exclusions)} free-form source_doc exclusions "
            f"from {exclusion_path}",
            file=sys.stderr,
        )

    if args.conversations:
        path = Path(args.conversations)
        if not path.exists():
            ap.error(f"Conversations file not found: {path}")
        print(f"Loading conversations from {path} ...", file=sys.stderr)
        conv_pairs = list(_load_conversations(path))
        if args.filter_system_prompts:
            conv_pairs, _ = _filter_system_prompt_pairs(
                conv_pairs, source="gemma4_conversations", strip_composite=False
            )
        source_counts["gemma4_conversations"] = len(conv_pairs)
        print(f"  → {len(conv_pairs)} pairs", file=sys.stderr)

    if args.mcq_rationale:
        path = Path(args.mcq_rationale)
        if not path.exists():
            ap.error(f"MCQ rationale file not found: {path}")
        _refuse_contaminated_mcq(path, ap)
        print(f"Loading MCQ+rationale pairs from {path} ...", file=sys.stderr)
        pairs = list(_load_mcq_with_rationale(path))
        source_counts["mcq_rationale"] = len(pairs)
        mcq_pairs.extend(pairs)
        print(f"  → {len(pairs)} pairs", file=sys.stderr)

    if args.qmd_mcq:
        path = Path(args.qmd_mcq)
        if not path.exists():
            ap.error(f"QMD MCQ file not found: {path}")
        print(f"Loading QMD MCQ from {path} (with cyclic shifts) ...", file=sys.stderr)
        try:
            pairs = list(_load_qmd_mcq(
                path,
                sft_format=args.sft_format,
                target_permutation_ids=target_permutation_ids,
                target_permutation_multiplier=args.target_permutation_multiplier,
                synth_item_count=args.synth_item_count,
                seed=args.seed,
                excluded_source_docs=freeform_source_exclusions,
                filter_system_prompts=args.filter_system_prompts,
            ))
        except ValueError as exc:
            ap.error(str(exc))
        source_counts["qmd_mcq"] = len(pairs)
        mcq_pairs.extend(pairs)
        synth_note = (
            f", raw QMD cap {args.synth_item_count}"
            if args.synth_item_count is not None
            else ""
        )
        print(
            f"  → {len(pairs)} pairs "
            f"(shift weights {_SHIFT_WEIGHTS}, D-pos 2x, "
            f"SFT format {args.sft_format}{synth_note})",
            file=sys.stderr,
        )

    if args.cloze_source:
        path = Path(args.cloze_source)
        if not path.exists():
            ap.error(f"Cloze source file not found: {path}")
        print(f"Loading audited cloze pairs from {path} ...", file=sys.stderr)
        try:
            cloze_pairs = list(_load_cloze_pairs(path))
        except ValueError as exc:
            ap.error(str(exc))
        if args.cloze_count is not None:
            if args.cloze_count > len(cloze_pairs):
                ap.error(
                    f"--cloze-count {args.cloze_count} exceeds audited cloze pool "
                    f"({len(cloze_pairs)})"
                )
            if args.cloze_count < len(cloze_pairs):
                rng = random.Random(args.seed)
                cloze_pairs = sorted(
                    rng.sample(cloze_pairs, args.cloze_count),
                    key=lambda pair: hashlib.sha256(
                        pair["instruction"].encode("utf-8")
                    ).hexdigest(),
                )
        source_counts["chat_cloze"] = len(cloze_pairs)
        print(f"  → {len(cloze_pairs)} audited pairs", file=sys.stderr)

    if args.adversarial_refusals:
        path = Path(args.adversarial_refusals)
        if not path.exists():
            ap.error(f"Adversarial refusal source not found: {path}")
        print(f"Loading reviewed adversarial refusals from {path} ...", file=sys.stderr)
        try:
            refusal_pairs = list(_load_adversarial_refusals(path))
        except ValueError as exc:
            ap.error(str(exc))
        if args.adversarial_refusal_count is not None:
            if args.adversarial_refusal_count > len(refusal_pairs):
                ap.error(
                    f"--adversarial-refusal-count {args.adversarial_refusal_count} "
                    f"exceeds refusal pool ({len(refusal_pairs)})"
                )
            if args.adversarial_refusal_count < len(refusal_pairs):
                rng = random.Random(args.seed)
                refusal_pairs = sorted(
                    rng.sample(refusal_pairs, args.adversarial_refusal_count),
                    key=lambda pair: pair["instruction"],
                )
        source_counts["adversarial_refusal"] = len(refusal_pairs)
        print(f"  → {len(refusal_pairs)} reviewed pairs", file=sys.stderr)

    if args.conversation_ratio is not None and conv_pairs and mcq_pairs:
        n_conv_before, n_mcq_before = len(conv_pairs), len(mcq_pairs)
        conv_pairs, mcq_pairs = apply_conversation_ratio(
            conv_pairs, mcq_pairs, args.conversation_ratio, args.seed
        )
        print(
            f"\nApplied --conversation-ratio {args.conversation_ratio}: "
            f"conversations {n_conv_before} → {len(conv_pairs)}, "
            f"MCQ {n_mcq_before} → {len(mcq_pairs)} "
            f"(subsample-only, letter-balance preserved on the MCQ side)",
            file=sys.stderr,
        )

    if args.filter_system_prompts:
        cloze_pairs, _ = _filter_system_prompt_pairs(
            cloze_pairs, source="chat_cloze", strip_composite=False
        )

    all_pairs: list[dict[str, str]] = (
        conv_pairs + mcq_pairs + cloze_pairs + refusal_pairs
    )
    print(f"\nTotal before filtering: {len(all_pairs)}", file=sys.stderr)

    keep_hashes: set[str] | None = None
    if args.judged_filter:
        judge_path = Path(args.judged_filter)
        if not judge_path.exists():
            ap.error(f"Judged filter file not found: {judge_path}")
        print(f"Loading judge filter from {judge_path} ...", file=sys.stderr)
        keep_hashes = _load_keep_instructions(judge_path)
        print(f"  → {len(keep_hashes)} keep-verdict instructions", file=sys.stderr)

    filtered = filter_and_dedup(
        all_pairs,
        min_output_chars=args.min_output_chars,
        max_total_chars=args.max_total_chars,
        keep_hashes=keep_hashes,
    )
    print(f"Total after filtering:  {len(filtered)}", file=sys.stderr)

    if not filtered:
        print("ERROR: No pairs remain after filtering.", file=sys.stderr)
        sys.exit(1)

    letter_dist = _letter_distribution(filtered)
    mcq_total = sum(letter_dist.values())
    print("\nAnswer letter distribution (MCQ pairs only):", file=sys.stderr)
    for letter, count in sorted(letter_dist.items()):
        pct = 100 * count / mcq_total if mcq_total else 0
        print(f"  {letter}: {count:5d}  ({pct:.1f}%)", file=sys.stderr)

    print("\nSource breakdown:", file=sys.stderr)
    for source, count in source_counts.items():
        print(f"  {source}: {count}", file=sys.stderr)

    output_dir = Path(args.output_dir)
    n_train, n_val = split_and_write(
        filtered,
        output_dir,
        args.split,
        args.seed,
        suffix=args.suffix,
        curriculum_order=args.curriculum_order,
        curriculum_report=(
            Path(args.curriculum_report) if args.curriculum_report else None
        ),
    )
    print(f"\nWrote {n_train} train pairs → {output_dir}/train{args.suffix}.jsonl",
          file=sys.stderr)
    print(f"Wrote {n_val}  val pairs   → {output_dir}/val{args.suffix}.jsonl",
          file=sys.stderr)


if __name__ == "__main__":
    main()
