#!/usr/bin/env python3
"""Annotate MCQ dev rows with J-space (Jacobian-lens) answer-letter signals.

Phase 5 / J1 (chainlink #55). For each scorable MCQ dev row this script reads
out the per-layer rank/margin of the gold answer letter through a fitted
Jacobian lens, probes option-rotation and answer-surface stability, and assigns
one of six buckets (robust / absent / late_weak / present_unsurfaced /
format_sensitive / position_unstable) plus a recommended intervention.

Proposer guidance only: nothing here feeds objective weights or gate logic.
Sealed test paths (eval_freeform/test.jsonl) are never read; --dataset is
checked against a denylist before any file IO.

    conda run -n qwen3-diagnostics python autoresearch/jspace_annotator.py \
        --model outputs/merged-model-v3-clean \
        --dataset data/val_v3.jsonl \
        --lens-path results/jspace/jacobian_lens.pt --fit-if-missing \
        --output-signals results/jspace_item_signals.jsonl \
        --output-summary results/jspace_category_summary.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from score_mcq_ll import (
    LETTERS,
    DEFAULT_DIAGNOSTICS_DIR,
    import_diagnostics_helpers,
    normalize_records,
    repo_root,
)

log = logging.getLogger(__name__)

DEFAULT_JLENS_DIR = Path("jacobian-lens")
DIGITS = ("1", "2", "3", "4")
#: Option-content rotations probed for position instability. Rotation k moves
#: the content of slot i to slot (i + k) % 4; the gold letter moves with it.
ROTATIONS = (1, 2, 3)

SEALED_PATH_FRAGMENTS = ("eval_freeform/test.jsonl",)

BUCKETS = (
    "robust",
    "absent",
    "late_weak",
    "present_unsurfaced",
    "format_sensitive",
    "position_unstable",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "prompt_mode": "chatml_empty_think",
    "target_tokens": "answer_letters",
    "lens_layers": "all_fitted",
    "lens_max_seq_len": 1024,
    "fit": {
        "prompt_source": "data/train_v3.jsonl",
        "prompt_count": 100,
        "max_seq_len": 128,
        "checkpoint_every": 10,
        "dim_batch": 32,
    },
    "thresholds": {
        # first layer (fraction of n_layers) by which gold must lead the
        # letter readout for the item to count as early/robust
        "robust_max_first_layer_frac": 0.7,
        # fraction of scored layers after first gold lead that must keep gold on top
        "robust_min_top1_frac_after_first": 0.8,
        # final-layer gold-vs-best-other-letter logprob margin below which a
        # correct item is only weakly held
        "weak_margin_nats": 1.0,
        # layers at/after this fraction of n_layers count as "late"
        "late_layer_frac": 0.85,
        # rotation flips (out of len(ROTATIONS)) at/above which the item is
        # position_unstable
        "position_flip_min": 2,
        # full-vocab gold rank that still counts as marginal evidence when the
        # gold letter never leads the letter readout
        "absent_rank_threshold": 20,
    },
    "interventions": {
        "robust": "do_not_target",
        "absent": "add_category_data",
        "late_weak": "add_rationale_targets",
        "present_unsurfaced": "adjust_answer_surface",
        "format_sensitive": "adjust_answer_surface",
        "position_unstable": "targeted_permutation",
    },
    "priority_categories": [
        "natural_fertilizers",
        "sustainable_water_management",
        "soil_science_and_microbiology",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_not_sealed(path: str | Path) -> Path:
    resolved = Path(path)
    as_posix = resolved.as_posix()
    for fragment in SEALED_PATH_FRAGMENTS:
        if fragment in as_posix:
            raise SystemExit(f"refusing to read sealed test path: {path}")
    return resolved


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def import_jlens(jlens_dir: Path):
    resolved = jlens_dir.expanduser().resolve()
    if not (resolved / "jlens").is_dir():
        raise FileNotFoundError(
            f"jacobian-lens repo not found at {resolved}; pass --jlens-dir"
        )
    sys.path.insert(0, str(resolved))
    import jlens  # type: ignore[import-not-found]

    return jlens


def load_category_map(output_dir: Path) -> dict[str, str]:
    """Map normalized question text -> generation category slug."""

    def norm(text: str) -> str:
        return " ".join(text.lower().split())

    mapping: dict[str, str] = {}
    for cat_file in sorted(output_dir.glob("*.jsonl")):
        for line in cat_file.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            question = json.loads(line).get("question") or ""
            if question:
                mapping[norm(question)] = cat_file.stem
    log.info("category map: %d questions from %s", len(mapping), output_dir)
    return mapping


def single_token_ids(tokenizer, symbols: tuple[str, ...]) -> dict[str, int]:
    """Single-token id per symbol, tolerating a required space prefix."""
    ids: dict[str, int] = {}
    for symbol in symbols:
        for candidate in (symbol, " " + symbol):
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(token_ids) == 1:
                ids[symbol] = token_ids[0]
                break
        else:
            raise ValueError(f"symbol {symbol!r} is not a single token")
    return ids


def rotate_item(item: dict[str, Any], shift: int) -> dict[str, Any]:
    """Move option content i -> slot (i+shift)%4 and relocate the gold letter."""
    rotated = dict(item)
    for i, letter in enumerate(LETTERS):
        rotated[LETTERS[(i + shift) % len(LETTERS)]] = item[letter]
    gold_index = LETTERS.index(item["answer"])
    rotated["answer"] = LETTERS[(gold_index + shift) % len(LETTERS)]
    return rotated


def format_digit_prompt(item: dict[str, Any], tokenizer) -> str:
    """The canonical surface with 1/2/3/4 option labels instead of A/B/C/D."""
    question_text = (
        f"{item['question']}\n\n"
        f"1. {item['A']}\n"
        f"2. {item['B']}\n"
        f"3. {item['C']}\n"
        f"4. {item['D']}"
    )
    messages = [{"role": "user", "content": question_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt + "<think>\n</think>\n"


def fit_prompts_from_source(source: Path, count: int) -> list[str]:
    """Deterministic stride sample of instruction texts from a held-in split.

    Texts under 20 words are dropped up front: jlens.fit skips prompts at or
    below its skip_first token threshold, and pre-filtering keeps the fitted
    prompt count equal to the requested count.
    """
    rows = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    texts = [str(row.get("instruction") or "").strip() for row in rows]
    texts = [text for text in texts if len(text.split()) >= 20]
    if len(texts) <= count:
        return texts
    stride = len(texts) / count
    return [texts[int(i * stride)] for i in range(count)]


def ensure_lens(args, config, jlens, lens_model) -> tuple[Any, dict[str, Any]]:
    """Load the lens at --lens-path, fitting it first when --fit-if-missing."""
    lens_path = Path(args.lens_path)
    metadata_path = lens_path.with_name(lens_path.stem + "_metadata.json")
    if not lens_path.exists():
        if not args.fit_if_missing:
            raise SystemExit(
                f"lens not found at {lens_path}; rerun with --fit-if-missing"
            )
        fit_cfg = config["fit"]
        source = check_not_sealed(fit_cfg["prompt_source"])
        prompts = fit_prompts_from_source(source, int(fit_cfg["prompt_count"]))
        log.info("fitting lens on %d prompts from %s", len(prompts), source)
        lens_path.parent.mkdir(parents=True, exist_ok=True)
        lens = jlens.fit(
            lens_model,
            prompts=prompts,
            max_seq_len=int(fit_cfg["max_seq_len"]),
            dim_batch=int(fit_cfg.get("dim_batch", 32)),
            checkpoint_path=str(lens_path.with_suffix(".ckpt.pt")),
            checkpoint_every=int(fit_cfg["checkpoint_every"]),
        )
        lens.save(str(lens_path))
        metadata = {
            "model": args.model,
            "prompt_source": source.as_posix(),
            "prompt_count": lens.n_prompts,
            "prompt_count_requested": len(prompts),
            "source_layers": lens.source_layers,
            "max_seq_len": int(fit_cfg["max_seq_len"]),
            "lens_sha256": sha256_file(lens_path),
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
    lens = jlens.JacobianLens.load(str(lens_path))
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.setdefault("source_layers", lens.source_layers)
    metadata["lens_sha256"] = sha256_file(lens_path)
    metadata["lens_path"] = lens_path.as_posix()
    metadata["n_prompts"] = lens.n_prompts
    return lens, metadata


def gold_letter_layer_signals(
    lens, lens_model, prompt: str, letter_ids: dict[str, int],
    gold: str, layers: list[int], max_seq_len: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Per-layer gold rank/margin at the answer position, plus final logprobs."""
    import torch
    import torch.nn.functional as F

    lens_logits, model_logits, _ = lens.apply(
        lens_model, prompt, layers=layers, positions=[-1], max_seq_len=max_seq_len
    )
    gold_id = letter_ids[gold]
    other_ids = [letter_ids[l] for l in LETTERS if l != gold]

    per_layer: list[dict[str, Any]] = []
    for layer in layers:
        logits = lens_logits[layer][0]
        gold_logit = logits[gold_id].item()
        best_other = max(logits[i].item() for i in other_ids)
        per_layer.append(
            {
                "layer": layer,
                "gold_margin": round(gold_logit - best_other, 4),
                "gold_vocab_rank": int((logits > logits[gold_id]).sum().item()),
            }
        )

    final_logprobs = F.log_softmax(model_logits[0], dim=-1)
    final = {l: final_logprobs[letter_ids[l]].item() for l in LETTERS}
    return per_layer, final


def assign_bucket(
    *,
    correct: bool,
    final_margin: float,
    per_layer: list[dict[str, Any]],
    n_layers: int,
    rotation_flips: int,
    surface_disagrees: bool,
    thresholds: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Total decision tree: every item receives exactly one bucket."""
    lead_layers = [row["layer"] for row in per_layer if row["gold_margin"] > 0]
    first_lead = lead_layers[0] if lead_layers else None
    late_start = int(thresholds["late_layer_frac"] * n_layers)
    best_vocab_rank = min(row["gold_vocab_rank"] for row in per_layer)

    evidence = {
        "first_gold_lead_layer": first_lead,
        "n_gold_lead_layers": len(lead_layers),
        "late_start_layer": late_start,
        "best_gold_vocab_rank": best_vocab_rank,
        "rotation_flips": rotation_flips,
        "surface_disagrees": surface_disagrees,
        "final_margin": round(final_margin, 4),
    }

    if rotation_flips >= int(thresholds["position_flip_min"]):
        return "position_unstable", evidence
    if surface_disagrees:
        return "format_sensitive", evidence

    if correct:
        early_enough = (
            first_lead is not None
            and first_lead <= int(thresholds["robust_max_first_layer_frac"] * n_layers)
        )
        stable = False
        if first_lead is not None:
            later = [row for row in per_layer if row["layer"] >= first_lead]
            stable = (
                sum(1 for row in later if row["gold_margin"] > 0) / len(later)
                >= float(thresholds["robust_min_top1_frac_after_first"])
            )
        strong = final_margin >= float(thresholds["weak_margin_nats"])
        if early_enough and stable and strong and rotation_flips == 0:
            return "robust", evidence
        return "late_weak", evidence

    if any(layer < late_start for layer in lead_layers):
        return "present_unsurfaced", evidence
    if lead_layers or best_vocab_rank <= int(thresholds["absent_rank_threshold"]):
        return "late_weak", evidence
    return "absent", evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate MCQ dev rows with Jacobian-lens answer-letter buckets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to merged model directory")
    parser.add_argument("--dataset", default="data/val_v3.jsonl")
    parser.add_argument("--lens-path", default="results/jspace/jacobian_lens.pt")
    parser.add_argument(
        "--fit-if-missing",
        action="store_true",
        help="Fit and save a lens at --lens-path when none exists",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Stop after the lens exists; skip annotation (resumable fit step)",
    )
    parser.add_argument("--config", default=None, help="JSON overrides for DEFAULT_CONFIG")
    parser.add_argument("--jlens-dir", default=str(DEFAULT_JLENS_DIR))
    parser.add_argument("--diagnostics-dir", default=str(DEFAULT_DIAGNOSTICS_DIR))
    parser.add_argument("--category-map-dir", default="mcq_generation/output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="First N rows; smoke checks")
    parser.add_argument("--output-signals", default="results/jspace_item_signals.jsonl")
    parser.add_argument("--output-summary", default="results/jspace_category_summary.json")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import torch

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    config = DEFAULT_CONFIG
    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config = deep_merge(config, overrides)
    thresholds = config["thresholds"]

    dataset_path = check_not_sealed(args.dataset)
    helpers = import_diagnostics_helpers(Path(args.diagnostics_dir))
    jlens = import_jlens(Path(args.jlens_dir))

    rows = helpers["load_benchmark"](dataset_path)
    records, skipped = normalize_records(rows)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        parser.error(f"no scorable MCQ rows found in {args.dataset}")

    category_map = load_category_map(Path(args.category_map_dir))

    def category_of(record: dict[str, Any]) -> str:
        return category_map.get(" ".join(record["question"].lower().split()), "unknown")

    model, tokenizer = helpers["load_model_and_tokenizer"](
        args.model, full_precision=True, device=args.device
    )
    # force_bos=False: the tokenizer is shared with score_symbol, and adding a
    # BOS would break parity with the frozen score_mcq_ll.py conventions.
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    lens, lens_metadata = ensure_lens(args, config, jlens, lens_model)
    if args.fit_only:
        print(f"lens ready at {args.lens_path}")
        print(f"lens_sha256={lens_metadata['lens_sha256']}")
        return 0

    layers = (
        lens.source_layers
        if config["lens_layers"] == "all_fitted"
        else list(config["lens_layers"])
    )
    letter_ids = helpers["get_letter_token_ids"](tokenizer)
    digit_ids = single_token_ids(tokenizer, DIGITS)
    n_layers = lens_model.n_layers
    max_seq_len = int(config["lens_max_seq_len"])

    signal_rows: list[dict[str, Any]] = []
    log.info("annotating %d MCQ rows from %s", len(records), args.dataset)
    for index, item in enumerate(records):
        prompt = helpers["format_mcq_prompt"](item, tokenizer, after_think=True)
        gold = item["answer"]

        per_layer, final_logprobs = gold_letter_layer_signals(
            lens, lens_model, prompt, letter_ids, gold, layers, max_seq_len
        )
        top_letter = max(LETTERS, key=lambda l: final_logprobs[l])
        correct = top_letter == gold
        final_margin = final_logprobs[gold] - max(
            final_logprobs[l] for l in LETTERS if l != gold
        )

        rotation_flips = 0
        for shift in ROTATIONS:
            rotated = rotate_item(item, shift)
            rotated_prompt = helpers["format_mcq_prompt"](
                rotated, tokenizer, after_think=True
            )
            rotated_logprobs = helpers["score_symbol"](
                model, tokenizer, rotated_prompt, letter_ids, args.device
            )
            rotated_top = max(LETTERS, key=lambda l: rotated_logprobs[l])
            if (rotated_top == rotated["answer"]) != correct:
                rotation_flips += 1

        digit_prompt = format_digit_prompt(item, tokenizer)
        digit_logprobs = helpers["score_symbol"](
            model, tokenizer, digit_prompt, digit_ids, args.device
        )
        gold_digit = DIGITS[LETTERS.index(gold)]
        digit_top = max(DIGITS, key=lambda d: digit_logprobs[d])
        surface_disagrees = (digit_top == gold_digit) != correct

        bucket, evidence = assign_bucket(
            correct=correct,
            final_margin=final_margin,
            per_layer=per_layer,
            n_layers=n_layers,
            rotation_flips=rotation_flips,
            surface_disagrees=surface_disagrees,
            thresholds=thresholds,
        )

        signal_rows.append(
            {
                "item_id": item["question_id"],
                "source_split": dataset_path.as_posix(),
                "category": category_of(item),
                "gold": gold,
                "top_letter": top_letter,
                "correct_by_argmax": correct,
                "gold_logprob": final_logprobs[gold],
                "final_margin": round(final_margin, 4),
                "per_layer": per_layer,
                "bucket": bucket,
                "recommended_intervention": config["interventions"][bucket],
                "evidence": evidence,
            }
        )
        if (index + 1) % 20 == 0:
            log.info("  %d/%d", index + 1, len(records))

    signals_path = Path(args.output_signals)
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    with signals_path.open("w", encoding="utf-8") as handle:
        for row in signal_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    bucket_counts = Counter(row["bucket"] for row in signal_rows)
    by_category: dict[str, dict[str, Any]] = {}
    for row in signal_rows:
        entry = by_category.setdefault(
            row["category"],
            {"n_items": 0, "n_correct": 0, "buckets": {b: 0 for b in BUCKETS}},
        )
        entry["n_items"] += 1
        entry["n_correct"] += int(row["correct_by_argmax"])
        entry["buckets"][row["bucket"]] += 1
    for name, entry in by_category.items():
        entry["accuracy"] = round(entry["n_correct"] / entry["n_items"], 4)
        entry["priority"] = name in config["priority_categories"]

    summary = {
        "model": args.model,
        "dataset": dataset_path.as_posix(),
        "scorable_mcq_rows": len(records),
        "skipped_non_mcq_rows": len(skipped),
        "bucketed_rows": len(signal_rows),
        "bucketed_fraction_of_scorable": round(len(signal_rows) / len(records), 4),
        "bucket_counts": dict(bucket_counts),
        "intervention_counts": dict(
            Counter(row["recommended_intervention"] for row in signal_rows)
        ),
        "by_category": dict(sorted(by_category.items())),
        "priority_categories": {
            name: by_category.get(name) for name in config["priority_categories"]
        },
        "lens": lens_metadata,
        "config": config,
        "sealed_test_paths_read": [],
        "output_signals": signals_path.as_posix(),
    }
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {signals_path.as_posix()}")
    print(f"wrote {summary_path.as_posix()}")
    print(f"bucketed {len(signal_rows)}/{len(records)} scorable MCQ rows")
    print(f"bucket_counts={json.dumps(dict(bucket_counts), sort_keys=True)}")
    print(f"lens_sha256={lens_metadata.get('lens_sha256', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
