#!/usr/bin/env python3
"""Autoresearch harness entrypoint.

This script is the integration point for Phase 4 search runs. It deliberately
invokes the data prep, training, and merge scripts as subprocesses rather than
importing them. In particular, train.py and merge_adapter.py both rely on
Unsloth import-time patching, so each step gets a clean Python process.

Default data-source choice for this skeleton:
  - conversations: data/sources/gemma4/hf_dataset/train.jsonl
  - QMD MCQs:      data/sources/qmd_mcq_all_phase3.jsonl
  - MCQ rationale: disabled by default, because the historical default is
    benchmark-contaminated and prepare_dataset_v3.py now refuses it by name.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONVERSATIONS = "data/sources/gemma4/hf_dataset/train.jsonl"
DEFAULT_QMD_MCQ = "data/sources/qmd_mcq_all_phase3.jsonl"
DEFAULT_DATASET_OUTPUT_DIR = "data"
DEFAULT_OUTPUT_ROOT = "outputs"

# J-space routing hooks (chainlink #58). Buckets must match the frozen
# jspace_config.json surface; the prefix map covers every _source_question_id
# prefix in the QMD MCQ pool (verified 311/311).
JSPACE_BUCKETS = (
    "robust",
    "absent",
    "late_weak",
    "present_unsurfaced",
    "format_sensitive",
    "position_unstable",
)
JSPACE_ID_PREFIX_CATEGORIES = {
    "AG": "agroecology",
    "CONV": "conversations",
    "IFS": "integrated_farming_systems",
    "ITK": "indigenous_and_traditional_knowledge",
    "NF": "natural_fertilizers",
    "OF": "organic_farming",
    "PC": "permaculture",
    "RA": "regenerative_agriculture",
    "SS": "soil_science_and_microbiology",
    "SWM": "sustainable_water_management",
}
SEALED_PATH_FRAGMENTS = ("eval_freeform/test.jsonl",)


def _run(argv: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, check=True)


def _dataset_suffix(args: argparse.Namespace) -> str:
    return args.dataset_suffix or f"_{args.run_name}"


def _dataset_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    suffix = _dataset_suffix(args)
    output_dir = Path(args.dataset_output_dir)
    return output_dir / f"train{suffix}.jsonl", output_dir / f"val{suffix}.jsonl"


def _prepare_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable,
        "prepare_dataset_v3.py",
        "--output-dir",
        args.dataset_output_dir,
        "--suffix",
        _dataset_suffix(args),
        "--split",
        str(args.split),
        "--seed",
        str(args.seed),
        "--min-output-chars",
        str(args.min_output_chars),
        "--max-total-chars",
        str(args.max_total_chars),
        "--sft-format",
        args.sft_format,
    ]
    if args.conversations:
        argv.extend(["--conversations", args.conversations])
    if args.mcq_rationale:
        argv.extend(["--mcq-rationale", args.mcq_rationale])
    if args.mcq_rationale_categories:
        argv.extend(["--mcq-rationale-categories", args.mcq_rationale_categories])
    if args.rationale_overlay:
        argv.append("--rationale-overlay")
    if args.qmd_mcq:
        argv.extend(["--qmd-mcq", args.qmd_mcq])
    if args.adversarial_refusals:
        argv.extend(["--adversarial-refusals", args.adversarial_refusals])
    if args.adversarial_refusal_count is not None:
        argv.extend(["--adversarial-refusal-count", str(args.adversarial_refusal_count)])
    if args.filter_system_prompts:
        argv.append("--filter-system-prompts")
    if args.cloze_source:
        argv.extend(["--cloze-source", args.cloze_source])
    if args.cloze_count is not None:
        argv.extend(["--cloze-count", str(args.cloze_count)])
    if args.conversation_ratio is not None:
        argv.extend(["--conversation-ratio", str(args.conversation_ratio)])
    if args.target_permutation_ids:
        argv.extend(["--target-permutation-ids", args.target_permutation_ids])
    if args.synth_item_count is not None:
        argv.extend(["--synth-item-count", str(args.synth_item_count)])
    if args.curriculum_order != "none":
        argv.extend(["--curriculum-order", args.curriculum_order])
    if args.curriculum_report:
        argv.extend(["--curriculum-report", args.curriculum_report])
    return argv


def _cloze_commands(args: argparse.Namespace) -> list[list[str]]:
    """Return the reproducible cloze generation/audit preflight commands."""
    if not args.cloze_conversations:
        return []
    suffix = args.run_name.replace("/", "_")
    raw_path = Path(args.dataset_output_dir) / f"cloze_raw_{suffix}.jsonl"
    clean_path = Path(args.dataset_output_dir) / f"cloze_clean_{suffix}.jsonl"
    report_path = Path("results") / f"cloze_contamination_{suffix}.json"
    args.cloze_source = str(clean_path)
    return [
        [
            sys.executable,
            "scripts/generate_cloze_from_conversations.py",
            "--input",
            args.cloze_conversations,
            "--output",
            str(raw_path),
            "--seed",
            str(args.seed),
            "--maximum-answer-frequency",
            str(args.cloze_maximum_answer_frequency),
        ],
        [
            sys.executable,
            "scripts/check_cloze_contamination.py",
            "--cloze",
            str(raw_path),
            "--benchmark",
            args.cloze_benchmark,
            "--output",
            str(clean_path),
            "--report",
            str(report_path),
        ],
    ]


def _train_argv(args: argparse.Namespace) -> list[str]:
    train_data, val_data = _dataset_paths(args)
    lora_output_dir = Path(args.output_root) / f"lora-adapter-{args.run_name}"
    argv = [
        sys.executable,
        "train.py",
        "--model",
        args.model,
        "--train-data",
        str(train_data),
        "--val-data",
        str(val_data),
        "--output-dir",
        str(lora_output_dir),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--lr-scheduler",
        args.lr_scheduler,
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--target-modules",
        args.target_modules,
        "--max-seq-len",
        str(args.max_seq_len),
        "--chat-template-variant",
        args.chat_template_variant,
        "--train-order",
        args.train_order,
        "--seed",
        str(args.training_seed),
    ]
    if args.early_stop_patience is not None:
        argv.extend(["--early-stop-patience", str(args.early_stop_patience)])
    if args.packing:
        argv.append("--packing")
    if args.wandb:
        argv.append("--wandb")
    if args.resume:
        argv.append("--resume")
    return argv


def _merge_argv(args: argparse.Namespace) -> list[str]:
    lora_output_dir = Path(args.output_root) / f"lora-adapter-{args.run_name}"
    merged_output_dir = Path(args.output_root) / f"merged-model-{args.run_name}"
    argv = [
        sys.executable,
        "merge_adapter.py",
        "--base",
        args.merge_base,
        "--adapter",
        str(lora_output_dir),
        "--output",
        str(merged_output_dir),
        "--dtype",
        args.merge_dtype,
        "--max-seq-len",
        str(args.max_seq_len),
    ]
    if args.force_peft:
        argv.append("--force-peft")
    return argv


def _jspace_category(record: dict[str, Any]) -> str:
    prefix = str(record.get("_source_question_id", "")).split("-")[0]
    return JSPACE_ID_PREFIX_CATEGORIES.get(prefix, "unknown")


def _jspace_select(args: argparse.Namespace) -> Path:
    """Filter the raw QMD MCQ source by J-space bucket/category evidence.

    Returns the path of the filtered source file to hand to
    prepare_dataset_v3.py. Letter balance is unaffected: it is enforced
    downstream by cyclic-shift expansion, exactly as with --synth-item-count.
    """
    signals_path = Path(args.jspace_signals)
    for fragment in SEALED_PATH_FRAGMENTS:
        if fragment in signals_path.as_posix():
            raise SystemExit(f"refusing to read sealed test path: {signals_path}")
    signals = [
        json.loads(line)
        for line in signals_path.open(encoding="utf-8")
        if line.strip()
    ]

    buckets = args.jspace_bucket.split(",") if args.jspace_bucket else []
    if args.jspace_category:
        target_categories = sorted(set(args.jspace_category.split(",")))
    else:
        target_categories = sorted(
            {
                row["category"]
                for row in signals
                if row["bucket"] in buckets and row["category"] != "unknown"
            }
        )
    evidence_counts: dict[str, dict[str, int]] = {}
    for row in signals:
        if row["category"] in target_categories and (
            not buckets or row["bucket"] in buckets
        ):
            per_cat = evidence_counts.setdefault(row["category"], {})
            per_cat[row["bucket"]] = per_cat.get(row["bucket"], 0) + 1

    source_path = Path(args.qmd_mcq)
    records = [
        json.loads(line)
        for line in source_path.open(encoding="utf-8")
        if line.strip()
    ]
    targeted_ids = sorted(
        str(r["_source_question_id"])
        for r in records
        if _jspace_category(r) in target_categories
    )

    capped_ids = targeted_ids
    if args.jspace_max_items is not None and len(targeted_ids) > args.jspace_max_items:
        capped_ids = sorted(
            random.Random(args.seed).sample(targeted_ids, args.jspace_max_items)
        )

    if args.jspace_action == "rationale_overlay":
        if not args.mcq_rationale:
            raise SystemExit(
                "--jspace-action rationale_overlay requires --mcq-rationale "
                "to point at an audit-approved rationale source"
            )
        # Keep the complete baseline QMD pool.  J-space selects the categories
        # whose clean rationale rows are replayed; it does not replace the
        # ordinary MCQ stream with a validation-derived subset.
        kept_ids = {
            str(r["_source_question_id"]) for r in records
        }
        args.mcq_rationale_categories = ",".join(target_categories)
        args.rationale_overlay = True
    elif args.jspace_action == "include":
        kept_ids = set(capped_ids)
    elif args.jspace_action == "exclude":
        kept_ids = {
            str(r["_source_question_id"])
            for r in records
            if _jspace_category(r) not in target_categories
        }
    else:  # cap: keep everything except targeted overflow beyond the cap
        dropped = set(targeted_ids) - set(capped_ids)
        kept_ids = {
            str(r["_source_question_id"]) for r in records
        } - dropped
    kept = [r for r in records if str(r["_source_question_id"]) in kept_ids]

    suffix = _dataset_suffix(args)
    filtered_path = Path(args.dataset_output_dir) / f"qmd_mcq_jspace{suffix}.jsonl"
    with filtered_path.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    kept_by_category: dict[str, int] = {}
    for record in kept:
        category = _jspace_category(record)
        kept_by_category[category] = kept_by_category.get(category, 0) + 1
    artifact = {
        "jspace_signals": signals_path.as_posix(),
        "action": args.jspace_action,
        "buckets": buckets,
        "target_categories": target_categories,
        "rationale_source": args.mcq_rationale if args.jspace_action == "rationale_overlay" else None,
        "rationale_overlay": args.jspace_action == "rationale_overlay",
        "signal_evidence_counts": evidence_counts,
        "seed": args.seed,
        "source": source_path.as_posix(),
        "source_records": len(records),
        "targeted_records": len(targeted_ids),
        "kept_records": len(kept),
        "kept_by_category": dict(sorted(kept_by_category.items())),
        "kept_targeted_ids": sorted(kept_ids & set(targeted_ids)),
        "dropped_ids": sorted(
            {str(r["_source_question_id"]) for r in records} - kept_ids
        ),
        "filtered_qmd_path": filtered_path.as_posix(),
    }
    artifact_path = Path(args.output_root) / f"jspace-selection-{args.run_name}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"jspace selection: action={args.jspace_action} buckets={buckets or 'n/a'} "
        f"categories={target_categories}"
    )
    print(
        f"jspace selection: kept {len(kept)}/{len(records)} QMD records "
        f"({len(targeted_ids)} targeted); artifact {artifact_path.as_posix()}"
    )
    return filtered_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Checksum-gated autoresearch fine-tune harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Bookkeeping / control.
    parser.add_argument("--run-name", default=None, help="Run suffix for output paths")
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="YAML config path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run checksum check, print subprocess argv, then stop",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-output-dir", default=DEFAULT_DATASET_OUTPUT_DIR)
    parser.add_argument(
        "--dataset-suffix",
        default=None,
        help="Suffix for prepared train/val JSONL files; defaults to _<run-name>",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-seed",
        type=int,
        default=42,
        help=(
            "Training-only initialization/sampler seed. Keep --seed fixed "
            "when confirming a candidate on the same validation split."
        ),
    )
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--min-output-chars", type=int, default=10)
    parser.add_argument("--max-total-chars", type=int, default=8192)

    # Data/source controls. The source defaults intentionally avoid the known
    # contaminated MCQ-rationale file; callers may pass a verified clean source.
    parser.add_argument("--conversations", default=DEFAULT_CONVERSATIONS)
    parser.add_argument("--mcq-rationale", default=None)
    parser.add_argument(
        "--mcq-rationale-categories",
        default=None,
        help="Comma-separated clean rationale source categories to replay",
    )
    parser.add_argument(
        "--rationale-overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preserve balanced cyclic rationale replay rows through deduplication",
    )
    parser.add_argument("--qmd-mcq", default=DEFAULT_QMD_MCQ)
    parser.add_argument(
        "--adversarial-refusals",
        default=None,
        help="Reviewed adversarial-refusal JSONL to add as a separate source",
    )
    parser.add_argument("--adversarial-refusal-count", type=int, default=None)
    parser.add_argument(
        "--filter-system-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Strip composite persona prefixes and drop persona-only rows",
    )
    parser.add_argument(
        "--cloze-source",
        default=None,
        help="Audit-approved cloze JSONL to mix into the training pool",
    )
    parser.add_argument(
        "--cloze-conversations",
        default=None,
        help="Raw ShareGPT JSONL from which the cloze preflight builds items",
    )
    parser.add_argument(
        "--cloze-benchmark",
        default=None,
        help="Benchmark JSONL used by the cloze contamination preflight",
    )
    parser.add_argument(
        "--cloze-maximum-answer-frequency",
        type=int,
        default=50,
        help="Cap repeated normalized cloze answers during preflight generation",
    )
    parser.add_argument("--cloze-count", type=int, default=None)

    # SEARCH DIMENSION: Tier 1 hyperparameters.
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("cosine", "linear", "constant"),
        default="cosine",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # SEARCH DIMENSION: Tier 2 data/format knobs.
    parser.add_argument("--conversation-ratio", type=float, default=None)
    parser.add_argument("--target-permutation-ids", default=None, metavar="PATH")
    parser.add_argument(
        "--sft-format",
        choices=("bare-letter", "answer-prefix", "sentence"),
        default="bare-letter",
    )
    parser.add_argument(
        "--synth-item-count",
        type=int,
        default=None,
        help="Cap/subsample raw QMD MCQ records before cyclic shift expansion",
    )
    parser.add_argument(
        "--curriculum-order",
        choices=("none", "easy-to-hard", "hard-to-easy"),
        default="none",
        help="Text-only curriculum ordering applied to the training split",
    )
    parser.add_argument(
        "--curriculum-report",
        default=None,
        help="JSON audit report path for the curriculum signal",
    )
    parser.add_argument(
        "--chat-template-variant",
        choices=("hardcoded-v1", "tokenizer-v1"),
        default="hardcoded-v1",
        help="Training ChatML rendering variant",
    )
    parser.add_argument(
        "--train-order",
        choices=("shuffled", "sequential"),
        default="shuffled",
        help="Trainer sampler order; sequential is for curriculum experiments",
    )

    # Tier 3 J-space routing hooks (chainlink #58). Opt-in and narrow:
    # baseline behavior is unchanged unless --jspace-signals is passed.
    parser.add_argument(
        "--jspace-signals",
        default=None,
        metavar="PATH",
        help="J-space item-signals JSONL (results/jspace_item_signals.jsonl)",
    )
    parser.add_argument(
        "--jspace-bucket",
        default=None,
        help="Comma-separated J-space buckets whose evidence picks target categories",
    )
    parser.add_argument(
        "--jspace-category",
        default=None,
        help="Comma-separated category override; defaults to categories with bucket evidence",
    )
    parser.add_argument(
        "--jspace-max-items",
        type=int,
        default=None,
        help="Cap on targeted QMD records (seeded deterministic subsample)",
    )
    parser.add_argument(
        "--jspace-action",
        choices=("include", "exclude", "cap", "rationale_overlay"),
        default="include",
        help=(
            "include: keep only targeted records; exclude: drop them; cap: "
            "subsample them; rationale_overlay: keep the baseline QMD pool and "
            "replay clean rationale rows for the selected categories"
        ),
    )

    # Pass-through model/merge controls; not search dimensions.
    parser.add_argument("--model", default="unsloth/Qwen3-0.6B")
    parser.add_argument("--merge-base", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--merge-dtype",
        choices=("float16", "fp16", "bfloat16", "bf16", "float32", "fp32"),
        default="float16",
    )
    parser.add_argument("--force-peft", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _load_config_defaults(
    config_path: Path,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        parser.error(f"{config_path}: expected a flat YAML key:value map")

    schema_version = data.get("schema_version")
    if schema_version != 1:
        parser.error(f"{config_path}: schema_version must be 1")

    allowed = {
        action.dest
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    config_keys = set(data) - {"schema_version"}
    unknown = sorted(config_keys - allowed)
    if unknown:
        parser.error(
            f"{config_path}: unknown config key(s): {', '.join(unknown)}"
        )

    return {key: data[key] for key in config_keys}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config:
        parser.set_defaults(
            **_load_config_defaults(Path(config_args.config), parser)
        )
    args = parser.parse_args(argv)
    return args, parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.run_name:
        parser.error("--run-name is required unless provided by --config")
    if args.synth_item_count is not None and args.synth_item_count < 1:
        parser.error("--synth-item-count must be >= 1")
    if args.cloze_count is not None and args.cloze_count < 1:
        parser.error("--cloze-count must be >= 1")
    if args.cloze_maximum_answer_frequency < 0:
        parser.error("--cloze-maximum-answer-frequency must be >= 0")
    if args.adversarial_refusal_count is not None and args.adversarial_refusal_count < 1:
        parser.error("--adversarial-refusal-count must be >= 1")
    if args.mcq_rationale_categories and not args.mcq_rationale:
        parser.error("--mcq-rationale-categories requires --mcq-rationale")
    if args.rationale_overlay and not args.mcq_rationale:
        parser.error("--rationale-overlay requires --mcq-rationale")
    if args.cloze_conversations and not args.cloze_benchmark:
        parser.error("--cloze-conversations requires --cloze-benchmark")
    if args.cloze_source and args.cloze_conversations:
        parser.error("use either --cloze-source or --cloze-conversations, not both")
    if not 0.0 < args.split < 1.0:
        parser.error("--split must be between 0 and 1 (exclusive)")
    if args.conversation_ratio is not None and not 0.0 < args.conversation_ratio < 1.0:
        parser.error("--conversation-ratio must be between 0 and 1 (exclusive)")
    if args.train_order == "sequential" and args.curriculum_order == "none":
        parser.error("--train-order sequential requires --curriculum-order")

    jspace_dependents = (
        args.jspace_bucket,
        args.jspace_category,
        args.jspace_max_items,
        args.jspace_action if args.jspace_action != "include" else None,
    )
    if args.jspace_signals is None:
        if any(value is not None for value in jspace_dependents):
            parser.error("--jspace-* flags require --jspace-signals")
    else:
        if args.jspace_bucket:
            unknown = sorted(set(args.jspace_bucket.split(",")) - set(JSPACE_BUCKETS))
            if unknown:
                parser.error(f"unknown --jspace-bucket value(s): {', '.join(unknown)}")
        if not args.jspace_bucket and not args.jspace_category:
            parser.error(
                "--jspace-signals requires --jspace-bucket and/or --jspace-category"
            )
        if args.jspace_max_items is not None and args.jspace_max_items < 1:
            parser.error("--jspace-max-items must be >= 1")
        if args.jspace_action == "cap" and args.jspace_max_items is None:
            parser.error("--jspace-action cap requires --jspace-max-items")
        if args.jspace_action == "rationale_overlay" and not args.mcq_rationale:
            parser.error(
                "--jspace-action rationale_overlay requires --mcq-rationale"
            )


def main() -> None:
    args, parser = parse_args()
    validate_args(args, parser)

    # Security boundary: checksum gate must be the first subprocess.
    subprocess.run(
        [sys.executable, "autoresearch/freeze_manifest.py", "check"],
        check=True,
    )

    # J-space selection runs in dry-run too, so the exact selected ids and
    # counts are visible before committing to a training run.
    if args.jspace_signals is not None:
        args.qmd_mcq = str(_jspace_select(args))

    commands = _cloze_commands(args) + [_prepare_argv(args), _train_argv(args), _merge_argv(args)]
    if args.dry_run:
        print("Dry run: checksum passed; planned subprocesses:")
    for argv in commands:
        _run(argv, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
