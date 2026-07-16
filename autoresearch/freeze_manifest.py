#!/usr/bin/env python3
"""Generate and verify checksums for files frozen by the autoresearch harness."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import NamedTuple


class FrozenPath(NamedTuple):
    path: str
    note: str


FROZEN_PATHS = [
    FrozenPath(
        "data/val_v3.jsonl",  # C1 initial: dev-split proxy, not yet a sealed C3/C4 test set
        "dev-split proxy, not yet a sealed C3/C4 test set",
    ),
    FrozenPath(
        "data/val.jsonl",  # C1 initial: dev-split proxy, not yet a sealed C3/C4 test set
        "dev-split proxy, not yet a sealed C3/C4 test set",
    ),
    FrozenPath(
        "autoresearch/levers.md",  # C1 initial: lever spec finetune.py must not silently drift from
        "lever spec finetune.py must not silently drift from",
    ),
    FrozenPath(
        "eval_freeform/dev.jsonl",
        "free-form dev split, loop-visible but fixed",
    ),
    FrozenPath(
        "eval_freeform/test.jsonl",
        "sealed free-form test split; grade exactly once at final evaluation",
    ),
    FrozenPath(
        "eval_freeform/split_manifest.json",
        "free-form dev/test split ids including sealed test ids",
    ),
    FrozenPath(
        "eval_freeform/scorer_config.json",
        "pinned free-form scorer model/version/config",
    ),
    FrozenPath(
        "eval_freeform/score_refll.py",
        "frozen free-form reference log-likelihood scorer code",
    ),
    FrozenPath(
        "eval_freeform/score_rubric.py",
        "frozen free-form rubric scorer code",
    ),
    FrozenPath(
        "eval_freeform/grader_anchors.jsonl",
        "free-form rubric grader anchor set",
    ),
    FrozenPath(
        "autoresearch/score_primary.py",
        "frozen primary objective combiner code",
    ),
    FrozenPath(
        "autoresearch/score_mcq_ll.py",
        "frozen MCQ option-set log-likelihood scorer code",
    ),
    FrozenPath(
        "autoresearch/objective_weights.json",
        "frozen primary objective policy; not tuned or learned by the loop",
    ),
    # TODO #23: add sealed MCQ test set once C3/C4 exists.
]

ALGORITHM = "sha256"
MANIFEST_PATH = Path("autoresearch/frozen_manifest.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: Path, root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(child for child in path.rglob("*") if child.is_file())
    for child in files:
        rel_path = child.relative_to(root).as_posix()
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_entry(relative_path: str, root: Path) -> tuple[str, str]:
    path = root / relative_path
    if path.is_file():
        return "file", sha256_file(path)
    if path.is_dir():
        return "dir", sha256_dir(path, root)
    raise FileNotFoundError(f"frozen path does not exist: {relative_path}")


def build_manifest(root: Path) -> dict:
    entries = {}
    for frozen in FROZEN_PATHS:
        kind, digest = hash_entry(frozen.path, root)
        entries[frozen.path] = {
            "sha256": digest,
            "kind": kind,
            "note": frozen.note,
        }
    return {
        "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "algorithm": ALGORITHM,
        "entries": entries,
    }


def load_manifest(root: Path) -> dict:
    path = root / MANIFEST_PATH
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def generate(args: argparse.Namespace) -> int:
    if not args.i_know_what_im_doing:
        print(
            "Refusing to regenerate frozen checksums without --i-know-what-im-doing",
            file=sys.stderr,
        )
        return 2

    root = repo_root()
    manifest = build_manifest(root)
    output_path = root / MANIFEST_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {MANIFEST_PATH.as_posix()}")
    return 0


def check(args: argparse.Namespace) -> int:
    root = repo_root()
    manifest = load_manifest(root)
    if manifest.get("algorithm") != ALGORITHM:
        print(
            f"MISMATCH: algorithm expected={ALGORITHM} actual={manifest.get('algorithm')}",
            file=sys.stderr,
        )
        sys.exit(1)

    mismatches: list[str] = []
    for relative_path, expected in manifest.get("entries", {}).items():
        expected_kind = expected.get("kind")
        expected_hash = expected.get("sha256")
        try:
            actual_kind, actual_hash = hash_entry(relative_path, root)
        except FileNotFoundError:
            actual_kind, actual_hash = "missing", "missing"

        if actual_kind != expected_kind:
            mismatches.append(
                f"MISMATCH: {relative_path} kind expected={expected_kind} actual={actual_kind}"
            )
        if actual_hash != expected_hash:
            mismatches.append(
                f"MISMATCH: {relative_path} expected={expected_hash} actual={actual_hash}"
            )

    for line in mismatches:
        print(line, file=sys.stderr)

    if mismatches:
        sys.exit(1)

    if args.verbose:
        print("frozen manifest check passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify the autoresearch frozen checksum manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="write frozen_manifest.json")
    generate_parser.add_argument(
        "--i-know-what-im-doing",
        action="store_true",
        help="required friction flag for regenerating frozen checksums",
    )
    generate_parser.set_defaults(func=generate)

    check_parser = subparsers.add_parser("check", help="verify frozen_manifest.json")
    check_parser.add_argument(
        "--verbose",
        action="store_true",
        help="print a success message when the manifest matches",
    )
    check_parser.set_defaults(func=check)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
