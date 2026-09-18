#!/usr/bin/env python3
"""#205 conversion post-step: emit suppress_generation_token_ids into a Cactus
bundle's components/manifest.json.

The v2.1.0 engine reads `suppress_generation_token_ids` as a top-level string in
components/manifest.json (Model::load_manifest -> read_string_key). The value is
a comma-separated list of vocab IDs the sampler must never emit. For LPM we
suppress the whole reserved pool (lpm.config.RESERVED_ID_START..END, 266 IDs),
not just the currently-baked subset, so the manifest does not need re-emitting as
more personas are trained.

The insertion is textual and byte-minimal: exactly one new line after the opening
brace, so diffing the bundle against an unmodified bundle shows only this key.

Usage:
    python scripts/cactus_set_suppress_ids.py <bundle_dir>            # write
    python scripts/cactus_set_suppress_ids.py <bundle_dir> --check    # verify only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lpm.config import RESERVED_ID_END, RESERVED_ID_START  # noqa: E402

MANIFEST_REL = Path("components") / "manifest.json"


def suppress_csv(start: int = RESERVED_ID_START, end: int = RESERVED_ID_END) -> str:
    return ",".join(str(i) for i in range(start, end + 1))


def read_manifest(path: Path) -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8")
    return raw, json.loads(raw)


def apply(bundle_dir: Path, *, check: bool = False, start: int = RESERVED_ID_START, end: int = RESERVED_ID_END) -> dict:
    manifest_path = bundle_dir / MANIFEST_REL
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    expected = suppress_csv(start, end)
    raw, obj = read_manifest(manifest_path)
    if not isinstance(obj, dict):
        raise SystemExit(f"manifest is not a JSON object: {manifest_path}")

    existing = obj.get("suppress_generation_token_ids")
    if check:
        if existing != expected:
            raise SystemExit(
                f"CHECK FAILED: suppress_generation_token_ids present={existing is not None} "
                f"ids={0 if existing is None else len(str(existing).split(','))} (expected 266)"
            )
        return {"status": "ok", "ids": len(expected.split(","))}

    if existing == expected:
        return {"status": "unchanged", "ids": len(expected.split(","))}

    # Byte-minimal textual insert right after the opening brace.
    if existing is None:
        if not re.match(r"^\{\s*\n", raw):
            raise SystemExit("manifest is not pretty-printed; refusing a non-minimal rewrite")
        line = f'  "suppress_generation_token_ids": "{expected}",\n'
        # Preserve any leading whitespace/BOM before the brace.
        brace = raw.index("{")
        new_raw = raw[: brace + 1] + "\n" + line + raw[brace + 2 :]
        manifest_path.write_text(new_raw, encoding="utf-8")
    else:
        # Key already present but wrong: JSON round-trip to fix.
        obj["suppress_generation_token_ids"] = expected
        manifest_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

    _, verify_obj = read_manifest(manifest_path)
    got = verify_obj.get("suppress_generation_token_ids")
    if got != expected:
        raise SystemExit("post-write verification failed")
    return {"status": "written", "ids": len(str(got).split(","))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--check", action="store_true", help="Verify the key without writing.")
    parser.add_argument("--start", type=int, default=RESERVED_ID_START)
    parser.add_argument("--end", type=int, default=RESERVED_ID_END)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = apply(args.bundle_dir, check=args.check, start=args.start, end=args.end)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
