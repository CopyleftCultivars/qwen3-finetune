#!/usr/bin/env python3
"""#225 (S2): build an armed / disarmed / malformed memory-prefix bundle variant.

Creates a cheap symlink tree pointing at ``--src-bundle`` (no weight copy), with
a real ``components/manifest.json`` carrying the ``memory_prefix_file`` /
``memory_prefix_rows`` keys, and (armed only) the raw FP32 prefix file copied
into the bundle at the relative path the fragment declares.

Modes:
  armed      keys present, prefix file copied in        -> loader arms the model
  malformed  keys present, prefix file NOT copied in    -> loader must hard-fail
  disarmed   keys absent (manifest copied unchanged)    -> loader is a no-op

The key values come from the checked-in fragment
``results/F179_M3_S2_A1_MANIFEST.json`` by default, so #226 consumes the same
manifest this issue's gate tested.

Usage:
  python scripts/apply_memory_prefix_bundle.py --mode armed \
      --src-bundle outputs/r179-m1/p193-suppress-deploy \
      --out-bundle outputs/r179-m1/s2a/gate_bundles/armed \
      --prefix-file outputs/r179-m1/s2a/a1_memory_prefix_p0.f32
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAGMENT = REPO_ROOT / "results" / "F179_M3_S2_A1_MANIFEST.json"


def link_tree(src: Path, dst: Path) -> None:
    """Symlink every entry of ``src`` into ``dst``; ``components/`` is expanded
    one level so its manifest.json can be replaced by a real file."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name == "components":
            comp_dst = dst / "components"
            comp_dst.mkdir(exist_ok=True)
            for ce in entry.iterdir():
                if ce.name == "manifest.json":
                    continue
                os.symlink(ce.resolve(), comp_dst / ce.name)
        else:
            os.symlink(entry.resolve(), dst / entry.name)


def inject_keys(manifest_text: str, rows: str, rel_file: str) -> str:
    """Insert the two keys right after the opening brace (valid: comma-joined)."""
    brace = manifest_text.index("{")
    injected = (
        "{\n"
        f'  "memory_prefix_file": {json.dumps(rel_file)},\n'
        f'  "memory_prefix_rows": {json.dumps(rows)},\n'
    )
    return injected + manifest_text[brace + 1:]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("armed", "malformed", "disarmed"), required=True)
    p.add_argument("--src-bundle", required=True)
    p.add_argument("--out-bundle", required=True)
    p.add_argument("--prefix-file", default="", help="raw FP32 file to copy in (armed mode)")
    p.add_argument("--fragment", default=str(DEFAULT_FRAGMENT))
    args = p.parse_args(argv)

    src = Path(args.src_bundle)
    dst = Path(args.out_bundle)
    if dst.exists():
        shutil.rmtree(dst)

    manifest_src = src / "components" / "manifest.json"
    if not manifest_src.is_file():
        raise SystemExit(f"source manifest not found: {manifest_src}")

    if args.mode == "disarmed":
        link_tree(src, dst)
        shutil.copyfile(manifest_src, dst / "components" / "manifest.json")
        print(f"disarmed bundle written to {dst} (keys absent)")
        return 0

    fragment = json.loads(Path(args.fragment).read_text())
    rows = str(fragment["memory_prefix_rows"])
    rel_file = fragment["memory_prefix_file"]

    link_tree(src, dst)
    (dst / "components" / "manifest.json").write_text(
        inject_keys(manifest_src.read_text(), rows, rel_file))

    if args.mode == "armed":
        if not args.prefix_file:
            raise SystemExit("--prefix-file is required for armed mode")
        dest_file = dst / rel_file
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.prefix_file, dest_file)
        print(f"armed bundle written to {dst} (rows={rows}, file={rel_file})")
    else:
        print(f"malformed bundle written to {dst} (keys present, file deliberately missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
