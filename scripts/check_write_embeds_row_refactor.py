#!/usr/bin/env python3
"""#224 regression check: prove write_embeds_row is a verbatim lift of the
write_media_row_at lambda, and that the media call sites pass identical args.

Pure static check -- no engine build/model needed. It compares the pre-refactor
model.cpp (fork HEAD) against the working tree:

  1. extracts the lambda body and the new function body, applies the documented
     capture->parameter substitutions to the lambda, and asserts the two bodies
     are line-for-line identical (so the media/audio/image path is unchanged);
  2. asserts both call sites still pass the same modality row/bytes/precision
     expressions, with only the three now-explicit context args prepended.

Usage: python scripts/check_write_embeds_row_refactor.py [FORK_REPO]
"""

from __future__ import annotations

import re
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_FORK = Path(os.environ.get("FORK_ROOT", "."))
REL = "cactus-engine/src/model.cpp"

SUBS = [
    (r"\bfeature_row_bytes\b", "row_bytes"),
    (r"\bfeature_precision\b", "row_precision"),
    (r"\bfeature_row\b", "row_data"),
    (r"\blm_encoder_media_chunk_\b", "comp"),
    (r"\bmedia_embeds_idx\b", "embeds_idx"),
    (r"\bmedia_chunk_tokens\b", "chunk_tokens"),
    # the lambda captured the member as a pointer (->); the lifted helper takes
    # a Component& (.), so the one expected non-textual adaptation is -> to .
    (r"\bcomp->", "comp."),
]


def normalize(line: str) -> str:
    for pat, repl in SUBS:
        line = re.sub(pat, repl, line)
    return line.strip()


def extract_lambda_body(text: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "auto write_media_row_at = [&]" in l)
    # signature spans two lines; body starts after the line ending in "{"
    open_i = next(i for i in range(start, len(lines)) if lines[i].rstrip().endswith("{"))
    end = next(i for i in range(open_i + 1, len(lines)) if lines[i].strip() == "};")
    body = []
    for l in lines[open_i + 1 : end]:
        if not l.strip():
            continue
        body.append(normalize(l))
    return body


def extract_function_body(text: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("void Model::write_embeds_row("))
    # signature spans three lines; body starts after the line ending in "{"
    open_i = next(i for i in range(start, len(lines)) if lines[i].rstrip().endswith("{"))
    end = next(i for i in range(open_i + 1, len(lines)) if lines[i] == "}")
    body = []
    for l in lines[open_i + 1 : end]:
        if not l.strip():
            continue
        body.append(normalize(l))
    return body


def extract_calls(text: str, name: str) -> list[str]:
    # collapse whitespace so a call split across lines compares as one string.
    # [^{;] prevents matching the function definition (which has "{") and keeps
    # the match inside a single call's argument list.
    joined = re.sub(r"\s+", " ", text)
    calls = []
    for m in re.finditer(re.escape(name) + r"\(([^{;]*?)\);", joined):
        args = re.sub(r"\s+", " ", m.group(1)).strip()
        calls.append(args)
    return calls


def main() -> int:
    fork = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FORK
    if len(sys.argv) > 2:
        base = sys.argv[2]
    else:
        # the commit that introduced the function; compare against its parent
        hits = subprocess.check_output(
            ["git", "-C", str(fork), "log", "--format=%H", "-S",
             "void Model::write_embeds_row(", "--", REL], text=True).split()
        if not hits:
            print("could not find the commit introducing write_embeds_row")
            return 1
        base = hits[-1] + "^"
    old = subprocess.check_output(
        ["git", "-C", str(fork), "show", f"{base}:{REL}"], text=True)
    new = (fork / REL).read_text()
    print(f"base ref: {base}")

    failures: list[str] = []

    old_body = extract_lambda_body(old)
    new_body = extract_function_body(new)
    if old_body != new_body:
        failures.append("lambda body and write_embeds_row body differ after substitution")
        for i, (a, b) in enumerate(zip(old_body, new_body)):
            if a != b:
                failures.append(f"  line {i}: old={a!r} new={b!r}")
        if len(old_body) != len(new_body):
            failures.append(f"  line counts differ: {len(old_body)} vs {len(new_body)}")

    old_calls = extract_calls(old, "write_media_row_at")
    new_calls = extract_calls(new, "write_embeds_row")
    if len(old_calls) != 2 or len(new_calls) != 2:
        failures.append(f"expected 2 call sites each; got {len(old_calls)} / {len(new_calls)}")
    else:
        prefix = "*lm_encoder_media_chunk_, media_embeds_idx, media_chunk_tokens, "
        for oc, nc in zip(old_calls, new_calls):
            if nc != prefix + oc:
                failures.append(f"call args changed: old=({oc}) new=({nc})")

    print(f"lambda body lines: {len(old_body)} | function body lines: {len(new_body)}")
    print(f"call sites: old={len(old_calls)} new={len(new_calls)}")
    for oc in old_calls:
        print(f"  media row: ({oc})")
    if failures:
        print("FAIL:")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASS: write_embeds_row is a verbatim lift; media call sites unchanged in effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
