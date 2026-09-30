#!/usr/bin/env python3
"""#238 G2: HF-level weight audit of the merged Qwen3.5 artifact vs the base.

Answers, directly from the safetensors files (no model load):
  * are tensor NAMES and SHAPES identical to the base checkpoint?
  * which tensors CHANGED (must be exactly the LoRA-adapted set), and which
    only changed DTYPE (the bf16 LoRA recipe loads in bf16, so the 36 F32
    GDN A_log/norm tensors round to bf16 on save) — a dtype round-trip must
    stay inside bf16 rounding, anything else is an unexplained change.

Reads the safetensors header + raw tensor bytes directly, so it needs only
numpy (no torch/safetensors) and stays unit-testable.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

BF16_EPS = 2.0 ** -7

_DTYPES: dict[str, tuple[str, int]] = {  # numpy view name, bytes per element
    "F64": ("<f8", 8), "F32": ("<f4", 4), "F16": ("<f2", 2),
    "I64": ("<i8", 8), "I32": ("<i4", 4), "I16": ("<i2", 2),
    "I8": ("|i1", 1), "U8": ("|u1", 1), "BOOL": ("|u1", 1),
    "BF16": ("<u2", 2),
}


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def safetensors_files(directory: Path) -> list[Path]:
    if (directory / "model.safetensors").is_file():
        return [directory / "model.safetensors"]
    index_files = sorted(directory.glob("*.safetensors.index.json"))
    if index_files:
        weight_map = json.loads(index_files[0].read_text(encoding="utf-8"))
        return [directory / name for name in
                sorted(set(weight_map["weight_map"].values()))]
    return sorted(directory.glob("*.safetensors"))


def load_tensors(directory: Path) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for file_path in safetensors_files(directory):
        header, data_start = read_header(file_path)
        with file_path.open("rb") as handle:
            for name, spec in header.items():
                dtype, width = _DTYPES[spec["dtype"]]
                begin, end = spec["data_offsets"]
                handle.seek(data_start + begin)
                raw = handle.read(end - begin)
                array = np.frombuffer(raw, dtype=dtype)
                if spec["dtype"] == "BF16":
                    array = bf16_to_f32(array)
                elif spec["dtype"] == "F64":
                    array = array.astype(np.float32)
                if width and len(spec["shape"]) > 0:
                    array = array.reshape(spec["shape"])
                tensors[name] = array
    return tensors


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """bf16 is the top 16 bits of an fp32 word: shift left, reinterpret."""
    widened = bits.astype(np.uint32) << 16
    return widened.view(np.float32)


def load_headers(directory: Path) -> dict[str, dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    for file_path in safetensors_files(directory):
        header, _ = read_header(file_path)
        headers.update(header)
    return headers


def classify_tensor(base_spec: dict[str, Any], adapted_spec: dict[str, Any],
                    base_values: np.ndarray | None,
                    adapted_values: np.ndarray | None) -> str:
    if list(base_spec["shape"]) != list(adapted_spec["shape"]):
        return "shape_changed"
    if base_spec["dtype"] == adapted_spec["dtype"]:
        if base_values is None or adapted_values is None:
            return "value_changed"
        return "identical" if np.array_equal(base_values, adapted_values) \
            else "value_changed"
    if base_values is None or adapted_values is None:
        return "dtype_changed"
    scale = np.maximum(np.abs(base_values), 1e-6)
    rel_err = float(np.max(np.abs(base_values - adapted_values) / scale))
    return "dtype_cast_only" if rel_err <= BF16_EPS else "dtype_changed"


NESTED_LANGUAGE_MODEL_RE = re.compile(r"(?:\.language_model){2,}")
MISNESTED_VISUAL_PREFIX = "model.language_model.visual."


def canonical_key(name: str) -> str:
    """Map a merged-save tensor name back to the upstream checkpoint form.

    The Unsloth FastLanguageModel + PEFT merge_and_unload save path nests
    everything under an extra ``language_model`` scope:

    * language tensors are written as
      ``model.language_model.language_model.language_model.<rest>`` while
      upstream (and every in-memory ``named_parameters`` view) uses
      ``model.language_model.<rest>``;
    * the vision tower is written as ``model.language_model.visual.<rest>``
      while upstream uses ``model.visual.<rest>``.

    ``transformers`` remaps both on load, so HF inference is unaffected, but
    raw-weight consumers (the audit, the Cactus converter) see mismatched
    names. Canonicalization is a key-string rewrite only; tensor bytes are
    untouched.
    """
    if name.startswith(MISNESTED_VISUAL_PREFIX):
        return "model.visual." + name[len(MISNESTED_VISUAL_PREFIX):]
    return NESTED_LANGUAGE_MODEL_RE.sub(".language_model", name)


def _read_header_with_metadata(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return header, 8 + size


def canonicalize_checkpoint(adapted_dir: Path, base_dir: Path) -> dict[str, Any]:
    """Rewrite a single-file merged checkpoint with canonical tensor names.

    * renames ``model.language_model.language_model...`` keys to the upstream
      ``model.language_model...`` form (bytes untouched);
    * restores base-only tensors the conditional-generation save drops
      (the 15 ``mtp.*`` multi-token-prediction head tensors) byte-verbatim
      from ``base_dir``;
    * orders keys in base order so the file diff is minimal and reviewable.

    Returns a JSON-serializable report; ``canonicalized`` is False when the
    directory is not a single-file checkpoint or a key collision is found
    (the file is left untouched in both cases).
    """
    report: dict[str, Any] = {
        "adapted_dir": str(adapted_dir),
        "base_dir": str(base_dir),
        "canonicalized": False,
    }
    adapted_files = safetensors_files(adapted_dir)
    if len(adapted_files) != 1 or adapted_files[0].suffix != ".safetensors":
        report["reason"] = ("expected a single model.safetensors file, found: "
                            + str([p.name for p in adapted_files]))
        return report
    target = adapted_files[0]

    header, data_start = _read_header_with_metadata(target)
    metadata = header.pop("__metadata__", None)
    canonical_map: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, spec in header.items():
        canon = canonical_key(name)
        if canon in canonical_map:
            report["reason"] = f"key collision after canonicalization: {canon}"
            return report
        canonical_map[canon] = (name, spec)
    report["renamed_key_count"] = sum(
        1 for name in header if canonical_key(name) != name)

    base_ordered: list[tuple[str, dict[str, Any]]] = []
    base_locations: dict[str, tuple[Path, int, int]] = {}
    for file_path in safetensors_files(base_dir):
        file_header, file_data_start = _read_header_with_metadata(file_path)
        file_header.pop("__metadata__", None)
        for name, spec in file_header.items():
            base_ordered.append((name, spec))
            begin, end = spec["data_offsets"]
            base_locations[name] = (file_path, file_data_start + begin,
                                    file_data_start + end)
    base_names = set(base_locations)

    with target.open("rb") as handle:
        handle.seek(data_start)
        adapted_blob = handle.read()

    base_cache: dict[Path, Any] = {}

    def base_bytes(name: str) -> bytes:
        file_path, begin, end = base_locations[name]
        handle = base_cache.get(file_path)
        if handle is None:
            handle = file_path.open("rb")
            base_cache[file_path] = handle
        handle.seek(begin)
        return handle.read(end - begin)

    new_header: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    restored: list[str] = []
    try:
        for name, base_spec in base_ordered:
            if name in canonical_map:
                orig_name, adapted_spec = canonical_map[name]
                begin, end = adapted_spec["data_offsets"]
                chunk = adapted_blob[begin:end]
                newspec = {"dtype": adapted_spec["dtype"],
                           "shape": adapted_spec["shape"]}
            else:
                chunk = base_bytes(name)
                newspec = {"dtype": base_spec["dtype"],
                           "shape": base_spec["shape"]}
                restored.append(name)
            newspec["data_offsets"] = [offset, offset + len(chunk)]
            new_header[name] = newspec
            chunks.append(chunk)
            offset += len(chunk)
        extras = sorted(set(canonical_map) - base_names)
        for name in extras:
            orig_name, adapted_spec = canonical_map[name]
            begin, end = adapted_spec["data_offsets"]
            chunk = adapted_blob[begin:end]
            newspec = {"dtype": adapted_spec["dtype"],
                       "shape": adapted_spec["shape"],
                       "data_offsets": [offset, offset + len(chunk)]}
            new_header[name] = newspec
            chunks.append(chunk)
            offset += len(chunk)
    finally:
        for handle in base_cache.values():
            handle.close()

    missing = sorted(base_names - set(new_header))
    report["restored_names"] = restored
    report["restored_count"] = len(restored)
    report["extra_names"] = extras
    report["missing_names"] = missing
    if missing or extras:
        report["reason"] = (f"missing={missing} extra={extras}; "
                            "not rewriting to avoid a lossy checkpoint")
        return report
    if report["renamed_key_count"] == 0 and not restored:
        report["canonicalized"] = True
        report["rewrite"] = "skipped (already canonical)"
        return report

    payload_header = dict(new_header)
    if metadata is not None:
        payload_header["__metadata__"] = metadata
    header_bytes = json.dumps(payload_header).encode("utf-8")
    tmp_path = target.with_suffix(".canonicalizing")
    with tmp_path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for chunk in chunks:
            handle.write(chunk)
    tmp_path.replace(target)
    report["canonicalized"] = True
    report["rewrite"] = (f"rewrote {target.name} with "
                         f"{len(new_header)} tensors")
    return report


def audit(base_dir: Path, adapted_dir: Path,
          expected_changed: list[str] | None = None) -> dict[str, Any]:
    base_headers = load_headers(base_dir)
    adapted_headers = load_headers(adapted_dir)
    base_names, adapted_names = set(base_headers), set(adapted_headers)
    shared = sorted(base_names & adapted_names)

    base_values = load_tensors(base_dir) if base_names else {}
    adapted_values = load_tensors(adapted_dir) if adapted_names else {}

    classes: dict[str, list[str]] = {"identical": [], "value_changed": [],
                                     "dtype_cast_only": [], "dtype_changed": [],
                                     "shape_changed": []}
    for name in shared:
        verdict = classify_tensor(base_headers[name], adapted_headers[name],
                                  base_values.get(name), adapted_values.get(name))
        classes[verdict].append(name)

    changed = sorted(classes["value_changed"] + classes["dtype_changed"]
                     + classes["shape_changed"])
    report: dict[str, Any] = {
        "base_dir": str(base_dir),
        "adapted_dir": str(adapted_dir),
        "base_tensor_count": len(base_names),
        "adapted_tensor_count": len(adapted_names),
        "names_identical": base_names == adapted_names,
        "missing_in_adapted": sorted(base_names - adapted_names),
        "extra_in_adapted": sorted(adapted_names - base_names),
        "shapes_all_identical": not classes["shape_changed"],
        "identical_count": len(classes["identical"]),
        "value_changed_count": len(classes["value_changed"]),
        "dtype_cast_only": {
            "count": len(classes["dtype_cast_only"]),
            "names": classes["dtype_cast_only"],
        },
        "dtype_changed_count": len(classes["dtype_changed"]),
        "changed_names": changed,
        "unexpected_changed_names": changed,
    }
    if expected_changed is not None:
        expected = sorted(expected_changed)
        report["expected_changed_count"] = len(expected)
        report["unexpected_changed_names"] = sorted(set(changed) - set(expected))
        report["expected_but_unchanged"] = sorted(set(expected) - set(changed))
        report["verdict"] = (
            "PASS" if report["names_identical"] and report["shapes_all_identical"]
            and not report["unexpected_changed_names"]
            and not report["expected_but_unchanged"]
            and not classes["dtype_changed"] and not classes["shape_changed"]
            else "FAIL")
    else:
        report["verdict"] = "PASS" if (report["names_identical"]
                                       and report["shapes_all_identical"]
                                       and not classes["dtype_changed"]
                                       and not classes["shape_changed"]) else "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base checkpoint directory")
    parser.add_argument("--adapted", required=True, help="merged checkpoint dir")
    parser.add_argument("--merge-json", default=None,
                        help="merge-stage JSON carrying changed_params")
    parser.add_argument("--out", default=None)
    parser.add_argument("--canonicalize", action="store_true",
                        help="rewrite the adapted checkpoint with canonical "
                             "tensor names (collapse repeated "
                             ".language_model nesting written by the "
                             "Unsloth+PEFT save path) and restore base-only "
                             "tensors (e.g. mtp.*) byte-verbatim before "
                             "auditing")
    args = parser.parse_args(argv)

    if args.canonicalize:
        canon = canonicalize_checkpoint(Path(args.adapted), Path(args.base))
        print(json.dumps(canon, indent=2, sort_keys=True))
        if not canon.get("canonicalized"):
            return 2

    expected = None
    if args.merge_json:
        merge = json.loads(Path(args.merge_json).read_text(encoding="utf-8"))
        expected = merge.get("changed_params")
    report = audit(Path(args.base), Path(args.adapted), expected)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
