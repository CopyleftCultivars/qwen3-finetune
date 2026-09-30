#!/usr/bin/env python3
"""#238 G2 unit tests: HF weight audit helpers (CPU-only, hermetic).

Covers scripts/g2_hf_weight_audit.py without a model, GPU, or network:
key canonicalization, tensor classification, checkpoint canonicalization on
synthetic safetensors files, and the audit verdict paths.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from g2_hf_weight_audit import (  # noqa: E402
    BF16_EPS,
    audit,
    bf16_to_f32,
    canonical_key,
    canonicalize_checkpoint,
    classify_tensor,
)


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray],
                       dtype: str = "BF16") -> None:
    """Write a minimal single-file safetensors checkpoint (bf16 path)."""
    header: dict = {}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        array = tensors[name].astype(np.float32)
        raw = array.view(np.uint32)
        top = (raw >> 16).astype(np.uint16)
        blob = top.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(blob)]}
        chunks.append(blob)
        offset += len(blob)
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for chunk in chunks:
            handle.write(chunk)


def _write_f32_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict = {}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        blob = tensors[name].astype("<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(tensors[name].shape),
                        "data_offsets": [offset, offset + len(blob)]}
        chunks.append(blob)
        offset += len(blob)
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for chunk in chunks:
            handle.write(chunk)


class CanonicalKeyTest(unittest.TestCase):
    def test_language_triple_nesting_collapses(self):
        self.assertEqual(
            canonical_key("model.language_model.language_model"
                          ".language_model.layers.0.mlp.down_proj.weight"),
            "model.language_model.layers.0.mlp.down_proj.weight")

    def test_misnested_visual_moves_to_model_visual(self):
        self.assertEqual(
            canonical_key("model.language_model.visual.blocks.0.norm1.weight"),
            "model.visual.blocks.0.norm1.weight")

    def test_canonical_names_unchanged(self):
        for name in ("model.language_model.layers.3.self_attn.q_proj.weight",
                     "model.visual.blocks.3.norm1.weight",
                     "mtp.fc.weight",
                     "model.language_model.embed_tokens.weight"):
            self.assertEqual(canonical_key(name), name)


class ClassifyTensorTest(unittest.TestCase):
    def test_shape_change(self):
        base = {"shape": [4, 4], "dtype": "BF16"}
        adapted = {"shape": [4, 8], "dtype": "BF16"}
        self.assertEqual(classify_tensor(base, adapted, None, None),
                         "shape_changed")

    def test_identical_and_value_changed(self):
        spec = {"shape": [4], "dtype": "BF16"}
        values = np.arange(4, dtype=np.float32)
        self.assertEqual(classify_tensor(spec, spec, values, values.copy()),
                         "identical")
        self.assertEqual(
            classify_tensor(spec, spec, values, values + 1.0),
            "value_changed")

    def test_f32_to_bf16_roundtrip_is_cast_only(self):
        base_spec = {"shape": [8], "dtype": "F32"}
        adapted_spec = {"shape": [8], "dtype": "BF16"}
        rng = np.random.default_rng(238)
        base = rng.normal(size=8).astype(np.float32)
        as_bf16 = bf16_to_f32((base.view(np.uint32) >> 16).astype(np.uint16))
        self.assertEqual(classify_tensor(base_spec, adapted_spec,
                                         base, as_bf16), "dtype_cast_only")

    def test_large_dtype_gap_is_dtype_changed(self):
        base_spec = {"shape": [8], "dtype": "F32"}
        adapted_spec = {"shape": [8], "dtype": "BF16"}
        base = np.zeros(8, dtype=np.float32)
        adapted = np.full(8, 10 * BF16_EPS + 0.5, dtype=np.float32)
        self.assertEqual(classify_tensor(base_spec, adapted_spec,
                                         base, adapted), "dtype_changed")


class CanonicalizeCheckpointTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(prefix="g2audit_")
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.base_dir = self.tmp / "base"
        self.adapted_dir = self.tmp / "adapted"
        self.base_dir.mkdir(exist_ok=True)
        self.adapted_dir.mkdir(exist_ok=True)

    def _base_tensors(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(7)
        return {
            "model.language_model.layers.0.mlp.down_proj.weight":
                rng.normal(size=(8, 8)).astype(np.float32),
            "model.language_model.embed_tokens.weight":
                rng.normal(size=(16, 8)).astype(np.float32),
            "model.visual.blocks.0.norm1.weight":
                rng.normal(size=(8,)).astype(np.float32),
            "mtp.fc.weight":
                rng.normal(size=(8, 8)).astype(np.float32),
        }

    def test_renames_and_restores_mtp(self):
        base = self._base_tensors()
        _write_safetensors(self.base_dir / "model.safetensors", base)
        adapted_tensors = {
            "model.language_model.language_model.language_model"
            ".layers.0.mlp.down_proj.weight":
                base["model.language_model.layers.0.mlp.down_proj.weight"]
                + 0.5,
            "model.language_model.language_model.language_model"
            ".embed_tokens.weight":
                base["model.language_model.embed_tokens.weight"],
            "model.language_model.visual.blocks.0.norm1.weight":
                base["model.visual.blocks.0.norm1.weight"],
        }
        _write_safetensors(self.adapted_dir / "model.safetensors",
                           adapted_tensors)
        report = canonicalize_checkpoint(self.adapted_dir, self.base_dir)
        self.assertTrue(report["canonicalized"], report)
        self.assertEqual(report["renamed_key_count"], 3)
        self.assertEqual(report["restored_names"], ["mtp.fc.weight"])
        self.assertEqual(report["extra_names"], [])
        self.assertEqual(report["missing_names"], [])

        result = audit(self.base_dir, self.adapted_dir,
                       expected_changed=[
                           "model.language_model.layers.0.mlp"
                           ".down_proj.weight"])
        self.assertTrue(result["names_identical"], result)
        self.assertEqual(result["verdict"], "PASS", result)
        self.assertEqual(
            result["changed_names"],
            ["model.language_model.layers.0.mlp.down_proj.weight"])

    def test_refuses_multi_file_checkpoint(self):
        (self.adapted_dir / "a.safetensors").write_bytes(b"")
        (self.adapted_dir / "b.safetensors").write_bytes(b"")
        report = canonicalize_checkpoint(self.adapted_dir, self.base_dir)
        self.assertFalse(report["canonicalized"])
        self.assertIn("reason", report)

    def test_audit_fails_on_unexpected_change(self):
        base = self._base_tensors()
        _write_safetensors(self.base_dir / "model.safetensors", base)
        tampered = dict(base)
        tampered["model.language_model.embed_tokens.weight"] = \
            tampered["model.language_model.embed_tokens.weight"] + 1.0
        _write_safetensors(self.adapted_dir / "model.safetensors", tampered)
        result = audit(self.base_dir, self.adapted_dir,
                       expected_changed=[
                           "model.language_model.layers.0.mlp"
                           ".down_proj.weight"])
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("model.language_model.embed_tokens.weight",
                      result["unexpected_changed_names"])
        self.assertIn("model.language_model.layers.0.mlp.down_proj.weight",
                      result["expected_but_unchanged"])


if __name__ == "__main__":
    unittest.main()
