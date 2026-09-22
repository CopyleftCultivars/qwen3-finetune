"""#209 regression guard for the cactus_engine_s3s4_verdict.py leak fallback.

The primary path (leak_patched_combined.json present) reproduces
results/F179_M3_S3S4_PATCHED_LEAK.json byte-for-byte. The no-combined fallback
reduces the shards itself and is only STRUCTURALLY equal: its by_variant key
order is natural insertion order (none, 1mem, 8mem), not the committed file's
historical 8mem/none/1mem (not derivable from the repo). This test compares
parsed JSON (order-independent), so it passes regardless of key order but fails
if any generations/tokens/reserved value actually diverges.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.cactus_engine_s3s4 import MEM_VARIANTS  # noqa: E402
from scripts.cactus_engine_s3s4_verdict import build_leak_verdicts  # noqa: E402

RAW_DIR = REPO_ROOT / "outputs" / "r179-m1" / "s3s4"
COMMITTED = REPO_ROOT / "results" / "F179_M3_S3S4_PATCHED_LEAK.json"


COMBINED = RAW_DIR / "leak_patched_combined.json"
SHARDS = sorted(RAW_DIR.glob("leak_patched_s*.json"))


class S3S4LeakVerdictTests(unittest.TestCase):
    def test_primary_path_matches_committed(self):
        if not (COMBINED.exists() and COMMITTED.exists()):
            self.skipTest("combined passthrough input / committed verdict not present")
        _, patched = build_leak_verdicts(RAW_DIR)
        self.assertEqual(patched, json.loads(COMMITTED.read_text(encoding="utf-8")))

    def test_fallback_is_field_equal_and_deterministic(self):
        # Deliberately does NOT require leak_patched_combined.json: this is the
        # guard for the fallback that runs precisely when it is absent.
        if not (COMMITTED.exists() and SHARDS):
            self.skipTest("shard inputs / committed verdict not present")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for name in ("leak_c1_48.json", "leak_vanilla_48.json"):
                shutil.copy(RAW_DIR / name, tmp_dir / name)
            shards = sorted(RAW_DIR.glob("leak_patched_s*.json")) + sorted(RAW_DIR.glob("leak_ext_*.json"))
            self.assertTrue(shards, "no patched shards to reduce")
            for shard in shards:
                shutil.copy(shard, tmp_dir / shard.name)
            self.assertFalse((tmp_dir / "leak_patched_combined.json").exists())
            _, patched = build_leak_verdicts(tmp_dir)

        # Structural (order-independent): values must match the committed file.
        self.assertEqual(patched, json.loads(COMMITTED.read_text(encoding="utf-8")))
        # Key order is deterministic and follows MEM_VARIANTS declaration order.
        self.assertEqual(list(patched["by_variant"]), [v for v in MEM_VARIANTS if v in patched["by_variant"]])


if __name__ == "__main__":
    unittest.main()
