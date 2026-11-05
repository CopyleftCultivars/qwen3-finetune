"""Unit tests for scripts/g3_compare.py (CPU-only, no models)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from g3_compare import compare, mcnemar_exact, paired_bootstrap_ci


def test_mcnemar_no_discordant_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_known_value():
    # b=7, c=36 (#232 Phase 3 shape): strongly significant.
    assert mcnemar_exact(7, 36) < 1e-4
    # Symmetric table is not significant.
    assert mcnemar_exact(15, 16) > 0.5


def test_bootstrap_ci_covers_true_delta():
    a = [True] * 80 + [False] * 20
    b = [True] * 90 + [False] * 10
    lo, hi = paired_bootstrap_ci(a, b, reps=2000, seed=42)
    assert lo < 0.10 < hi
    # Deterministic under the same seed.
    assert paired_bootstrap_ci(a, b, reps=2000, seed=42) == (lo, hi)


def test_compare_end_to_end():
    a = {f"Q-{i}": i < 8 for i in range(10)}
    b = {f"Q-{i}": i < 9 for i in range(10)}
    res = compare(a, b)
    assert res["n"] == 10
    assert res["acc_a"] == 0.8
    assert res["acc_b"] == 0.9
    assert res["delta_pp"] == 10.0
    assert res["b_only"] == 1 and res["a_only"] == 0
