"""Determinism and leakage guards for the local RLM slice.

LOCAL_RLM_SLICE_PLAN.md sec.2: no finetune, no weight change, no frozen-file
edits, never read the sealed free-form test split, and decode deterministically
(temperature = 0, fixed seed).

These guards live in `rlm_slice/` and are *additive*: nothing here modifies
`autoresearch/score_mcq_trace_ll.py`, `autoresearch/score_mcq_ll.py`, or
`autoresearch/frozen_manifest.json`.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

DEFAULT_SEED = 42

# Superset of score_mcq_trace_ll.SEALED_PATH_FRAGMENTS. The sealed-test wall is
# the one guard that must never be relaxed for convenience, so the slice keeps
# its own copy rather than depending on an import staying in place.
SEALED_PATH_FRAGMENTS = (
    "eval_freeform/test.jsonl",
    "eval_freeform\\test.jsonl",
)


class SealedPathError(RuntimeError):
    """Raised when any slice code tries to open the sealed test split."""


def check_not_sealed(path: str | Path) -> Path:
    """Refuse to touch the sealed free-form test split.

    Call this before *any* file IO on a caller-supplied dataset path.
    """
    resolved = Path(path)
    as_posix = resolved.as_posix()
    for fragment in SEALED_PATH_FRAGMENTS:
        if fragment.replace("\\", "/") in as_posix:
            raise SealedPathError(f"refusing to read sealed test path: {path}")
    return resolved


def set_determinism(seed: int = DEFAULT_SEED) -> None:
    """Pin every RNG the decode path can reach.

    Pass-1 generation is greedy (do_sample=False), so the seed is belt-and-braces
    rather than load-bearing -- but Arm C's multi-pass shape and any future
    sampling arm would silently lose reproducibility without it.
    """
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_greedy(temperature: float | None, top_p: float | None = None) -> None:
    """Reject any non-greedy decode config.

    sec.2.5 makes greedy decode a slice-wide invariant: an arm that samples is
    not comparable to the v5clean trace-LL baseline, which is greedy.
    """
    if temperature not in (0, 0.0, None):
        raise ValueError(
            f"non-greedy temperature={temperature!r}: the RLM slice requires "
            "temperature=0 (LOCAL_RLM_SLICE_PLAN.md sec.2.5)"
        )
    if top_p not in (None, 1, 1.0):
        raise ValueError(
            f"non-greedy top_p={top_p!r}: the RLM slice requires greedy decode "
            "(LOCAL_RLM_SLICE_PLAN.md sec.2.5)"
        )
