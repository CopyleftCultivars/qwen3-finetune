"""LPM (Latent Personal Memory) module for epic #179 / M2 (#189, #190).

PersonaMem-config slot memory (arXiv 2606.20911): N=32 slots per head, H=8
heads, d_mem=256, plus the hinge entropy load-balancing regularizer. The
shared projection network maps the frozen encoder's query vector to per-head
dynamic soft prompts, and a 256->1024 slot projector produces the static
vectors that M3/M4 bake into reserved embedding rows.

This __init__ deliberately imports no torch so that the config / reserved-ID
mapping (lpm.config) stays importable in any Python env. Torch submodules are
imported explicitly (lpm.encoder, lpm.memory, lpm.data).
"""

from lpm.config import (  # noqa: F401
    RESERVED_ID_END,
    RESERVED_ID_START,
    LPMConfig,
    entropy_target,
    reserved_id_map,
    slot_to_reserved_id,
)

__all__ = [
    "LPMConfig",
    "RESERVED_ID_START",
    "RESERVED_ID_END",
    "entropy_target",
    "reserved_id_map",
    "slot_to_reserved_id",
]
