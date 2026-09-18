"""Configuration + reserved-embedding-ID mapping for the LPM module (#190).

No torch here: this module is the single source of truth for the numbers #192,
M3 and M4 need, and it is imported by tests that must run without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

# --------------------------------------------------------------------------- #
# Reserved vocab IDs (M0 #183): the unused, tokenizer-unreachable range of the
# tied embedding table, model.embed_tokens.weight [151936, 1024].
# --------------------------------------------------------------------------- #
RESERVED_ID_START = 151670
RESERVED_ID_END = 151935
RESERVED_ID_COUNT = RESERVED_ID_END - RESERVED_ID_START + 1  # 266

# --------------------------------------------------------------------------- #
# PersonaMem config, read from the LPM paper rather than guessed.
#
# "N=32 slots, d_mem=256, H=8 heads" in the issue is *slots per head*: the
# paper's footnote 1 says "~65K unique parameters per user across all heads",
# and 8 heads * 32 slots * 256 dim = 65,536 exactly. Section 6 confirms it:
# "we average the trained slot matrices across 8 heads to obtain
# M_u in R^{32 x 256}". So the module has H*N = 256 slot vectors per persona,
# not 32.
# --------------------------------------------------------------------------- #
DEFAULT_N_SLOTS_PER_HEAD = 32
DEFAULT_D_MEM = 256
DEFAULT_N_HEADS = 8
DEFAULT_D_MODEL = 1024  # Qwen3-0.6B hidden_size, and the embedding-row width
DEFAULT_D_KEY = 256
ENTROPY_FRACTION = 0.4  # paper: H_target = 0.4 * log(k), k = slots per head
M1_SAFE_NORM = 4.0  # M1 #186: norm <= 4.0 did not leak; norm ~8.0 did


def entropy_target(n_slots_per_head: int = DEFAULT_N_SLOTS_PER_HEAD, fraction: float = ENTROPY_FRACTION) -> float:
    """H_target = fraction * ln(k), k = slots per head (paper Sec. 3.1)."""
    return fraction * math.log(n_slots_per_head)


def slot_index(head: int, slot: int, n_slots_per_head: int = DEFAULT_N_SLOTS_PER_HEAD) -> int:
    """Flat slot index in [0, H*N). Row-major: head-major, slot-minor."""
    return head * n_slots_per_head + slot


def slot_to_reserved_id(
    head: int,
    slot: int,
    n_slots_per_head: int = DEFAULT_N_SLOTS_PER_HEAD,
) -> int:
    """Map slot (head, slot) to a reserved vocab ID.

    H*N = 8*32 = 256 slots map 1:1 onto 151670..151925, leaving
    151926..151935 (10 IDs) spare. Deploy-time baking writes slot vector
    (h, n) into row ``slot_to_reserved_id(h, n)`` and the prompt references
    ``<|mem{h}_{n}|>`` at that position. Unambiguous in both directions via
    :func:`reserved_id_map`.
    """
    flat = slot_index(head, slot, n_slots_per_head)
    if flat >= RESERVED_ID_COUNT:
        raise ValueError(
            f"slot ({head},{slot}) -> flat {flat} exceeds the {RESERVED_ID_COUNT}-ID "
            f"reserved pool ({RESERVED_ID_START}..{RESERVED_ID_END})"
        )
    return RESERVED_ID_START + flat


def reserved_id_map(
    n_heads: int = DEFAULT_N_HEADS,
    n_slots_per_head: int = DEFAULT_N_SLOTS_PER_HEAD,
) -> dict[int, tuple[int, int]]:
    """reserved vocab ID -> (head, slot), the inverse of slot_to_reserved_id."""
    return {
        slot_to_reserved_id(h, n, n_slots_per_head): (h, n)
        for h in range(n_heads)
        for n in range(n_slots_per_head)
    }


@dataclass
class LPMConfig:
    """PersonaMem-config LPM hyperparameters, adapted to this deployment."""

    n_slots: int = DEFAULT_N_SLOTS_PER_HEAD  # per head
    d_mem: int = DEFAULT_D_MEM
    n_heads: int = DEFAULT_N_HEADS
    d_model: int = DEFAULT_D_MODEL
    d_key: int = DEFAULT_D_KEY
    mlp_hidden: int = 512
    dropout: float = 0.0

    # Entropy load-balancing (paper Eq. 3; removing it costs accuracy, so on).
    alpha_entropy: float = 0.1
    entropy_fraction: float = ENTROPY_FRACTION

    # Magnitude constraint toward the M1-observed norm<=4.0 safe zone. Applied
    # to the deployable slot projections; see lpm.memory.LPMModule.
    # (#192 dry run: a mean-hinge penalty at lambda_mag=1e-2 let projected_norm_max
    # reach 5.9. Default now = max-hinge penalty AND a differentiable projection
    # of every slot vector onto the norm ball, so the baked rows are guaranteed
    # <= max_vector_norm, not merely nudged.)
    lambda_mag: float = 1e-2
    max_vector_norm: float = M1_SAFE_NORM
    mag_penalty_mode: str = "max"  # "max" | "mean" | "none"
    clamp_memory_norm: bool = True

    # Training-objective decision for #191's is_first_chunk records. False =
    # drop them from the NTP loss (default: a t=0 record's prompt IS its target,
    # so training on it only teaches the module to copy). #192 can flip this
    # for the paper-faithful ablation, but the default is recall-only.
    include_first_chunk: bool = False

    reserved_id_start: int = RESERVED_ID_START
    reserved_id_end: int = RESERVED_ID_END

    @property
    def total_slots(self) -> int:
        return self.n_heads * self.n_slots

    @property
    def entropy_target(self) -> float:
        return entropy_target(self.n_slots, self.entropy_fraction)

    def validate(self) -> None:
        if self.n_heads * self.n_slots > RESERVED_ID_COUNT:
            raise ValueError(
                f"{self.n_heads} heads x {self.n_slots} slots = "
                f"{self.total_slots} exceeds the {RESERVED_ID_COUNT}-ID reserved pool"
            )
        if self.d_model != DEFAULT_D_MODEL:
            raise ValueError(
                f"d_model={self.d_model} but deployment rows are {DEFAULT_D_MODEL}-dim"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "n_slots_per_head": self.n_slots,
            "d_mem": self.d_mem,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "d_key": self.d_key,
            "mlp_hidden": self.mlp_hidden,
            "dropout": self.dropout,
            "alpha_entropy": self.alpha_entropy,
            "entropy_fraction": self.entropy_fraction,
            "entropy_target": self.entropy_target,
            "lambda_mag": self.lambda_mag,
            "max_vector_norm": self.max_vector_norm,
            "mag_penalty_mode": self.mag_penalty_mode,
            "clamp_memory_norm": self.clamp_memory_norm,
            "include_first_chunk": self.include_first_chunk,
            "total_slots": self.total_slots,
            "reserved_id_start": self.reserved_id_start,
            "reserved_id_end": self.reserved_id_end,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LPMConfig":
        """Rebuild from :meth:`to_dict` output (the checkpoint's ``config`` field).

        ``to_dict`` renames ``n_slots`` -> ``n_slots_per_head`` and adds the
        computed-only ``entropy_target`` / ``total_slots``; both are handled here
        so a checkpoint can be reloaded in a fresh process without hand-mapping.
        Unknown keys are rejected rather than silently dropped, so a stale or
        malformed checkpoint fails loudly.
        """
        data = dict(payload)
        if "n_slots_per_head" in data:
            data["n_slots"] = data.pop("n_slots_per_head")
        for computed_only in ("entropy_target", "total_slots"):
            data.pop(computed_only, None)
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown LPMConfig field(s) in dict: {unknown}")
        return cls(**data)  # type: ignore[arg-type]
