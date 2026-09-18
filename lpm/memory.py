"""LPM slot-memory module (PersonaMem config) — #190.

Implements:

  * per-persona slots ``M_u`` of shape [H, N, d_mem] (H=8, N=32 per head,
    d_mem=256; H*N = 256 slots = 65,536 params/persona, matching the paper);
  * the shared cross-attention projection network (paper Eq. 1-2), producing H
    input-conditioned soft prompts of width d_model=1024;
  * the 256->1024 slot projector ``w_v`` used both in the dynamic path and to
    produce the static vectors M3/M4 bake into reserved embedding rows;
  * the hinge entropy load-balancing regularizer (paper Sec. 3.1) and a
    magnitude penalty toward the M1-observed norm<=4.0 safe zone (#187).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from lpm.config import LPMConfig, entropy_target


class SlotMemory(nn.Module):
    """Per-persona latent slots ``M_u in R^{H x N x d_mem}``."""

    def __init__(self, n_personas: int, config: LPMConfig):
        super().__init__()
        self.config = config
        config.validate()
        self.slots = nn.Parameter(
            torch.randn(n_personas, config.n_heads, config.n_slots, config.d_mem) * 0.02
        )

    def forward(self, persona_ids: torch.Tensor) -> torch.Tensor:
        """persona_ids [B] -> slots [B, H, N, d_mem]."""
        return self.slots[persona_ids]


class ProjectionNetwork(nn.Module):
    """Shared projection network: query in R^{d_model}, per-head slots -> H soft prompts.

    W_Qh, W_Kh, W_Vh are per head (paper: "each with its own slots and
    projection"). The MLP + output LayerNorm are shared across heads to keep the
    trainable count near the paper's ~1.3M instead of H x that. W_Vh is exactly
    the 256->1024 slot projector: its output for slot (h, n) is the vector that
    M3 bakes into a reserved embedding row.
    """

    def __init__(self, config: LPMConfig):
        super().__init__()
        self.config = config
        self.w_q = nn.ModuleList(
            nn.Linear(config.d_model, config.d_key) for _ in range(config.n_heads)
        )
        self.w_k = nn.ModuleList(
            nn.Linear(config.d_mem, config.d_key) for _ in range(config.n_heads)
        )
        self.w_v = nn.ModuleList(
            nn.Linear(config.d_mem, config.d_model) for _ in range(config.n_heads)
        )
        self.out_ln = nn.ModuleList(
            nn.LayerNorm(config.d_model) for _ in range(config.n_heads)
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query_emb: torch.Tensor, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """query_emb [B, d_model], slots [B, H, N, d_mem].

        Returns ``prompts`` [B, H, d_model], per-head ``attention`` [B, H, N],
        and ``projected_slots`` [B, H, N, d_model] (the bake vectors).
        """
        scale = math.sqrt(self.config.d_key)
        prompts: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        projected: list[torch.Tensor] = []
        for head in range(self.config.n_heads):
            slot_h = slots[:, head]  # [B, N, d_mem]
            query = self.w_q[head](query_emb)  # [B, d_key]
            keys = self.w_k[head](slot_h)  # [B, N, d_key]
            values = self.w_v[head](slot_h)  # [B, N, d_model]
            scores = (query.unsqueeze(1) @ keys.transpose(1, 2)).squeeze(1) / scale  # [B, N]
            attn = torch.softmax(scores, dim=-1)  # [B, N]
            p_tilde = attn.unsqueeze(1) @ values  # [B, 1, d_model]
            p_tilde = p_tilde.squeeze(1)
            p = self.out_ln[head](p_tilde + self.mlp(p_tilde))
            prompts.append(self.dropout(p))
            attentions.append(attn)
            projected.append(values)
        return {
            "prompts": torch.stack(prompts, dim=1),  # [B, H, d_model]
            "attention": torch.stack(attentions, dim=1),  # [B, H, N]
            "projected_slots": torch.stack(projected, dim=1),  # [B, H, N, d_model]
        }


def _entropy(attention: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Shannon entropy in nats of the last-dim attention distribution."""
    return -(attention * torch.log(attention + eps)).sum(dim=-1)


def magnitude_penalty(vectors: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Mean squared hinge on the L2 norm above ``max_norm`` (M1 #186 safe zone)."""
    norms = vectors.norm(dim=-1)
    excess = torch.relu(norms - max_norm)
    return (excess**2).mean()


class LPMModule(nn.Module):
    """Slot memory + projection network + regularizers.

    ``forward`` is the training path (dynamic, query-conditioned soft prompts);
    ``bake_vectors`` is the deployment path (static per-slot vectors for the
    reserved embedding rows M3/M4 write).
    """

    def __init__(self, n_personas: int, config: LPMConfig | None = None):
        super().__init__()
        self.config = config or LPMConfig()
        self.config.validate()
        self.slot_memory = SlotMemory(n_personas, self.config)
        self.projection = ProjectionNetwork(self.config)

    def forward(self, persona_ids: torch.Tensor, query_emb: torch.Tensor) -> dict[str, Any]:
        slots = self.slot_memory(persona_ids)
        out = self.projection(query_emb, slots)
        attention = out["attention"]
        entropy = _entropy(attention).mean()  # scalar, per-head mean over batch
        target = entropy_target(self.config.n_slots, self.config.entropy_fraction)
        entropy_penalty = torch.relu(target - entropy) * self.config.alpha_entropy

        projected = out["projected_slots"]
        mag_penalty = magnitude_penalty(projected, self.config.max_vector_norm)
        out.update(
            {
                "entropy": entropy,
                "entropy_target": torch.as_tensor(target, device=attention.device),
                "entropy_penalty": entropy_penalty,
                "mag_penalty": mag_penalty,
                "bake_vectors": projected,
                "projected_norm_mean": projected.norm(dim=-1).mean(),
                "projected_norm_max": projected.norm(dim=-1).amax(),
            }
        )
        return out

    def bake_vectors(self, persona_ids: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """Static per-slot vectors to bake into reserved embedding rows.

        Shape [B, H, N, d_model]; row order is (head, slot) and maps to reserved
        IDs via ``lpm.config.slot_to_reserved_id``. Detached by default because
        baking is a deployment operation.
        """
        slots = self.slot_memory(persona_ids)
        vectors = torch.stack(
            [self.projection.w_v[h](slots[:, h]) for h in range(self.config.n_heads)], dim=1
        )
        return vectors.detach() if detach else vectors

    def loss_terms(self, out: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Regularizer terms #192 adds to the NTP loss."""
        return {
            "entropy_penalty": out["entropy_penalty"],
            "mag_penalty": out["mag_penalty"] * self.config.lambda_mag,
        }

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_module(n_personas: int, config: LPMConfig | None = None) -> LPMModule:
    return LPMModule(n_personas, config)
