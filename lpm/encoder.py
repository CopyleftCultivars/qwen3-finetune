"""Qwen3-0.6B early-layer encoder for LPM (#189).

Reuses the frozen distilled backbone's first ``n_layers`` transformer layers as
the LPM encoder stub, instead of introducing a separate embedding model. The
same merged checkpoint (``outputs/merged-model-distill-v2-rationale``) backs the
NTP forward and the Cactus bundle, so the encoder adds zero new weights: it is a
prefix of the deployed model.

The layer list is truncated in place; the final norm is applied after the last
stub layer, and the pooled representation is returned in R^{d_model=1024}, which
is exactly the width LPM's W_Q expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

DEFAULT_MODEL_PATH = "outputs/merged-model-distill-v2-rationale"
DEFAULT_N_LAYERS = 6


@dataclass
class EncoderSpec:
    model_path: str = DEFAULT_MODEL_PATH
    n_layers: int = DEFAULT_N_LAYERS
    pooling: str = "mean"  # "mean" | "last"
    dtype: str = "auto"


class Qwen3EarlyLayerEncoder(nn.Module):
    """Frozen prefix of Qwen3-0.6B used as LPM's embedding model.

    ``trainable_layers`` controls the frozen/fine-tuned decision: 0 freezes the
    whole stub; >0 unfreezes the last N stub layers (and the final norm) for the
    fine-tuned arm of #189's comparison. The default is fully frozen, because
    these are the same weights the NTP backbone uses -- unfreezing them perturbs
    the backbone LPM must stay compatible with.
    """

    def __init__(
        self,
        spec: EncoderSpec | None = None,
        *,
        device: torch.device | str = "cpu",
        trainable_layers: int = 0,
    ):
        super().__init__()
        from transformers import AutoModel

        self.spec = spec or EncoderSpec()
        if self.spec.pooling not in ("mean", "last"):
            raise ValueError(f"unknown pooling: {self.spec.pooling!r}")

        dtype = self._resolve_dtype(self.spec.dtype, device)
        backbone = AutoModel.from_pretrained(self.spec.model_path, torch_dtype=dtype)
        n_layers = min(self.spec.n_layers, backbone.config.num_hidden_layers)
        backbone.layers = nn.ModuleList(list(backbone.layers)[:n_layers])
        backbone.config.num_hidden_layers = n_layers
        self.hidden_size = backbone.config.hidden_size
        self.backbone = backbone.to(device)
        self.device = torch.device(device)

        self.freeze_all()
        if trainable_layers > 0:
            for layer in list(self.backbone.layers)[-trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
        self.backbone.eval()

    @staticmethod
    def _resolve_dtype(dtype: str, device: torch.device | str) -> torch.dtype:
        if dtype == "float32":
            return torch.float32
        if dtype in ("bfloat16", "auto"):
            if dtype == "auto" and torch.device(device).type != "cuda":
                return torch.float32
            if torch.device(device).type == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float32
        raise ValueError(f"unknown dtype: {dtype!r}")

    def freeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "Qwen3EarlyLayerEncoder":
        # Keep the frozen backbone in eval mode; dropout stays off. The caller
        # controls grad via trainable_layers.
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids/attention_mask [B,T] -> pooled [B, d_model], no grad."""
        return self._pool(self._forward_hidden(input_ids, attention_mask), attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Grad-enabled pooled encoding (used only when trainable_layers > 0)."""
        return self._pool(self._forward_hidden(input_ids, attention_mask), attention_mask)

    def _forward_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.spec.pooling == "last":
            lengths = attention_mask.sum(dim=1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def encode_texts(
        self,
        tokenizer: object,
        texts: Iterable[str],
        *,
        max_length: int = 256,
        trainable: bool | None = None,
    ) -> torch.Tensor:
        """Tokenize + encode a batch of strings on the encoder's device.

        ``trainable`` forces grad on/off; by default grad runs only if some stub
        layer was unfrozen, so the frozen smoke path stays under ``no_grad`` and
        the fine-tuned #189 arm gets real gradients.
        """
        batch = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        if trainable is None:
            trainable = any(p.requires_grad for p in self.parameters())
        context = torch.enable_grad() if trainable else torch.no_grad()
        with context:
            return self._pool(
                self._forward_hidden(batch["input_ids"], batch["attention_mask"]),
                batch["attention_mask"],
            )


def load_encoder(
    model_path: str = DEFAULT_MODEL_PATH,
    *,
    n_layers: int = DEFAULT_N_LAYERS,
    device: torch.device | str = "cpu",
    trainable_layers: int = 0,
    pooling: str = "mean",
) -> Qwen3EarlyLayerEncoder:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"backbone not found: {model_path}")
    return Qwen3EarlyLayerEncoder(
        EncoderSpec(model_path=model_path, n_layers=n_layers, pooling=pooling),
        device=device,
        trainable_layers=trainable_layers,
    )
