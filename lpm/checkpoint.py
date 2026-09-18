"""Self-service load of the #192 LPM checkpoint in a fresh process.

The training script writes a plain dict (module state, config, persona map,
reserved-ID map). This reconstructs a ready-to-use module from it, so #193 and
later M3/M4 work do not have to hand-map the config fields (the gap found on
#192: ``LPMConfig.to_dict`` renames ``n_slots`` and adds computed-only keys, so
the raw dict is not constructor-compatible).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lpm.config import LPMConfig
from lpm.memory import LPMModule


@dataclass
class LoadedLPMCheckpoint:
    module: LPMModule
    config: LPMConfig
    persona_to_slot_index: dict[str, int]
    reserved_id_map: dict[int, tuple[int, int]]
    step: int
    summary: dict[str, Any]


def load_lpm_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedLPMCheckpoint:
    """Load a checkpoint written by ``scripts/lpm_train.py``.

    Strict ``load_state_dict`` (shape/config mismatches raise rather than
    silently partial-load). Returned module is in eval mode on ``device``.
    """
    raw = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = LPMConfig.from_dict(raw["config"])
    persona_to_slot_index = {str(k): int(v) for k, v in raw["persona_to_slot_index"].items()}
    module = LPMModule(n_personas=len(persona_to_slot_index), config=config)
    module.load_state_dict(raw["module_state_dict"], strict=True)
    module.to(device)
    module.eval()
    reserved_id_map = {
        int(key): (int(value[0]), int(value[1]))
        for key, value in raw.get("reserved_id_map", {}).items()
    }
    return LoadedLPMCheckpoint(
        module=module,
        config=config,
        persona_to_slot_index=persona_to_slot_index,
        reserved_id_map=reserved_id_map,
        step=int(raw.get("step", 0)),
        summary=raw.get("summary", {}),
    )
