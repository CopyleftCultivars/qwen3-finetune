#!/usr/bin/env python3
"""#231 Phase 2 smoke: Qwen3-1.7B EncoderSpec + LPMConfig(d_model=2048) shape check.

Component-level validation only -- no training (Phase 3) and no engine/bundle. On
the host/CPU it:

  1. instantiates the 1.7B EncoderSpec (Qwen3EarlyLayerEncoder over the Qwen3-1.7B
     backbone, first 6 of 28 layers), runs a forward pass, and asserts the pooled
     output is R^2048;
  2. instantiates LPMConfig.qwen3_1_7b() (d_model=2048, same H/N/d_mem as the
     paper default), builds LPMModule, runs forward()/bake_vectors(), and asserts
     the projected/baked slot shape is [B, H, N, 2048];
  3. empirically confirms the reserved-ID range (151670..151935) is unreachable on
     the 1.7B tokenizer (VOCAB_SIZE=151936, tie_word_embeddings=true) rather than
     assuming it from the 0.6B derivation.

Writes results/F179_M3_BACKBONE_SCALE_PHASE2_SMOKE.json. CPU/host only.

Run: [LOCAL_PATH] \
       scripts/lpm_backbone_scale_phase2_smoke.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from lpm.config import (  # noqa: E402
    QWEN3_0_6B_D_MODEL,
    QWEN3_1_7B_D_MODEL,
    RESERVED_ID_COUNT,
    RESERVED_ID_END,
    RESERVED_ID_START,
    LPMConfig,
    reserved_id_map,
    slot_to_reserved_id,
)
from lpm.encoder import MODEL_PATH_QWEN3_1_7B, Qwen3EarlyLayerEncoder, encoder_spec_qwen3_1_7b  # noqa: E402
from lpm.memory import LPMModule  # noqa: E402

TEXTS = [
    "The best fertilizer for tomatoes is a combination of nutrients that supports healthy growth",
    "Korean natural farming inputs include FPJ, FFJ, IMO, and LAB cultures",
    "Describe how to make biochar from agricultural waste and why it helps soil",
    "My tomato leaves are turning yellow, what should I do about the nutrient deficiency",
]


def _disable_progress_bars() -> None:
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001
        pass


def check_encoder(device: torch.device) -> dict:
    from transformers import AutoTokenizer

    spec = encoder_spec_qwen3_1_7b(n_layers=6, pooling="mean")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_QWEN3_1_7B)
    encoder = Qwen3EarlyLayerEncoder(spec, device=device)
    query = encoder.encode_texts(tokenizer, TEXTS, max_length=32).float()

    checks = {
        "model_path": spec.model_path,
        "n_layers": spec.n_layers,
        "hidden_size": int(encoder.hidden_size),
        "stub_layers": len(encoder.backbone.layers),
        "output_shape": list(query.shape),
        "frozen_params": encoder.frozen_parameter_count(),
        "trainable_params": encoder.trainable_parameter_count(),
        "finite": bool(torch.isfinite(query).all()),
        "dtype": str(query.dtype),
    }
    assert query.shape == (len(TEXTS), QWEN3_1_7B_D_MODEL), query.shape
    assert encoder.hidden_size == QWEN3_1_7B_D_MODEL
    assert len(encoder.backbone.layers) == 6
    assert checks["finite"]
    return checks, query


def check_lpm(query_emb: torch.Tensor) -> dict:
    config = LPMConfig.qwen3_1_7b()
    config.validate()
    assert config.d_model == QWEN3_1_7B_D_MODEL
    assert QWEN3_0_6B_D_MODEL != QWEN3_1_7B_D_MODEL  # 0.6B path still distinct

    b = query_emb.shape[0]
    module = LPMModule(n_personas=b, config=config)
    persona_ids = torch.arange(b)
    out = module(persona_ids, query_emb)
    bake = module.bake_vectors(persona_ids)

    d = config.d_model
    checks = {
        "config_d_model": d,
        "config_n_heads": config.n_heads,
        "config_n_slots": config.n_slots,
        "config_d_mem": config.d_mem,
        "total_slots": config.total_slots,
        "prompts_shape": list(out["prompts"].shape),
        "attention_shape": list(out["attention"].shape),
        "projected_slots_shape": list(out["projected_slots"].shape),
        "bake_shape": list(bake.shape),
        "attention_sum_max_abs_err": round(float((out["attention"].sum(dim=-1) - 1.0).abs().max()), 8),
        "finite": bool(torch.isfinite(out["prompts"]).all()),
        "w_q_in_features": module.projection.w_q[0].in_features,
        "w_v_out_features": module.projection.w_v[0].out_features,
        "mlp_in_features": module.projection.mlp[0].in_features,
        "mlp_out_features": module.projection.mlp[-1].out_features,
        "out_ln_normalized_shape": int(module.projection.out_ln[0].normalized_shape[0]),
        "bake_matches_forward_max_abs_diff": float((bake - out["bake_vectors"].detach()).abs().max()),
        "module_trainable_params": module.trainable_parameter_count(),
    }
    assert out["prompts"].shape == (b, config.n_heads, d)
    assert out["attention"].shape == (b, config.n_heads, config.n_slots)
    assert out["projected_slots"].shape == (b, config.n_heads, config.n_slots, d)
    assert bake.shape == (b, config.n_heads, config.n_slots, d)
    assert checks["finite"]
    assert checks["w_q_in_features"] == d
    assert checks["w_v_out_features"] == d
    assert checks["mlp_in_features"] == d and checks["mlp_out_features"] == d
    assert checks["out_ln_normalized_shape"] == d
    assert checks["bake_matches_forward_max_abs_diff"] == 0.0
    assert checks["attention_sum_max_abs_err"] < 1e-5

    # Reserved-ID mapping is width-independent: 256 slots -> 151670..151925.
    mapping = reserved_id_map(config.n_heads, config.n_slots)
    checks["reserved_ids_used"] = len(mapping)
    checks["reserved_id_min"] = min(mapping)
    checks["reserved_id_max"] = max(mapping)
    assert len(mapping) == config.total_slots == 256
    assert slot_to_reserved_id(0, 0) == RESERVED_ID_START
    assert min(mapping) == 151670 and max(mapping) == 151925
    return checks


def check_reserved_ids() -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH_QWEN3_1_7B)
    vocab = tok.get_vocab()
    max_vocab_id = max(vocab.values())
    reserved = range(RESERVED_ID_START, RESERVED_ID_END + 1)

    none_token = 0
    roundtrip = 0
    decode_nonempty = 0
    for i in reserved:
        if tok.convert_ids_to_tokens(i) is None:
            none_token += 1
        text = tok.decode([i])
        if text:
            decode_nonempty += 1
        if i in tok(text, add_special_tokens=False)["input_ids"]:
            roundtrip += 1

    nf_ids: list[int] = []
    for t in TEXTS:
        nf_ids += tok(t, add_special_tokens=False)["input_ids"]
    reserved_in_text = [x for x in nf_ids if RESERVED_ID_START <= x <= RESERVED_ID_END]

    checks = {
        "tokenizer_class": type(tok).__name__,
        "tokenizer_vocab_size": int(tok.vocab_size),
        "tokenizer_len": len(tok),
        "tokenizer_num_vocab_entries": len(vocab),
        "tokenizer_max_vocab_id": int(max_vocab_id),
        "reserved_id_start": RESERVED_ID_START,
        "reserved_id_end": RESERVED_ID_END,
        "reserved_id_count": RESERVED_ID_COUNT,
        "reserved_ids_below_start_max_id": int(max_vocab_id) < RESERVED_ID_START,
        "reserved_convert_ids_to_tokens_none": none_token,
        "reserved_decode_nonempty": decode_nonempty,
        "reserved_roundtrip_to_self": roundtrip,
        "reserved_ids_in_encoded_text": reserved_in_text,
        "text_max_id": max(nf_ids),
    }
    # The range must be entirely above the tokenizer's highest id, map to no
    # token, and never reappear when its decode is re-encoded.
    assert int(max_vocab_id) < RESERVED_ID_START, (max_vocab_id, RESERVED_ID_START)
    assert none_token == RESERVED_ID_COUNT
    assert decode_nonempty == 0
    assert roundtrip == 0
    assert reserved_in_text == []

    # Cross-check the 0.6B tokenizer, if its local checkpoint is present.
    ckpt = REPO_ROOT / "outputs" / "merged-model-distill-v2-rationale"
    if ckpt.exists():
        try:
            tok06 = AutoTokenizer.from_pretrained(str(ckpt))
            v06 = tok06.get_vocab()
            checks["qwen3_0_6b_tokenizer_max_vocab_id"] = int(max(v06.values()))
            checks["qwen3_0_6b_reserved_unreachable"] = int(max(v06.values())) < RESERVED_ID_START
            assert int(max(v06.values())) < RESERVED_ID_START
        except Exception as exc:  # noqa: BLE001
            checks["qwen3_0_6b_tokenizer_error"] = f"{type(exc).__name__}: {exc}"
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/F179_M3_BACKBONE_SCALE_PHASE2_SMOKE.json")
    args = parser.parse_args()

    _disable_progress_bars()
    torch.manual_seed(0)
    device = torch.device(args.device)

    encoder_checks, query = check_encoder(device)
    print("encoder:", json.dumps(encoder_checks))
    lpm_checks = check_lpm(query)
    print("lpm:", json.dumps({k: v for k, v in lpm_checks.items() if k.endswith("_shape")}))
    reserved_checks = check_reserved_ids()
    print("reserved:", json.dumps({
        "max_vocab_id": reserved_checks["tokenizer_max_vocab_id"],
        "start": reserved_checks["reserved_id_start"],
        "unreachable": reserved_checks["reserved_ids_below_start_max_id"],
    }))

    report = {
        "issue": 231,
        "phase": 2,
        "device": str(device),
        "checks": {
            "encoder": encoder_checks,
            "lpm": lpm_checks,
            "reserved_ids": reserved_checks,
        },
        "pass": True,
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
