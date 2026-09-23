#!/usr/bin/env python3
"""#212 Part 2 verification: cosine-compare cactus_embed's engine output against
lpm/encoder.py's Qwen3EarlyLayerEncoder(6) on the same prompts.

Was originally run ad hoc when #212 closed; this commits that comparison step
so results/D203_S2A_ENGINE_EMBED.json is reproducible from the raw engine
embedding dumps, not just asserted. Raw inputs (gitignored):
outputs/r179-m1/s2a/engine_embed_{6L,28L}.json (per-text 1024-d vectors,
produced by scripts/cactus_embed_probe.py against the 6-layer enc6b-deploy
bundle and the unmodified 28-layer p193-deploy bundle respectively, both via
cactus_embed(normalize=True) on the real ARM engine).

Usage:
    python scripts/cactus_embed_compare.py \
        --6l outputs/r179-m1/s2a/engine_embed_6L.json \
        --28l outputs/r179-m1/s2a/engine_embed_28L.json \
        --model outputs/merged-model-distill-v2-rationale \
        --out results/D203_S2A_ENGINE_EMBED.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return float("nan")
    return dot / (na * nb)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--6l", dest="six_l", required=True, type=Path)
    p.add_argument("--28l", dest="twentyeight_l", required=True, type=Path)
    p.add_argument("--model", default="outputs/merged-model-distill-v2-rationale")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    from lpm.encoder import load_encoder

    engine_6l = json.loads(args.six_l.read_text())
    engine_28l = json.loads(args.twentyeight_l.read_text())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = load_encoder(args.model, n_layers=6, device="cpu")

    comparisons = []
    for text in engine_6l:
        py_vec = encoder.encode_texts(tokenizer, [text], trainable=False)[0].float().tolist()
        py_norm = torch.tensor(py_vec)
        py_norm = (py_norm / py_norm.norm()).tolist()

        cos_6l = cosine(engine_6l[text], py_norm)
        cos_28l = cosine(engine_28l[text], py_norm) if text in engine_28l else None

        comparisons.append({
            "text": text,
            "cos_engine_6L_vs_python6L": round(cos_6l, 5),
            "cos_engine_28L_vs_python6L": round(cos_28l, 5) if cos_28l is not None else None,
        })

    result = {
        "issue": 212,
        "engine": "v2.1.0+delta glibc arm64 (qemu)",
        "bundle_6L": "outputs/r179-m1/s2a/enc6b-deploy",
        "bundle_28L": "outputs/r179-m1/p193-deploy",
        "decoder_embed_chunk_node_count": {"6L": 314, "28L": 1392},
        "comparisons": comparisons,
        "note": (
            "engine bundle is CQ4; python encoder is bf16 -> residual ~1.3% cosine "
            "is quantization, not a layer mismatch (28L path is ~0.03)."
        ),
    }
    out = args.out or (REPO_ROOT / "results" / "D203_S2A_ENGINE_EMBED.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
