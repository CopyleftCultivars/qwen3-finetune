#!/usr/bin/env python3
"""#225 (S2): generate the tier-A1 static memory-prefix file.

Runs #216's S0 selection + post-transform bake for ONE val persona and dumps the
8 x d_model post-transform object ``out_ln[h](v + mlp(v))`` as raw row-major
FP32 (no header) -- the object the bundle manifest ships for S3's
``inputs_embeds`` prepend path.

This is deliberately NOT ``bake_vectors``' raw pre-transform output (the
falsified #211/#212 object) and NOT norm-clamped (#222 found no viable clamp for
the tied-row path; this path never touches the tied embedding table, so the true
norm~37 object is shipped).

Slot fitting is GPU-nondeterministic run-to-run (#216), so this is a new shipped
artifact, not a bit-reproduction of any prior JSON measurement.

Default persona is the one #214's effect probe used: val persona offset 0,
persona index 0, issue-default fit budget 40/128/192.

Outputs:
  --out        raw FP32, [H, d_model] row-major (default H=8, d_model=1024)
  --meta-out   provenance JSON (persona/budget/seed, s0_top, norm stats, sha256)

Usage:
  python scripts/gen_memory_prefix_a1.py \
      --out outputs/r179-m1/s2a/a1_memory_prefix_p0.f32 \
      --meta-out outputs/r179-m1/s2a/a1_memory_prefix_p0.meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lpm.checkpoint import load_lpm_checkpoint  # noqa: E402
from lpm.data import load_corpus  # noqa: E402
from lpm.encoder import load_encoder  # noqa: E402
from lpm_quant_gap import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_MODEL,
    DEFAULT_VAL,
    device_of,
    fit_val_slots,
    l1_transform_selected,
    s0_select,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--val-corpus", default=DEFAULT_VAL)
    p.add_argument("--encoder-layers", type=int, default=6)
    p.add_argument("--n-personas", type=int, default=12)
    p.add_argument("--persona-offset", type=int, default=0)
    p.add_argument("--persona-index", type=int, default=0,
                   help="index within the fitted slice (0 = #214's test persona at offset 0)")
    p.add_argument("--fit-steps", type=int, default=40)
    p.add_argument("--fit-lr", type=float, default=1e-3)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--fit-max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    p.add_argument("--meta-out", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = device_of(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ck = load_lpm_checkpoint(args.checkpoint, device=device)
    config = ck.config
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    encoder = load_encoder(args.model, n_layers=args.encoder_layers, device=device, trainable_layers=0)

    t0 = time.time()
    val_module, eligible, probes, probe_q, fit_q, fit_pid = fit_val_slots(
        ck, encoder, model, tokenizer, load_corpus(args.val_corpus), config,
        n_personas=args.n_personas, persona_offset=args.persona_offset,
        fit_steps=args.fit_steps, lr=args.fit_lr, max_length=args.max_length,
        fit_max_length=args.fit_max_length, device=device, seed=args.seed,
    )
    print(f"fitted {len(eligible)} val personas in {time.time()-t0:.1f}s")

    top, selected, attn = s0_select(val_module, fit_q, fit_pid, len(eligible), device)
    transformed = l1_transform_selected(val_module, selected)  # [K, H, d]

    k = args.persona_index
    if not (0 <= k < transformed.shape[0]):
        raise SystemExit(f"persona-index {k} out of range for K={transformed.shape[0]}")
    rows = np.ascontiguousarray(transformed[k].detach().to(torch.float32).cpu().numpy())
    H, d = rows.shape

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.tofile(out_path)

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    norm = np.linalg.norm(rows, axis=1)
    meta = {
        "issue": 225,
        "artifact": "a1_static_memory_prefix",
        "format": "raw FP32, row-major, H rows x d_model floats, no header",
        "rows": int(H),
        "dim": int(d),
        "bytes": int(out_path.stat().st_size),
        "sha256": digest,
        "persona": f"val offset{args.persona_offset}[{k}] (persona_id={eligible[k]})",
        "fit_budget": f"{args.fit_steps}/{args.fit_max_length}/{args.max_length}",
        "seed": args.seed,
        "selection": "S0 (#211 math) -> l1_transform_selected out_ln[h](v+mlp(v))",
        "clamped": False,
        "norm_mean": round(float(norm.mean()), 4),
        "norm_max": round(float(norm.max()), 4),
        "s0_top": [int(top[k, h]) for h in range(H)],
        "attn_max_per_head": [round(float(attn[k, h].max()), 4) for h in range(H)],
        "source_script": "scripts/gen_memory_prefix_a1.py",
    }
    if args.meta_out:
        Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta_out).write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, sha256 {digest[:16]}...)")
    if args.meta_out:
        print(f"wrote {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
