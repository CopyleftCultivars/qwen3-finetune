#!/usr/bin/env python3
"""#228 (S5/G6): generate the random-8 control for the effect probe.

This is the ONLY variable in the selected-vs-random contrast: the *slot picked
per head*, not the object type. It reuses ``gen_memory_prefix_a1.py``'s exact
pipeline (same checkpoint/encoder/val corpus, same fit budget 40/128/192, same
persona) up to ``s0_select``; then instead of shipping S0's argmax slot per head
it picks, per head, a uniformly random slot in ``[0, N)`` **excluding that
head's S0 choice**, exactly mirroring ``d203_effect_probe_build.py``'s
``random_slots`` logic (#214's methodology). Those raw bake vectors are then run
through the SAME post-transform ``l1_transform_selected`` (#216) that the real
A1 file uses, so both objects are in-distribution ``out_ln[h](v + mlp(v))``
rows; only the slot index differs.

The random seed is recorded in the meta and is deliberately different from
#214's 20260923 so this is a new artifact, not a rerun of an old measurement.

Outputs:
  --out        raw FP32, [H, d_model] row-major (default H=8, d_model=1024)
  --meta-out   provenance JSON (persona/budget/seed, s0_top, random_slots, norms, sha256)
  --matched-out optional: the SAME fit's S0-selected post-transform object, for
               provenance only (lets a reviewer gauge fit run-to-run drift vs
               the shipped a1 file without building another bundle)

Usage:
  python scripts/gen_memory_prefix_random8.py \
      --out outputs/r179-m1/s2a/random8_memory_prefix_p0.f32 \
      --meta-out outputs/r179-m1/s2a/random8_memory_prefix_p0.meta.json \
      --matched-out outputs/r179-m1/s2a/matched_selected_memory_prefix_p0.f32 \
      --random-seed 20260924
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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

# #214's seed; a different one is required so this is not a rerun of an old artifact.
PRIOR_RANDOM_SEED_214 = 20260923
DEFAULT_RANDOM_SEED = 20260924


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
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED,
                   help=f"random slot picker seed (must differ from #214's {PRIOR_RANDOM_SEED_214})")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    p.add_argument("--meta-out", default="")
    p.add_argument("--matched-out", default="",
                   help="optional: dump this same fit's S0-selected post-transform object")
    return p


def dump_f32(rows: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(rows).tofile(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.random_seed == PRIOR_RANDOM_SEED_214:
        raise SystemExit(
            f"--random-seed {args.random_seed} is #214's seed; pick a different one "
            f"(default {DEFAULT_RANDOM_SEED})"
        )
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

    # S0 (same math as the real A1 export): argmax slot per head over fit queries.
    top, selected_raw, attn = s0_select(val_module, fit_q, fit_pid, len(eligible), device)
    # raw pre-transform bake for the same fit; #214 picked random slots from here.
    bake = val_module.bake_vectors(torch.arange(len(eligible), device=device))  # [K, H, N, d]

    k = args.persona_index
    if not (0 <= k < len(eligible)):
        raise SystemExit(f"persona-index {k} out of range for K={len(eligible)}")
    H = config.n_heads
    N = config.n_slots
    s0_top_k = [int(top[k, h]) for h in range(H)]

    rng = random.Random(args.random_seed)
    random_slots: list[int] = []
    for h in range(H):
        choices = [s for s in range(N) if s != s0_top_k[h]]
        random_slots.append(rng.choice(choices))

    rand_raw = torch.stack([bake[k, h, random_slots[h]] for h in range(H)])  # [H, d]
    # SAME post-transform as the shipped A1 object: p_h = out_ln[h](v + mlp(v)).
    rand_t = l1_transform_selected(val_module, rand_raw.unsqueeze(0))[0]  # [H, d]
    rows = np.ascontiguousarray(rand_t.detach().to(torch.float32).cpu().numpy())
    Hd, d = rows.shape
    if (Hd, d) != (H, d):
        raise SystemExit(f"unexpected random object shape {(Hd, d)} != {(H, d)}")

    out_path = Path(args.out)
    digest = dump_f32(rows, out_path)
    norm = np.linalg.norm(rows, axis=1)

    matched_out = ""
    matched = {}
    if args.matched_out:
        sel_t = l1_transform_selected(val_module, selected_raw[k : k + 1])[0]
        sel_rows = np.ascontiguousarray(sel_t.detach().to(torch.float32).cpu().numpy())
        m_path = Path(args.matched_out)
        m_sha = dump_f32(sel_rows, m_path)
        matched_out = str(m_path)
        sel_norm = np.linalg.norm(sel_rows, axis=1)
        matched = {
            "file": str(m_path),
            "sha256": m_sha,
            "s0_top": s0_top_k,
            "norm_mean": round(float(sel_norm.mean()), 4),
            "norm_max": round(float(sel_norm.max()), 4),
        }

    meta = {
        "issue": 228,
        "artifact": "random8_memory_prefix_control",
        "format": "raw FP32, row-major, H rows x d_model floats, no header",
        "rows": int(H),
        "dim": int(d),
        "bytes": int(out_path.stat().st_size),
        "sha256": digest,
        "persona": f"val offset{args.persona_offset}[{k}] (persona_id={eligible[k]})",
        "fit_budget": f"{args.fit_steps}/{args.fit_max_length}/{args.max_length}",
        "seed": args.seed,
        "random_seed": args.random_seed,
        "prior_random_seed_214": PRIOR_RANDOM_SEED_214,
        "n_slots": int(N),
        "selection": "S0 slots (#211) -> per-head random slot excluding S0 -> l1_transform_selected out_ln[h](v+mlp(v))",
        "s0_top": s0_top_k,
        "random_slots": random_slots,
        "attn_max_per_head": [round(float(attn[k, h].max()), 4) for h in range(H)],
        "clamped": False,
        "norm_mean": round(float(norm.mean()), 4),
        "norm_max": round(float(norm.max()), 4),
        "matched_selected": matched,
        "source_script": "scripts/gen_memory_prefix_random8.py",
    }
    if args.meta_out:
        Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta_out).write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, sha256 {digest[:16]}...)")
    if args.meta_out:
        print(f"wrote {args.meta_out}")
    if matched_out:
        print(f"wrote {matched_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
