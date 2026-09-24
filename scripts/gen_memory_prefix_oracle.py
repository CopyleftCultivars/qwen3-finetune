#!/usr/bin/env python3
"""#230: generate the ORACLE-selector memory prefix for the disambiguation probe.

Same pipeline, same persona, same fit budget as ``gen_memory_prefix_a1.py`` /
``gen_memory_prefix_random8.py``, but the slot picked per head comes from
``oracle_pack`` (scripts/lpm_quant_gap.py:170) -- the #211/#216 upper-bound
mechanism:

    attn = val_module(persona_id, probe_q)          # probe_q = encoding of the
    top  = attn.argmax(dim=-1)                       # persona's OWN held-out probe
    v    = bake[persona, head, top[head]]            # hard-selected RAW vector

This is the best-case, *cheating* selection: it uses the persona's own held-out
( future ) content, which a truly novel live query would never have. That is
exactly the point: it isolates SELECTION QUALITY -- "if selection were as good as
it could possibly be for this persona, does the model surface persona content?" --
from the corpus / backbone. It is NOT deployable and is labeled as such in the
meta.

The raw selected vectors are then run through ``l1_transform_selected`` (#216),
the same post-transform every shipped bundle uses -- NOT the falsified raw
pre-transform object. Dumped as raw FP32, [H, d_model] row-major.

Outputs:
  --out        raw FP32, [H, d_model] row-major (default H=8, d_model=1024)
  --meta-out   provenance JSON (persona/budget/seed, oracle_top, norms, sha256)

Usage:
  python scripts/gen_memory_prefix_oracle.py \
      --out outputs/r179-m1/s2a/oracle_memory_prefix_p0.f32 \
      --meta-out outputs/r179-m1/s2a/oracle_memory_prefix_p0.meta.json
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
    oracle_pack,
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
                   help="index within the fitted slice (0 = #214/#228's test persona at offset 0)")
    p.add_argument("--fit-steps", type=int, default=40)
    p.add_argument("--fit-lr", type=float, default=1e-3)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--fit-max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    p.add_argument("--meta-out", default="")
    p.add_argument("--soft-out", default="",
                   help="optional SUPPLEMENTARY control: dump the same fit's soft attention-mixture "
                        "post-transform object out_ln[h](mix+mlp(mix)) -- the true query-conditioned "
                        "upper bound #216 scored at 0.25-0.31, vs oracle_pack's HARD argmax")
    return p


def dump_f32(rows: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(rows).tofile(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    # ORACLE: query-conditioned, per-persona held-out probe -> argmax per head.
    pack = oracle_pack(val_module, eligible, probes, probe_q, device)

    k = args.persona_index
    if not (0 <= k < len(eligible)):
        raise SystemExit(f"persona-index {k} out of range for K={len(eligible)}")
    H = config.n_heads
    oracle_top_k = [int(pack["top"][k, h]) for h in range(H)]

    raw_k = pack["vectors"][k].to(device)  # [H, d] raw pre-transform, hard-selected
    oracle_t = l1_transform_selected(val_module, raw_k.unsqueeze(0))[0]  # [H, d] post-transform
    rows = np.ascontiguousarray(oracle_t.detach().to(torch.float32).cpu().numpy())
    Hd, d = rows.shape
    if (Hd, d) != (H, d):
        raise SystemExit(f"unexpected oracle object shape {(Hd, d)} != {(H, d)}")

    out_path = Path(args.out)
    digest = dump_f32(rows, out_path)
    norm = np.linalg.norm(rows, axis=1)

    soft = {}
    if args.soft_out:
        # SUPPLEMENTARY: the exact trained query-conditioned prompt for this
        # persona's held-out probe -- ProjectionNetwork.forward returns
        # p = out_ln[h](mix + mlp(mix)) already post-transformed, so out["prompts"]
        # is the true soft-mixture object (val_module is in eval mode, dropout=0).
        ids = torch.arange(len(eligible), device=device)
        sout = val_module(ids, probe_q)
        soft_t = sout["prompts"][k]  # [H, d] post-transform soft mixture
        soft_attn = sout["attention"][k]  # [H, N]
        s_rows = np.ascontiguousarray(soft_t.detach().to(torch.float32).cpu().numpy())
        s_path = Path(args.soft_out)
        s_sha = dump_f32(s_rows, s_path)
        s_norm = np.linalg.norm(s_rows, axis=1)
        soft = {
            "file": str(s_path),
            "sha256": s_sha,
            "selection": "soft attention mixture over all N slots (query-conditioned), post-transform: out_ln[h](mix+mlp(mix))",
            "soft_note": "SUPPLEMENTARY upper bound (the #216 soft-oracle object, recall 0.25-0.31). Deployable only via the A2 runtime setter (#213), not a static A1 bundle; included here only as a control on the hard-argmax oracle.",
            "attn_max_per_head": [round(float(soft_attn[h].max()), 4) for h in range(H)],
            "norm_mean": round(float(s_norm.mean()), 4),
            "norm_max": round(float(s_norm.max()), 4),
        }

    meta = {
        "issue": 230,
        "artifact": "oracle_select_memory_prefix",
        "format": "raw FP32, row-major, H rows x d_model floats, no header",
        "rows": int(H),
        "dim": int(d),
        "bytes": int(out_path.stat().st_size),
        "sha256": digest,
        "persona": f"val offset{args.persona_offset}[{k}] (persona_id={eligible[k]})",
        "fit_budget": f"{args.fit_steps}/{args.fit_max_length}/{args.max_length}",
        "seed": args.seed,
        "selection": "ORACLE oracle_pack: persona's own held-out probe_q -> attention.argmax per head -> bake[k,h,top] -> l1_transform_selected out_ln[h](v+mlp(v))",
        "deployable": False,
        "cheating": True,
        "oracle_note": "Uses the persona's own held-out/future content as the query. NOT available to a live novel query; this is the selection upper bound, used only to isolate selection quality from corpus/backbone.",
        "oracle_top": oracle_top_k,
        "attn_max_per_head": [float(x) for x in pack["personas"][k]["attn_max_per_head"]],
        "clamped": False,
        "norm_mean": round(float(norm.mean()), 4),
        "norm_max": round(float(norm.max()), 4),
        "soft_mixture": soft,
        "source_script": "scripts/gen_memory_prefix_oracle.py",
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
