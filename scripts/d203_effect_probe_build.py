#!/usr/bin/env python3
"""#214 effect-probe bundle builder.

Builds ONE patched model dir with two persona-slots (the #193 eval-packing
formula PACK_BASE + persona_idx*8 + head, i.e. <|mem0_0..7|> and <|mem1_0..7|>):

  slot 0 = SELECTED: one val persona's S0-selected 8 vectors (issue-default fit
           40/128/192), gathered as bake[persona, h, s0_top[persona, h]] using
           the recorded s0_top from results/D203_S0_BASELINE_offset<N>.json.
  slot 1 = RANDOM:   for each head, a random slot index in [0,32) excluding the
           selected one, gathered the same way -- genuinely random candidates
           from bake_vectors, not another persona's meaningful slots.

Writes the patched HF model dir (patch_model_dir) + a metadata JSON. Convert
that dir to CQ weights, then swap token_embeddings.weights into a copy of
p193-suppress-deploy (which carries the #205 suppression manifest key) to get a
patched-engine bundle for the effect probe.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402

from lpm.checkpoint import load_lpm_checkpoint  # noqa: E402
from lpm.encoder import load_encoder  # noqa: E402
from lpm.data import load_corpus  # noqa: E402
from lpm_quant_gap import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_MODEL,
    DEFAULT_VAL,
    fit_val_slots,
    patch_model_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--val-corpus", default=DEFAULT_VAL)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--n-personas", type=int, default=12)
    parser.add_argument("--persona-offset", type=int, default=0)
    parser.add_argument("--persona-index", type=int, default=0, help="Which persona in the slice to use.")
    parser.add_argument("--fit-steps", type=int, default=40)
    parser.add_argument("--fit-lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--fit-max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260923)
    parser.add_argument("--s0-json", default=None, help="Default: results/D203_S0_BASELINE_offset<N>.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-model", default="outputs/r179-m1/s2a/effect-model")
    parser.add_argument("--out-meta", default="results/D203_S2A_EFFECT_BUILD.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    s0_path = Path(args.s0_json) if args.s0_json else REPO_ROOT / "results" / f"D203_S0_BASELINE_offset{args.persona_offset}.json"
    s0 = json.loads(s0_path.read_text(encoding="utf-8"))
    s0_top = s0["s0_top"]
    p = args.persona_index

    ck = load_lpm_checkpoint(args.checkpoint, device=device)
    config = ck.config
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    encoder = load_encoder(args.model, n_layers=args.encoder_layers, device=device, trainable_layers=0)
    val_module, eligible, probes, probe_q, fit_q, fit_pid = fit_val_slots(
        ck, encoder, model, tokenizer, load_corpus(args.val_corpus), config,
        n_personas=args.n_personas, persona_offset=args.persona_offset, fit_steps=args.fit_steps, lr=args.fit_lr,
        max_length=args.max_length, fit_max_length=args.fit_max_length, device=device, seed=args.seed,
    )
    H = config.n_heads
    N = config.n_slots
    bake = val_module.bake_vectors(torch.tensor([p], device=device))[0]  # [H, N, d]
    selected = torch.stack([bake[h, int(s0_top[p][h])] for h in range(H)])  # [H, d]

    rng = random.Random(args.random_seed)
    random_slots = []
    for h in range(H):
        choices = [s for s in range(N) if s != int(s0_top[p][h])]
        random_slots.append(rng.choice(choices))
    rand = torch.stack([bake[h, random_slots[h]] for h in range(H)])  # [H, d]

    vectors = torch.stack([selected, rand]).float().cpu()  # [2, H, d]
    patch_model_dir(Path(args.model), Path(args.out_model), vectors)

    meta = {
        "issue": 214,
        "source_s0_json": str(s0_path),
        "persona_offset": args.persona_offset,
        "persona_index": p,
        "persona_id": eligible[p],
        "n_heads": H,
        "n_slots": N,
        "fit_steps": args.fit_steps,
        "fit_max_length": args.fit_max_length,
        "max_length": args.max_length,
        "selected_slots": [int(s0_top[p][h]) for h in range(H)],
        "random_slots": random_slots,
        "random_seed": args.random_seed,
        "packing": "slot 0 = PACK_BASE+0*8+h (<|mem0_h|>), slot 1 = PACK_BASE+1*8+h (<|mem1_h|>)",
        "selected_norm_mean": round(float(selected.norm(dim=-1).mean()), 4),
        "random_norm_mean": round(float(rand.norm(dim=-1).mean()), 4),
    }
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_meta).write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out_model} and {args.out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
