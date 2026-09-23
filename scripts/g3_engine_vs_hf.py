#!/usr/bin/env python3
"""#226 (S3) G3: engine-vs-HF directional parity for the armed memory prefix.

Reads the engine first-token records produced by
``cactus-engine/tests/test_memory_prefix_g3.cpp`` (armed vs disarmed) and compares
them to the HF-side next-token reference computed by prepending the SAME A1
vectors to the SAME raw prompt tokens (the ``forward_ntp`` geometry).

Parity is directional, not bitwise: the engine bundle is CQ4-quantized and the
engine prepends to raw prompt tokens here (the chat-template app path is S4/S6),
so exact argmax equality is not required. We check:
  * conditioning present (engine): armed first token != disarmed first token
  * conditioning present (HF): memory argmax != no-memory argmax
  * directional consistency: the engine's armed token is (weakly) more likely
    under the HF memory prefix than under no memory (rank_mem <= rank_nomem),
    and matches the HF memory argmax at least as often as the disarmed token
    matches the HF no-memory argmax.

Usage:
  python scripts/g3_engine_vs_hf.py --engine-out /tmp/g3_engine.txt \
      --prefix-file outputs/r179-m1/s2a/a1_memory_prefix_p0.f32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "outputs" / "merged-model-distill-v2-rationale"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-out", required=True)
    p.add_argument("--prefix-file", required=True)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu")
    from transformers import AutoModelForCausalLM

    mem = np.fromfile(args.prefix_file, dtype=np.float32)
    if mem.size != args.rows * args.dim:
        raise SystemExit(f"prefix file has {mem.size} floats, expected {args.rows*args.dim}")
    mem_t = torch.from_numpy(mem.reshape(args.rows, args.dim)).to(device=device, dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    embed = model.get_input_embeddings()

    rows = []
    for line in Path(args.engine_out).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        armed_first, dis_first, ids = line.split("\t")
        prompt_ids = [int(x) for x in ids.split(",") if x]
        rows.append((int(armed_first), int(dis_first), prompt_ids))

    @torch.no_grad()
    def next_logits(prompt_ids, use_mem):
        tid = torch.tensor([prompt_ids], device=device)
        emb = embed(tid)
        if use_mem:
            inputs = torch.cat([mem_t.unsqueeze(0), emb], dim=1)
        else:
            inputs = emb
        logits = model(inputs_embeds=inputs).logits[0, -1, :].float()
        return logits

    n = len(rows)
    engine_shift = hf_shift = 0
    match_armed = match_dis = 0
    rank_improved = 0
    detail = []
    for armed_first, dis_first, prompt_ids in rows:
        lm = next_logits(prompt_ids, True)
        ln = next_logits(prompt_ids, False)
        hf_mem = int(torch.argmax(lm).item())
        hf_nomem = int(torch.argmax(ln).item())
        if armed_first != dis_first:
            engine_shift += 1
        if hf_mem != hf_nomem:
            hf_shift += 1
        if armed_first == hf_mem:
            match_armed += 1
        if dis_first == hf_nomem:
            match_dis += 1
        rank_mem = int((lm > lm[armed_first]).sum().item())
        rank_nomem = int((ln > ln[armed_first]).sum().item())
        if rank_mem <= rank_nomem:
            rank_improved += 1
        detail.append({
            "prompt_len": len(prompt_ids),
            "engine_armed_first": armed_first,
            "engine_disarmed_first": dis_first,
            "hf_memory_argmax": hf_mem,
            "hf_nomemory_argmax": hf_nomem,
            "engine_armed_matches_hf_memory": armed_first == hf_mem,
            "engine_disarmed_matches_hf_nomemory": dis_first == hf_nomem,
            "rank_of_engine_armed_under_memory": rank_mem,
            "rank_of_engine_armed_under_nomemory": rank_nomem,
        })

    result = {
        "issue": 226,
        "gate": "G3",
        "n_prompts": n,
        "engine_conditioning_shift": f"{engine_shift}/{n}",
        "hf_conditioning_shift": f"{hf_shift}/{n}",
        "engine_armed_matches_hf_memory": f"{match_armed}/{n}",
        "engine_disarmed_matches_hf_nomemory": f"{match_dis}/{n}",
        "engine_armed_rank_improves_under_memory": f"{rank_improved}/{n}",
        "detail": detail,
    }
    print(json.dumps(result, indent=2))

    # Accept: conditioning present on the engine and HF sides for a majority, and
    # the engine's armed token is no worse under the HF memory prefix than the
    # disarmed token is under no memory.
    majority = n // 2 + 1
    ok = engine_shift >= majority and hf_shift >= majority and rank_improved >= majority
    print(f"\nG3 {'PASS' if ok else 'FAIL'}: engine_shift {engine_shift}/{n}, "
          f"hf_shift {hf_shift}/{n}, rank_improved {rank_improved}/{n} (need >= {majority})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
