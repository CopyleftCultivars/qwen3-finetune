#!/usr/bin/env python3
"""#189 / #190 smoke test + encoder decision probe.

Validates the LPM module implementation (shapes, attention normalization,
entropy and magnitude regularizers, gradient flow, reserved-ID mapping) on a
handful of real corpus records, and optionally runs the #189 frozen-vs-fine-tuned
encoder comparison. This is NOT the #192 training pass: a few dozen steps on a
couple dozen personas, clearly labelled, to produce the concrete tradeoff #189
asks for.

Run under the qwen3-train env::

  python scripts/lpm_smoke.py --probe
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lpm.config import LPMConfig, reserved_id_map, slot_to_reserved_id  # noqa: E402
from lpm.data import PersonaIndex, load_corpus, select_subset  # noqa: E402
from lpm.encoder import load_encoder  # noqa: E402
from lpm.memory import LPMModule  # noqa: E402


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_module_smoke(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from transformers import AutoTokenizer

    records = load_corpus(args.corpus)
    config = LPMConfig(include_first_chunk=args.include_first_chunk)
    subset = select_subset(
        records,
        config,
        n_personas=args.n_personas,
        records_per_persona=args.records_per_persona,
        min_chunks_per_persona=2,
    )
    if not subset:
        raise SystemExit("no usable corpus records (check --corpus / filter)")
    index = PersonaIndex.build(subset)
    persona_ids = torch.tensor([index.id_of(r["persona_id"]) for r in subset], device=device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    encoder = load_encoder(args.model_path, n_layers=args.n_layers, device=device)
    query_emb = encoder.encode_texts(
        tokenizer, [r["prompt_text"] for r in subset], max_length=args.max_length
    ).float()

    module = LPMModule(n_personas=len(index), config=config).to(device)
    out = module(persona_ids, query_emb)

    n = len(subset)
    checks: dict[str, Any] = {}
    checks["prompts_shape"] = list(out["prompts"].shape)
    checks["attention_shape"] = list(out["attention"].shape)
    checks["bake_shape"] = list(out["bake_vectors"].shape)
    assert out["prompts"].shape == (n, config.n_heads, config.d_model)
    assert out["attention"].shape == (n, config.n_heads, config.n_slots)
    assert out["bake_vectors"].shape == (n, config.n_heads, config.n_slots, config.d_model)

    attn_sum = out["attention"].sum(dim=-1)
    checks["attention_sum_max_abs_err"] = float((attn_sum - 1.0).abs().max())
    assert checks["attention_sum_max_abs_err"] < 1e-5
    checks["entropy"] = float(out["entropy"])
    checks["entropy_target"] = float(out["entropy_target"])
    assert torch.isfinite(out["prompts"]).all()
    assert checks["entropy"] <= config.entropy_target + 1e-6 or out["entropy_penalty"] == 0

    checks["projected_norm_mean"] = float(out["projected_norm_mean"])
    checks["projected_norm_max"] = float(out["projected_norm_max"])
    checks["mag_penalty"] = float(out["mag_penalty"])
    checks["entropy_penalty"] = float(out["entropy_penalty"])
    assert checks["mag_penalty"] >= 0 and checks["entropy_penalty"] >= 0

    # Gradient flow to both per-persona slots and the shared projection.
    loss_terms = module.loss_terms(out)
    loss = (
        out["prompts"].pow(2).mean()
        + loss_terms["entropy_penalty"]
        + loss_terms["mag_penalty"]
    )
    module.zero_grad(set_to_none=True)
    loss.backward()
    slot_grad = module.slot_memory.slots.grad
    proj_grad = module.projection.w_v[0].weight.grad
    checks["slot_grad_norm"] = float(slot_grad.norm()) if slot_grad is not None else 0.0
    checks["projection_grad_norm"] = float(proj_grad.norm()) if proj_grad is not None else 0.0
    assert checks["slot_grad_norm"] > 0, "no gradient reached the slots"
    assert checks["projection_grad_norm"] > 0, "no gradient reached the 256->1024 projector"

    # Reserved-ID mapping: 256 slots -> 151670..151925, bijective.
    mapping = reserved_id_map(config.n_heads, config.n_slots)
    checks["reserved_ids_used"] = len(mapping)
    checks["reserved_id_range"] = [min(mapping), max(mapping)]
    assert len(mapping) == config.total_slots == 256
    assert min(mapping) == 151670 and max(mapping) == 151925
    assert slot_to_reserved_id(0, 0) == 151670
    assert slot_to_reserved_id(0, 31) == 151701
    assert slot_to_reserved_id(1, 0) == 151702
    checks["slot_0_0_id"] = slot_to_reserved_id(0, 0)
    checks["slot_h1_0_id"] = slot_to_reserved_id(1, 0)
    checks["slot_last_id"] = slot_to_reserved_id(config.n_heads - 1, config.n_slots - 1)

    # bake_vectors must be the projector output (same values as forward).
    bake = module.bake_vectors(persona_ids).to(device)
    checks["bake_matches_forward_max_abs_diff"] = float(
        (bake - out["bake_vectors"].detach()).abs().max()
    )
    assert checks["bake_matches_forward_max_abs_diff"] == 0.0

    checks["n_personas"] = len(index)
    checks["n_records"] = n
    checks["n_slots_per_head"] = config.n_slots
    checks["module_trainable_params"] = module.trainable_parameter_count()
    checks["slot_params_per_persona"] = config.total_slots * config.d_mem
    checks["encoder_frozen_params"] = encoder.frozen_parameter_count()
    checks["encoder_trainable_params"] = encoder.trainable_parameter_count()
    return checks


# --------------------------------------------------------------------------- #
# #189 encoder probe: frozen vs fine-tuned, retrieval proxy + drift + cost
# --------------------------------------------------------------------------- #

def _retrieval_pairs(
    records: list[dict[str, Any]], index: PersonaIndex, persona_ids: list[str]
) -> list[tuple[str, str]]:
    """One (prompt, target) pair per persona, taken from its first cross-chunk record."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["persona_id"] in persona_ids:
            groups.setdefault(record["persona_id"], []).append(record)
    pairs: list[tuple[str, str]] = []
    for pid in persona_ids:
        group = sorted(groups.get(pid, []), key=lambda r: r["chunk_index"])
        if group:
            pairs.append((group[0]["prompt_text"], group[0]["target_text"]))
    return pairs


def _run_arm(
    arm: str,
    train_pairs: list[tuple[str, str]],
    eval_pairs: list[tuple[str, str]],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(0)
    encoder = load_encoder(
        args.model_path,
        n_layers=args.n_layers,
        device=device,
        trainable_layers=1 if arm == "finetuned" else 0,
    )
    head = torch.nn.Linear(encoder.hidden_size, encoder.hidden_size).to(device)
    param_groups = [{"params": list(head.parameters()), "lr": args.probe_lr}]
    if arm == "finetuned":
        encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        # Discriminative LR: the stub is a pretrained prefix, a head-sized LR
        # would wreck it (measured: cos 0.83 / 58% L2 shift at 1e-3).
        param_groups.append({"params": encoder_params, "lr": args.probe_encoder_lr})
    optimizer = torch.optim.Adam(param_groups)
    params = [p for group in param_groups for p in group["params"]]
    temperature = 0.07

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def encode(pairs: list[tuple[str, str]], grad: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [p for p, _ in pairs]
        targets = [t for _, t in pairs]
        trainable = (arm == "finetuned") if grad is None else grad
        with torch.set_grad_enabled(trainable):
            queries = encoder.encode_texts(
                tokenizer, prompts, max_length=args.max_length, trainable=trainable
            ).float()
            keys = encoder.encode_texts(
                tokenizer, targets, max_length=args.max_length, trainable=trainable
            ).float()
        return head(queries), head(keys)

    labels = torch.arange(len(train_pairs), device=device)
    start = time.time()
    for _step in range(args.probe_steps):
        queries, keys = encode(train_pairs)
        logits = (queries @ keys.t()) / temperature
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start

    with torch.no_grad():
        eval_q, eval_k = encode(eval_pairs, grad=False)
        eval_logits = (eval_q @ eval_k.t()) / temperature
        eval_acc = float((eval_logits.argmax(dim=-1) == torch.arange(len(eval_pairs), device=device)).float().mean())

    return {
        "arm": arm,
        "eval_top1": round(eval_acc, 4),
        "final_train_loss": round(float(loss), 4),
        "steps": args.probe_steps,
        "seconds": round(elapsed, 2),
        "seconds_per_step": round(elapsed / args.probe_steps, 4),
        "trainable_params": sum(p.numel() for p in params if p.requires_grad),
        "peak_gpu_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1e6, 1) if device.type == "cuda" else None
        ),
        "encoder": encoder,
    }


def run_encoder_probe(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from transformers import AutoTokenizer

    records = load_corpus(args.corpus)
    config = LPMConfig(include_first_chunk=False)
    subset = select_subset(
        records, config, n_personas=args.probe_train_personas + args.probe_eval_personas,
        records_per_persona=None, min_chunks_per_persona=2,
    )
    index = PersonaIndex.build(subset)
    personas = sorted({r["persona_id"] for r in subset})
    train_personas = personas[: args.probe_train_personas]
    eval_personas = personas[args.probe_train_personas :]
    train_pairs = _retrieval_pairs(subset, index, train_personas)
    eval_pairs = _retrieval_pairs(subset, index, eval_personas)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    frozen = _run_arm("frozen", train_pairs, eval_pairs, tokenizer, args, device)
    frozen_encoder = frozen.pop("encoder")
    finetuned = _run_arm("finetuned", train_pairs, eval_pairs, tokenizer, args, device)
    finetuned_encoder = finetuned.pop("encoder")

    # Drift: how much did fine-tuning the stub move the pooled representations
    # the 28-layer NTP backbone shares? Mean cosine on the eval prompts.
    with torch.no_grad():
        eval_texts = [p for p, _ in eval_pairs]
        a = frozen_encoder.encode_texts(tokenizer, eval_texts, max_length=args.max_length)
        b = finetuned_encoder.encode_texts(tokenizer, eval_texts, max_length=args.max_length)
        cos = F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()
        rel = ((a.float() - b.float()).norm(dim=-1) / a.float().norm(dim=-1).clamp(min=1e-6)).mean().item()

    return {
        "n_train_personas": len(train_personas),
        "n_eval_personas": len(eval_personas),
        "n_layers": args.n_layers,
        "probe_steps": args.probe_steps,
        "frozen": frozen,
        "finetuned": finetuned,
        "eval_representation_cosine_to_frozen": round(cos, 4),
        "eval_representation_relative_l2_shift": round(rel, 4),
        "decision": "frozen",
        "decision_reason": (
            "same weights back the 28-layer NTP forward; fine-tuning the stub changes "
            "the encoder the module reads from only if the shared early layers move, and "
            "the drift + cost below quantify that risk"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", default="outputs/merged-model-distill-v2-rationale")
    parser.add_argument("--corpus", default="data/persona_memory/lpm_corpus_train.jsonl")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-personas", type=int, default=8)
    parser.add_argument("--records-per-persona", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--include-first-chunk", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--probe", action="store_true", help="Run the #189 frozen-vs-finetuned comparison.")
    parser.add_argument("--probe-steps", type=int, default=150)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument(
        "--probe-encoder-lr",
        type=float,
        default=1e-5,
        help="Discriminative LR for the unfrozen encoder stub (#189 fine-tuned arm).",
    )
    parser.add_argument("--probe-train-personas", type=int, default=40)
    parser.add_argument("--probe-eval-personas", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/F179_M2_SMOKE.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)

    report: dict[str, Any] = {"device": str(device), "smoke": build_module_smoke(args, device)}
    if args.probe:
        report["encoder_probe"] = run_encoder_probe(args, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "encoder_probe"}, indent=2))
    if "encoder_probe" in report:
        print(json.dumps(report["encoder_probe"], indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
