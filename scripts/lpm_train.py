#!/usr/bin/env python3
"""#192 (P2.4): train the LPM module against the frozen Qwen3-0.6B backbone.

Wiring that did not exist before #192:

  prompt_text --frozen encoder--> query_emb [B,1024]
              --LPM module------> soft prompts [B,H,1024]
              + tokenized target_text --> backbone inputs_embeds
              --> NTP cross-entropy on target positions only
              + entropy_penalty + lambda*mag_penalty

The full backbone is frozen (weights unchanged, exactly as Cactus deployment);
only the LPM module's per-persona slots and shared projection receive gradients.
The encoder is frozen and prompt_text is fixed, so query_emb is precomputed once
per split rather than re-forwarded every step.

Primary metric is NOT the loss: the loss can fall via non-recall shortcuts. The
validation proxy is a recall@1 over the persona-disjoint val split via test-time
slot personalization (freeze the trained projection, fit each val persona's
slots for a few steps, then check whether each probe target is best predicted
under its own soft prompts).

Dry run (wiring validation, expected to be minutes)::

  [LOCAL_PATH] scripts/lpm_train.py --dry-run

Full run (separately approved)::

  [LOCAL_PATH] scripts/lpm_train.py \
      --steps 3000 --out-dir outputs/lpm-persona-memory-192
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lpm.config import LPMConfig, SUPPORTED_D_MODELS, reserved_id_map  # noqa: E402
from lpm.data import (  # noqa: E402
    PersonaIndex,
    filter_records,
    group_by_persona,
    load_corpus,
)
from lpm.encoder import MODEL_PATH_QWEN3_1_7B, load_encoder  # noqa: E402
from lpm.memory import LPMModule  # noqa: E402

DEFAULT_MODEL = "outputs/merged-model-distill-v2-rationale"
DEFAULT_TRAIN = "data/persona_memory/lpm_corpus_train.jsonl"
DEFAULT_VAL = "data/persona_memory/lpm_corpus_val.jsonl"

# #231 backbone profiles: 0.6b is the #192 default (d_model=1024, byte-for-byte
# unchanged); 1.7b routes to LPMConfig.qwen3_1_7b() (d_model=2048) and the
# Qwen3-1.7B backbone. "1.7b" is the issue's backbone-scale disambiguation arm.
BACKBONE_0_6B = "0.6b"
BACKBONE_1_7B = "1.7b"
DEFAULT_BACKBONE = BACKBONE_0_6B
BACKBONE_MODEL = {
    BACKBONE_0_6B: DEFAULT_MODEL,
    BACKBONE_1_7B: MODEL_PATH_QWEN3_1_7B,
}
BACKBONE_CONFIG = {
    BACKBONE_0_6B: LPMConfig.qwen3_0_6b,
    BACKBONE_1_7B: LPMConfig.qwen3_1_7b,
}


@dataclass
class Batch:
    persona_ids: torch.Tensor
    query_emb: torch.Tensor
    target_input_ids: torch.Tensor
    target_attention_mask: torch.Tensor


class TargetEncoder:
    """Pre-tokenized target_text, shared by train and val."""

    def __init__(self, tokenizer: Any, records: list[dict[str, Any]], max_length: int):
        self.records = records
        enc = tokenizer(
            [r["target_text"] for r in records],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def gather(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[indices], self.attention_mask[indices]


def select_train_records(
    records: list[dict[str, Any]], config: LPMConfig, min_chunks: int
) -> list[dict[str, Any]]:
    """Records for personas with >= min_chunks cross-chunk records (the #192 subset)."""
    filtered = filter_records(records, config)
    groups = group_by_persona(filtered)
    keep = {pid for pid, group in groups.items() if len(group) >= min_chunks}
    kept = [r for r in filtered if r["persona_id"] in keep]
    return kept


def encode_texts_chunked(
    encoder: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    chunk: int,
) -> torch.Tensor:
    """Encode texts in mini-batches to bound encoder activation memory.

    ``chunk <= 0`` (or ``chunk >= len(texts)``) reproduces the original single-call
    behavior exactly, so the 0.6B path is unchanged. A positive chunk bounds the
    per-forward batch for the 1.7B encoder, whose 1215-record single batch would
    otherwise materialize attention matrices far beyond a 12 GB GPU. Padding is
    per-chunk but pooling is masked, so the pooled vectors are identical.
    """
    if chunk <= 0 or len(texts) <= chunk:
        return encoder.encode_texts(tokenizer, texts, max_length=max_length, trainable=False).float()
    parts = [
        encoder.encode_texts(tokenizer, texts[start : start + chunk], max_length=max_length, trainable=False).float()
        for start in range(0, len(texts), chunk)
    ]
    return torch.cat(parts, dim=0)


def forward_ntp(
    model: Any,
    prompts: torch.Tensor,
    target_input_ids: torch.Tensor,
    target_attention_mask: torch.Tensor,
    *,
    return_per_sample: bool = False,
) -> Any:
    """Prepend soft prompts [B,H,d] to target embeddings; CE over target positions only."""
    batch, n_soft, _ = prompts.shape
    embeds = model.get_input_embeddings()(target_input_ids)
    inputs_embeds = torch.cat([prompts.to(embeds.dtype), embeds], dim=1)
    attention_mask = torch.cat(
        [
            torch.ones(batch, n_soft, dtype=target_attention_mask.dtype, device=embeds.device),
            target_attention_mask,
        ],
        dim=1,
    )
    labels = torch.cat(
        [
            torch.full((batch, n_soft), -100, dtype=torch.long, device=embeds.device),
            target_input_ids.masked_fill(target_attention_mask == 0, -100),
        ],
        dim=1,
    )
    if not return_per_sample:
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return out.loss
    logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(batch, -1)
    denom = (shift_labels != -100).sum(dim=1).clamp(min=1)
    return per_token.sum(dim=1) / denom


def build_train_batches(
    records: list[dict[str, Any]],
    index: PersonaIndex,
    query_emb: torch.Tensor,
    targets: TargetEncoder,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> list[Batch]:
    """Deterministic batches over the subset (one pass = one epoch schedule unit)."""
    persona_ids = torch.tensor([index.id_of(r["persona_id"]) for r in records])
    batches: list[Batch] = []
    perm = torch.randperm(len(records), generator=generator)
    for start in range(0, len(records), batch_size):
        idx = perm[start : start + batch_size]
        tid, tam = targets.gather(idx)
        batches.append(
            Batch(
                persona_ids=persona_ids[idx],
                query_emb=query_emb[idx],
                target_input_ids=tid,
                target_attention_mask=tam,
            )
        )
    return batches


def split_fit_probe(
    records: list[dict[str, Any]], *, probes_per_persona: int = 1
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reserve the last ``probes_per_persona`` cross-chunk records of each persona.

    The probe records are never trained on, so their recall@1 measures genuine
    within-persona generalization on a stable, large set of train personas
    (unlike the persona-disjoint val proxy, which needs test-time slot fitting).
    """
    groups = group_by_persona(records)
    fit: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for pid in sorted(groups):
        group = groups[pid]
        if len(group) <= probes_per_persona:
            fit.extend(group)
            continue
        fit.extend(group[: -probes_per_persona])
        probes.extend(group[-probes_per_persona:])
    return fit, probes


def compute_recall_at1(
    module: LPMModule,
    model: Any,
    tokenizer: Any,
    probe_records: list[dict[str, Any]],
    probe_query_emb: torch.Tensor,
    persona_ids: list[int],
    *,
    max_length: int,
    device: torch.device,
    target_chunk: int = 8,
) -> tuple[float, torch.Tensor]:
    """B x B NTP-loss recall@1: own soft prompts should predict own target best.

    ``persona_ids[j]`` is the slot-memory index for probe j (train probes come
    from arbitrary personas, so position in the list is not the slot index). The
    B x B matrix is filled in target chunks and one source at a time so the
    [chunk, T, vocab] logits never blow up GPU memory.
    """
    n = len(probe_records)
    target_enc = TargetEncoder(tokenizer, probe_records, max_length)
    loss_matrix = torch.zeros(n, n)
    module.eval()
    with torch.no_grad():
        for source in range(n):
            source_id = torch.tensor([persona_ids[source]], device=device)
            prompts = module(source_id, probe_query_emb[source : source + 1])["prompts"]
            for start in range(0, n, target_chunk):
                stop = min(start + target_chunk, n)
                expanded = prompts.expand(stop - start, -1, -1)
                loss_matrix[start:stop, source] = forward_ntp(
                    model,
                    expanded,
                    target_enc.input_ids[start:stop].to(device),
                    target_enc.attention_mask[start:stop].to(device),
                    return_per_sample=True,
                ).cpu()
    recall = (loss_matrix.argmin(dim=1) == torch.arange(n)).float().mean().item()
    return recall, loss_matrix


def evaluate_val_recall(
    train_module: LPMModule,
    encoder: Any,
    model: Any,
    tokenizer: Any,
    val_records: list[dict[str, Any]],
    config: LPMConfig,
    *,
    n_personas: int,
    fit_steps: int,
    lr: float,
    max_length: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Test-time personalization recall@1 on persona-disjoint val personas.

    Freeze the trained projection, from-scratch fit each selected val persona's
    slots on its earlier chunks, then rank each held-out probe target among all
    selected personas' soft prompts. recall@1 = own prompts give the lowest NTP
    loss. A shortcut-free module should beat chance (1/n_personas).
    """
    records = filter_records(val_records, config)
    groups = group_by_persona(records)
    eligible = sorted(pid for pid, group in groups.items() if len(group) >= 2)[:n_personas]
    if len(eligible) < 2:
        return {"val_personas": len(eligible), "val_recall_at1": None, "reason": "too few val personas"}

    val_index = {pid: i for i, pid in enumerate(eligible)}
    fit_records: list[dict[str, Any]] = []
    fit_persona_ids: list[int] = []
    probes: list[dict[str, Any]] = []
    for pid in eligible:
        group = sorted(groups[pid], key=lambda r: r["chunk_index"])
        for record in group[:-1]:
            fit_records.append(record)
            fit_persona_ids.append(val_index[pid])
        probes.append(group[-1])

    val_module = LPMModule(n_personas=len(eligible), config=config).to(device)
    val_module.projection.load_state_dict(train_module.projection.state_dict())
    for param in val_module.projection.parameters():
        param.requires_grad = False
    for param in val_module.slot_memory.parameters():
        param.requires_grad = True

    fit_targets = TargetEncoder(tokenizer, fit_records, max_length)
    fit_query = encoder.encode_texts(
        tokenizer, [r["prompt_text"] for r in fit_records], max_length=max_length, trainable=False
    ).float()
    optimizer = torch.optim.Adam(val_module.slot_memory.parameters(), lr=lr)
    fit_persona_t = torch.tensor(fit_persona_ids, device=device)
    batch_size = min(8, len(fit_records))
    val_module.train()
    for _step in range(fit_steps):
        order = torch.randperm(len(fit_records), generator=generator)[:batch_size]
        out = val_module(fit_persona_t[order], fit_query[order])
        tid = fit_targets.input_ids[order].to(device)
        tam = fit_targets.attention_mask[order].to(device)
        ntp = forward_ntp(model, out["prompts"], tid, tam)
        terms = val_module.loss_terms(out)
        loss = ntp + terms["entropy_penalty"] + terms["mag_penalty"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    val_module.eval()
    probe_query = encoder.encode_texts(
        tokenizer, [r["prompt_text"] for r in probes], max_length=max_length, trainable=False
    ).float()
    n = len(probes)
    recall, loss_matrix = compute_recall_at1(
        val_module, model, tokenizer, probes, probe_query, list(range(n)),
        max_length=max_length, device=device,
    )
    own = loss_matrix.diagonal()
    return {
        "val_personas": n,
        "val_fit_steps": fit_steps,
        "val_fit_records": len(fit_records),
        "val_recall_at1": round(recall, 4),
        "val_chance": round(1.0 / n, 4),
        "val_own_ntp_loss": round(float(own.mean()), 4),
        "val_loss_matrix_mean": round(float(loss_matrix.mean()), 4),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backbone", choices=(BACKBONE_0_6B, BACKBONE_1_7B), default=DEFAULT_BACKBONE,
                        help="Backbone profile: 0.6b (d_model=1024; #192 default) or 1.7b (d_model=2048; #231).")
    parser.add_argument("--model", default=None,
                        help="Override the backbone path/HF id (default: the --backbone's own).")
    parser.add_argument("--backbone-path", default=None,
                        help="Explicit backbone/encoder path or HF id, taking precedence over both "
                             "--backbone's built-in model and --model (e.g. the #232 adapted 1.7B "
                             "checkpoint outputs/merged-model-distill-v2-rationale-1.7b). The LPM "
                             "d_model profile still comes from --backbone; a hidden-size mismatch "
                             "is rejected loudly.")
    parser.add_argument("--train-corpus", default=DEFAULT_TRAIN)
    parser.add_argument("--val-corpus", default=DEFAULT_VAL)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--encoder-chunk", type=int, default=0,
                        help="Mini-batch size for the encoder precompute (0 = single batch, the 0.6B "
                             "default). Set a positive value for the 1.7B encoder to bound its activation "
                             "memory; pooled vectors are unchanged (masked pooling).")
    parser.add_argument("--min-chunks", type=int, default=3, help="#192 persona subset: >= N chunks")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--lambda-mag", type=float, default=None, help="Override LPMConfig.lambda_mag")
    parser.add_argument("--alpha-entropy", type=float, default=None, help="Override LPMConfig.alpha_entropy")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-personas", type=int, default=12)
    parser.add_argument(
        "--probe-personas", type=int, default=32,
        help="Size of the within-persona held-out recall probe (train personas).",
    )
    parser.add_argument("--val-fit-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable backbone gradient checkpointing (saves compute, costs activation memory).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/lpm-persona-memory-192")
    parser.add_argument("--curve-out", default="results/F179_M2_TRAINING_CURVES.json")
    parser.add_argument("--dry-run", action="store_true", help="300 steps, current dir checkpoint tag")
    parser.add_argument("--save-checkpoint", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        args.steps = min(args.steps, 300)
        args.eval_every = min(args.eval_every, 100)
        if args.out_dir.endswith("lpm-persona-memory-192"):
            args.out_dir = args.out_dir + "-dryrun"
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    model_path = args.backbone_path or args.model or BACKBONE_MODEL[args.backbone]
    config_factory = BACKBONE_CONFIG[args.backbone]
    override = " (backbone-path override)" if args.backbone_path else ""
    print(f"backbone={args.backbone} model={model_path}{override}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    for param in model.parameters():
        param.requires_grad = False
    if not args.no_grad_checkpoint:
        # Frozen backbone, but gradients still flow through it to the soft
        # prompts; checkpointing trades recompute for the activation memory that
        # a 28-layer backward would otherwise hold. HF only checkpoints in
        # train(); dropout is 0 for Qwen3, so train() is behaviorally inert here.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False
        model.train()
    else:
        model.eval()

    config_overrides = {}
    if args.lambda_mag is not None:
        config_overrides["lambda_mag"] = args.lambda_mag
    if args.alpha_entropy is not None:
        config_overrides["alpha_entropy"] = args.alpha_entropy
    config = config_factory(**config_overrides)
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is not None and hidden_size != config.d_model:
        raise ValueError(
            f"backbone {model_path!r} hidden_size={hidden_size} != LPMConfig.d_model="
            f"{config.d_model} for --backbone {args.backbone}; pick the matching --backbone "
            f"profile or a supported width {SUPPORTED_D_MODELS}"
        )

    train_all = load_corpus(args.train_corpus)
    train_records = select_train_records(train_all, config, args.min_chunks)
    train_index = PersonaIndex.build(train_records)
    fit_records, probe_records = split_fit_probe(train_records, probes_per_persona=1)
    print(
        f"train subset: {len(train_records)} records, {len(train_index)} personas "
        f"(>= {args.min_chunks} cross-chunk records); "
        f"fit {len(fit_records)} / held-out probes {len(probe_records)}"
    )

    encoder = load_encoder(model_path, n_layers=args.encoder_layers, device=device, trainable_layers=0)
    with torch.no_grad():
        query_emb = encode_texts_chunked(
            encoder, tokenizer, [r["prompt_text"] for r in fit_records],
            max_length=args.max_length, chunk=args.encoder_chunk,
        ).to(device)
        probe_query_emb = encode_texts_chunked(
            encoder, tokenizer, [r["prompt_text"] for r in probe_records],
            max_length=args.max_length, chunk=args.encoder_chunk,
        ).to(device)
    targets = TargetEncoder(tokenizer, fit_records, args.max_length)

    module = LPMModule(n_personas=len(train_index), config=config).to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr)

    val_records = load_corpus(args.val_corpus)
    curves: list[dict[str, Any]] = []
    step = 0
    start_wall = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        batches = build_train_batches(
            fit_records, train_index, query_emb, targets,
            batch_size=args.batch_size, generator=generator,
        )
        for batch in batches:
            if step >= args.steps:
                break
            batch = Batch(
                persona_ids=batch.persona_ids.to(device),
                query_emb=batch.query_emb.to(device),
                target_input_ids=batch.target_input_ids.to(device),
                target_attention_mask=batch.target_attention_mask.to(device),
            )
            module.train()
            out = module(batch.persona_ids, batch.query_emb)
            ntp = forward_ntp(model, out["prompts"], batch.target_input_ids, batch.target_attention_mask)
            terms = module.loss_terms(out)
            loss = ntp + terms["entropy_penalty"] + terms["mag_penalty"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(module.parameters(), args.grad_clip)
            slot_grad = module.slot_memory.slots.grad
            proj_grad = module.projection.w_v[0].weight.grad
            optimizer.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                record = {
                    "step": step,
                    "loss": round(float(loss.detach()), 4),
                    "ntp_loss": round(float(ntp.detach()), 4),
                    "entropy_penalty": round(float(terms["entropy_penalty"].detach()), 5),
                    "mag_penalty_scaled": round(float(terms["mag_penalty"].detach()), 5),
                    "entropy": round(float(out["entropy"].detach()), 4),
                    "projected_norm_mean": round(float(out["projected_norm_mean"].detach()), 4),
                    "projected_norm_max": round(float(out["projected_norm_max"].detach()), 4),
                    "slot_grad_norm": round(float(slot_grad.norm()), 6) if slot_grad is not None else 0.0,
                    "proj_grad_norm": round(float(proj_grad.norm()), 6) if proj_grad is not None else 0.0,
                }
                curves.append(record)
                if record["projected_norm_max"] > config.max_vector_norm:
                    record["LEAKAGE_WATCH_projected_norm_exceeds_max"] = True
                print(json.dumps(record))

            if step % args.eval_every == 0:
                # Stable within-persona recall: held-out chunks of train
                # personas, their slots trained on the rest of the persona.
                probe_n = min(args.probe_personas, len(probe_records))
                probe_slice = probe_records[:probe_n]
                train_recall, _ = compute_recall_at1(
                    module, model, tokenizer, probe_slice, probe_query_emb[:probe_n],
                    [train_index.id_of(r["persona_id"]) for r in probe_slice],
                    max_length=args.max_length, device=device,
                )
                # Persona-disjoint transfer: test-time slot personalization on val.
                eval_out = evaluate_val_recall(
                    module, encoder, model, tokenizer, val_records, config,
                    n_personas=args.val_personas, fit_steps=args.val_fit_steps,
                    lr=args.lr, max_length=args.max_length, device=device, generator=generator,
                )
                eval_out["train_probe_recall_at1"] = round(train_recall, 4)
                eval_out["train_probe_personas"] = probe_n
                eval_out["step"] = step
                eval_out["wall_seconds"] = round(time.time() - start_wall, 1)
                curves.append({"step": step, "eval": eval_out})
                print("EVAL " + json.dumps(eval_out))

    wall = time.time() - start_wall
    peak = round(torch.cuda.max_memory_allocated(device) / 1e9, 3) if device.type == "cuda" else None
    summary = {
        "issue": 192,
        "backbone": args.backbone,
        "model": model_path,
        "backbone_path": args.backbone_path,
        "backbone_hidden_size": hidden_size,
        "d_model": config.d_model,
        "encoder_chunk": args.encoder_chunk,
        "steps": step,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "min_chunks": args.min_chunks,
        "train_records": len(train_records),
        "fit_records": len(fit_records),
        "held_out_probes": len(probe_records),
        "train_personas": len(train_index),
        "module_params": module.trainable_parameter_count(),
        "mag_penalty_mode": config.mag_penalty_mode,
        "clamp_memory_norm": config.clamp_memory_norm,
        "slot_params": config.total_slots * config.d_mem * len(train_index),
        "wall_seconds": round(wall, 1),
        "gpu_hours": round(wall / 3600, 3) if device.type == "cuda" else 0.0,
        "peak_gpu_gb": peak,
        "final_ntp_loss": round(float(ntp.detach()), 4),
        "final_loss": round(float(loss.detach()), 4),
        "config": config.to_dict(),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "module_state_dict": module.state_dict(),
        "config": config.to_dict(),
        "persona_to_slot_index": dict(train_index.persona_to_id),
        "reserved_id_map": {str(k): list(v) for k, v in reserved_id_map(config.n_heads, config.n_slots).items()},
        "subset": {
            "min_chunks": args.min_chunks,
            "train_records": len(train_records),
            "fit_records": len(fit_records),
            "held_out_probes": len(probe_records),
        },
        "step": step,
        "summary": summary,
    }
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    curve_path = Path(args.curve_out)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps({"summary": summary, "curves": curves}, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"wrote {checkpoint_path} and {curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
