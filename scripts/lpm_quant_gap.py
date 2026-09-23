#!/usr/bin/env python3
"""#193 (P3.1 GATE): bf16 vs CQ4 recall gap for the trained LPM memory.

Key methodology facts discovered during this issue (documented in the results
doc, not assumed):

  * ``cactus_score_window`` is a dead stub in the shipped engine
    (``model.cpp::score_tokens_window_logprob`` returns 0.0, tokens_scored=0).
    Verified with a real ARM-engine call: logprob 0, tokens 0. So no token-loss
    metric is possible on-engine; the gate's quantitative recall comparison is
    done in HF against the *dequantized* CQ4 weights (reconstructed from the
    convert pipeline's ``weights_manifest.json`` + ``dequantize_cq_file``), which
    isolates model quantization from engine kernels and is exactly
    apples-to-apples with the bf16 arm. The real ARM engine is still used for the
    required memory-injection + reserved-ID leakage generate() test.

  * #203: bake_vectors() is query-independent, so there is no on-device
    selection. This eval uses an ORACLE: the trained module's query-conditioned
    attention selects the argmax slot per head, and only those baked rows are
    injected. This isolates the quantization question from the deployable-
    selection problem (which is #203's, not #193's).

Subcommands:
  prepare : fit val-persona slots (TTP, as #192), oracle-select per head, pack
            K personas' selected rows into reserved IDs, patch the merged model
            into model-p193/ and write p193_pack.json.
  eval    : run the loss-based KxK recall@1 for a given HF model (bf16 or the
            dequantized CQ4 model) using the pack.
  dynamic : #192's dynamic soft-prompt oracle recall (probe_q-conditioned).
  s0      : #211 Gate 1 -- S0 bake-time fixed-query argmax slots + T0 always-on
            injection, scored on both the dynamic path (matched to #193's 0.250)
            and the reserved-row path (literal deployment form).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lpm.checkpoint import load_lpm_checkpoint  # noqa: E402
from lpm.config import LPMConfig, slot_to_reserved_id  # noqa: E402
from lpm.data import filter_records, group_by_persona, load_corpus  # noqa: E402
from lpm.encoder import load_encoder  # noqa: E402
from lpm.memory import LPMModule  # noqa: E402
from lpm_train import TargetEncoder, forward_ntp  # noqa: E402

DEFAULT_MODEL = "outputs/merged-model-distill-v2-rationale"
DEFAULT_CKPT = "outputs/lpm-persona-memory-192/checkpoint.pt"
DEFAULT_VAL = "data/persona_memory/lpm_corpus_val.jsonl"
PACK_BASE = 151670
PACK_PATH = REPO_ROOT / "outputs" / "lpm-persona-memory-192" / "p193_pack.json"
PACK_VECTORS = REPO_ROOT / "outputs" / "lpm-persona-memory-192" / "p193_pack_vectors.pt"
PATCHED_MODEL = REPO_ROOT / "outputs" / "r179-m1" / "model-p193"


def device_of(name: str) -> torch.device:
    return torch.device("cuda" if (name == "auto" and torch.cuda.is_available()) else name if name != "auto" else "cpu")


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #

def fit_val_slots(
    ck: Any,
    encoder: Any,
    model: Any,
    tokenizer: Any,
    val_records: list[dict[str, Any]],
    config: LPMConfig,
    *,
    n_personas: int,
    persona_offset: int,
    fit_steps: int,
    lr: float,
    max_length: int,
    fit_max_length: int,
    device: torch.device,
    seed: int,
) -> tuple[LPMModule, list[str], list[dict[str, Any]], torch.Tensor, torch.Tensor, list[int]]:
    """TTP-fit persona-disjoint val slots with the trained projection (#192 method).

    Returns ``(val_module, eligible, probes, probe_q, fit_q, fit_pid)``. The last
    two are the *training-time* queries (each persona's non-held-out chunks,
    ``fit_q``) and their persona index (``fit_pid``), which #211's S0 selector
    mean-pools into a fixed bake-time query. They are deliberately the fitting
    rows, never the held-out probe, so the S0 argmax never sees an eval query.
    """
    records = filter_records(val_records, config)
    groups = group_by_persona(records)
    all_eligible = sorted(pid for pid, g in groups.items() if len(g) >= 2)
    eligible = all_eligible[persona_offset : persona_offset + n_personas]
    val_index = {pid: i for i, pid in enumerate(eligible)}
    fit_records: list[dict[str, Any]] = []
    fit_pid: list[int] = []
    probes: list[dict[str, Any]] = []
    for pid in eligible:
        group = sorted(groups[pid], key=lambda r: r["chunk_index"])
        for record in group[:-1]:
            fit_records.append(record)
            fit_pid.append(val_index[pid])
        probes.append(group[-1])

    val_module = LPMModule(n_personas=len(eligible), config=config).to(device)
    val_module.projection.load_state_dict(ck.module.projection.state_dict())
    for p in val_module.projection.parameters():
        p.requires_grad = False
    for p in val_module.slot_memory.parameters():
        p.requires_grad = True

    gen = torch.Generator(device="cpu").manual_seed(seed)
    fit_targets = TargetEncoder(tokenizer, fit_records, fit_max_length)
    fit_q = encoder.encode_texts(
        tokenizer, [r["prompt_text"] for r in fit_records], max_length=max_length, trainable=False
    ).float().to(device)
    pid_t = torch.tensor(fit_pid, device=device)
    opt = torch.optim.Adam(val_module.slot_memory.parameters(), lr=lr)
    batch = min(2, len(fit_records))
    val_module.train()
    for _ in range(fit_steps):
        order = torch.randperm(len(fit_records), generator=gen)[:batch]
        out = val_module(pid_t[order], fit_q[order])
        tid = fit_targets.input_ids[order].to(device)
        tam = fit_targets.attention_mask[order].to(device)
        embeds = model.get_input_embeddings()(tid)
        inputs = torch.cat([out["prompts"].to(embeds.dtype), embeds], dim=1)
        am = torch.cat([torch.ones(batch, config.n_heads, dtype=tam.dtype, device=device), tam], dim=1)
        labels = torch.cat(
            [torch.full((batch, config.n_heads), -100, dtype=torch.long, device=device),
             tid.masked_fill(tam == 0, -100)], dim=1)
        loss = model(inputs_embeds=inputs, attention_mask=am, labels=labels).loss
        terms = val_module.loss_terms(out)
        loss = loss + terms["entropy_penalty"] + terms["mag_penalty"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    val_module.eval()
    probe_q = encoder.encode_texts(
        tokenizer, [r["prompt_text"] for r in probes], max_length=max_length, trainable=False
    ).float().to(device)
    return val_module, eligible, probes, probe_q, fit_q, fit_pid


@torch.no_grad()
def oracle_pack(val_module: LPMModule, eligible: list[str], probes: list[dict[str, Any]], probe_q: torch.Tensor, device: torch.device) -> dict[str, Any]:
    ids = torch.arange(len(eligible), device=device)
    out = val_module(ids, probe_q)
    attn = out["attention"]  # [K, H, N]
    top = attn.argmax(dim=-1)  # [K, H]
    bake = val_module.bake_vectors(ids)  # [K, H, N, d]
    K, H = top.shape
    selected_vectors = torch.stack([bake[k, h, int(top[k, h])] for k in range(K) for h in range(H)]).reshape(K, H, -1)
    personas = []
    for k, pid in enumerate(eligible):
        sel = [(h, int(top[k, h])) for h in range(H)]
        personas.append({
            "persona_id": pid,
            "base_reserved_id": PACK_BASE + k * H,
            "selected_heads_slots": sel,
            "canonical_reserved_ids": [slot_to_reserved_id(h, n) for h, n in sel],
            "attn_max_per_head": [round(float(attn[k, h].max()), 4) for h in range(H)],
        })
    return {"K": K, "n_heads": H, "top": top.cpu(), "vectors": selected_vectors.cpu(), "personas": personas}


@torch.no_grad()
def s0_select(
    val_module: LPMModule,
    fit_q: torch.Tensor,
    fit_pid: list[int],
    n_personas: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """S0: bake-time, query-independent argmax slot per head (#211).

    Unlike ``oracle_pack``, the query is **not** a held-out probe. For each
    persona we mean-pool that persona's own *training-time* query encodings
    (``fit_q`` rows) into one fixed vector, run the projection network exactly
    once, and take ``attention.argmax`` per head. The result is a fixed
    ``[K, H]`` slot-index set frozen before any eval query exists.

    Returns ``(top, selected_vectors, attention)`` where ``selected_vectors`` is
    ``[K, H, d_model]`` gathered exactly like ``oracle_pack``
    (``bake[k, h, top[k, h]]``), so it drops straight into ``patch_model_dir``.
    """
    K = n_personas
    fixed = torch.zeros(K, fit_q.size(-1), dtype=fit_q.dtype, device=device)
    counts = torch.zeros(K, dtype=fit_q.dtype, device=device)
    pid_t = torch.tensor(fit_pid, device=device)
    fixed.index_add_(0, pid_t, fit_q.to(device))
    counts.index_add_(0, pid_t, torch.ones_like(pid_t, dtype=fixed.dtype))
    fixed = fixed / counts.clamp(min=1).unsqueeze(1)

    ids = torch.arange(K, device=device)
    out = val_module(ids, fixed)
    attn = out["attention"]  # [K, H, N]
    top = attn.argmax(dim=-1)  # [K, H]
    bake = val_module.bake_vectors(ids)  # [K, H, N, d]
    H = top.shape[1]
    selected = torch.stack(
        [bake[k, h, int(top[k, h])] for k in range(K) for h in range(H)]
    ).reshape(K, H, -1)
    return top, selected, attn


def patch_model_dir(model_path: Path, out_dir: Path, vectors: torch.Tensor) -> None:
    from safetensors.torch import load_file, save_file

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"):
        src = model_path / name
        if src.exists():
            shutil.copy(src, out_dir / name)
    weights = load_file(str(model_path / "model.safetensors"))
    embed = weights["model.embed_tokens.weight"].clone()
    K, H = vectors.shape[0], vectors.shape[1]
    for k in range(K):
        for h in range(H):
            embed[PACK_BASE + k * H + h] = vectors[k, h].to(embed.dtype)
    weights["model.embed_tokens.weight"] = embed
    save_file(weights, str(out_dir / "model.safetensors"))
    print(f"patched model written to {out_dir} ({K*H} reserved rows)")


def stage_prepare(args: argparse.Namespace) -> None:
    device = device_of(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ck = load_lpm_checkpoint(args.checkpoint, device=device)
    config = ck.config
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    for p in model.parameters():
        p.requires_grad = False
    # Fit needs gradients to the soft prompts through the frozen 28 layers.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    encoder = load_encoder(args.model, n_layers=args.encoder_layers, device=device, trainable_layers=0)

    t0 = time.time()
    val_module, eligible, probes, probe_q, _, _ = fit_val_slots(
        ck, encoder, model, tokenizer, load_corpus(args.val_corpus), config,
        n_personas=args.n_personas, persona_offset=args.persona_offset, fit_steps=args.fit_steps, lr=args.fit_lr,
        max_length=args.max_length, fit_max_length=args.fit_max_length, device=device, seed=args.seed,
    )
    print(f"fitted {len(eligible)} val personas in {time.time()-t0:.1f}s")
    packed = oracle_pack(val_module, eligible, probes, probe_q, device)
    PACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": packed["vectors"], "top": packed["top"]}, PACK_VECTORS)
    meta = {
        "K": packed["K"], "n_heads": packed["n_heads"], "pack_base": PACK_BASE,
        "pack_rule": "reserved_id = 151670 + persona_index*8 + head",
        "origin": "oracle: trained module attention argmax slot per head (#203 workaround)",
        "n_personas": args.n_personas, "fit_steps": args.fit_steps,
        "probe_prompts": [r["prompt_text"] for r in probes],
        "probe_targets": [r["target_text"] for r in probes],
        "personas": packed["personas"],
    }
    PACK_PATH.write_text(json.dumps(meta, indent=2))
    patch_model_dir(Path(args.model), PATCHED_MODEL, packed["vectors"])
    print(f"wrote {PACK_PATH} and {PACK_VECTORS}")


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #

def per_sample_ntp(model: Any, input_ids: torch.Tensor, mem_len: int, attention_mask: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, :mem_len] = -100
    labels[attention_mask == 0] = -100
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(input_ids.size(0), -1)
    denom = (shift_labels != -100).sum(dim=1).clamp(min=1)
    return per_token.sum(dim=1) / denom


@torch.no_grad()
def recall_matrix(model: Any, tokenizer: Any, pack: dict[str, Any], *, max_length: int, device: torch.device, chunk: int) -> torch.Tensor:
    K, H = pack["K"], pack["n_heads"]
    enc = tokenizer(pack["probe_targets"], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tid = enc["input_ids"].to(device)
    tam = enc["attention_mask"].to(device)
    M = torch.zeros(K, K)
    for j in range(K):
        # Persona j's H selected vectors were packed one-per-head at consecutive
        # reserved IDs, so injection uses base + (0..H-1); `top` only selected
        # *which* vector got baked, it is not needed for the ID.
        mem = torch.tensor([PACK_BASE + j * H + h for h in range(H)], device=device)  # [H]
        for s in range(0, K, chunk):
            e = min(K, s + chunk)
            ids = torch.cat([mem.unsqueeze(0).expand(e - s, H), tid[s:e]], dim=1)
            mask = torch.cat([torch.ones(e - s, H, dtype=tam.dtype, device=device), tam[s:e]], dim=1)
            M[s:e, j] = per_sample_ntp(model, ids, H, mask).cpu()
    return M


@torch.no_grad()
def no_memory_loss(model: Any, tokenizer: Any, pack: dict[str, Any], *, max_length: int, device: torch.device) -> torch.Tensor:
    enc = tokenizer(pack["probe_targets"], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tid = enc["input_ids"].to(device)
    tam = enc["attention_mask"].to(device)
    return per_sample_ntp(model, tid, 0, tam).cpu()


def summarize(M: torch.Tensor, no_mem: torch.Tensor) -> dict[str, Any]:
    K = M.size(0)
    own = M.diagonal()
    recall = (M.argmin(dim=1) == torch.arange(K)).float().mean().item()
    # chance-level reference: mean over all off-diagonal columns
    off = (M.sum() - own.sum()) / (K * (K - 1))
    return {
        "K": K,
        "recall_at1": round(recall, 4),
        "chance": round(1.0 / K, 4),
        "own_ntp_loss_mean": round(float(own.mean()), 4),
        "offdiag_ntp_loss_mean": round(float(off), 4),
        "own_minus_offdiag": round(float(own.mean() - off), 4),
        "no_memory_loss_mean": round(float(no_mem.mean()), 4),
        "own_vs_no_memory_delta": round(float(own.mean() - no_mem.mean()), 4),
    }


def load_dequantized_cq4(weights_dir: Path, base_model: str, dtype: torch.dtype) -> Any:
    """Reconstruct the CQ4-quantized HF model from the convert weights dir."""
    from cactus.convert.export.qdq import dequantize_cq_file, dequantize_fp_file, read_header
    from transformers import AutoModelForCausalLM

    manifest = json.loads((weights_dir / "weights_manifest.json").read_text())["weights"]
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    state = model.state_dict()
    applied, skipped = 0, 0
    for entry in manifest:
        hf_name = entry["hf_name"]
        if hf_name not in state:
            skipped += 1
            continue
        path = weights_dir / entry["output_name"]
        header = read_header(path)
        if entry["precision"] == "FP16":
            tensor = dequantize_fp_file(path, header, dtype)
        else:
            tensor = dequantize_cq_file(path, header, dtype, row_batch_size=4096)
        tensor = tensor.reshape(state[hf_name].shape)
        with torch.no_grad():
            state[hf_name].copy_(tensor)
        applied += 1
    print(f"dequantized CQ4 model: {applied} tensors applied, {skipped} skipped (not in state_dict)")
    model.eval()
    return model


def stage_eval(args: argparse.Namespace) -> None:
    device = device_of(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pack = json.loads(PACK_PATH.read_text())
    vectors = torch.load(PACK_VECTORS, weights_only=False)["vectors"]
    pack["_vectors"] = vectors
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.arm == "bf16":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        model.eval()
        embed = model.get_input_embeddings().weight.data
        for k in range(pack["K"]):
            for h in range(pack["n_heads"]):
                embed[PACK_BASE + k * pack["n_heads"] + h] = vectors[k, h].to(embed.dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.cq4_model, torch_dtype=torch.bfloat16).to(device)
        model.eval()
    no_mem = no_memory_loss(model, tokenizer, pack, max_length=args.max_length, device=device)
    M = recall_matrix(model, tokenizer, pack, max_length=args.max_length, device=device, chunk=args.chunk)
    result = {"arm": args.arm, "model": args.model if args.arm == "bf16" else args.weights_dir, **summarize(M, no_mem)}
    out = REPO_ROOT / "results" / f"F179_M3_QUANT_GAP_{args.arm}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


@torch.no_grad()
def dynamic_recall_matrix(
    val_module: LPMModule, model: Any, tokenizer: Any, probes: list[dict[str, Any]],
    probe_q: torch.Tensor, *, max_length: int, device: torch.device, chunk: int = 8,
) -> torch.Tensor:
    """#192's dynamic soft-prompt recall: no reserved rows touched, so this is the
    actual LPM mechanism and isolates backbone quantization from the tied-head
    row-patching confound."""
    K = len(probes)
    enc = tokenizer([r["target_text"] for r in probes], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tid = enc["input_ids"].to(device)
    tam = enc["attention_mask"].to(device)
    M = torch.zeros(K, K)
    for j in range(K):
        prompts = val_module(torch.tensor([j], device=device), probe_q[j : j + 1])["prompts"]
        for s in range(0, K, chunk):
            e = min(K, s + chunk)
            M[s:e, j] = forward_ntp(
                model, prompts.expand(e - s, -1, -1), tid[s:e], tam[s:e], return_per_sample=True
            ).cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return M


@torch.no_grad()
def fixed_prompt_recall_matrix(
    model: Any, tokenizer: Any, probes: list[dict[str, Any]], prompts: torch.Tensor,
    *, max_length: int, device: torch.device, chunk: int = 8,
) -> torch.Tensor:
    """T0 recall on the dynamic soft-prompt path with S0-selected prompts.

    Same scoring code as ``dynamic_recall_matrix`` (``forward_ntp`` per pair),
    but the prompt for persona ``j`` is the S0-selected vector set
    ``prompts[j]`` -- fixed once per persona, no eval query touches selection --
    instead of the probe-conditioned attention mixture. This is the number
    directly comparable to #193's 0.250 oracle-fit reference.
    """
    K = len(probes)
    enc = tokenizer([r["target_text"] for r in probes], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tid = enc["input_ids"].to(device)
    tam = enc["attention_mask"].to(device)
    M = torch.zeros(K, K)
    for j in range(K):
        prompt_j = prompts[j : j + 1]
        for s in range(0, K, chunk):
            e = min(K, s + chunk)
            M[s:e, j] = forward_ntp(
                model, prompt_j.expand(e - s, -1, -1), tid[s:e], tam[s:e], return_per_sample=True
            ).cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return M


def stage_dynamic(args: argparse.Namespace) -> None:
    device = device_of(args.device)
    from transformers import AutoTokenizer

    ck = load_lpm_checkpoint(args.checkpoint, device=device)
    config = ck.config
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.arm == "bf16":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.cq4_model, torch_dtype=torch.bfloat16).to(device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()

    encoder = load_encoder(args.model, n_layers=args.encoder_layers, device=device, trainable_layers=0)
    val_module, eligible, probes, probe_q, _, _ = fit_val_slots(
        ck, encoder, model, tokenizer, load_corpus(args.val_corpus), config,
        n_personas=args.n_personas, persona_offset=args.persona_offset, fit_steps=args.fit_steps, lr=args.fit_lr,
        max_length=args.max_length, fit_max_length=args.fit_max_length, device=device, seed=args.seed,
    )
    model.eval()
    M = dynamic_recall_matrix(val_module, model, tokenizer, probes, probe_q, max_length=args.max_length, device=device)
    enc = tokenizer([r["target_text"] for r in probes], return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
    zero_prompts = torch.zeros(len(probes), config.n_heads, config.d_model, device=device, dtype=torch.bfloat16)
    no_mem = forward_ntp(
        model, zero_prompts, enc["input_ids"].to(device), enc["attention_mask"].to(device), return_per_sample=True
    ).cpu()
    result = {"arm": args.arm, "path": "dynamic_soft_prompt", "n_personas": len(eligible), **summarize(M, no_mem)}
    out = REPO_ROOT / "results" / f"F179_M3_QUANT_GAP_dynamic_{args.arm}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


def stage_s0(args: argparse.Namespace) -> None:
    """#211 Gate 1: S0 selection (bake-time fixed query) + T0 always-on injection.

    Slot-fitting is byte-for-byte the same as ``dynamic``/``prepare`` -- only the
    argmax query source changes (per-persona mean of training queries instead of
    each probe's own held-out encoding). The S0 slots are scored on both paths:

      * ``dynamic_soft_prompt``: the 8 selected vectors injected as soft prompts,
        scored with the same ``forward_ntp`` code as #193's 0.250 -- the matched
        comparison;
      * ``reserved_rows``: the same 8 vectors written to reserved embedding rows
        and scored with ``recall_matrix`` -- the literal T0 deployment form, which
        carries the known tied-head contamination (#193 §4 / #204).
    """
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
        n_personas=args.n_personas, persona_offset=args.persona_offset, fit_steps=args.fit_steps, lr=args.fit_lr,
        max_length=args.max_length, fit_max_length=args.fit_max_length, device=device, seed=args.seed,
    )
    print(f"fitted {len(eligible)} val personas in {time.time()-t0:.1f}s")

    top, selected, attn = s0_select(val_module, fit_q, fit_pid, len(eligible), device)
    model.eval()

    enc = tokenizer([r["target_text"] for r in probes], return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
    tid = enc["input_ids"].to(device)
    tam = enc["attention_mask"].to(device)

    # --- matched comparison to #193's 0.250: dynamic soft-prompt path ---------
    prompts = selected.to(torch.bfloat16)
    M_dyn = fixed_prompt_recall_matrix(
        model, tokenizer, probes, prompts, max_length=args.max_length, device=device, chunk=args.chunk
    )
    zero_prompts = torch.zeros(len(probes), config.n_heads, config.d_model, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        no_mem_dyn = forward_ntp(model, zero_prompts, tid, tam, return_per_sample=True).float().cpu()
    dyn = {"path": "dynamic_soft_prompt_s0", **summarize(M_dyn, no_mem_dyn)}

    # Same fitted module + settings, but the #193 oracle query (probe_q): the
    # matched control for the S0-vs-oracle selection gap at this fit budget.
    M_oracle = dynamic_recall_matrix(
        val_module, model, tokenizer, probes, probe_q, max_length=args.max_length, device=device, chunk=args.chunk
    )
    oracle = {"path": "dynamic_soft_prompt_oracle_probe_q", **summarize(M_oracle, no_mem_dyn)}

    # --- literal T0 deployment form: reserved rows (tied-head confound) -------
    embed = model.get_input_embeddings().weight.data
    for k in range(len(eligible)):
        for h in range(config.n_heads):
            embed[PACK_BASE + k * config.n_heads + h] = selected[k, h].to(embed.dtype)
    pack = {"K": len(eligible), "n_heads": config.n_heads, "probe_targets": [r["target_text"] for r in probes]}
    M_res = recall_matrix(model, tokenizer, pack, max_length=args.max_length, device=device, chunk=args.chunk)
    no_mem_res = no_memory_loss(model, tokenizer, pack, max_length=args.max_length, device=device)
    res = {"path": "reserved_rows_s0", **summarize(M_res, no_mem_res)}

    result = {
        "issue": 211,
        "arm": "bf16",
        "n_personas": len(eligible),
        "persona_offset": args.persona_offset,
        "fit_steps": args.fit_steps,
        "max_length": args.max_length,
        "fit_max_length": args.fit_max_length,
        "selection": "S0: per-persona mean of training-query encodings -> attention argmax per head",
        "s0_top": [[int(top[k, h]) for h in range(config.n_heads)] for k in range(len(eligible))],
        "attn_max_per_head": [[round(float(attn[k, h].max()), 4) for h in range(config.n_heads)] for k in range(len(eligible))],
        **dyn,
        "oracle_probe_q_dynamic": oracle,
        "reserved_rows": res,
    }
    out = REPO_ROOT / "results" / f"D203_S0_BASELINE_offset{args.persona_offset}{args.out_tag}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


def stage_s2a(args: argparse.Namespace) -> None:
    """#212 Gate 2: S2a exact-latent selector -- off-device reference.

    The literal on-device-computable selector math: encode the live query with
    the 6-layer encoder (``Qwen3EarlyLayerEncoder``), run w_q/w_k per-head
    attention, take a HARD argmax slot per head, and gather that already-baked
    vector (``oracle_pack``). #211 scored only the SOFT attention mixture via
    ``dynamic_recall_matrix``; this scores the HARD gather via
    ``fixed_prompt_recall_matrix`` (same ``forward_ntp`` code) and also reports
    the matched soft-oracle control, so hard-vs-soft is an explicit comparison.
    """
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
    val_module, eligible, probes, probe_q, _, _ = fit_val_slots(
        ck, encoder, model, tokenizer, load_corpus(args.val_corpus), config,
        n_personas=args.n_personas, persona_offset=args.persona_offset, fit_steps=args.fit_steps, lr=args.fit_lr,
        max_length=args.max_length, fit_max_length=args.fit_max_length, device=device, seed=args.seed,
    )
    print(f"fitted {len(eligible)} val personas in {time.time()-t0:.1f}s")

    pack = oracle_pack(val_module, eligible, probes, probe_q, device)  # HARD argmax from live probe_q
    model.eval()

    enc = tokenizer([r["target_text"] for r in probes], return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
    tid = enc["input_ids"].to(device)
    tam = enc["attention_mask"].to(device)
    zero_prompts = torch.zeros(len(probes), config.n_heads, config.d_model, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        no_mem = forward_ntp(model, zero_prompts, tid, tam, return_per_sample=True).float().cpu()

    M_hard = fixed_prompt_recall_matrix(
        model, tokenizer, probes, pack["vectors"].to(device=device, dtype=torch.bfloat16),
        max_length=args.max_length, device=device, chunk=args.chunk,
    )
    hard = {"path": "dynamic_fixed_prompt_s2a_hard_argmax", **summarize(M_hard, no_mem)}

    M_soft = dynamic_recall_matrix(
        val_module, model, tokenizer, probes, probe_q, max_length=args.max_length, device=device, chunk=args.chunk
    )
    soft = {"path": "dynamic_soft_prompt_oracle_probe_q", **summarize(M_soft, no_mem)}

    result = {
        "issue": 212,
        "arm": "bf16",
        "n_personas": len(eligible),
        "persona_offset": args.persona_offset,
        "fit_steps": args.fit_steps,
        "max_length": args.max_length,
        "fit_max_length": args.fit_max_length,
        "selection": "S2a: live probe_q -> attention argmax per head -> gather bake[k,h,top]",
        "s2a_top": [[int(pack["top"][k, h]) for h in range(config.n_heads)] for k in range(len(eligible))],
        "hard_selected_dynamic": hard,
        "soft_oracle_dynamic": soft,
    }
    out = REPO_ROOT / "results" / f"D203_S2A_REFERENCE_offset{args.persona_offset}{args.out_tag}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--model", default=DEFAULT_MODEL)
    p_prep.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p_prep.add_argument("--val-corpus", default=DEFAULT_VAL)
    p_prep.add_argument("--encoder-layers", type=int, default=6)
    p_prep.add_argument("--n-personas", type=int, default=24)
    p_prep.add_argument("--persona-offset", type=int, default=0)
    p_prep.add_argument("--fit-steps", type=int, default=40)
    p_prep.add_argument("--fit-lr", type=float, default=1e-3)
    p_prep.add_argument("--max-length", type=int, default=192)
    p_prep.add_argument("--fit-max-length", type=int, default=128)
    p_prep.add_argument("--seed", type=int, default=0)
    p_prep.add_argument("--device", default="auto")
    p_prep.set_defaults(func=stage_prepare)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--arm", choices=("bf16", "cq4"), required=True)
    p_eval.add_argument("--model", default=DEFAULT_MODEL)
    p_eval.add_argument("--weights-dir", default="outputs/r179-m1/p193-weights")
    p_eval.add_argument("--cq4-model", default="outputs/r179-m1/p193-qdq")
    p_eval.add_argument("--max-length", type=int, default=192)
    p_eval.add_argument("--chunk", type=int, default=8)
    p_eval.add_argument("--device", default="auto")
    p_eval.set_defaults(func=stage_eval)

    p_dyn = sub.add_parser("dynamic")
    p_dyn.add_argument("--arm", choices=("bf16", "cq4"), required=True)
    p_dyn.add_argument("--model", default=DEFAULT_MODEL)
    p_dyn.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p_dyn.add_argument("--val-corpus", default=DEFAULT_VAL)
    p_dyn.add_argument("--weights-dir", default="outputs/r179-m1/p193-weights")
    p_dyn.add_argument("--cq4-model", default="outputs/r179-m1/p193-qdq")
    p_dyn.add_argument("--encoder-layers", type=int, default=6)
    p_dyn.add_argument("--n-personas", type=int, default=24)
    p_dyn.add_argument("--persona-offset", type=int, default=0)
    p_dyn.add_argument("--fit-steps", type=int, default=40)
    p_dyn.add_argument("--fit-lr", type=float, default=1e-3)
    p_dyn.add_argument("--max-length", type=int, default=192)
    p_dyn.add_argument("--fit-max-length", type=int, default=128)
    p_dyn.add_argument("--chunk", type=int, default=8)
    p_dyn.add_argument("--seed", type=int, default=0)
    p_dyn.add_argument("--device", default="auto")
    p_dyn.set_defaults(func=stage_dynamic)

    p_s0 = sub.add_parser("s0")
    p_s0.add_argument("--model", default=DEFAULT_MODEL)
    p_s0.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p_s0.add_argument("--val-corpus", default=DEFAULT_VAL)
    p_s0.add_argument("--encoder-layers", type=int, default=6)
    p_s0.add_argument("--n-personas", type=int, default=12)
    p_s0.add_argument("--persona-offset", type=int, default=0)
    p_s0.add_argument("--fit-steps", type=int, default=40)
    p_s0.add_argument("--fit-lr", type=float, default=1e-3)
    p_s0.add_argument("--max-length", type=int, default=192)
    p_s0.add_argument("--fit-max-length", type=int, default=128)
    p_s0.add_argument("--chunk", type=int, default=8)
    p_s0.add_argument("--seed", type=int, default=0)
    p_s0.add_argument("--device", default="auto")
    p_s0.add_argument("--out-tag", default="")
    p_s0.set_defaults(func=stage_s0)

    p_s2a = sub.add_parser("s2a")
    p_s2a.add_argument("--model", default=DEFAULT_MODEL)
    p_s2a.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p_s2a.add_argument("--val-corpus", default=DEFAULT_VAL)
    p_s2a.add_argument("--encoder-layers", type=int, default=6)
    p_s2a.add_argument("--n-personas", type=int, default=12)
    p_s2a.add_argument("--persona-offset", type=int, default=0)
    p_s2a.add_argument("--fit-steps", type=int, default=40)
    p_s2a.add_argument("--fit-lr", type=float, default=1e-3)
    p_s2a.add_argument("--max-length", type=int, default=192)
    p_s2a.add_argument("--fit-max-length", type=int, default=128)
    p_s2a.add_argument("--chunk", type=int, default=8)
    p_s2a.add_argument("--seed", type=int, default=0)
    p_s2a.add_argument("--device", default="auto")
    p_s2a.add_argument("--out-tag", default="")
    p_s2a.set_defaults(func=stage_s2a)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
