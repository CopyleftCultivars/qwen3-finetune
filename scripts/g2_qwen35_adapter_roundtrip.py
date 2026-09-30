#!/usr/bin/env python3
"""#238 G2: prove the tiny Qwen3.5 adaptation survives save/reload/merge.

Each stage runs in its own process (fresh interpreter, no PEFT state carried
over), so "reload" means a real reload rather than an in-memory continuation:

  attach  load base bf16, attach the saved adapter with PEFT, run the probes
  merge   merge_and_unload, save merged model + tokenizer + config, verify
          tensor names/shapes and tokenizer/config identity vs the base repo
  verify  fresh process: load the merged directory with plain transformers
          (no PEFT) and run the same probes
  compare attach vs verify must be identical (round-trip fidelity); base vs
          attach must measurably differ (adaptation-sensitive probes)

Run inside the #236 retry isolated venv:
  localllm-qwen3.5/retry-evidence/q1-isolated/venv/bin/python \
      scripts/g2_qwen35_adapter_roundtrip.py --stage attach ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

PROBE_PROMPTS = [
    {"name": "train_seen",
     "text": "What is the mineral content of Anacardium occidentale "
             "(Anacardium occidentale)?"},
    {"name": "nf_unseen",
     "text": "How do I prepare a fermented plant juice fertilizer at home?"},
    {"name": "generic_tomato",
     "text": "The best fertilizer for tomatoes depends on the"},
    {"name": "neutral_soil",
     "text": "Write a short paragraph about soil structure."},
]
BASE_MODEL_DEFAULT = "Qwen/Qwen3.5-2B"
BASE_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                        "chat_template.jinja", "special_tokens_map.json",
                        "config.json", "generation_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["attach", "merge", "verify", "compare"])
    parser.add_argument("--model", default=BASE_MODEL_DEFAULT)
    parser.add_argument("--adapter", default="outputs/lora-adapter-g2-qwen35-tiny")
    parser.add_argument("--merged", default="outputs/merged-model-g2-qwen35-tiny")
    parser.add_argument("--work-dir", default="outputs/g2_roundtrip")
    return parser.parse_args()


def text_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return getattr(tokenizer, "tokenizer", tokenizer)


def load_model(model_id: str, dtype=torch.bfloat16) -> torch.nn.Module:
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
    model.eval().cuda()
    return model


def probe_batch(tokenizer, prompt: str) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt")
    return {k: v.cuda() for k, v in encoded.items()}


@torch.no_grad()
def run_probes(model, tokenizer, prompts: list[dict[str, str]]) -> dict[str, Any]:
    generations: dict[str, str] = {}
    logits: dict[str, np.ndarray] = {}
    for probe in prompts:
        batch = probe_batch(tokenizer, probe["text"])
        outputs = model(**batch)
        logits[probe["name"]] = outputs.logits[0, -1].float().cpu().numpy()
        generated = model.generate(**batch, max_new_tokens=32, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(generated[0][batch["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        generations[probe["name"]] = text
    return {"generations": generations, "logits": logits}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digests(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    """Byte-level sha256 + shape per parameter (bf16 viewed as raw bytes)."""
    digests: dict[str, dict[str, Any]] = {}
    for name, param in model.named_parameters():
        raw = param.detach().cpu().contiguous().view(torch.uint8).numpy()
        digests[name] = {"shape": list(param.shape),
                         "sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
    return digests


def write_stage(name: str, payload: dict[str, Any], work_dir: Path,
                logits: dict[str, np.ndarray] | None = None) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = {"stage": name, **payload}
    (work_dir / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if logits is not None:
        np.savez(work_dir / f"{name}_logits.npz", **logits)
    print(json.dumps({k: v for k, v in payload.items() if k != "generations"},
                     indent=2, sort_keys=True))


def stage_attach(args: argparse.Namespace) -> int:
    from peft import PeftModel

    started = time.time()
    tokenizer = text_tokenizer(args.model)
    base = load_model(args.model)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n}
    lora_norm = float(sum(p.detach().float().norm() ** 2 for p in lora_params.values())
                      ** 0.5)
    probes = run_probes(model, tokenizer, PROBE_PROMPTS)
    write_stage("attach", {
        "adapter": args.adapter,
        "adapter_loaded_lora_tensors": len(lora_params),
        "adapter_lora_l2_norm": round(lora_norm, 6),
        "adapter_nonzero": lora_norm > 0,
        "generations": probes["generations"],
        "wall_s": round(time.time() - started, 2),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }, Path(args.work_dir), probes["logits"])
    return 0


def stage_merge(args: argparse.Namespace) -> int:
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = load_model(args.model)
    base_digests = tensor_digests(base)
    model = PeftModel.from_pretrained(base, args.adapter)
    reference = capture_lora_reference(model)
    merged = model.merge_and_unload()
    merge_check = verify_merged_weights(merged, reference)
    del reference
    merged_digests = tensor_digests(merged)

    out_dir = Path(args.merged)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    canonicalization: dict[str, Any] = {"canonicalized": False,
                                        "reason": "not attempted"}
    try:
        from huggingface_hub import hf_hub_download

        from g2_hf_weight_audit import canonicalize_checkpoint

        base_index = Path(hf_hub_download(args.model,
                                          "model.safetensors.index.json"))
        canonicalization = canonicalize_checkpoint(out_dir, base_index.parent)
    except Exception as exc:  # keep the in-memory evidence; audit will show it
        canonicalization = {"canonicalized": False,
                            "reason": f"{type(exc).__name__}: {exc}"}

    same_keys = set(base_digests) == set(merged_digests)
    same_shapes = all(
        base_digests[name]["shape"] == merged_digests[name]["shape"]
        for name in base_digests if name in merged_digests)
    changed = sorted(name for name in base_digests
                     if base_digests[name]["sha256"]
                     != merged_digests.get(name, {}).get("sha256"))
    unchanged = len(base_digests) - len(changed)

    merged_tokenizer = text_tokenizer(args.merged)
    probes = run_probes(merged, merged_tokenizer, PROBE_PROMPTS)
    token_match = tokenizer_match_report(args.model, out_dir)
    write_stage("merge", {
        "merged_dir": str(out_dir),
        "merged_files": sorted(p.name for p in out_dir.iterdir()),
        "canonicalization": canonicalization,
        "param_names_identical_to_base": same_keys,
        "param_shapes_identical_to_base": same_shapes,
        "n_params": len(base_digests),
        "changed_param_count": len(changed),
        "unchanged_param_count": unchanged,
        "changed_params": changed,
        "merge_arithmetic_check": merge_check,
        "tokenizer_config_match": token_match,
        "generations": probes["generations"],
    }, Path(args.work_dir), probes["logits"])
    return 0


def capture_lora_reference(peft_model: torch.nn.Module
                           ) -> dict[str, dict[str, torch.Tensor]]:
    """Base weight + fp32 LoRA delta for every adapted module, on CPU."""
    reference: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in peft_model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if not (hasattr(lora_a, "keys") and "default" in lora_a):
            continue
        base_layer = getattr(module, "base_layer", None)
        if base_layer is None:
            continue
        scale = float(getattr(module, "scaling", {}).get("default", 1.0))
        a_mat = lora_a["default"].weight.detach().float().cpu()      # (r, in)
        b_mat = module.lora_B["default"].weight.detach().float().cpu()  # (out, r)
        plain_name = name.removeprefix("base_model.model.")
        reference[plain_name] = {
            "weight": base_layer.weight.detach().float().cpu(),
            "delta": b_mat.matmul(a_mat) * scale,
        }
    return reference


def verify_merged_weights(merged: torch.nn.Module,
                          reference: dict[str, dict[str, torch.Tensor]]
                          ) -> dict[str, Any]:
    """Merged weight must equal base + scaling * B@A up to bf16 rounding."""
    merged_by_name = dict(merged.named_modules())
    worst = {"tensor": "", "abs": 0.0, "rel": 0.0}
    checked = 0
    for name, expected in reference.items():
        weight = merged_by_name[name].weight.detach().float().cpu()
        target = expected["weight"] + expected["delta"]
        abs_err = (weight - target).abs()
        scale = target.abs().clamp_min(1e-6)
        rel = float((abs_err / scale).max())
        checked += 1
        if rel > worst["rel"]:
            worst = {"tensor": name, "abs": float(abs_err.max()), "rel": rel}
    bf16_eps = 2 ** -7
    return {
        "tensors_checked": checked,
        "expected_max_rel_err_bf16": bf16_eps,
        "worst_tensor": worst["tensor"],
        "worst_max_abs_err": worst["abs"],
        "worst_max_rel_err": worst["rel"],
        "within_bf16_rounding": worst["rel"] <= bf16_eps,
    }


def tokenizer_match_report(model_id: str, merged_dir: Path) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    report: dict[str, Any] = {}
    for filename in BASE_TOKENIZER_FILES:
        entry: dict[str, Any] = {}
        try:
            base_path = Path(hf_hub_download(model_id, filename))
            entry["base_sha256"] = sha256_file(base_path)
        except Exception as exc:  # file not in the base repo
            report[filename] = {"base": f"absent ({type(exc).__name__})"}
            continue
        merged_path = merged_dir / filename
        if not merged_path.is_file():
            report[filename] = {"merged": "absent"}
            continue
        entry["merged_sha256"] = sha256_file(merged_path)
        entry["byte_identical"] = entry["base_sha256"] == entry["merged_sha256"]
        if filename.endswith(".json"):
            base_json = json.loads(Path(
                hf_hub_download(model_id, filename)).read_text(encoding="utf-8"))
            merged_json = json.loads(merged_path.read_text(encoding="utf-8"))
            entry["json_identical"] = base_json == merged_json
        report[filename] = entry
    return report


def stage_verify(args: argparse.Namespace) -> int:
    started = time.time()
    tokenizer = text_tokenizer(args.merged)
    model = load_model(args.merged)
    probes = run_probes(model, tokenizer, PROBE_PROMPTS)
    write_stage("verify", {
        "merged_dir": args.merged,
        "loaded_with_peft": False,
        "generations": probes["generations"],
        "wall_s": round(time.time() - started, 2),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }, Path(args.work_dir), probes["logits"])
    return 0


def load_stage(name: str, work_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    payload = json.loads((work_dir / f"{name}.json").read_text(encoding="utf-8"))
    logits_path = work_dir / f"{name}_logits.npz"
    if not logits_path.is_file():
        return payload, {}
    with np.load(logits_path) as data:
        logits = {key: data[key] for key in data.files}
    return payload, logits


def logit_diff(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.abs(left - right)
    p = _softmax(left)
    q = _softmax(right)
    kl = float(np.sum(p * np.log(np.maximum(p, 1e-30) /
                                 np.maximum(q, 1e-30))))
    return {
        "max_abs_logit_delta": float(delta.max()),
        "mean_abs_logit_delta": float(delta.mean()),
        "kl_base_vs_other": kl,
        "top1_agrees": bool(int(left.argmax()) == int(right.argmax())),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def tokenizer_equivalence(model_id: str, merged_dir: str,
                          prompts: list[dict[str, str]]) -> dict[str, Any]:
    """Base tokenizer vs merged-reload tokenizer: same ids on probe text,
    chat-rendered prompt, and unicode/punctuation edge cases."""
    base_tok = text_tokenizer(model_id)
    merged_tok = text_tokenizer(merged_dir)
    cases = {probe["text"]: "raw:" + probe["name"] for probe in prompts}
    for probe in prompts:
        rendered = base_tok.apply_chat_template(
            [{"role": "user", "content": probe["text"]}],
            tokenize=False, add_generation_prompt=True)
        cases[rendered] = "chat:" + probe["name"]
    cases["emoji/CJK: 🌱 山田 soil pH 6.5–7.0"] = "unicode"
    cases["tab\tnewline\nquote's \"mixed\" <> &"] = "punctuation"

    mismatches = []
    for text, label in cases.items():
        base_ids = base_tok(text, add_special_tokens=False)["input_ids"]
        merged_ids = merged_tok(text, add_special_tokens=False)["input_ids"]
        if base_ids != merged_ids:
            mismatches.append({"case": label, "base_len": len(base_ids),
                               "merged_len": len(merged_ids),
                               "base_head": base_ids[:8],
                               "merged_head": merged_ids[:8]})
    rendered_equal = True
    for probe in prompts:
        messages = [{"role": "user", "content": probe["text"]}]
        base_render = base_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        merged_render = merged_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        rendered_equal &= base_render == merged_render
    return {"cases": len(cases), "all_ids_equal": not mismatches,
            "chat_render_strings_equal": rendered_equal,
            "mismatches": mismatches,
            "base_class": type(base_tok).__name__,
            "merged_class": type(merged_tok).__name__}


def config_equivalence_report(model_id: str, merged_dir: str) -> dict[str, Any]:
    """Config match at two levels: raw files, and what the runtime loads.

    The raw files may differ because transformers re-serializes (materialized
    defaults such as ``partial_rotary_factor``, ``dtype``, version stamp). The
    gating check is the LOADED config: if AutoConfig sees the same values on
    both sides, the merged artifact is configuration-identical for inference.
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    base_cfg = json.loads(Path(hf_hub_download(model_id, "config.json")
                               ).read_text(encoding="utf-8"))
    merged_cfg = json.loads((Path(merged_dir) / "config.json"
                             ).read_text(encoding="utf-8"))
    file_diffs: list[str] = []

    def walk(prefix: str, left: Any, right: Any, sink: list[str]) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                walk(f"{prefix}.{key}", left.get(key), right.get(key), sink)
        elif left != right:
            sink.append(prefix)

    walk("config", base_cfg, merged_cfg, file_diffs)
    loaded_base = AutoConfig.from_pretrained(model_id).to_dict()
    loaded_merged = AutoConfig.from_pretrained(merged_dir).to_dict()
    loaded_diffs: list[str] = []
    walk("", loaded_base, loaded_merged, loaded_diffs)
    volatile_basenames = {"dtype", "torch_dtype", "transformers_version",
                          "_name_or_path"}
    loaded_model_diffs = sorted(
        path for path in loaded_diffs
        if path.removeprefix(".").rsplit(".", 1)[-1] not in volatile_basenames)
    materialized = sorted(
        {path.removeprefix("config.") for path in file_diffs}
        - {path.removeprefix("config.") for path in loaded_model_diffs})
    return {
        "file_diff_paths": file_diffs,
        "loaded_config_diffs": loaded_diffs,
        "loaded_model_defining_diffs": loaded_model_diffs,
        "file_diffs_not_visible_after_load": materialized,
        "model_defining_fields_equal": not loaded_model_diffs,
        "verdict": "PASS" if not loaded_model_diffs else "FAIL",
    }


def bf16_ulp(*logits: np.ndarray) -> float:
    """Smallest bf16 spacing at the largest magnitude seen (eps = 2**-7)."""
    magnitude = max(float(np.abs(arr).max()) for arr in logits)
    return 2.0 ** (np.floor(np.log2(max(magnitude, 1e-6))) - 7)


def compare_pairs(name: str, left_payload: dict, left_logits: dict[str, np.ndarray],
                  right_payload: dict, right_logits: dict[str, np.ndarray]
                  ) -> dict[str, Any]:
    """Generations + logits for two probe runs of the same artifact."""
    result: dict[str, Any] = {
        "generations_identical":
            left_payload["generations"] == right_payload["generations"],
        "per_prompt": {},
    }
    worst = 0.0
    worst_ulps = 0.0
    for prompt in left_logits:
        diff = logit_diff(left_logits[prompt], right_logits[prompt])
        ulp = bf16_ulp(left_logits[prompt], right_logits[prompt])
        diff["max_abs_delta_in_bf16_ulps"] = diff["max_abs_logit_delta"] / ulp
        worst = max(worst, diff["max_abs_logit_delta"])
        worst_ulps = max(worst_ulps, diff["max_abs_delta_in_bf16_ulps"])
        result["per_prompt"][prompt] = diff
    result["name"] = name
    result["worst_max_abs_logit_delta"] = worst
    result["worst_delta_in_bf16_ulps"] = worst_ulps
    result["all_top1_agree"] = all(p["top1_agrees"]
                                   for p in result["per_prompt"].values())
    return result


def stage_compare(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir)
    attach, attach_logits = load_stage("attach", work_dir)
    merge_payload, merge_logits = load_stage("merge", work_dir)
    verify, verify_logits = load_stage("verify", work_dir)

    # (1) Disk round-trip: the just-merged model (merge stage, in memory) vs a
    # fresh process reading the saved directory with plain transformers.
    reload_cmp = compare_pairs("merge_in_memory_vs_disk_reload",
                               merge_payload, merge_logits, verify, verify_logits)
    reload_cmp["exact_bitwise"] = reload_cmp["worst_max_abs_logit_delta"] == 0.0
    reload_cmp["criterion"] = "bitwise-equal logits and identical greedy text"
    reload_cmp["verdict"] = ("PASS" if reload_cmp["exact_bitwise"]
                             and reload_cmp["generations_identical"]
                             else "FAIL")

    # (2) PEFT attach arithmetic (W x + s*B A x) vs merged weights ((W+s*B A) x):
    # the same parameters summed in a different order, so logit deltas must sit
    # at the bf16 ulp scale and top-1 must not move. 32-token greedy text is
    # REPORTED here rather than gated: a 1-ulp difference can flip a near-tie
    # mid-sequence. The artifact-stability gate is (1), which is bitwise.
    attach_cmp = compare_pairs("peft_attach_vs_merged_weights",
                               attach, attach_logits, merge_payload, merge_logits)
    attach_cmp["greedy_text_identical"] = attach_cmp["generations_identical"]
    attach_cmp["criterion"] = ("top1 agreement on all probes and worst logit "
                               "delta <= 2 bf16 ulps (greedy text reported, "
                               "not gated)")
    attach_cmp["verdict"] = (
        "PASS" if attach_cmp["all_top1_agree"]
        and attach_cmp["worst_delta_in_bf16_ulps"] <= 2.0
        else "FAIL")

    tok_eq = tokenizer_equivalence(args.model, args.merged, PROBE_PROMPTS)
    cfg_eq = config_equivalence_report(args.model, args.merged)

    tokenizer = text_tokenizer(args.model)
    base_model = load_model(args.model)
    base_probes = run_probes(base_model, tokenizer, PROBE_PROMPTS)
    del base_model
    torch.cuda.empty_cache()

    # (3) Adaptation-sensitive probes: base vs adapted must be far larger than
    # the arithmetic noise in (2).
    sensitivity: dict[str, Any] = {"per_prompt": {}}
    any_delta = False
    any_text_change = False
    for name, base_logits in base_probes["logits"].items():
        diff = logit_diff(base_logits, attach_logits[name])
        text_changed = (base_probes["generations"][name]
                        != attach["generations"][name])
        diff["greedy_text_changed_vs_base"] = text_changed
        any_delta |= diff["max_abs_logit_delta"] > 0
        any_delta |= diff["kl_base_vs_other"] > 0
        any_text_change |= text_changed
        sensitivity["per_prompt"][name] = diff
    sensitivity["base_generations"] = base_probes["generations"]
    sensitivity["adapted_generations"] = attach["generations"]
    sensitivity["any_logit_delta"] = any_delta
    sensitivity["any_greedy_text_change"] = any_text_change
    sensitivity["worst_max_abs_logit_delta"] = max(
        p["max_abs_logit_delta"] for p in sensitivity["per_prompt"].values())
    sensitivity["noise_to_signal_ratio"] = round(
        attach_cmp["worst_max_abs_logit_delta"]
        / max(sensitivity["worst_max_abs_logit_delta"], 1e-9), 4)
    sensitivity["verdict"] = (
        "PASS" if any_delta
        and sensitivity["worst_max_abs_logit_delta"]
        > 10 * attach_cmp["worst_max_abs_logit_delta"]
        else "FAIL")

    fidelity_pass = (reload_cmp["verdict"] == "PASS"
                     and attach_cmp["verdict"] == "PASS"
                     and tok_eq["all_ids_equal"]
                     and tok_eq["chat_render_strings_equal"]
                     and cfg_eq["verdict"] == "PASS")
    payload = {
        "stage": "compare",
        "adapter": args.adapter,
        "merged_dir": args.merged,
        "tokenizer_equivalence": tok_eq,
        "config_equivalence": cfg_eq,
        "disk_reload_fidelity": reload_cmp,
        "adapter_arithmetic": attach_cmp,
        "adaptation_sensitivity": sensitivity,
        "merge_summary": {
            "changed_param_count": merge_payload.get("changed_param_count"),
            "param_names_identical_to_base":
                merge_payload.get("param_names_identical_to_base"),
            "param_shapes_identical_to_base":
                merge_payload.get("param_shapes_identical_to_base"),
            "merge_arithmetic_check":
                merge_payload.get("merge_arithmetic_check"),
            "tokenizer_config_match":
                merge_payload.get("tokenizer_config_match"),
        },
        "verdicts": {
            "save_reload_merge_fidelity": "PASS" if fidelity_pass else "FAIL",
            "adaptation_sensitive_probe": sensitivity["verdict"],
        },
    }
    (work_dir / "compare.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["verdicts"], indent=2, sort_keys=True))
    print("disk reload worst delta:", reload_cmp["worst_max_abs_logit_delta"])
    print("attach-vs-merge worst delta (bf16 ulps):",
          round(attach_cmp["worst_delta_in_bf16_ulps"], 3))
    print("base-vs-adapted worst delta:",
          sensitivity["worst_max_abs_logit_delta"])
    return 0 if all(v == "PASS" for v in payload["verdicts"].values()) else 1


STAGES = {
    "attach": stage_attach,
    "merge": stage_merge,
    "verify": stage_verify,
    "compare": stage_compare,
}


def main() -> int:
    args = parse_args()
    return STAGES[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
