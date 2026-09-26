# #236 Phase 0 RETRY — Qwen3.5-2B re-verified against upstream Cactus v2.2.1

Date: 2026-09-26. Parent #236 (epic: distill + benchmark Qwen3.5-2B). This is the
retry the issue called for after the original Phase 0 NO-GO was reopened as
PENDING RETRY: **re-run the same four Phase 0 questions against the current
upstream engine instead of the project's stale v2.0/v2.0.1 pins.** No training,
no LoRA config, no data prep, no benchmark, no on-device test.

**Verdict: NO-GO for Phase 1 (unchanged), now confirmed on the current engine.**
The training half is real and was verified live (Q1 = YES). The blocker is, and
remains, that Cactus cannot produce a **text-only runtime bundle** for Qwen3.5-2B
— but the *reason* is now precise and not an artifact of a stale checkout: the
upstream graph builder is mid-rewrite (upstream's own docs say so), and no
`qwen3_5` profile with the required export patches exists. Both previously
observed failures reproduce **byte-for-byte in wording** on v2.2.1.

Reused prior context (not redone): `localllm-qwen3.5/` workspace, fresh
`cactus-sdk` clone, and `phase0-reference/`. Built on: the earlier addendum
(`qwen3-finetune/results/QWEN3_5_2B_ARCH_VERIFICATION_ENGINE_UPDATE.md`), which
was an in-place engine build + new-transpiler probe; this retry additionally
**rebuilds the container from the fresh checkout** and re-runs the **legacy**
Phase 0 graph-builder paths.

## 0. Engine / container verified

| item | value |
|---|---|
| upstream repo | `github.com/cactus-compute/cactus`, `main` `git pull` (already current) |
| version / commit | **`CACTUS_VERSION=v2.2.1`**, `fa094ce86c99117f351b100a6ebe51021c11499b` |
| container | **`cactus-arm64-convert:v221`** rebuilt from `localllm-qwen3.5/cactus-sdk` |
| image id | `sha256:29ed51100c94947ac597c1cc1180c06f0e9442a76f4354f26c8774381442ff56` (arm64) |
| build | `Dockerfile.arm64-convert`, `cactus build --python` exit 0; `libcactus_engine.so` 4.5 MB |
| python / torch / transformers | 3.12.14 / 2.8.0+cpu / **5.5.4** (same pins as the old image) |
| GDN support present | `cactus-kernels/src/attention_hybrid.cpp`, `transpile/fusion/deltanet.py` (`match_gated_deltanet`), transpiler `gated_deltanet_*` fusions/lowerings |

The old image (`cactus-arm64-convert:built`, created 2026-09-10) was built from
the old SDK pin; the retry image bakes in the v2.2.1 checkout so container code
and engine are unambiguously current.

## 1. Q1 — training path: YES (verified live, stronger than Phase 0's "qualified yes")

Phase 0 said Unsloth supports Qwen3.5 but `qwen3-train` had `transformers 4.57.6`
and Unsloth raised `"minimum required version is 5.2.0"`. Re-checked in an
**isolated venv** (`--system-site-packages` over `qwen3-train`, so the real env was
never modified):

- **`transformers==5.5.0` and `5.5.4` both make `AutoConfig("Qwen/Qwen3.5-2B")`
  succeed** (`model_type=qwen3_5`, arch `Qwen3_5ForConditionalGeneration`,
  text_config `qwen3_5_text`, 24 layers = 18 linear_attention + 6 full_attention).
  Because `AutoConfig` no longer errors, Unsloth's `autoconfig_error`-based
  "does not support Qwen3.5" gate (loader.py:517/1190) is **not entered**.
- **Regression:** `unsloth/Qwen3-1.7B` and `unsloth/Qwen3-0.6B` `AutoConfig` +
  tokenizer still load fine — the existing #232 pipeline is unaffected at the
  config/tokenizer level.
- **Live load (the decisive check), `transformers==5.5.0`:**
  `unsloth.FastLanguageModel.from_pretrained("Qwen/Qwen3.5-2B", load_in_4bit=False)`
  → `LOADED Qwen3_5ForConditionalGeneration` on the RTX 4070 Ti
  (`"Switching to 16bit LoRA"`), forward pass OK `logits (1, 18, 248320)` bf16,
  then `get_peft_model(...)` attached **5,455,872 trainable params** with the
  standard `q/k/v/o/gate/up/down` target modules.

**Important version caveat:** Unsloth 2026.6.9's own metadata requires
`transformers>=4.51.3,<=5.5.0`. The conversion container's **5.5.4 is outside
Unsloth's declared range** (it satisfies only the loader's runtime `>=5.2.0`
check). The safe upgrade target for `qwen3-train` is therefore **`transformers==5.5.0`**,
not 5.5.4.

Other caveats carried forward: GDN fast path needs `flash-linear-attention` +
`causal-conv1d` (else slow torch fallback); Unsloth warns that explicit
`target_modules` are filtered by its vision/language/attention/mlp split, so
Phase 1 must re-derive Qwen3.5 module names (Q1 ≠ "recipe unchanged").

Recorded: conversion container is a *separate* concern from the training env —
the container needs transformers only to run the transpiler/torch.export; it does
not need Unsloth.

## 2. Q2 — raw HF + PEFT + bitsandbytes QLoRA

Unchanged and now moot: a raw `transformers>=5.2` + `peft` + `bnb` QLoRA path is
nominally possible for the `Qwen3_5ForCausalLM` text model, but Unsloth explicitly
advises against QLoRA for Qwen3.5, GDN×bnb 4-bit compatibility is unverified, and
it forfeits Unsloth's memory/speed gains. bf16 LoRA (≈5 GB, verified loadable) is
the path.

## 3. Q3 — Cactus conversion (the decisive one): text-only bundle still impossible

All commands in `cactus-arm64-convert:v221`, v2.2.1, plain `Qwen/Qwen3.5-2B`,
text-only (no image). Full logs: `retry-evidence/logs/_driver.log`.

### 3a. Weights-only — **SUCCEEDED** (matches Phase 0)
```
python3 -m cactus convert Qwen/Qwen3.5-2B <out> --bits 4 --weights-only --skip-model-load
  converted 285 | fallback 332 | ignored 0 | unrecognized 15 | total 1,183,691,808 B
  components: embedding 2, language 333, vision 297   (632 .weights files)
```
Verified from `weights_manifest.json`: the **15 unrecognized entries are all
`mtp.*`** (multi-token-prediction head, `status:"unrecognized"`). Core weights
convert cleanly, exactly as in Phase 0. (A GPTQ-Hessian-calibration warning is
printed but is non-fatal.)

### 3b. Default `cactus convert` path (new "Crassus v1.0.0" transpiler) — **FAILED**
```
Resolves Qwen3.5 to generic contract 'generic_text' (no qwen3_5 profile)
Conversion error: Could not guard on data-dependent expression Eq(u0, 1)
  (unhinted: Eq(u0, 1)). (Size-like symbols: none)
  at transformers/models/qwen3_5/modeling_qwen3_5.py:1301
     _update_linear_attn_mask: attention_mask is not None and torch.all(attention_mask == 1)
```
`torch.export` cannot trace/guard the Python conditional; no `components/` bundle
is produced. Root cause: **no `qwen3_5` profile** in
`transpiler/ModelProfiles/profiles.py` (Qwen3.5 falls through to
`GENERIC_TEXT_PROFILE`, `export_patches=()`), whereas other families carry export
patches (`gemma4_audio_mask`, `lfm2_vl_image_features`, …).

### 3c. Legacy `cactus.transpile.hf_model`, default invocation — **FAILED**
```
RuntimeError: Qwen3.5 multimodal transpile requires at least one --image-file
  at hf_model.py:2316 (_prepare_qwen3_5_multimodal_inputs)
```
Identical to Phase 0 (line 2298 there). `QWEN_PROFILE` still sets
`default_task="multimodal_causal_lm_logits"`, `needs_image=True`, default
components include `vision_encoder`, and `text_component_plans` maps only
`"qwen3forcausallm"` (`Qwen3ForCausalLM`), not Qwen3.5's
`Qwen3_5ForConditionalGeneration`. **The legacy plain-text 6-component path #231
used still does not exist for Qwen3.5, on v2.2.1.**

### 3d. Legacy `hf_model`, forced text-only task — **FAILED (same error)**
```
--task causal_lm_logits --components decoder,decoder_step
adapter_family=qwen3_5, adapter_module=component_pipeline, input_ids=[1,49]
cactus.transpile.capture_pytorch.CapturePhaseError: [capture:export] torch.export failed
  for model=_ComponentCaptureWrapper strict=True: Dynamo failed to run FX node with fake
  tensors: call_function <built-in function matmul>(...): got RuntimeError('Unhandled
  FakeTensor Device Propagation for aten.bmm.default, found two different devices cpu, meta')
  model_adapters.py:4913 forward -> :4941 debug_forward ->
    position_embeddings = self.backbone.rotary_emb(hidden_states, multimodal_position_ids) ->
    transformers/models/qwen3_5/modeling_qwen3_5.py:149
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
```
Exactly Phase 0's RoPE `bmm` `cpu`/`meta` mismatch.

### Why this is not "stale checkout" noise

A source-level diff of the pinned fork (`cactus-sdk-v210-suppress`, v2.0.1,
a82155c) vs v2.2.1 shows the Qwen3.5-relevant code is **unchanged**:
`transpile/model_profiles.py`, `capture_pytorch.py`, `fusion/deltanet.py`,
`fusion/rope.py`, and the Qwen3.5 adapter path are byte-identical; the only
non-audio changes are generic-prefill GQA attention fusion + conv mask handling,
and the only non-Parakeet/audio diff is the fork's own #212 `CACTUS_QWEN_EMBED_LAYERS`
patch being absent upstream. The old image *did* already contain the GDN fusion
pass. The Phase 0 failures were a real, current-engine limitation — not an old
transpiler that simply lacked Qwen3.5 support. Upstream's `list.txt` "Qwen 3/3.5"
family listing and `gated DeltaNet prefill/decode` fusion entry describe the
*kernel/transpiler* feature set, which is exactly what converts weights but does
not yet capture a runnable text graph.

Corroborating upstream documentation (v2.2.1):
- `docs/cactus_engine.md:32`, `docs/cactus_quants.md:187`, `docs/finetuning.md:71`:
  *"local runtime bundle generation is unavailable while the graph builder is
  being rewritten"* — deploy a prebuilt Cactus-Compute bundle or wait.
- `docs/cactus_transpiler.md:478`: supported family table lists `Qwen 3/3.5 | text,
  vision | component split: no`.

## 4. Step-3 bundle smoke test

**Not applicable — no text-only bundle is produced by any path.** There is no
`components/manifest.json`, so `cactus_init` / unarmed generation /
`inputs_embeds`-prefix arming cannot be run. (The Phase 0 §3a converted CQ weights
are not themselves a runnable bundle.)

## 5. Q4 — genuine text-only HF mode: YES (unchanged)

In `cactus-arm64-convert:v221` (`transformers 5.5.4`), the Phase 0 probe runs
clean:
```
transformers 5.5.4
LOADED via Qwen3_5ForCausalLM in 3s -> Qwen3_5ForCausalLM
GENERATED 8 tokens in 53s
OUT: 'The best fertilizer for tomatoes depends on the'
TEXT_ONLY_LOAD: OK via Qwen3_5ForCausalLM
```
Caveat unchanged: GDN fast path needs `flash-linear-attention` + `causal-conv1d`.

## 6. GO / NO-GO for Phase 1

**NO-GO — and a harder NO-GO than the original, because it now holds against the
current upstream engine, not a stale pin.**

The bar is *both* a working training path *and* a working Cactus conversion path
to a plain-text bundle:
- **Training: YES.** Verified live: transformers 5.5.0 + Unsloth 2026.6.9 loads
  Qwen3.5-2B bf16, forward passes, and attaches LoRA (5.46M params).
- **Cactus text-only bundle: NO.** All three runnable paths fail on v2.2.1:
  default Crassus path (data-dependent guard in `_update_linear_attn_mask`),
  legacy default (`--image-file` required), legacy forced-text (`torch.export`
  RoPE `bmm` device mismatch). The public local graph builder is mid-rewrite and
  has no `qwen3_5` export-patch profile.

Qwen3.5-2B is **not** architecturally ruled out (kernels, fusions, family table,
and prebuilt Cactus-Compute bundles all exist), but this project cannot yet build
its own distilled, plain-text, memory-prefix-compatible bundle. Phase 1 does not
start; do not build around the vision path (explicit non-goal).

## 7. Fallbacks (unchanged)

Plain-transformer Qwen3 family, already proven end-to-end through this toolchain
(0.6B champion, 1.7B #232 port), no Gated DeltaNet, existing legacy 6-component
text path and engine `inputs_embeds` support:
- **Qwen3-4B** — intermediate step.
- **Qwen3-8B** — #235 research target.

## 8. Recommendation / tracks

1. **Direct unblock:** watch upstream for the replacement graph builder + a
   `qwen3_5` model profile (with an `_update_linear_attn_mask` export patch),
   then re-run §3. The training half is ready now.
2. **If an earlier deliverable is needed:** use the Qwen3-4B/8B fallbacks.
3. If pursuing Qwen3.5, upgrade `qwen3-train` to `transformers==5.5.0` (in
   Unsloth's range) and add `flash-linear-attention` + `causal-conv1d`.

## 9. Reproduce / evidence

```bash
NEW=[LOCAL_PATH]
# 1) rebuild container from the fresh checkout
cd $NEW && DOCKER_BUILDKIT=1 docker build --platform linux/arm64 \
  -t cactus-arm64-convert:v221 -f Dockerfile.arm64-convert .
# 2) all four Q3 tests + Q4 probe (A weights / B default / C legacy / D legacy-text / E probe)
bash $NEW/retry-evidence/run_retry_tests.sh
# (driver output: $NEW/retry-evidence/logs/_driver.log)
# Q1 isolated check
$NEW/retry-evidence/q1-isolated/venv/bin/python $NEW/retry-evidence/q1-isolated/check_q1.py
$NEW/retry-evidence/q1-isolated/venv/bin/python $NEW/retry-evidence/q1-isolated/load_qwen35_unsloth.py
```

Artifacts: this note; `QWEN3_5_2B_ARCH_VERIFICATION_RETRY.json`; raw logs in
`localllm-qwen3.5/retry-evidence/logs/_driver.log`; Q1 logs in
`localllm-qwen3.5/retry-evidence/q1-isolated/`. Converted CQ weights and failed
graph dirs are under `localllm-qwen3.5/retry-evidence/weights/` (not committed).
