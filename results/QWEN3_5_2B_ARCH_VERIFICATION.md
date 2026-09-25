# #236 Phase 0 — does this project's toolchain support Qwen3.5-2B's architecture?

Date: 2026-09-25. Parent #236 (epic: distill + benchmark Qwen3.5-2B). Scope:
**architecture/tooling verification only** — no training, no LoRA config, no data
prep, no benchmark, no on-device test. Mirrors #231 Phase 1's "does the bundle
even load" discipline before any compute.

**Verdict: NO-GO for Phase 1.** Qwen3.5-2B is trainable in principle (Unsloth
supports it, with a transformers-v5 upgrade and bf16 LoRA rather than the
project's QLoRA recipe), but **there is no working Cactus path to a text-only
bundle**: the legacy graph builder refuses to build a text graph without an image,
and the explicit text task fails inside `torch.export`. A model this project can
train but cannot deploy on-device as a plain-text LM is not useful for this epic.
Fallback candidates named in §6.

## 0. What the model actually is (config, direct from HF)

`Qwen/Qwen3.5-2B/config.json`:
- `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"` —
  a **unified vision-language model**, not a plain text LM.
- `text_config.model_type: "qwen3_5_text"`, hidden 2048, 24 layers,
  `layer_types` = **18 × `linear_attention` + 6 × `full_attention`** (every 4th
  layer full), `linear_conv_kernel_dim: 4`, `mamba_ssm_dtype: "float32"`,
  `attn_output_gate: true`, vocab 248320 tied, `mtp_num_hidden_layers: 1`.
- `vision_config`: depth 24, hidden 1024, patch 16.
- HF pipeline tag `image-text-to-text`; `other=qwen3_5`.

So the #232 "base-model swap" assumption is confirmed false: only 6/24 layers are
Qwen3-style attention; the other 18 are Gated DeltaNet.

## 1. Q1 — Does Unsloth's FastLanguageModel support Qwen3.5 / Gated DeltaNet?

**Yes for training, but not with the installed environment and not with the
project's QLoRA recipe.**

Concrete evidence:

1. **Unsloth code has Qwen3.5 support.** Installed `unsloth 2026.6.9`:
   - `unsloth/models/loader.py:121` — `FORCE_FLOAT32 = [..., "qwen3_5",  # Qwen3.5
     GDN layers produce NaN grad norms in float16 training]`.
   - `unsloth/save.py:463` `_is_qwen3_5_vlm(...)`, `:471` handles
     `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration`,
     `:475` `model_type in ("qwen3_5","qwen3_5_moe")`.
   - `unsloth/models/loader.py:517,1190` raise an explicit error when the
     transformers version is too old:
     > `Unsloth: Your transformers version of <v> does not support Qwen3.5.`
     > `The minimum required version is 5.2.0.`
     > `Try pip install --upgrade "transformers>=5.2.0"`
2. **Unsloth's own docs confirm fine-tuning.** `docs.unsloth.ai/models/qwen3.5/fine-tune`
   (fetched 2026-09-25):
   > "You can now fine-tune Qwen3.5 model family (0.8B, 2B, 4B, 9B, 27B, 35B‑A3B,
   > 122B‑A10B) with Unsloth. Support includes both vision, text and RL
   > fine-tuning."
   > "Qwen3.5 bf16 LoRA VRAM use: 0.8B: 3GB • **2B: 5GB** • 4B: 10GB • 9B: 22GB …"
   > "**Please use transformers v5 for Qwen3.5. Older versions will not work.**"
   > "**It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models,
   > no matter MoE or dense, due to higher than normal quantization differences.**"
   > "Qwen3.5 use custom Mamba Triton kernels. Compiling those kernels can take
   > longer than normal."
3. **No prequantized 4-bit base exists.** `unsloth/Qwen3.5-2B-unsloth-bnb-4bit` →
   `RepositoryNotFoundError` (404). The Unsloth catalog lists Qwen3.5 0.8B/2B/4B/
   9B/27B/35B-A3B/122B-A10B/397B-A17B under **GGUF only**; the "Instruct (4-bit)"
   column is `—` for every size. (`unsloth/Qwen3.5-2B` exists but is a
   full-precision VLM checkpoint, with `preprocessor_config.json` /
   `video_preprocessor_config.json`.)
4. **Current env is too old.** `qwen3-train` has `transformers 4.57.6`, `unsloth
   2026.6.9`; no `Qwen3_5` classes, no `qwen3_5` in `CONFIG_MAPPING_NAMES`, no
   `AutoModelForMultimodalLM`. Training therefore requires a `transformers>=5.2.0`
   upgrade of the working env (which currently drives the 0.6B/1.7B pipeline).

**Q1 = qualified yes**: Unsloth can fine-tune Qwen3.5-2B, but only after upgrading
`transformers` to v5, only via bf16 LoRA (not the project's QLoRA recipe), and the
loader forces float32 for the GDN path (fp16 NaN) — a VRAM cost on the 12 GB card
(Unsloth quotes 2B bf16 LoRA at 5 GB, so it likely still fits).

## 2. Q2 — If not Unsloth, a raw HF + PEFT + bitsandbytes QLoRA path?

Moot for "Unsloth doesn't support it" (Q1 is a yes), but recorded:
- A raw `transformers>=5.2` + `peft` + `bitsandbytes` QLoRA path is *nominally*
  possible for the `Qwen3_5ForCausalLM` text model, but it is exactly what Unsloth
  explicitly advises against for Qwen3.5 ("higher than normal quantization
  differences"), and the GatedDeltaNet layers use custom kernels whose
  compatibility with `bitsandbytes` 4-bit `Linear` wrapping is unverified. It would
  also forfeit Unsloth's memory/speed gains — material given #232's 1.7B plain
  transformer already peaked at 8.9 GB of 12 GB.
- The recommended path is therefore **bf16 LoRA (2B ≈ 5 GB)**, not QLoRA.

## 3. Q3 — Does Cactus recognize/convert the hybrid layout? (the decisive one)

**Weights convert; the text-only graph does not build.** All commands run in the
pinned arm64 container (`cactus-arm64-convert:built`, torch 2.8.0 / transformers
5.5.4), fork `cactus-sdk-v210-suppress`.

### 3a. `cactus convert Qwen/Qwen3.5-2B --bits 4 --weights-only --skip-model-load` — SUCCEEDED

The converter recognizes Qwen3.5: `detect_model_type` normalizes `qwen3_5` →
`qwen` (so `detect_family(auto)` = `qwen`, in `SUPPORTED_FAMILIES`), and the
weight patterns include the Qwen3.5 linear-attention tensors
(`linear_attn.in_proj_qkv/in_proj_a/in_proj_b/in_proj_z/out_proj/norm/conv1d/
A_log/dt_bias`, plus deltanet gate/beta aliases).

```
converted     285
fallback      332   (norm tensor 184, bias tensor 99, non-2d tensor 55, ...)
unrecognized  15
total bytes   1,183,691,808
components:   embedding 2, language 333, vision 297
```
The **15 unrecognized tensors are all `mtp.*`** (the multi-token-prediction head:
`mtp.fc`, `mtp.layers.0.*`, `mtp.norm`, `mtp.pre_fc_norm_*`) — auxiliary to basic
generation, so the core weights convert cleanly. `weights_manifest.json` written.

### 3b. Legacy `cactus.transpile.hf_model` (the #231 path) — FAILS, two ways

**Default invocation** (mirroring #231 Phase 1):
```
python3 -m cactus.transpile.hf_model --model-id Qwen/Qwen3.5-2B \
  --weights-dir .../qwen3.5-2b-cq4-conv --artifact-dir .../qwen3.5-2b-legacy \
  --low-memory-load --skip-execute --skip-reference-compare
```
→
```
RuntimeError: Qwen3.5 multimodal transpile requires at least one --image-file
  at hf_model.py:2298 (_prepare_qwen3_5_multimodal_inputs)
```
Root cause: `model_profiles.QWEN_PROFILE` sets `default_task =
"multimodal_causal_lm_logits"`, `needs_image=True`, and a default component set
that **includes `vision_encoder`**. Its `text_component_plans` only maps
`"qwen3forcausallm"` → the 6 text components; Qwen3.5's architecture is
`Qwen3_5ForConditionalGeneration`, so **no text-only 6-component plan exists for
it**. The legacy plain-text path #231 used simply does not apply.

**Explicit text task** (forcing the text path):
```
... --task causal_lm_logits --components decoder,decoder_step ...
```
→
```
cactus.transpile.capture_pytorch.CapturePhaseError: [capture:export] torch.export
failed for model=_ComponentCaptureWrapper strict=True: Dynamo failed to run FX node
with fake tensors: call_function <built-in function matmul>(...): got
RuntimeError('Unhandled FakeTensor Device Propagation for aten.bmm.default, found
two different devices cpu, meta')

from user code:
  model_adapters.py:4913 forward -> debug_forward
  model_adapters.py:4941 position_embeddings = self.backbone.rotary_emb(hidden_states, multimodal_position_ids)
  transformers/models/qwen3_5/modeling_qwen3_5.py:149
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
```
i.e. the Qwen3.5 RoPE (`Qwen3_5` rotary embedding) does a `bmm` that mixes `cpu`
and `meta` fake tensors, so `torch.export` cannot capture even a 2-component text
graph.

**Q3 = hard no for this project's need.** Cactus *recognizes* the hybrid
architecture for weight conversion, and has Gated DeltaNet fusion/adapters, but it
cannot produce a **text-only** runtime graph here: the default path demands an
image (multimodal `vision_encoder + lm_encoder + decoder`), and the forced text
path dies in `torch.export`. A multimodal bundle (untested further — vision is an
explicit non-goal) would be a different shape than the plain-text engine path this
project uses, and would carry a vision tower that is dead weight for natural-farming
text.

## 4. Q4 — Is there a genuine text-only mode, and does it work?

**Yes, via `Qwen3_5ForCausalLM`** (not `Qwen3ForCausalLM`), verified in the
container (`transformers 5.5.4`, torch 2.8 CPU; `scripts/qwen35_2b_textonly_probe.py`):

```
transformers 5.5.4
has Qwen3_5ForCausalLM True   has Qwen3_5ForConditionalGeneration True
has AutoModelForCausalLM True has AutoModelForMultimodalLM True
rendered prompt: '<|im_start|>user\nThe best fertilizer for tomatoes is<|im_end|>\n<|im_start|>assistant\n thinking\n\n</think>\n\n'
LOADED via Qwen3_5ForCausalLM in 4s -> Qwen3_5ForCausalLM
GENERATED 8 tokens in 105s
OUT: 'The best fertilizer for tomatoes depends on the'
TEXT_ONLY_LOAD: OK via Qwen3_5ForCausalLM
```
Caveat printed by transformers:
```
The fast path is not available because one of the required library is not installed.
Falling back to torch implementation. To install follow .../flash-linear-attention ...
and .../causal-conv1d
```
So text-only inference works, but the GatedDeltaNet fast kernels require
`flash-linear-attention` + `causal-conv1d`; without them the GDN layers run a slow
torch fallback (relevant to both training throughput and on-device latency).

## 5. GO / NO-GO for Phase 1

**NO-GO.** The bar is a working training path **and** a working Cactus conversion
path:
- Training: qualified yes (Unsloth; requires `transformers>=5.2.0` upgrade and
  bf16 LoRA instead of the project's QLoRA). Not a blocker by itself.
- **Cactus: hard no for a text-only bundle** — default transpile requires an
  image; explicit text transpile fails in `torch.export`. This is the blocker.

Because a model this project can train but not deploy as a plain-text on-device LM
is not useful for the epic, Phase 1 does not start. Do not force Qwen3.5-2B
forward; do not build around the vision path (explicit non-goal).

## 6. Fallback candidates

Per the issue's own guidance, prefer the **plain-transformer Qwen3 family**, which
is already proven end-to-end through this toolchain (0.6B champion, 1.7B #232
port) — same architecture family, no Gated DeltaNet, existing Cactus legacy
6-component text path and engine `inputs_embeds` support:
- **Qwen3-4B** — intermediate step; larger than 1.7B but still plain transformer.
- **Qwen3-8B** — the #235 research target (fit + Cactus conversion both still
  unverified, but the architecture is the known-good family).

Both are untested at these sizes by this project, but neither introduces the
Qwen3.5 architectural risk that this phase found to be a hard blocker.

## 7. Reproduce / evidence

```bash
FORK=[LOCAL_PATH]
# Q1 evidence (installed env):
python -c "import transformers;print(transformers.__version__)"            # 4.57.6
grep -n "qwen3_5" $FORK/../../..  # see unsloth site-packages loader.py:121,517
# Q3a: weights convert (succeeded)
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTHONPATH=/src/cactus-sdk/python \
  -e CACTUS_LIB_PATH=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so \
  --entrypoint python3 cactus-arm64-convert:built -m cactus convert Qwen/Qwen3.5-2B \
  /src/cactus-sdk/weights/qwen3.5-2b-cq4-conv --bits 4 --weights-only --skip-model-load
# Q3b: legacy graph build (fails: requires --image-file)
docker run --rm --platform linux/arm64 ... -m cactus.transpile.hf_model \
  --model-id Qwen/Qwen3.5-2B --weights-dir .../qwen3.5-2b-cq4-conv \
  --artifact-dir .../qwen3.5-2b-legacy --low-memory-load --skip-execute --skip-reference-compare
# Q3b': forced text task (fails: CapturePhaseError)
docker run --rm --platform linux/arm64 ... -m cactus.transpile.hf_model \
  --model-id Qwen/Qwen3.5-2B --weights-dir .../qwen3.5-2b-cq4-conv \
  --artifact-dir .../qwen3.5-2b-text-legacy --task causal_lm_logits \
  --components decoder,decoder_step --low-memory-load --skip-execute --skip-reference-compare
# Q4: text-only generation
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk \
  -v "$PWD/scripts":/work/scripts:ro -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint python3 cactus-arm64-convert:built /work/scripts/qwen35_2b_textonly_probe.py
```

Artifacts: this note, `results/QWEN3_5_2B_ARCH_VERIFICATION.json`,
`scripts/qwen35_2b_textonly_probe.py`. Converted CQ weights + failed graph dirs live
under the fork's `weights/qwen3.5-2b-*` (not committed).
