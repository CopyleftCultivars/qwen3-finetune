# #236 Phase 0 addendum — updated-engine (Cactus v2.2.1) re-verification of Qwen3.5-2B

Date: 2026-09-25. Follow-up to `QWEN3_5_2B_ARCH_VERIFICATION.md`, after project-owner
direction: Cactus-Compute ships a Qwen3.5-2B model, so the earlier NO-GO — which
was scoped to a **stale pinned fork** — needed re-checking against the current
upstream toolchain. New workspace: `localllm-qwen3.5/` (kept separate from the
`cactus-sdk-v210-suppress` path).

**Bottom line:** the updated engine changes *why*, not *whether*. Qwen3.5 is a
first-class Cactus model family and its weights convert on both forks, but **local
runtime-bundle generation for Qwen3.5 still fails** — now in the new transpiler
stack ("Crassus v1.0.0") at a `torch.export` data-dependent guard in Qwen3.5's
`_update_linear_attn_mask`. Upstream's own docs state local bundle generation is
disabled while the graph builder is rewritten. The training path (Unsloth, bf16
LoRA) is real; the missing piece is a `qwen3_5` model profile with export patches.

## 1. Workspace setup (`localllm-qwen3.5/`)

| item | value |
|---|---|
| fork | `localllm-qwen3.5/cactus-sdk` — clean clone of `github.com/cactus-compute/cactus` `main` |
| version | **`CACTUS_VERSION = v2.2.1`** (commit `fa094ce8`), vs the old `v2.1.0-suppress-patch` (a82155c) |
| engine | **built** in the arm64 container: `cactus-engine/build/libcactus_engine.so` (4.5 MB, includes `attention_hybrid.cpp` / Gated DeltaNet kernels) |
| prior work | `phase0-reference/` holds the Phase-0 note/JSON/probe from `qwen3-finetune/results/` |

Kept separate from the existing `LocalLLM/finetune/cactus-sdk-v210-suppress` so the
memory-prefix/LPM-era engine stays intact.

## 2. Why Cactus ships a Qwen3.5-2B model (the owner's question)

Cactus **does** support the hybrid architecture; the public local *builder* is the
gap, not the architecture.

- **Family support is documented.** `docs/cactus_transpiler.md` "Supported Model
  Families": **`Qwen 3/3.5 | text, vision | component split: no`**. Qwen3.5 is a
  tested adapter target, alongside Gemma3/4, LFM2/2.5, Whisper, Parakeet.
- **The engine has the kernels.** v2.2.1 compiles `cactus-kernels/src/attention_hybrid.cpp`,
  and the new transpiler has `gated_deltanet_decode` / `gated_deltanet_prefill`
  fusions + lowerings (`transpiler/Fusions/…`, `Generator/lowerings/lowering_special_ops.py`).
- **They ship versioned prebuilt weights.** `huggingface.co/Cactus-Compute/Qwen3.5-2B`
  carries `weights/qwen3.5-2b-int4.zip` + `int8.zip`, tags **v1.11 / v1.12 / v1.14**
  (their runtime↔weights versioning scheme, `docs/compatibility.md`). Built with
  their internal/prebuilt pipeline.
- **The public local graph builder is mid-rewrite.** `docs/cactus_engine.md:32`,
  `docs/cactus_quants.md:187`, and `docs/finetuning.md:70` all say: *"local runtime
  bundle generation is unavailable while the graph builder is being rewritten."*
  The README now describes `cactus convert` as **"HuggingFace → Cactus CQ weights"**
  and `cactus download`/`cactus run` as the bundle path (prebuilt, else build).

So Cactus can ship Qwen3.5 because their pipeline supports it; this machine's local
public toolchain cannot yet *rebuild* it.

## 3. Updated-engine test (v2.2.1, arm64 container)

`cactus convert Qwen/Qwen3.5-2B /src/cactus-sdk/weights/qwen3.5-2b-v221 --bits 4`

**Weights: SUCCEEDED** (same as the old fork):
```
converted 285, fallback 332, ignored 2, unrecognized 15, total 1,183,691,808 B
components: embedding 2, language 335, vision 297
```
(the 15 unrecognized are again all `mtp.*`).

**Graph: FAILED** in the new stack:
```
Transpiling Qwen/Qwen3.5-2B with generic contract 'generic_text' and modalities: text
Running converter (Crassus v1.0.0)
Conversion error: Could not guard on data-dependent expression Eq(u0, 1)
  (unhinted: Eq(u0, 1)). (Size-like symbols: none)
...
  File ".../transformers/models/qwen3_5/modeling_qwen3_5.py", line 1301, in _update_linear_attn_mask
    attention_mask is not None and torch.all(attention_mask == 1)
```
`torch.export` cannot trace the Python conditional in Qwen3.5's
`_update_linear_attn_mask`. No `components/` dir is produced.

**Root cause / missing piece:** v2.2.1's new stack resolves Qwen3.5 to the
**`generic_text`** profile (`plan_name=generic_text_no_cache`,
`execution_strategy=full_context_recompute`) because there is **no `qwen3_5` profile**
in `transpiler/ModelProfiles/profiles.py`, and `generic_text` has
`export_patches=()`. A proper `qwen3_5` profile would supply the export patch that
handles `_update_linear_attn_mask` (other families already carry patches, e.g.
`gemma4_audio_mask`, `lfm2_vl_image_features`, `transformers_moe_grouped_mm_fallback`).

## 4. Updated SFT/distill → Cactus API (the owner's second question)

From `docs/finetuning.md` ("Deploying Unsloth Fine-Tunes to Cactus for Phones") and
`docs/cactus_transpiler.md`:

- **Supported base models:** `Qwen3, Qwen3.5, Gemma3, LFM2, LFM2.5`.
- **Train (Unsloth):** transformers **v5** required; for Qwen3.5 the docs recommend
  **bf16 LoRA** (`load_in_4bit=False`), *not* QLoRA ("higher than normal quantization
  differences"); 2B bf16 LoRA ≈ **5 GB** VRAM (fits the 12 GB card).
- **Convert back to Cactus format:**
  ```bash
  cactus convert Qwen/Qwen3.5-2B ./out --lora ./my-lora-adapter
  # or from the Hub:
  cactus convert Qwen/Qwen3.5-2B ./out --lora username/my-lora-adapter
  ```
  `cactus convert` merges the LoRA and writes **CQ weights only** (`--weights-only`
  stops earlier still). There is currently **no local graph-build step**.
- **Deploy:** "deploy the fine-tuned model after the replacement graph builder
  lands, or run a prebuilt Cactus-Compute bundle with `cactus run <model>`."
  `cactus run <hf-id|bundle-path>` downloads/loads a prebuilt bundle; `cactus build
  --apple/--android/--python` builds the native engine.

So the project's recipe maps cleanly up to **CQ weights**; the last mile
(weights → runnable bundle) is blocked upstream, not by Qwen3.5 specifically.

## 5. Revised assessment vs the original Phase-0 note

| claim (original Phase 0) | updated status |
|---|---|
| Qwen3.5 is trainable with Unsloth (transformers v5, bf16 LoRA) | **unchanged** |
| no `unsloth/Qwen3.5-2B-unsloth-bnb-4bit`; QLoRA discouraged | **unchanged** |
| legacy `cactus.transpile.hf_model` fails (multimodal-only / RoPE bmm) | **superseded** — that was the stale v2.1.0 legacy path |
| "no working Cactus path to a text-only bundle" | **still true today**, but the reason is upstream-wide: the graph builder is mid-rewrite, and the new stack lacks a `qwen3_5` export-patch profile |
| verdict NO-GO | **NO-GO stands for local bundle generation today**, but it is not a Qwen3.5 dead end: the architecture is supported (kernels + fusions + family table + prebuilt weights); the blocker is an in-progress upstream builder |

**Recommendation:** do not force a full Phase 1 training run yet, but do not treat
Qwen3.5-2B as ruled out. Two viable tracks:
1. **Watch upstream** for the replacement graph builder + a `qwen3_5` model profile
   (export patches), then re-run §3 — this is the direct unblock.
2. If an earlier deliverable is needed, the plain-transformer fallbacks
   (**Qwen3-4B / Qwen3-8B**) already build on the existing path.

The training half (Unsloth bf16 LoRA → `cactus convert --lora` → CQ weights) can be
validated now if desired, independent of the bundle blocker.

## 6. Evidence / reproduce

```bash
NEW=[LOCAL_PATH]
# build engine (v2.2.1, arm64)
docker run --rm --platform linux/arm64 -v "$NEW":/src/cactus-sdk \
  -e PYTHONPATH=/src/cactus-sdk/python --entrypoint bash cactus-arm64-convert:built \
  -lc "cd /src/cactus-sdk && python3 -m cactus build --python"
# updated-engine conversion (weights OK, graph fails as in §3)
docker run --rm --platform linux/arm64 -v "$NEW":/src/cactus-sdk \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTHONPATH=/src/cactus-sdk/python \
  -e CACTUS_LIB_PATH=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so \
  --entrypoint python3 cactus-arm64-convert:built \
  -m cactus convert Qwen/Qwen3.5-2B /src/cactus-sdk/weights/qwen3.5-2b-v221 --bits 4
```

Artifacts: this note; `results/QWEN3_5_2B_ARCH_VERIFICATION_ENGINE_UPDATE.json`;
the original Phase-0 note/JSON. Workspace: `localllm-qwen3.5/cactus-sdk` (v2.2.1,
engine built), `localllm-qwen3.5/phase0-reference/`.
