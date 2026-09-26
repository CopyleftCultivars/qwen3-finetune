# #237 G1 — Pin and smoke-test the hosted Qwen3.5-2B Cactus bundle; stock baseline

Date: 2026-09-26. Parent #236, task #237. Scope: determine whether the **hosted**
`Cactus-Compute/Qwen3.5-2B` artifact supports our text-only natural-farming
workload and, if so, establish its unadapted baseline on the frozen 210-item
ladder. No training, no custom graph generation, no LPM/RLM, no memory prefix.

**Separate verdicts:**

| gate | verdict |
|---|---|
| Hosted text-only runtime (ARM) | **FAIL / BLOCKED** — no runnable hosted bundle exists to load |
| Stock baseline on the 210-item ladder | **BLOCKED** — depends on the runtime; not run (stop-dependent-eval) |
| Physical-device smoke (Pixel 7) | **PENDING** — no device attached (`adb devices` empty) |

The blocker is concrete and reproducible: the hosted artifact is a **legacy
flat-CQ-weights zip** (`-int4`/`-int8`, tag `v1.14`), **not a runnable bundle**
(no `components/manifest.json`); the current runtime (v2.2.1) requires that
manifest and refuses it; and the supported `cactus download`/`run` path cannot even
resolve the archive because the resolver only matches the newer `-cqN` naming.

## 1. Artifact pinning and provenance (verified vs unspecified)

### 1.1 Hosted artifact — `Cactus-Compute/Qwen3.5-2B`

| field | value |
|---|---|
| repo | `huggingface.co/Cactus-Compute/Qwen3.5-2B` |
| tag / revision used | **`v1.14`** (latest ≤ runtime; `main` == `v1.14`) |
| commit sha | **`f906dab20c19645665a3149cfd75ae44454714da`** |
| available tags | `v1.11`, `v1.12`, `v1.14` |
| int4 archive | `weights/qwen3.5-2b-int4.zip` — **1,327,893,436 B**, sha256 **`e57c9e50ece5f9fd3ca366a5a0db5cc0e79303e595cf004912e9c6fa871feeb4`** |
| int8 archive | `weights/qwen3.5-2b-int8.zip` — 2,177,355,521 B, sha256 `05d02c3b1b07a9ffc50ddb44942ac1b1e18605a5f53dc71d427fe7f65840d993` |
| `config.json` | `{"model_type":"Qwen3.5-2B","precisions":["int4"],"fingerprint":"acb4c997c7b6afcc4e818ecc9d9faf673dddc28ac4b324df91570fe1a5ca7c7b"}` |
| README | `base_model: Qwen/Qwen3.5-2B`; "hybrid vision-language model with DeltaNet for on-device multimodal inference" |
| pipeline tag | `image-text-to-text` |

The int4 zip was **fully downloaded and its sha256 recomputed locally**: it matches
the HF LFS digest exactly (`e57c9e50…feeb4`). Artifact is pinned and immutable.

### 1.2 Bundle structure (inspected, not assumed)

Extracted `qwen3.5-2b-int4.zip` → **396 files, 389 `*.weights` + 7 sidecars**
(`config.txt`, `tokenizer.json`, `tokenizer_config.txt`, `vocab.txt`, `merges.txt`,
`special_tokens.json`, `chat_template.jinja2`).

- **No `components/manifest.json`. No `graph.cactus`. No `runtime_plan.json`.**
- The int8 zip has the identical shape (396 entries, flat weights, no manifest).

So the hosted artifact is the **converted CQ weights only** — the same flat format
as the project's own `cactus convert --weights-only` output, and the same flat
format the Pixel app's legacy `qwen3-nf-finetuned.zip` uses. It is **not** a
runnable bundle.

### 1.3 Corresponding HF source checkpoint

`Qwen/Qwen3.5-2B` (instruct; commit `15852e8c16360a2fea060d615a32b45270f8a8fc`,
`image-text-to-text`). A base variant `Qwen/Qwen3.5-2B-Base`
(`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`) also exists.

### 1.4 Compatible runtime — inspected, and it does **not** match

Runtime tested: fresh upstream clone `localllm-qwen3.5/cactus-sdk`,
**`CACTUS_VERSION=v2.2.1`** (commit `fa094ce8`), arm64 container
**`cactus-arm64-convert:v221`** (`sha256:29ed5110…`, py 3.12.14 / torch 2.8.0+cpu /
transformers 5.5.4), engine built with hybrid kernels.

Two independent incompatibilities:

1. **The CLI cannot resolve the hosted archive.** `CQ_VARIANT_RE = -cq(\d+)$`
   (`python/cactus/cli/utils.py:19`); the hosted files are `-int4`/`-int8`.
   `archives_from_repo_files` skips any archive whose variant doesn't parse, so:
   ```
   suggested_cq_repo('Qwen/Qwen3.5-2B') -> Cactus-Compute/Qwen3.5-2B
   resolve_weight_revision(repo)         -> v1.14
   list_hf_cq_archives(repo, rev=v1.14)  -> ()
   download_bundle('Qwen/Qwen3.5-2B',4)  -> RuntimeError: no bundles published at Cactus-Compute/Qwen3.5-2B
   ```
   The newer `-cqN` naming *is* a runnable-bundle format (verified:
   `Cactus-Compute/needle/needle-cq4.zip` contains `components/manifest.json`), but
   **Qwen3.5-2B publishes no `-cqN` archive at any tag** (only `-int4`/`-int8`).
2. **The engine refuses the flat weights.** `cactus-engine/src/model.cpp:5885-5888`
   requires `components/manifest.json`:
   ```
   [ERROR] [model] Not a runnable Cactus bundle (no components/manifest.json at
     /work/g1-evidence/bundle-int4-v114). Run `cactus download <hf_model>` to fetch one.
   [ERROR] [init] Failed to create model - check config.txt exists at: …
   cactus_init RAISED: RuntimeError Failed to create model - check config.txt exists at: …
   ```
   The engine's own remediation message ("Run `cactus download`") is circular: that
   command fails at (1). The `cactus download` fallback then attempts a local build,
   which is the already-failed Qwen3.5 graph path (retry evidence, §5) — not
   repeated here.

**Unspecified by upstream:** the hosted repo does not state a runtime version or
that the zip is weights-only; the `config.json` "fingerprint" is opaque; the
`v1.x` tag scheme is far below the v2.2.1 runtime and the compatibility doc
(`docs/compatibility.md`) does not describe an `-intN` ↔ v2 runtime path.

## 2. Step 2 — real text-only smoke tests: **not run (blocked)**

There is no bundle to `cactus_init`, so init / no-image generation / reset /
multi-turn / prompt-rendering / thinking-separation / stop / truncation / answer
extraction **cannot be exercised**. No `components/manifest.json` ⇒ the engine
fails at init (§1.4). This is the exact artifact/runtime blocker the task said to
record and stop on; emulation is not substituted for real evidence.

## 3. Step 3 — stock baseline: **blocked, not run**

The frozen 210-item protocol was prepared for use (`scripts/cactus_engine_ladder.py`
`prepare` → prompts; the same frozen `SYSTEM_PROMPT`/`PROMPT_TEMPLATE`/
`extract_answer` imported from `benchmark-workspace/evaluate.py`; greedy
`temperature=0`, `max_tokens=8`, matching F8/#232's ladder transport). But the
baseline is **dependent evaluation** on the hosted runtime, which does not load.
Per the task ("If the hosted bundle fails … stop dependent evaluation"), the
Cactus baseline was **not** run and **no score is reported**. Historical F8/#232
comparators (distilled 0.6B 0.8714, base-0.6B 0.4286, qwen3-8b 0.8571) remain the
reference table and were **not** compared against a non-existent Cactus run.

An **HF-runtime** stock baseline (transformers 5.5.0 + `Qwen3_5ForCausalLM`) is
possible and would be a *separate, non-equivalent* result (different runtime and
quantization: bf16 HF vs CQ4 Cactus) — it was deliberately **not** produced here to
respect the stop rule and #237's Cactus-runtime scope. Recommended as an explicit
follow-up (§6).

## 4. Step 2b — physical device: **pending**

`adb devices` shows no attached device, so the Pixel 7 fit/inference smoke via the
existing harness (`benchmark-workspace/oc_eval/run_cactus_pixel.py`,
`benchmark-eval-workspace/evaluate_pixel7.py`) could not be attempted. Device
validation is **pending**, not substituted with emulation.

## 5. Why this is not a repeated local-graph attempt

The retry (`results/QWEN3_5_2B_ARCH_VERIFICATION_RETRY.md`) already established,
on the same v2.2.1 engine, that **local graph generation for Qwen3.5-2B fails** by
all three paths (default Crassus `_update_linear_attn_mask` guard; legacy default
`--image-file` required; legacy forced-text RoPE `bmm` FakeTensor mismatch). #237's
job was the **hosted** artifact, and it independently fails earlier: the hosted
artifact is not a bundle and the CLI cannot fetch it. No local graph path was
re-run.

## 6. Blocker statement and next actions

**Blocker:** no runnable hosted text-only Cactus bundle exists for Qwen3.5-2B.
Concretely: (a) the hosted zip is flat CQ weights (`-int4`, v1.14), not a bundle;
(b) the v2.2.1 CLI resolves only `-cqN` bundle archives and reports "no bundles
published"; (c) the v2.2.1 engine requires `components/manifest.json` and refuses
the flat weights; (d) the local fallback graph build fails for Qwen3.5.

**Next actions (not started here):**
1. **Upstream (unblock path):** ask Cactus to publish a `-cqN` runnable bundle for
   Qwen3.5-2B (the format newer models already use), or a `qwen3_5` model profile
   with the `_update_linear_attn_mask` export patch so the local builder can
   transpile the flat weights. Then re-run #237 §2–3.
2. **Separate HF stock baseline** (clearly labelled non-equivalent): load
   `Qwen/Qwen3.5-2B` with `transformers==5.5.0` + `Qwen3_5ForCausalLM`, run the
   frozen ladder with the same prompt/extractor, and report it as an HF-runtime
   result — not a Cactus deployment result. This informs #239.
3. **Device:** attach the Pixel 7 and re-attempt the short fit/inference smoke.
4. Do **not** start #238+ from this task.

## 7. Reproduce / evidence

```bash
NEW=[LOCAL_PATH]
IMG=cactus-arm64-convert:v221          # sha256:29ed51100c94947ac597c1cc1180c06f0e9442a76f4354f26c8774381442ff56
LIB=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so

# 1) pinned download + local sha256 verification (host)
python - <<'PY'
from huggingface_hub import hf_hub_download; import hashlib,os
p=hf_hub_download("Cactus-Compute/Qwen3.5-2B","weights/qwen3.5-2b-int4.zip",revision="v1.14",
                  local_dir="localllm-qwen3.5/g1-evidence/artifacts")
print(os.path.getsize(p)); print(hashlib.sha256(open(p,'rb').read()).hexdigest())
PY

# 2) supported upstream path fails to resolve the hosted archive
docker run --rm --platform linux/arm64 -v "$NEW":/work \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CACTUS_LIB_PATH="$LIB" -e PYTHONPATH=/src/cactus-sdk/python \
  --entrypoint python3 "$IMG" -c "
from cactus.cli.download import download_bundle
download_bundle('Qwen/Qwen3.5-2B', bits=4)"     # -> RuntimeError: no bundles published

# 3) engine refuses the flat-weights artifact
docker run --rm --platform linux/arm64 -v "$NEW":/work -e CACTUS_LIB_PATH="$LIB" \
  -e PYTHONPATH=/src/cactus-sdk/python --entrypoint python3 "$IMG" -c "
from cactus.bindings import cactus as c
c.cactus_init('/work/g1-evidence/bundle-int4-v114')"   # -> RuntimeError (no components/manifest.json)
```

Raw evidence: `localllm-qwen3.5/g1-evidence/` — `artifacts/weights/qwen3.5-2b-int4.zip`
(sha256-verified), `bundle-int4-v114/` (extracted), `logs/prebuilt_download_error.log`,
`logs/cli_download_userfacing.log`, `logs/cactus_init_flat_weights.log`.

Artifacts (this repo): `results/QWEN3_5_2B_HOSTED_BUNDLE_G1.md`,
`results/QWEN3_5_2B_HOSTED_BUNDLE_G1.json`.
