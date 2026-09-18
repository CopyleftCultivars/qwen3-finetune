# F179 / M3 — #205 (S1-S2): cactus-engine v2.1.0 build + sampling-path Δ + manifest key

Milestone #14 (Epic #179). Build + wire-up + sanity-load only. The leak A/B and
the **MUST-VERIFY argmax-parity** check are #206's job and are **not** claimed
here.

Status: **gate met.** Patch applied and built for Android arm64-v8a; a glibc
arm64 build of the identical source loads both a baseline and the
p193+suppress bundle under the M1 qemu harness with no init regression; the
manifest key is emitted, read back (266 IDs), and is the only bundle diff.

## 1. Patch (one site)

Worktree: `LocalLLM/finetune/cactus-sdk-v210-suppress` at tag **v2.1.0**
(`ba2ee756f940549b7a1cf355bcba5deb5ae3db85`), detached. Patch saved to
`results/F179_M3_SUPPRESS_PATCH.diff` (sha256
`547be60660ef4f2f9d280971de5a1b39ba96b36ef5cda712dacd48457dde6618`); `cactus-engine/src/model.cpp`,
13 insertions / 1 deletion.

`Model::prepare_sampling_context` now includes
`!generation_suppressed_token_ids_.empty()` in the dense-bias condition and folds
the set into `samp_bias_dense_` as `-65504.0f` (then `samp_has_bias_ = true`),
using the same mechanism as `tool_bias`/`vocab_bias`. This makes the set survive
the `samp_ctx_active_` branch, which otherwise applies only the scalar
`suppressed_token_id_`. No ABI change (function body only).

**Second trap found and fixed during implementation:** the dense bias buffer is
consumed by `update_bias_dense(vocab)` as `min(bias_len, vocab)`, and
`tokenizer_->get_vocab_size()` is **151670** while the reserved IDs are
**151670..151935** — i.e. at/above the tokenizer vocab. Sizing the buffer to the
tokenizer vocab zero-pads exactly the IDs we need to suppress (silent no-op). The
patch therefore extends `n` to `max(tokenizer_vocab, tool_dense_size,
largest_suppressed_id + 1)`. A naive "fold into dense bias" without this would
look correct in source and do nothing at runtime.

## 2. Builds

| artifact | how | size | sha256 (prefix) | deps |
|---|---|---|---|---|
| Android arm64-v8a | `ANDROID_HOME=[LOCAL_PATH] ./android/build.sh` (NDK **29.0.14206865**) | 54.96 MB | `3f81ad0f…` | bionic: liblog, libandroid, libm, libdl, libc |
| glibc arm64 (qemu gate) | `cactus build --python` in `cactus-arm64-convert:built` (arm64) | 4.54 MB | `548575ac…` | libcurl.so.4, libstdc++, libm, libgcc_s, libc |
| glibc arm64 vanilla v2.1.0 | same, unpatched worktree | 4.54 MB | `aa7801fe…` | same |

Android `.so` exports 30 `cactus_*`/`Java_com_cactus_*` symbols (`cactus_init`,
`cactus_complete`, …), `file` = ELF 64-bit ARM aarch64. The `suppress_generation_token_ids`
reader string is present in the built binaries.

**Deviation, flagged:** the Android NDK artifact is **bionic-linked** and cannot
`dlopen` in the M1 qemu harness (`arm64v8/ubuntu:24.04`, glibc). The qemu gate
therefore runs the **glibc build of the exact same patched source** (M1 itself
used a glibc build of the then-shipped engine). The patch lives in shared
`model.cpp`, so the glibc build exercises the sampling change; the Android build
validates the target compilation. #206 should run its leak/argmax test on the
glibc `.so` under qemu (and the Android artifact on-device/JNI later). Also: the
image's pre-existing `.so` is a different SDK revision (it lacks
`cactus_set_backend`, which the v2.1.0 binding requires), so a fresh **vanilla
v2.1.0** build was made for the no-regression comparison rather than reusing it.

## 3. Bundle-side change

`scripts/cactus_set_suppress_ids.py` (authoritative; reads
`lpm.config.RESERVED_ID_START/END`) emits a top-level
`"suppress_generation_token_ids": "151670,…,151935"` (266 IDs — the pool, not the
192 currently baked) into `components/manifest.json`. It does a byte-minimal
textual insert (one line after the opening brace) so the bundle diff stays clean.
Wired as a post-step into `LocalLLM/conversion/convert_to_cactus.sh` (self-contained
inline equivalent; uncommitted LocalLLM-side change — the qwen3-finetune commit
carries the script).

`outputs/r179-m1/p193-suppress-deploy` = copy of `p193-deploy` + the key.

## 4. Gate results

- **Loads under qemu (patched glibc `.so`):** baseline
  `distill-calibrated-deploy` and `p193-suppress-deploy` both `cactus_init` OK;
  `p193-suppress-deploy` tokenizes `<|mem0_0|>` → `151670` (bundle live).
- **Init time, patched vs vanilla v2.1.0** (2 runs each): patched 1.94/1.83s and
  1.79/1.79s; vanilla 1.90/1.84s and 1.84/1.81s. Comparable, ~2s, no regression.
- **Manifest key:** present and correct — read back as exactly
  `list(range(151670, 151936))`, 266 IDs, no truncation.
- **Bundle diff:** `diff -rq p193-deploy p193-suppress-deploy` reports **only**
  `components/manifest.json`; its diff is the single inserted line.

## 5. What is NOT claimed here (#206)

- That suppression actually works (leak A/B) — not run.
- argmax parity with the unpatched sampler — not run. This is the load-bearing
  correctness check: the dense-bias path must not change argmax on non-suppressed
  tokens.
- On-device (Android `.so`) behavior — no device attached.

## Reproduction

```
# patch + builds
git -C .../cactus-sdk worktree add ../cactus-sdk-v210-suppress v2.1.0
# apply results/F179_M3_SUPPRESS_PATCH.diff
ANDROID_HOME=[LOCAL_PATH] ./android/build.sh                     # Android
docker run --rm --platform linux/arm64 -v <worktree>:/src/cactus-sdk \
  cactus-arm64-convert:built bash -lc "cd /src/cactus-sdk && cactus build --python"   # glibc

# bundle key
python scripts/cactus_set_suppress_ids.py outputs/r179-m1/p193-suppress-deploy
python scripts/cactus_set_suppress_ids.py outputs/r179-m1/p193-suppress-deploy --check

# qemu load/time
docker run --rm --platform linux/arm64 -e CACTUS_LIB_PATH=/work/lib.so \
  -v <patched>/cactus-engine/build/libcactus_engine.so:/work/lib.so:ro \
  -v <baseline>:/work/baseline:ro -v outputs/r179-m1/p193-suppress-deploy:/work/p193:ro \
  cactus-arm64-convert:built python3 /work/t.py
```

Artifacts: `outputs/r179-m1/v210-suppress/` (both `.so` + vanilla, gitignored),
`outputs/r179-m1/p193-suppress-deploy/` (gitignored). Provenance:
`results/F179_M3_S1S2_BUILD.json`, patch `results/F179_M3_SUPPRESS_PATCH.diff`.
