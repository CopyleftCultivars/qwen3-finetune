# #242 RESULT — qwen3_5 rotary fix cherry-picked onto Cactus v2.2.1: F1 CapturePhaseError cleared, command D captures both components

Date: 2026-09-30. Parent #236 (epic). Machine record:
`results/QWEN3_5_2B_ROTARY_FIX_242.json`.
Raw evidence (gitignored, on this host):
`localllm-qwen3.5/rotary-fix-evidence/` (`logs/`, `artifacts/`,
`run_command_d.sh`, `qwen35_rotary_helper_check.py`, `README.md`); baseline for
contrast: `localllm-qwen3.5/retry-evidence/logs/_driver.log` (test D) and its
empty `artifacts/`.

**Verdict: F1 PASS.** With the upstream qwen3_5 rotary fix
(`9d07a3bc` → `17dded31`) rebound into the same arm64 conversion container,
command D got past `capture_component_begin=decoder` — the exact line where the
#236 retry died with `CapturePhaseError` (mixed cpu/meta `aten.bmm`) — and
**no error of any kind occurred afterwards**: the run wrote the complete
two-component artifact bundle (both `graph.cactus` files + `components/manifest.json`,
written last), i.e. capture, lowering, optimization and graph build all
completed for `decoder` **and** `decoder_step`. The predicted next failure
(F2, the data-dependent guard in `_update_linear_attn_mask`) did **not**
materialize. **There is no "next error" to record because no traceback exists
— see the evidence-gap disclosure in §4, which this verdict does not paper over.**

## 1. Provenance of the patch

| item | value |
|---|---|
| upstream commit | `9d07a3bc6c30145ad20f42f113578f0b0fa09d2f` "fix(transpile): qwen3_5 rotary under --low-memory-load" (on `origin/bonsai-masked-gemv`, `origin/bonsai-repack` only; confirmed NOT in v2.2.1 / v2.2.2) |
| base | **`v2.2.1` = `fa094ce86c99117f351b100a6ebe51021c11499b`** |
| branch / result | `qwen35-rotary-fix`, cherry-pick applied cleanly (auto-merge, no conflicts) → **`17dded31`** |
| patch-id (both) | **`7bf700c97301c1570853b5800f6eeb594ede905f`** (identical → faithful pick) |
| diff | `python/cactus/transpile/model_adapters.py`, **+18/−8**, single file |
| effect | new shared helper `_qwen35_rotary_embeddings` (model_adapters.py:4902) reroutes **all 7** qwen3_5 rotary call sites (4951, 5022, 5235, 5328, 5405, 5475, 5570) — including the exact F1 site `rotary_emb(hidden_states, multimodal_position_ids)` at (pre-pick) model_adapters.py:4941 — running the rotary on the `inv_freq` device, then moving cos/sin back to the activation device; plus one device-consistency fix (`base_position_ids` on `input_ids.device`) |

Working tree on the branch is clean (`git status --short` empty).

## 2. Rebind into the arm64 conversion container

Pure Python, no `.so` rebuild: the patched checkout's `python/` tree was
bind-mounted over `/src/cactus-sdk/python` (the editable-install target) inside
**`cactus-arm64-convert:v221`** (the #236 retry image, engine
`libcactus_engine.so` + site-packages untouched).

* Preflight (`logs/preflight.log`, exit 0): in-container
  `md5sum(model_adapters.py) = 9bad62554b8dd7911aa8b2503a6a02d5` ==
  host checkout, 8 references to `_qwen35_rotary_embeddings` (1 def + 7 call sites).
* Host-side prerequisite: the `qemu-aarch64` binfmt registration had been lost
  (reboot); restored with `docker run --privileged --rm tonistiigi/binfmt
  --install arm64` before any `--platform linux/arm64` run.

## 3. Command D — the run

```
python3 -m cactus.transpile.hf_model --model-id Qwen/Qwen3.5-2B \
  --weights-dir /work/weights/qwen3.5-2b-cq4-conv \
  --artifact-dir /work/artifacts/qwen3.5-2b-text-fix \
  --task causal_lm_logits --components decoder,decoder_step \
  --low-memory-load --skip-execute --skip-reference-compare
```

Same command, same image, same weights (`qwen3.5-2b-cq4-conv` from the #236
retry) as the F1-failed baseline. Started `2026-09-30T07:35:24Z`, last
artifact written `2026-09-30T08:01:51Z` (~26.5 min wall under qemu-aarch64
emulation). No `--task multimodal_causal_lm_logits`, no `--image-file` (per
task instructions, F3 avoided).

### ACCEPT evidence — artifact bundle vs. baseline

| | #236 baseline (unpatched, test D) | #242 patched run |
|---|---|---|
| log after `capture_component_begin=decoder` | immediate `CapturePhaseError` | proceeded normally, no traceback anywhere in captured log |
| `artifacts/` | **0 files** (failed at capture of `decoder`) | **13 files, 127 MB** |
| `components/manifest.json` | absent | present, **written last at 08:01:51Z**, `component_order: ["decoder", "decoder_step"]`, `family: qwen3_5`, `task: causal_lm_logits` |
| `components/decoder/graph.cactus` | absent | **9,657,834 B** (+ raw_ir 18,577,768 B, optimized_ir 1,206,793 B) |
| `components/decoder_step/graph.cactus` | absent | **238,298 B** (+ raw_ir 18,479,263 B, optimized_ir 1,201,206 B) |
| top-level IR | absent | raw_ir.json 40,652,609 B, optimized_ir.json 2,595,687 B (+ per-component copies) |

Baseline error (verbatim, `retry-evidence/logs/_driver.log`, EXIT_CODE=1):

```
cactus.transpile.capture_pytorch.CapturePhaseError: [capture:export] torch.export
failed for model=_ComponentCaptureWrapper strict=True: Dynamo failed to run FX
node with fake tensors: call_function <built-in function matmul>(*(FakeTensor(...
size=(3, 1, 32, 1)), FakeTensor(..., device='meta', size=(3, 1, 1, 49))), **{}):
got RuntimeError('Unhandled FakeTensor Device Propagation for aten.bmm.default,
found two different devices cpu, meta')
  ... model_adapters.py:4941, in debug_forward
      position_embeddings = self.backbone.rotary_emb(hidden_states, multimodal_position_ids)
  ... transformers/models/qwen3_5/modeling_qwen3_5.py:149
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
```

### Focused reproduction (confirms the mechanism, independent of command D)

`qwen35_rotary_helper_check.py` run inside the same container
(`logs/helper_check.log`, exit 0), under `FakeTensorMode` to mirror capture:

```
BASELINE_ERROR: RuntimeError : Unhandled FakeTensor Device Propagation for
  aten.bmm.default, found two different devices cpu, meta
helper outputs: cos (1, 49, 64) meta | sin (1, 49, 64) meta
ROTARY_HELPER_OK: mixed-device bmm avoided, cos/sin rejoined to activation device
```

i.e. the pre-fix call reproduces F1's exact error signature, and the patched
helper avoids it.

## 4. Evidence gap (disclosed, not papered over)

The run's **post-capture stdout and process exit code were lost**:

* `logs/command_D_rotary_fix.log` (1078 B) and `logs/driver.log` (1248 B) are
  frozen at `07:36Z`, ending mid-stream at `capture_component_begin=decoder` —
  no `EXIT_CODE_D_ROTARY_FIX=` line, no `end=` timestamp, no trailing
  `hf_model.py` summary lines. Cause: the driving shell was killed while the
  container kept running, and `python`'s block-buffered stdout through `tee`
  was never flushed.
* The container was launched with `--rm`, so when it finished (~08:01:51Z) it
  was removed and its **exit code is unrecoverable** (`docker ps -a` shows no
  such container).

Consequences, stated exactly:

1. **F1 is cleared** — this is established by the captured log (it advanced
   well past the failure line without any traceback) plus the artifact bundle
   that can only be written by a successful capture+lowering of both
   components (the baseline, failing at capture, left zero files).
2. **No next error exists to record.** Both `graph.cactus` files and the
   manifest (written last) are present, and with `--skip-execute
   --skip-reference-compare` nothing substantive runs after the manifest; so
   neither F2 (`_update_linear_attn_mask` data-dependent guard,
   modeling_qwen3_5.py:1292–1304) nor F3 occurred before pipeline completion.
   This is derived from artifacts, **not from a traceback** — no traceback
   was produced/captured.
3. The exact process exit code (expected 0) is **not** recorded.

Mitigation applied: `run_command_d.sh` is hardened for any re-run —
`python3 -u` + `PYTHONUNBUFFERED=1` (unbuffered stdout), container **not**
`--rm`'d and given a fixed name (`cactus242-cmd-d`, recoverable via
`docker inspect -f '{{.State.ExitCode}}'`), and the exit code persisted to
`logs/exit_code.txt`.

## 5. Lint / checks

* `ruff check python/cactus/transpile/model_adapters.py`: **identical findings
  vs v2.2.1** (13 pre-existing E731, same codes/counts) — the cherry-pick adds
  no new lint issues.
* Syntax: `py_compile` of the patched file OK.
* No upstream unit test references `qwen3_5` / this adapter (grep over
  `python/tests` and `python/cactus/convert/tests`) — command D plus the
  focused helper check are the effective tests for this change.

## 6. Status / next

* F1 (qwen3_5 rotary under `--low-memory-load`) — **FIXED and verified**.
* Pipeline now completes both components on v2.2.1 + cherry-pick; the
  remaining #236 GO/NO-GO blockers (executable bundle / runtime parity, GDN
  kernels, `qwen3_5` export profile for the default path) are unchanged and
  out of scope for #242.
* If a captured exit code is required for the record, re-run
  `rotary-fix-evidence/run_command_d.sh` (hardened, ~27 min under qemu) — not
  re-run for this result because the artifact evidence is already decisive.
