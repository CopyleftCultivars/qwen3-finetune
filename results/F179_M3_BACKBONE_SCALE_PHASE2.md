# F179 / M3 — #231 Phase 2: Qwen3-1.7B LPMConfig profile (d_model=2048) + encoder forward-shape smoke

Parent #179; sibling to #216/#222/#223/#230; follows Phase 1
(`results/F179_M3_BACKBONE_SCALE_PHASE1.md`, PASS). Scope: **config + shape
validation only** — no training (Phase 3), no re-run of the #228/#230 effect
probe (Phase 4), no engine/bundle needed.

**Verdict: PASS.** The 1.7B encoder stub forward returns `[B, 2048]`; the
`LPMConfig.qwen3_1_7b()` module forward/bake returns `[B, H, N, 2048]`; the
reserved-ID range is confirmed unreachable on the 1.7B tokenizer. All 21 existing
LPM tests still pass, and the 0.6B default is unchanged.

## 1. Code changes (both blockers resolved)

### `lpm/config.py` — multi-width validation + 1.7B profile

The single-global-constant reject is replaced by a set of supported deployment
widths, so the 0.6B default is untouched and 2048 validates:

```python
QWEN3_0_6B_D_MODEL = 1024  # Qwen3-0.6B hidden_size (current M3 champion)
QWEN3_1_7B_D_MODEL = 2048  # Qwen3-1.7B hidden_size (#231)
DEFAULT_D_MODEL = QWEN3_0_6B_D_MODEL
SUPPORTED_D_MODELS = (QWEN3_0_6B_D_MODEL, QWEN3_1_7B_D_MODEL)
```

`LPMConfig.validate()` now checks `self.d_model not in SUPPORTED_D_MODELS`
instead of `!= DEFAULT_D_MODEL`. Profile constructors added:
`LPMConfig.qwen3_0_6b()` and `LPMConfig.qwen3_1_7b()` (same H=8 / N=32 / d_mem=256
as the paper default; only `d_model` differs). `to_dict`/`from_dict` are
unchanged — 2048 is just a value of the existing `d_model` field.

### `lpm/encoder.py` — 1.7B EncoderSpec

- `MODEL_PATH_QWEN3_1_7B = "Qwen/Qwen3-1.7B"` (HF repo id; resolves via the HF
  cache or a local path) alongside `MODEL_PATH_QWEN3_0_6B = DEFAULT_MODEL_PATH`.
- `encoder_spec_qwen3_1_7b(n_layers, pooling, dtype)` factory.
- `load_encoder` now accepts an HF repo id as well as a local path (it previously
  hard-required a filesystem path via `Path.exists()`), so the 1.7B backbone can
  be referenced by id.
- `Qwen3EarlyLayerEncoder` itself needed no change: it derives `hidden_size` from
  the backbone config, so the 1.7B stub produces R^2048 for free.

**Layer-count choice (N=6, documented rather than copied blindly).** Both Qwen3-0.6B
and Qwen3-1.7B have **28 layers** (Phase 1 confirmed; only width differs), so "same
fraction" and "same absolute count" coincide — there is no fraction-vs-absolute
trade-off to make. N=6 keeps (a) the validated #189/#212 stub depth, and (b) the
engine's `CACTUS_QWEN_EMBED_LAYERS=6` early-layer exposure (the runtime selector
must read the same early-layer representation the encoder was built against), and
(c) exact 0.6B↔1.7B comparability for Phase 4. Changing depth would require an
engine embedding-layer flag change and break the comparison. 6/28 = 0.214 for
both backbones.

## 2. Smoke results (host/CPU, no engine)

`scripts/lpm_backbone_scale_phase2_smoke.py`, `torch.manual_seed(0)`.

| check | value |
|---|---|
| encoder stub | `Qwen3EarlyLayerEncoder` over `Qwen/Qwen3-1.7B`, 6 of 28 layers |
| encoder hidden size | **2048** |
| encoder output shape | **[4, 2048]** ✓ |
| encoder frozen / trainable params | 613,182,976 / 0 |
| LPM config | `d_model=2048`, `H=8`, `N=32`, `d_mem=256`, `total_slots=256` |
| `forward()["prompts"]` | **[4, 8, 2048]** ✓ |
| `forward()["attention"]` | [4, 8, 32], row sums 1 (±1.2e-7) ✓ |
| `forward()["projected_slots"]` | **[4, 8, 32, 2048]** ✓ |
| `bake_vectors()` | **[4, 8, 32, 2048]**, max abs diff vs forward = 0.0 ✓ |
| projector dims | `w_q.in=2048`, `w_v.out=2048`, `mlp in/out=2048`, `out_ln=2048` |
| module trainable params | 11,328,000 |

The module-trainable count is deliberately dominated by the `d_model`-scaled
projector (`w_q`, `w_v`, `mlp`, `out_ln`); the per-persona slot table
(`H*N*d_mem = 65,536`/persona) is unchanged from 0.6B.

## 3. Reserved-ID range — verified, not assumed

The 151670..151935 range was derived from the 0.6B tied embedding table. Phase 1
confirmed the 1.7B also has `vocab_size=151936`, `tie_word_embeddings=true`, but
the tokenizer reachability was re-checked empirically on the 1.7B tokenizer:

| check | value |
|---|---|
| tokenizer class / `len` | `Qwen2Tokenizer` / 151,669 |
| highest id present in the vocab | **151,668** (one below `RESERVED_ID_START=151670`) |
| reserved ids (266) mapping to a token | 0 (`convert_ids_to_tokens` → `None` for all 266) |
| reserved ids with non-empty decode | 0 |
| reserved ids that round-trip (decode→encode) to themselves | 0 |
| reserved ids appearing when encoding the 4 smoke texts | none (text max id 84,071) |
| 0.6B cross-check (highest vocab id) | 151,669 < 151,670 → also unreachable |

So the reserved range assumption holds on **both** tokenizers. The config-level
mapping is width-independent: 256 slots → 151670..151925 (`reserved_id_map`),
leaving 10 spare IDs in the 266-ID pool — identical to 0.6B.

## 4. Regression + determinism

- `python -m unittest tests.test_lpm_module`: **21/21 pass** (the config
  validation change does not break the oversized-slot-pool rejection, the
  bijective reserved-ID map, `to_dict`/`from_dict`, or the module shape tests).
  The 0.6B default (`LPMConfig()`) still validates at d_model=1024.
- Smoke re-run: byte-identical output, sha256
  `ca3e0e6af91472d19a50283dd578df3ba9e7bb4aab4d4d03ade18ed07b4b124f`
  (`results/F179_M3_BACKBONE_SCALE_PHASE2_SMOKE.json`).

## 5. Go/no-go for Phase 3

**GO.** Both Phase-2 blockers are resolved cleanly and additively:
- `LPMConfig.qwen3_1_7b()` builds and validates at d_model=2048; the projector and
  `out_ln`/`mlp` come out at the right width.
- The 1.7B encoder stub produces R^2048 with the frozen-first-6-layers contract
  intact.
- No training, no engine change, no 0.6B regression.

Phase 3 (a #192-equivalent training pass at d_model=2048 against the 1.7B
backbone, using the legacy 6-component bundle decision from Phase 1) can proceed.
The main cost unknown is still per-step time on the 1.7B forward, to be measured
in Phase 3 (the issue asks for a real timing, not the 0.6B 0.156 GPU-h estimate).

## 6. Reproduce

```bash
cd [LOCAL_PATH]
HF_HUB_DISABLE_PROGRESS_BARS=1 \
  [LOCAL_PATH] scripts/lpm_backbone_scale_phase2_smoke.py
# regression:
[LOCAL_PATH] -m unittest tests.test_lpm_module
```

Artifacts: `results/F179_M3_BACKBONE_SCALE_PHASE2_SMOKE.json`; script
`scripts/lpm_backbone_scale_phase2_smoke.py`; code changes in `lpm/config.py`,
`lpm/encoder.py`. The 1.7B backbone resolves from the HF cache
(`~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B`); no engine/bundle needed
for this phase.
