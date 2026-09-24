# F179 / M3 — #231 Phase 3: train an LPM module against Qwen3-1.7B (d_model=2048)

Parent #179; follows Phase 1 (bundle/engine plumbing, PASS) and Phase 2
(config + encoder shape smoke, PASS). Scope: **training only.** No engine, no
bundle, no CQ4 pass, no #228/#230 effect-probe re-run (that is Phase 4).

**Verdict: PASS — a loadable 1.7B LPM checkpoint exists, the recall proxies are
meaningfully above chance (6.6×/8.8×), and real per-step timing is measured.
GO for Phase 4.** Honest caveat up front: at the recall-proxy level the 1.7B run
is **comparable to, not better than,** #192's 0.6B run — the decisive backbone
question is Phase 4's generation-legibility probe, which this phase does not
answer.

## 1. Recipe (identical to #192, one memory-only deviation)

`scripts/lpm_train.py` gained `--backbone {0.6b,1.7b}` (default `0.6b`, so the
#192 path is unchanged — verified below) and an `--encoder-chunk` knob.

| setting | value |
|---|---|
| backbone | `Qwen/Qwen3-1.7B` (`--backbone 1.7b`), `LPMConfig.qwen3_1_7b()` |
| d_model / H / N / d_mem | 2048 / 8 / 32 / 256 |
| subset | same as #192: `include_first_chunk=False`, ≥3 cross-chunk → **273 personas, 1488 records**, 1215 fit + 273 held-out probes |
| steps / batch / max_length / lr | 4000 / 2 / 192 / 1e-3 |
| regularizers | `lambda_mag=0.01`, `mag_penalty_mode=max`, `clamp_memory_norm=true` |
| grad checkpointing | on (as #192) |
| eval | every 500: val-personas 40 (fit 40 steps) + within-persona probes 48 |
| seed | 0 |
| module params | 28,957,184 (273×65,536 slots + 11.3M shared projection) |

**The one deviation: `--encoder-chunk 64`.** With batch 2 the run OOM'd — not in
training, but in the **encoder precompute**: `encoder.encode_texts` forwarded all
1215 fit prompts as a single batch, and the 2048-wide 6-layer encoder's attention
matrices (`[1215, 16, 192, 192]` fp32 ≈ 2.9 GB) plus the 1.7B backbone exceeded
the 11.6 GB GPU. `encode_texts_chunked` mini-batches the (no-grad, deterministic)
encoder pass; masked mean-pooling makes the pooled vectors chunk-size-invariant.
`--encoder-chunk 0` (the default) reproduces the original single call, so the
0.6B path is byte-for-byte unchanged. Training batch stayed at #192's **2**
(after the encoder fix, peak training memory was 5.9 GB in the probe), so the
recipe is otherwise untouched.

## 2. Cost (the unknown Phase 2 flagged)

| metric | #231 Phase 3 (1.7B) | #192 (0.6B) |
|---|---|---|
| wall clock | **854.5 s** | 560.6 s |
| **GPU-hours** | **0.237** | 0.156 |
| seconds/step | **0.214** | 0.140 |
| peak GPU | **8.39 GB** | 5.18 GB |
| total samples seen | 8000 | 8000 |

The 1.7B forward is ~1.5× the 0.6B wall clock and ~1.6× the peak memory for the
same 4000×2 recipe — a genuinely modest cost, not the 2.8× width naive scaling,
because batch 2 at 192 tokens with gradient checkpointing keeps activations small.
Comfortably inside a single 12 GB GPU. No further batch-size reduction was needed.

## 3. Metrics

| metric | #231 Phase 3 (1.7B) | #192 (0.6B) |
|---|---|---|
| NTP loss | 5.23 (step 1) → **2.77** final | 9.14 → 2.92 |
| projected norm max | **4.00, 0 leakage-watch triggers** | 4.00, 0 |
| projected norm mean | 1.72 → **3.91** (near cap) | 1.20 → 3.65 |
| entropy | 3.47 → **1.44** (target 1.386; penalty fires late) | 3.47 → 1.55 |
| slot grad norm | 0.021 … 0.983 (final 0.156) | 0.031 … 0.124 |
| projection grad norm | 0.0015 … 0.053 (final 0.006) | 0.0024 … 0.016 |

The clamp holds exactly as designed: `projected_norm_max` reaches 4.0 and never
exceeds it (0 leakage-watch triggers), and the mean rises to 3.91 — the memory is
pinned at the safe-zone cap, same qualitative picture as #192 (3.65).

## 4. Recall proxies (0.6B vs 1.7B, same protocol)

| eval step | val recall@1 (chance 0.042) | within-persona probe recall@1 (chance 0.021) |
|---|---|---|
| 500 | 0.1667 | 0.0208 |
| 1000 | 0.2083 | 0.1458 |
| 1500 | 0.2500 | 0.1875 |
| 2000 | 0.2917 | 0.1250 |
| 2500 | 0.2917 | 0.1458 |
| 3000 | 0.3333 | 0.2292 |
| 3500 | 0.3750 | 0.3125 |
| 4000 | 0.2917 | 0.2917 |
| **mean** | **0.2761 (6.6× chance)** | **0.1823 (8.8× chance)** |
| #192 0.6B mean | 0.339 (8.1× chance) | 0.140 (6.7× chance) |

Both proxies are clearly above chance and mirror #192's ~7–8× bar, so the module
learns persona-discriminative memory and the checkpoint is a valid Phase-4 input.
**Honest read:** the 1.7B val (persona-disjoint transfer) mean is *lower* than
0.6B (0.276 vs 0.339) while the within-persona probe mean is *higher* (0.182 vs
0.140) — i.e. no recall-level backbone win. These are noisy proxies (batch 2, 24
val personas) and, per #230, recall was never the binding constraint anyway; the
binding question is generation legibility (Phase 4).

## 5. Checkpoint

`outputs/lpm-persona-memory-231-p3/checkpoint.pt` (gitignored, 115.9 MB).
Verified round-trip via `lpm.checkpoint.load_lpm_checkpoint` (strict load):

- `config.d_model = 2048`, H=8, N=32, d_mem=256
- `slot_memory.slots.shape = [273, 8, 32, 256]`, 273 personas, `step = 4000`
- `bake_vectors([0]).shape = [1, 8, 32, 2048]`, norm max **4.0**
- reserved-ID map: 256 ids, 151670..151925
- `summary.backbone = "1.7b"`, `summary.d_model = 2048`

## 6. Go/no-go for Phase 4

**GO.** All three Phase-4 preconditions are met:
1. a loadable 1.7B checkpoint with correct slot shape and clamped norm;
2. recall proxies meaningfully above chance (val 6.6×, probe 8.8×);
3. real per-step timing (0.214 s/step; 0.237 GPU-h for the full recipe).

Phase 4 = the #228/#230 S0-static + oracle effect probe (post-transform bake,
inputs_embeds injection via the unmodified engine) at d_model=2048, against the
**same persona** as 0.6B, to answer whether persona-distinctive content appears
at 1.7B where 0.6B was generic. The Phase-1 legacy 6-component bundle is the
deployment target for that.

## 7. Reproduce

```bash
cd [LOCAL_PATH]
PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1 \
[LOCAL_PATH] scripts/lpm_train.py \
    --backbone 1.7b --steps 4000 --batch-size 2 --max-length 192 --encoder-chunk 64 \
    --eval-every 500 --val-personas 40 --val-fit-steps 40 --probe-personas 48 \
    --out-dir outputs/lpm-persona-memory-231-p3 \
    --curve-out results/F179_M3_BACKBONE_SCALE_PHASE3_TRAINING_CURVES.json

# 0.6B path unchanged (sanity, 3 steps):
#   ... scripts/lpm_train.py --steps 3 --eval-every 1000000
```

Artifacts: `results/F179_M3_BACKBONE_SCALE_PHASE3_TRAINING_CURVES.json`,
`results/F179_M3_BACKBONE_SCALE_PHASE3.md`, script diff in
`scripts/lpm_train.py`; checkpoint under `outputs/lpm-persona-memory-231-p3/`.

## 8. Notes

- No CQ4 quantization pass (mirrors #193; that would be Phase 3.5/4 if the
  effect probe warrants it).
- The only code additions are `--backbone`, `--encoder-chunk`,
  `encode_texts_chunked`, and provenance fields in the summary; `lpm/config.py`
  and `lpm/encoder.py` were already changed in Phase 2. The 0.6B default was
  exercised end-to-end (3 steps; d_model=1024, encoder_chunk=0, same subset) to
  confirm the path is intact.
