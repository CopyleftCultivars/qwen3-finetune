# F179 / M3 — #193 (P3.1 GATE): bf16 vs CQ4 recall gap + reserved-row leakage

Milestone #14 (Epic #179). GATE issue: measure the quantization gap explicitly.
Result: **the backbone bf16-vs-CQ4 recall gap is negligible (~0.00 recall@1),
but the baked reserved-row mechanism is unusable for a much more basic reason —
tied-embedding output contamination at 192-256 rows, independent of
quantization.** #194 (calibration pass) should be SKIPPED; a separate blocker
must be resolved first.

Methodology is honest about one hard engine limitation found up front.

## 1. `cactus_score_window` is a dead stub — no loss metric on the engine

`Model::score_tokens_window_logprob` (`cactus-engine/src/model.cpp:4310`) ignores
its arguments and returns `0.0`, `tokens_scored=0`. Verified with a **real
ARM-engine call** (docker arm64 + qemu, shipped bundle), not a source read:

```
score_window start=1 end=7 context=16: {"success": true, "logprob": 0, "tokens": 0}
```

Same #172 landmine class as M1's `cactus_submit` (#186): the symbol exists and
looks usable, but does nothing. So the CQ4 arm cannot be scored by token loss on
the engine. Instead the quantitative gap is measured in HF against the CQ4
weights reconstructed from the conversion pipeline (`cactus.convert qdq`), which
is exactly apples-to-apples with bf16 (same code path, only weights differ) and
avoids engine-kernel confounds. The real engine is still used for the required
injection + leakage `generate()` test (§4).

## 2. Oracle (the #203 workaround), stated explicitly

`bake_vectors()` is query-independent (#203), so there is no on-device selection.
This eval uses an **oracle**: the trained module's query-conditioned attention
selects the argmax slot per head, and only those baked rows are injected. **This
is not a deployable mechanism** — it exists solely to isolate the quantization
question #193 asks from the unsolved selection problem (#203).

## 3. Quantization gap — dynamic soft-prompt path (the real LPM mechanism)

`bake_vectors`/reserved rows are not touched on this path, so it isolates
*backbone* quantization. Same TTP recall proxy as #192 (fit val-persona slots with
the trained projection, then evaluate), applied identically to both arms. 24
persona-disjoint val personas in two disjoint K=12 slices, `fit_steps=20`,
`fit_max_length=96`, `max_length=160`.

| arm | recall@1 slice0 | recall@1 slice1 | aggregate (24) | own-mem NTP loss | own−offdiag margin |
|---|---|---|---|---|---|
| bf16 | 0.333 | 0.167 | **0.250** (6/24) | 2.37 / 3.29 | −0.208 / −0.035 |
| CQ4 (QDQ) | 0.250 | 0.250 | **0.250** (6/24) | 2.49 / 3.49 | −0.175 / −0.031 |

Chance = 0.083.

**Gap: Δrecall@1 = 0.000** (6/24 vs 6/24). CQ4 does cost a small, consistent
amount of likelihood: own-memory NTP loss +0.12 to +0.20 nats, and the
discriminative margin (own loss below other-persona loss) shrinks slightly. The
recall decision itself is unchanged. Absolute values are lower than #192's 0.339
because this uses a deliberately cheaper fit (20 steps, 96-token fit windows) to
fit the memory budget — the arms are matched, which is what the gate needs.

**→ Negligible gap: SKIP #194** (its own description: "skip entirely if the gap
is negligible"). Quantization calibration is not the binding constraint.

## 4. The GATE surprise — learned reserved rows break the tied output head

`model.embed_tokens.weight` is tied to `lm_head` (M0 #183). Writing learned
vectors into reserved rows therefore changes the output softmax denominator. M1
tested at most 6 rows at ≤4.0; the design needs 256. At 192 rows (K=24 val
personas × 8 heads, all learned, norm ≤4.0 via the #192 clamp):

| embedding rows 151670..151861 | no-memory NTP loss |
|---|---|
| untouched (original norm ~0.30) | 1.89 |
| 192 rows zeroed | 4.16 (unchanged vs patched baseline) |
| **192 learned rows** (norm mean 3.27) | **12.85** |
| 192 norm-matched *random* rows | 19.13 |
| 8 learned rows (one persona) | 5.03 |

The effect is the tied-head denominator: injected high-norm rows dominate the
softmax. It is independent of quantization (bf16 patched 12.85 vs CQ4 patched
13.74; QDQ adds only ~0.9).

**Real ARM engine, real `generate()`** (`scripts/lpm_cq4_engine_leak.py`,
p193-deploy bundle with the 192 learned rows + `<|memK_H|>` special tokens):

```
tokenize "<|mem0_0|>" -> [151670]           # transport works
[no memory]  The best fertilizer for tomatoes is ...  gen_ids all reserved  16/16
[1 mem]      ... + <|mem0_0|>                          gen_ids all reserved  16/16
[8 mem]      ... + <|mem0_0..7|>                       gen_ids all reserved  16/16
[no memory]  One fact about soil: ...                  gen_ids all reserved  16/16
```

Every generated token is a reserved ID, with **or without** memory in the prompt;
responses are empty strings (reserved IDs have no decoded text). This reproduces
and amplifies M1 #187's FAIL at design scale: the emitter is saturated, so the
memory signal is unobservable.

### Embedding-row fidelity (secondary)

The QDQ CQ4 round-trip preserves the learned rows well: cos(bf16, CQ4) mean
0.9951 / min 0.9929; relative L2 shift mean 9.9% / max 11.9%; norm mean 3.266 →
3.250, max 4.001 → 4.020 (2/192 rows marginally cross the 4.0 line). Full-table
relative MSE 2.5%. So the blocker is **not** row fidelity.

## 5. Gate decision

- bf16-vs-CQ4 **recall gap is negligible** → **#194 SKIPPED**.
- The reserved-row deployment mechanism is **blocked by tied-head contamination**
  at the required row count, not by quantization. Fixing it needs a real engine
  change (banned-token suppression at sampling, per #187/#188) or an untied head
  / different injection mechanism. This must be resolved before M5 deployment and
  before #197's LPM-vs-RAG comparison is meaningful (#203 also remains open).
- #195 (CQ sweeps) is unaffected by this blocker as a backbone-only question, but
  is a separate follow-on.

## Reproduction

```
# convert the patched model (reserved rows baked) and reconstruct the CQ4 HF model
VENV=.../cactus-sdk/venv/bin/python
$VENV -c "from cactus.convert.cli import main; main(['convert','--model','outputs/r179-m1/model-p193','--out','outputs/r179-m1/p193-weights','--bits','4','--calibration-manifest','.../calibration/manifest.json','--hessian-cache-in','.../distill-calibrated-weights','--force'])"
$VENV -c "from cactus.convert.cli import main; main(['qdq','outputs/r179-m1/control-weights','--out','outputs/r179-m1/p193-control-qdq','--dtype','bfloat16','--force'])"

# dynamic-path gap (24 val personas in 2 slices, per arm)
python scripts/lpm_quant_gap.py dynamic --arm bf16 --n-personas 12 --persona-offset {0,12} ...
python scripts/lpm_quant_gap.py dynamic --arm cq4  --cq4-model outputs/r179-m1/p193-control-qdq ...

# real ARM engine leakage
docker run --rm --platform linux/arm64 -v outputs/r179-m1/p193-deploy:/work/bundle:ro \
  -v scripts/lpm_cq4_engine_leak.py:/work/engine_leak.py:ro \
  cactus-arm64-convert:built python3 /work/engine_leak.py
```

Machine-readable: `results/F179_M3_QUANT_GAP.json`. Conversion logs:
`/tmp/opencode/p193-convert.log`, `/tmp/opencode/p193-qdq.log`.
