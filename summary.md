# v2-clean Evaluation Summary

Status: **Complete** (Stage A + Stage B, all diagnostics run against the full
1603-question benchmark).

## What v2-clean is

The first trustworthy checkpoint produced by this project. Per `MCQ_DIAGNOSTIC_PLAN.md`
Phase 1, it was trained on `data/sources/gemma4/hf_dataset/train.jsonl` **only** —
3,669 conversation pairs (3,302 train / 367 val), zero MCQ-formatted data of any kind.
Verified by direct check: 0/3,669 pairs overlap with any question in
`benchmark-eval-workspace/benchmark_combined.jsonl` (the eval benchmark).

This matters because every prior checkpoint (the original v2 and v3) was trained on
100%/99.9% of the eval benchmark itself (see `MCQ_DIAGNOSTIC_PLAN.md`'s Critical Finding),
so their eval numbers measured memorization, not generalization. v2-clean is the first
checkpoint whose benchmark score means what it looks like it means.

Artifacts: `outputs/lora-adapter-v2-clean/final`, `outputs/merged-model-v2-clean`
(1137MB, single safetensors file). Training: 26:33, 1242 steps, default hyperparameters
(6 epochs, lr 2e-4, LoRA r=32/alpha=64).

## Stage A results

Both runs against the full 1603-question `benchmark_combined.jsonl`.

| Metric | Value |
|---|---|
| Generation accuracy | **42.55%** (682/1603 correct, 253 ambiguous) |
| Macro-avg accuracy | 42.69% |
| Logprob accuracy | **31.8%** (509/1603) |
| Mean entropy | 0.645 bits |
| Error buckets | correct 31.8%, confident_wrong 0%, uniform 13.0%, **biased_letter 55.2%** |
| Letter win distribution (logprob top choice) | A: 1237 (77.2%), B: 271, C: 31, D: 64 |

Baselines: random chance 25.0%, majority-class ('A' at 26.1% of gold answers) 26.1%.
v2-clean beats both by a wide margin (+16.4pp over majority on generation accuracy),
despite never seeing an MCQ during training.

### Reasoning

**Generation (42.55%) exceeding logprob (31.8%) is the opposite pattern from the
contaminated checkpoints** (where logprob > generation, e.g. v3's 83.4% logprob vs 88.3%
generation was close, but v2-original's 78.3% logprob vs 56.3% generation showed a large
gap the other way — logprob ahead). Here logprob is *lower* than generation. The likely
explanation: `calibration.py` scores the model's raw next-token distribution over the
literal letters A/B/C/D with an empty closed `<think>` block — a format the model has
*never seen* (no MCQ exposure at all), so that single-forward-pass measurement is probing
a completely out-of-distribution prompt shape. Full generation, by contrast, lets the
model produce free-form text before settling on an answer, which apparently recovers more
signal than the constrained single-token logprob read. This is a different phenomenon
from the contaminated checkpoints' "present-not-surfaced" story — there, the model *had*
been trained on that exact prompt shape (because the MCQ-formatted benchmark was in
training), so the logprob read was reliable; here it isn't, because the format itself is
novel to the model.

**The A-bias (77.2% of logprob top-choices) is the headline finding.** With zero
MCQ-format training, the model has no learned signal that says "weigh all four options
by content" — so when forced into the `<think>\n</think>\n` + letter-logit prompt shape,
it falls back on some generic prior (plausibly: A is often a reasonable first-listed
default completion in general text) rather than reasoning about the actual question.
This is *exactly* the failure mode the original project's cyclic-shift augmentation
(4 rotations per MCQ so the gold letter lands roughly evenly on A/B/C/D across training)
was built to fix — we're now seeing the problem cleanly, without it being tangled up in
the benchmark-contamination issue. When Phase 3/4 add a clean, balanced MCQ source back
in, this bias should correct the same way the original v2 fix corrected it — just without
training on the eval set this time.

## Stage B: logit lens (complete)

50 correct + 50 wrong questions sampled from v2-clean's own generation results, layer-by-layer
top-1 letter check across all 28 layers:

| Split | Median first-correct layer | Never-correct |
|---|---|---|
| correct | 1/27 | 17/50 |
| wrong | 28/27 (never, by definition of the metric) | 26/50 |

### Reasoning

**This is a fundamentally different pattern from the contaminated checkpoints.** There
(v2-original, v3), "wrong" questions were diagnosed as **present-not-surfaced**: the
correct letter was decodable from the residual stream by around layer 8 on average, but
the model's final output still picked the wrong one — a surfacing/routing failure, not a
knowledge gap. Here, the median first-correct layer for the wrong split is 28 (i.e. never,
for at least half the sample), and 26/50 never decode to the correct letter at *any*
layer. That's **knowledge genuinely absent**, not present-and-unsurfaced — an expected and
honest result for a model that was never shown MCQ-formatted natural farming facts, only
free-form conversations.

Also notable: even the *correct* split has 17/50 "never-correct" cases — meaning some
answers the model got right in generation were not obviously decodable via layer-wise
logit lens either. Combined with the A-bias finding above, a plausible read is that some
fraction of "correct" answers are correct by coincidence (the model's generic A-leaning
prior happening to land on the right letter for that particular question), not because it
reasoned through the content. This should be kept in mind when interpreting the 42.55%
generation accuracy — it's a real, uncontaminated number, but likely somewhat inflated by
letter-guessing luck on questions where A happens to be correct (26.1% of the benchmark).

## Stage B: linear probes (complete)

`diagnostics/linear_probes.py` against `unsloth/Qwen3-0.6B` (base) and
`outputs/merged-model-v2-clean` (finetuned) on the full 1603-question benchmark:

| Model | Peak knowledge accuracy | Peak layer | Max knowledge-prediction gap |
|---|---|---|---|
| base (Qwen3-0.6B) | 30.7% | 17 | −0.077 |
| v2-clean finetuned | **25.8%** | 24 | −0.131 |

Generation accuracy as measured by this script: 46.5% (differs slightly from `evaluate.py`'s
42.55% — this script defaults ambiguous/missing answers to letter 'A' rather than scoring
them as simply wrong, and A happens to be correct 26.1% of the time, which nudges the
number up; not a discrepancy worth chasing further).

Decision guidance printed by the script: *"Max knowledge probe accuracy = 0.258 (near
chance) — knowledge is ABSENT. Invest in domain data."*

### Reasoning

The base model's own peak (30.7% at layer 17) reproduced identically across two separate
runs this project (this run and the earlier v3-vs-base run) — a solid reproducibility
anchor for the pipeline itself.

**The genuinely notable and slightly uncomfortable result: v2-clean's finetuned peak
(25.8%) is *lower* than the untouched base model's (30.7%).** Conversation-only fine-tuning
appears to have made MCQ-relevant knowledge *less* linearly decodable from the residual
stream than it was before fine-tuning, not more. This isn't necessarily evidence the model
"knows less" about natural farming in any practical sense (generation accuracy is still
well above chance, and linear-probe decodability under this specific empty-think-block
prompt format is a narrow measurement) — but it's a real, honest data point, consistent
with the calibration finding that this prompt format is out-of-distribution for a model
with zero MCQ exposure. Combined with logit lens's "knowledge genuinely absent" verdict,
the overall Stage B picture for v2-clean is coherent: this checkpoint has real
conversational natural-farming knowledge, but close to none of it is organized in a way
that's legible through an MCQ-shaped probe — which is exactly what training with zero MCQ
examples would predict, and is *not* the same failure mode ("present-not-surfaced") that
motivated the original v2→v3 rationale-quality fix.

## What this means for Phase 3/4

- The conversations-only baseline transfers real natural-farming knowledge (42.55% is well
  above chance/majority), confirming the `hf_dataset/train.jsonl` expansion work
  (2865 → 3998 conversations) has genuine signal, not just noise.
- The dominant failure mode to fix in v3-clean is **lack of MCQ-format exposure entirely**
  (letter bias + near-chance linear decodability), not "present-not-surfaced." The original
  v2→v3 fix (swap thin rationale for genuine CoT) was solving a different problem than the
  one v2-clean actually has — v3-clean's design should prioritize simply *having* balanced,
  well-formed MCQ training data over rationale-quality nuance, though genuine rationale is
  still worth keeping once MCQ data exists at all.
- A clean, balanced (cyclic-shifted or otherwise letter-balanced) MCQ source, built without
  touching `benchmark_combined.jsonl`, is the natural next step — see Phase 3 in
  `MCQ_DIAGNOSTIC_PLAN.md`. As of 2026-07-10, `qmd` is working again (Node ABI rebuild) and
  its indexed collections map onto the same 9 topics as the benchmark categories
  (Permaculture, Organic_Farming, Natural_Fertilizers, Regenerative_Agriculture,
  Agroecology, Soil_Science_and_Microbiology, Sustainable_Water_Management,
  Integrated_Farming_Systems, Indigenous_and_Traditional_Knowledge — 53 to 143 files each),
  making it straightforward to generate category-balanced MCQs directly from source
  documents rather than from the benchmark itself.
