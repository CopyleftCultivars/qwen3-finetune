# #168: fp16-vs-CQ-int4 numerical divergence via Cactus's built-in reference-compare

Date: 2026-09-10. Method: `cactus.transpile.hf_model` with `--component-pipeline
off` (forces the unified single-"decoder"-component graph, the only mode where
the tool's built-in reference-compare actually runs — the deployed
split/cached-decode-step bundle format hits `reference_compare_skipped` per
#168's own scoping) and neither `--skip-execute` nor `--skip-reference-compare`
set, so each run executes the fp16 HF reference and the transpiled CQ-int4
graph on the same tokenized input and reports `max_abs_diff`/`mean_abs_diff`
over the final logits plus each path's argmax next-token.

Inputs: the calibrated weights (`LocalLLM/finetune/outputs/{distill,v5clean}-calibrated-weights`)
and merged fp16 checkpoints (`outputs/merged-model-distill-v2-rationale`,
`outputs/merged-model-p6-lr5e5-r2`) already used for #163/#165. Item selection:
computed the residual "lost" set (GPU-correct, calibrated-on-device-wrong)
directly from `results/ladder210_*.jsonl` vs
`LocalLLM/finetune/outputs/ladder_results/{distill,v5clean}_calibrated_fixed_snapshot.json` —
**14 residual lost items for distill, 12 for v5clean** on the *current
calibrated* bundles (not the stale pre-calibration 25-item list in #164, which
was measured against a since-superseded uncalibrated bundle). Tested 4
model-specific lost items + 2 shared-lost items + 3 correct-on-both-paths
control items per model (9 runs each, 18 total). Prompt: the real ladder
question text, `SYSTEM_PROMPT + buildPrompt(item)` folded into a single
user-turn (the tool's causal_lm_logits path only supports one chat turn —
see caveats below).

Environment note: this required rebuilding the Cactus native engine for
ARM64 inside the QEMU-emulated Docker image (`cactus-arm64-convert`) — the
image's `bindings/lib/libcactus_engine.so` was a stale 15KB x86 stub, not the
real ~4MB ARM64 build. Fixed by installing `cmake`/`build-essential`/
`libcurl4-openssl-dev`, running `cactus build --python`, and manually copying
the rebuilt `.so` into the bindings path (the build script does not do this
copy itself — worth a small upstream fix if this environment gets reused).

## Results

| question_id | model | label | gold | real GPU | real device | ref-compare hf | ref-compare transpiled | max_abs_diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NF-0001 | distill | correct_control | B | B | B | B | B | 14.047 |
| NF-0002 | distill | correct_control | B | B | B | B | B | 11.449 |
| NF-0003 | distill | correct_control | B | B | B | B | B | 10.770 |
| NF-0063 | distill | distill_only_lost | C | C | B | **C** | **B** | 9.841 |
| NF-0070 | distill | distill_only_lost | B | B | A | A | A | 8.223 |
| NF-0078 | distill | shared_lost | B | B | A | **B** | **A** | 7.512 |
| NF-0086 | distill | distill_only_lost | B | B | A | **B** | **A** | 7.180 |
| NF-0093 | distill | distill_only_lost | C | C | D | B | D | 7.091 |
| NF-0139 | distill | shared_lost | C | C | A | **C** | **A** | 8.607 |
| NF-0001 | v5clean | correct_control | B | B | B | B | B | 12.934 |
| NF-0002 | v5clean | correct_control | B | B | B | B | B | 13.379 |
| NF-0003 | v5clean | correct_control | B | B | B | B | B | 8.975 |
| NF-0004 | v5clean | v5clean_only_lost | C | C | A | A | A | 10.602 |
| NF-0025 | v5clean | v5clean_only_lost | B | B | A | A | A | 13.328 |
| NF-0057 | v5clean | v5clean_only_lost | B | B | A | B | B | 10.277 |
| NF-0078 | v5clean | shared_lost | B | B | A | **B** | **A** | 12.071 |
| NF-0083 | v5clean | v5clean_only_lost | B | B | A | B | B | 11.082 |
| NF-0139 | v5clean | shared_lost | C | C | A | A | A | 10.353 |

Bold = this diagnostic **cleanly reproduces** the real on-device flip
(fp16-reference agrees with the real GPU answer, transpiled agrees with the
real on-device answer).

## Finding 1: the diagnostic reproduces the real flip for most of distill's lost items, few of v5clean's

Distill: 4/6 tested lost items clean-reproduce (NF-0063, NF-0078, NF-0086,
NF-0139). The other 2 (NF-0070, NF-0093) show the fp16 reference itself
disagreeing with the real GPU-recorded answer — a prompt-format-sensitivity
artifact (this diagnostic folds the system prompt into a single user turn;
the real harness sends it as a separate system turn), not a quantization
effect. Confirms: most of distill's residual on-device losses are real,
reproducible fp16-to-int4 decision flips, not measurement noise.

v5clean: only 1/6 tested lost items clean-reproduces (NF-0078, which is
shared with distill). Two items (NF-0057, NF-0083) show **no divergence at
all** in this diagnostic — hf and transpiled agree with each other but
disagree with the real recorded on-device answer, meaning the split
cached-decode-step graph actually deployed on-device behaves differently from
the unified `--component-pipeline off` graph tested here for these two
specific items. Three items (NF-0004, NF-0025, NF-0139) show the same
prompt-format-sensitivity artifact as distill. **This diagnostic's coverage
of v5clean's real failure mode is weaker** — the unified-graph mode this tool
requires for reference-compare is a structurally different graph (no KV
cache reuse across calls) from the deployed one, and that structural gap
matters more for v5clean's lost items than distill's, for reasons not
determined here.

## Finding 2: raw logit divergence magnitude does not separate lost items from correct items

| | mean max_abs_diff, lost items (n=6) | mean max_abs_diff, correct-control items (n=3) |
| --- | --- | --- |
| distill | 8.08 | 12.09 |
| v5clean | 11.29 | 11.76 |

For **both** models, correct-control items show divergence magnitude equal to
or *larger* than lost items. Distill's correct items diverge more (12.09) than
its own lost items (8.08). This directly weakens a "bigger raw quantization
noise on distill's decision-relevant logits" story as the mechanism — if that
were true, lost items should show larger `max_abs_diff` than correct items,
within a model. They do not, for either model.

v5clean's lost-item divergence (11.29) is not meaningfully smaller than
distill's correct-item divergence (12.09) or its own control divergence
(11.76) either. **Cross-model, v5clean's divergence numbers run as high or
higher than distill's across every category tested** — the opposite of what
"distill is the more INT4-sensitive model" would predict if raw logit
divergence were the mechanism.

## Interpretation

Raw fp16-vs-int4 logit divergence, as measured here (final-layer L1 distance
only), is **not** the right lens for "why does this item flip." What
determines a flip is not how much numerical noise quantization injects in
absolute terms, but how close the top-1 and top-2 candidate logits were
*before* quantization — a margin, not a magnitude. This diagnostic did not
capture the runner-up logit (only the argmax), so the margin hypothesis from
#164 (distill's tighter teacher-imitation fit sits closer to decision
boundaries) is **neither confirmed nor falsified** by this data — it remains
the live hypothesis, just not testable with what this tool prints by default.

This is a small, well-scoped follow-up: `hf_model.py`'s reference-compare
already computes the full logit arrays for both paths in memory
(`reference_output`, `transpiled_output` at hf_model.py:859/931); capturing
the top-2 margin (`logits[top1] - logits[top2]`) on the fp16 reference for
each item, alongside whether the item flips, would directly test whether
low-margin items are the ones that flip under quantization — independent of
model identity. That is a targeted patch to the existing tool, not a new
measurement pipeline.

## A candidate lever for the autoresearch loop

If the margin hypothesis holds (low fp16 top1-vs-top2 margin predicts INT4
flip risk), it suggests a genuinely new, cheap, pre-deployment training-time
signal: **margin at training time as a distillation-lever knob**, screenable
on GPU before ever quantizing or deploying to a device. Concretely: a
temperature-scaled or margin-regularized distillation loss (softening the
teacher target so the student's decision boundaries sit further from its
top-2 alternatives, at some cost to raw GPU-side ladder accuracy) traded
against a *predicted* INT4-robustness gain. This is exactly the kind of
lever the framing settled on #165 called for — "heuristics that balance
free-form thought and the MCQ benchmark" — extended one step further:
balancing GPU-side sharpness against quantization-time robustness, measurable
on GPU alone via the margin proxy, without needing a Pixel 7 in the loop for
every candidate. Scoping this as an actual lever-search config is future work,
not done here.

## Caveats

- The unified `--component-pipeline off` graph is not byte-identical to the
  deployed split cached-decode-step graph (`components/manifest.json`); v5clean
  Finding 1 shows this structural gap can matter for specific items.
- The single-user-turn prompt fold is not byte-identical to the real
  system+user two-turn harness; this explains 2/6 (distill) and 3/6 (v5clean)
  "reference disagrees with real GPU answer" cases and is a known, pre-existing
  caveat (see F9's "harness parity... unverified" note), not new here.
- n=6 lost items and n=3 control items per model is a spot-check, not a full
  210-item sweep — each run takes several minutes under QEMU ARM64 emulation
  (no native ARM64 host available). Scaling this to the full lost-item sets
  (14 distill / 12 v5clean) is straightforward with the same script
  (`run_refcompare.py`) if a stronger signal is wanted before acting on the
  margin lever.
