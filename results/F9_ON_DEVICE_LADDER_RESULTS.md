# F9 (#146): on-device ladder score, Pixel 7

Date: 2026-09-07. Benchmark:
`benchmark-workspace/benchmarks/natural_fertilizers/natural_fertilizers_benchmark_v1_graded.jsonl`
(210 items). Protocol ported from `benchmark-workspace/evaluate.py`
(`SYSTEM_PROMPT`, `PROMPT_TEMPLATE`, `extract_answer` regex reproduced verbatim
in Dart) by `LocalLLM/natural_farming_chat/integration_test/ladder_benchmark_test.dart`,
run as a Flutter `integration_test` against the real installed app
(`ModelService`/`CactusLM`, same code path as production) on a physical Pixel 7
over USB, greedy (`temperature: 0`), `maxTokens: 1024`. Both models converted
to the Cactus CQ int4 `components/manifest.json` format via the fixed ARM64
conversion pipeline (chainlink #162).

## The result

| Model | GPU score (this repo) | On-device score (Pixel 7) | Wilson 95% (on-device) | Δ GPU→device |
| --- | --- | --- | --- | --- |
| v5clean (champion) | 0.7810 | **0.8095** (170/210) | 0.7510–0.8569 | **+2.9pp** |
| distill-v2-rationale (candidate) | 0.8714 | **0.7905** (166/210) | 0.7305–0.8401 | **−8.1pp** |

Two findings:

**1. The distill candidate's GPU score does not survive deployment.** 0.8714 is
outside its own on-device 95% CI (0.7305–0.8401) — this is a real, measured
drop, not n=210 sampling noise. This is exactly the risk the epic's own risk
register flagged (risk C.4, `LADDER_AND_DISTILLATION_PLAN.md`: "on-device score
may fall below GPU score") and is the reason F9 was scoped as a required gate
rather than a formality.

**2. On-device, v5clean and distill are statistically indistinguishable.**
170/210 vs 166/210, two-proportion z = 0.488 (not significant at any
conventional threshold). The +9.0pp GPU-side edge that got distill accepted
under the #160(B) policy has effectively evaporated on the actual deployment
target. Neither model's on-device CI excludes the other's point estimate.

v5clean, by contrast, moved in the other direction (+2.9pp, GPU to on-device),
consistent with its GPU number simply being noisy in that direction rather
than the champion actually improving under quantization.

## Confound checked and ruled out

The GPU-side protocol (`LADDER_210Q_RESULTS.md`) ran think-off; this harness
does not explicitly disable thinking, and distill was trained with rationale
("think ON") data (F5, #142), raising a real risk that `maxTokens: 1024` was
truncating distill's reasoning before it reached an answer letter -- which
would make this a harness bug, not a real finding. Checked directly against
both result sets: **zero** null-extracted answers, **zero** unclosed
`<think>` tags, and average response length 36 (v5clean) / 21 (distill)
characters -- both models answered directly with no visible reasoning trace
under this app's default init params. Truncation is not the explanation.

## What is NOT yet ruled out

- **Harness parity with `evaluate.py` is not byte-identical.** The system/user
  message plumbing goes through Cactus's own `ChatMessage`/chat-template
  path rather than `evaluate.py`'s raw string formatting. If that template
  differs from the GPU harness's, it applies to both models equally, so it
  does not by itself explain the *asymmetric* drop, but it is an unverified
  assumption behind the absolute numbers.
- **Quantization-sensitivity mechanism is not identified.** The observed
  pattern (distill drops, champion doesn't) is consistent with the
  distilled candidate's decision boundaries being less robust to INT4
  quantization than v5clean's -- plausible given it was trained to imitate a
  much larger teacher's fine-grained outputs -- but that is a hypothesis, not
  something measured here.

## Replication (2026-09-07)

Both models were re-run on-device, full 210 items, independent app
install/model-push each time:

| Model | Run 1 | Run 2 | Items that differ |
| --- | --- | --- | --- |
| v5clean | 170/210 (0.8095) | 170/210 (0.8095) | **0 / 210** |
| distill-v2-rationale | 166/210 (0.7905) | 166/210 (0.7905) | **0 / 210** |

**Both models are fully deterministic on this device at `temperature: 0`** --
not just the same aggregate score, byte-identical predictions on every one of
210 items across independent runs. This rules out run-to-run noise entirely
(the #160-era concern that run-to-run SD could dominate item-sampling SE does
not apply here: there is no run-to-run component to measure). The gap between
v5clean and distill is a fixed, exactly-reproducible property of these two
conversions, not a fluke draw. The one remaining source of uncertainty is
item-sampling -- would a different 210-question draw from the same domain
show the same gap -- which is what the Wilson CIs already bound, and those
still overlap substantially.

## Recommendation

Do not promote distill-v2-rationale to champion on this evidence. The
promotion basis in #145/#160 was GPU-only; the one number this epic was built
to produce -- what actually happens on the Pixel 7 -- shows no measured
advantage over v5clean (confirmed deterministic, not a noisy single run) and
a real, reproducible regression from the candidate's own GPU score. Keep
v5clean as champion.

## CORRECTION (2026-09-07): the first calibrated-conversion results below were invalid

The originally-reported calibrated scores (v5clean 168/210, distill 168/210 --
identical) were **measured against a broken bundle and are void.** Calling
`cactus.transpile.hf_model` directly with `--weights-dir` and `--artifact-dir`
pointing at two *different* directories produced a graph-only bundle missing
`config.txt` and all tokenizer/config metadata. `ModelService.downloadModel()`'s
"already downloaded" check (`config.txt` existence) therefore failed silently
on every one of those runs, and the app fell through to downloading the
stock GitHub-hosted v5-ios bundle instead of the intended calibrated one
whenever the device had network -- explaining why both "different" models
landed on the identical 168/210. Root cause: `--weights-dir` and
`--artifact-dir` must point at the *same* directory (matching how the
standard `cactus convert` pipeline always runs them in-place); a separate
`--artifact-dir` never receives the source directory's metadata files.
Fixed by rebuilding both bundles in-place and re-running. Corrected results
replace the invalid ones below.

### Corrected calibrated results

| Model | GPU | On-device (uncalibrated) | On-device (calibrated, corrected) |
| --- | --- | --- | --- |
| v5clean | 0.7810 | 0.8095 [0.7510, 0.8569] | 0.7810 [0.7202, 0.8316] (164/210) |
| distill-v2-rationale | 0.8714 | 0.7905 [0.7305, 0.8401] | **0.8381 [0.7823, 0.8818]** (176/210) |

**Calibration substantially helped distill after all.** 79.05% -> 83.81%
(+4.76pp, 166->176/210), closing most of the -8.1pp gap to its own GPU score
down to -3.3pp. It now clearly leads v5clean-calibrated (+5.7pp, two-proportion
z=1.49 -- suggestive, not quite significant at the conventional 0.05 level).
v5clean's calibrated score lands exactly on its own GPU score (0.7810 both),
a clean result.

This reverses the original (invalid) conclusion that fix (1) barely helped.
The corrected picture supports "data-free conversion" as a real, substantial
contributor to distill's on-device drop after all -- calibration recovered
most, though not all, of the gap. The residual -3.3pp (and whatever
#166/#167's constrained-decoding experiment finds) is what remains to
explain.

## Fix attempt: calibrated INT4 conversion (2026-09-07, #165) -- SUPERSEDED, see correction above

Root cause (#164, `results/QUANT_SENSITIVITY_ROOTCAUSE_164.md`) identified the
CQ conversion as data-free (empty Hessian metadata) and hypothesized this as
a contributing cause. Both models were reconverted with a real GPTQ-Hessian
calibration pass (256 contamination-checked agronomy MCQ samples from the F5
teacher-data pool, `LocalLLM/finetune/calibration/`) via `cactus.convert.cli`
called directly (the documented `cactus convert` CLI never wires
`--calibration-manifest` through -- confirmed empty Hessian metadata in every
prior bundle, this repo included). Hessian metadata now shows real per-layer
sample counts (37,062 samples/layer, both models) confirming calibration
actually ran.

| Model | GPU | On-device (uncalibrated) | On-device (calibrated) |
| --- | --- | --- | --- |
| v5clean | 0.7810 | 0.8095 [0.7510, 0.8569] | 0.8000 [0.7407, 0.8485] |
| distill-v2-rationale | 0.8714 | 0.7905 [0.7305, 0.8401] | 0.8000 [0.7407, 0.8485] |

**Calibration recovered only 2 of 210 items (+0.95pp) for distill** --
79.05% -> 80.0%, still 7.1pp below its own GPU score. v5clean moved the
other direction (81.0% -> 80.0%), so the two models now land on an identical
score, but not because distill closed the gap -- both simply converged
toward the middle. Per #165's own sequencing: **fix (1) does not close the
gap.** This weighs against "data-free conversion" as the primary mechanism
and toward causes (b)/(c) from #164 (rationale-trained verbosity
destabilizing under quantization; decision-boundary fragility from tight
teacher distillation) -- i.e., toward fix (2), answer-format robustness,
as the next thing to try, not a calibration tuning pass.

Recommendation unchanged: keep v5clean as champion. Distill remains
rejectable on-device evidence even after the calibrated-conversion fix
attempt.

**VOID -- see correction above.** This paragraph and the "does NOT close the
gap" framing above it were based on the invalid broken-bundle measurement.

## Constrained-decoding diagnostic (#166/#167, 2026-09-07)

Tested whether forcing a short, clean termination (`maxTokens: 16`, no custom
stop sequences -- the custom stop-sequence array was tested first and ruled
out as unnecessary) recovers any of distill-calibrated's residual gap to its
GPU score. This is diagnostic only: production (`ModelService.generateCompletion`)
uses `maxTokens: 2048` unconstrained, so a recovered score here would not by
itself justify promotion (#167).

Result: **176/210 = 0.8381 -- identical to the unconstrained calibrated
result**, with `avg_response_chars: 1.0` (bare letters, 0 null-extracted
answers). Constraining decoding on top of calibration recovered nothing
further. This settles the mechanism question from #167: the residual
-3.3pp gap to GPU (0.8714 vs 0.8381) is not a termination/verbosity problem
-- calibration already fixed that (per #164's original finding of 39.9-char
run-on responses, which was specific to the *uncalibrated* bundle). What's
left is INT4 precision loss that decoding constraints cannot touch.

Note: this diagnostic took 5 attempts to actually execute -- 4 failures were
infrastructure noise (screen dozing / USB power management causing VM-service
and test-heartbeat timeouts on long-running on-device tests), not a real
engine issue with short maxTokens. Resolved with `adb shell svc power stayon
usb`. Worth keeping this setting for any future on-device runs longer than
a few minutes.

## Updated recommendation (post-correction)

With the corrected numbers, distill-calibrated (0.8381, 176/210) leads
v5clean-calibrated (0.7810, 164/210) by +5.7pp on-device -- suggestive
(two-proportion z=1.49) though not conventionally significant at n=210. This
is a real reversal from the pre-correction picture and reopens the
promotion question in #163: unlike the original uncalibrated on-device
result (statistically indistinguishable from v5clean), the calibrated
distill bundle now shows a real, if not airtight, on-device lead. Whether
this clears the bar for promotion is #163's call, not asserted here --
what's settled is that "calibration doesn't help" (the prior conclusion) was
wrong, and calibration should be part of whatever bundle is considered for
promotion going forward, for either model.
