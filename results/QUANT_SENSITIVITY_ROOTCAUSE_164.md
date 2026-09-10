# #164 (under #163): why distill-v2-rationale is more INT4-sensitive than v5clean

Date: 2026-09-07. Inputs: `results/ladder210_{distill-v2-rationale,v5clean}.jsonl`
(GPU per-item), `LocalLLM/finetune/outputs/ladder_results/{distill,v5clean}_ladder_snapshot.json`
(on-device per-item, Pixel 7, INT4), and the conversion bundles'
`hessian_metadata.json`. Feeds the #163 promotion call.

## Summary

The larger on-device drop of `distill-v2-rationale` is real and has three
compounding causes, in priority order:

1. Genuine boundary fragility under INT4. The distilled model loses items it
   answered correctly on GPU, and alternative answer-parsing does not recover
   them. This is real accuracy loss, not a harness artifact.
2. Rationale-trained verbosity that INT4 destabilizes. On the lost items the
   model changes shape from a clean single letter on GPU to a 40-character
   run-on on device. v5clean, trained bare-letter, stays a single letter and is
   nearly immune.
3. The INT4 conversion is data-free. Both bundles' `hessian_metadata.json` is
   empty, so the quantizer had no calibration data to protect sensitive weights.

The drop is plausibly fixable. The fastest lever is a calibrated INT4
conversion. This keeps the door open for a deployable distill candidate.

## Already ruled out (this session and prior)

1. Harness truncation (F9): zero null answers, zero unclosed `<think>` tags.
2. Run-to-run noise (F9): byte-identical predictions across two independent
   device runs for both models.
3. Answer-parsing artifact (this analysis, see below): an alternative parser
   recovers only 4 of 25 lost items, and it makes v5clean worse. The losses are
   genuine, not mis-parsed.

## The measured pattern

| Model | GPU correct | Device correct | Net | Lost (GPU right, device wrong) | Gained |
| --- | --- | --- | --- | --- | --- |
| v5clean | 164/210 | 170/210 | +6 | 13 | 19 |
| distill-v2-rationale | 183/210 | 166/210 | -17 | 25 | 8 |

Distill loses almost twice as many items as v5clean and gains far fewer.

### Finding 1 - the losses are genuine, not a parser bug

- Of the 25 distill lost items, an alternative parser (last letter, or the
  final `Answer: X` tag) recovers only 4.
- Applying a last-letter parser to all 210 items moves distill 166 -> 169 (+3)
  but moves v5clean 170 -> 165 (-5). So a parser change is not a clean fix and
  is not the explanation.
- Device-response category of the 25 lost items: single clean letter 12,
  verbose "X ... Answer: X" (self-consistent, still wrong) 6, run-on into a new
  hallucinated question 3, two conflicting letters 2, verbose-then-letter 2.

### Finding 2 - INT4 changes the distilled model's output shape

- On the same 25 lost items, the GPU response length averages 1.0 characters (a
  clean single letter, which was correct), while the device response length
  averages 39.9 characters (verbose or run-on, and wrong).
- Same weights, same items: fp16 on GPU produced a terminal letter; INT4 on
  device produced run-on text. Three lost items emit `<|endoftext|>` and then a
  new hallucinated question.
- v5clean's 13 lost items are all single-letter (length 1.0) on device. It stays
  terminal on both GPU and INT4. Bare-letter training gives quantization almost
  no surface to derail.

### Finding 3 - a first-option default under INT4

- First letter of the device response on the 25 distill lost items: A 14, D 7,
  B 3, C 1. On items it previously got right, the INT4 distill model leans
  heavily to option A. This is consistent with reduced confidence under
  quantization defaulting to the first option.

### Finding 4 - the Hessian "fast lead" is empty

- `hessian_metadata.json` for `distill-v2-rationale-cact-int4`,
  `distill-arm64-cact-int4`, and `v5clean-arm64-cact-int4` all have
  `rows = {}`, `sample_counts = {}`, `errors = {}`.
- So the conversion ran data-free (no calibration samples, no per-layer Hessian
  sensitivity). There is no per-layer signal to compare, and, more importantly,
  the quantizer could not protect the most sensitive weights. Data-free INT4
  penalizes the more sensitive (distilled) model the most.

### Not the cause: topic and hard-tier

- The 25 lost items spread across topics (crop-specific 4, vermicomposting 4,
  green manures 3, pest control 3, NPK 3). No single-topic knowledge hole.
- By difficulty, losses are 18 easy and 7 hard. Hard is over-represented
  (28% of losses vs 16% of the set), but v5clean shows the same hard skew
  (4/13 = 31%). Hard-tier fragility is not distill-specific. The distill-specific
  signals are the higher loss count, the verbosity, and the first-option default.

## Interpretation

Distillation pushed the 0.6B student to imitate an 8B teacher's finer-grained,
higher-entropy output distribution, including a rationale (think-ON) style. Two
effects follow under data-free INT4:

1. The tighter teacher-imitation fit sits closer to decision boundaries, so
   INT4 rounding flips more items (Finding 1) and biases uncertain items to the
   first option (Finding 3).
2. The rationale style leaves a latent tendency to continue generating. fp16
   still terminates cleanly, but INT4 destabilizes the stop behavior into
   run-on output (Finding 2).

v5clean's bare-letter fit is coarser, more margin-robust, and terminates in one
token, so it is nearly immune on all three axes.

## Recommended fixes (ranked) - feeds #163

1. Calibrated INT4 conversion. Populate the Hessian with real agronomy-domain
   calibration samples (GPTQ/AWQ-style), then re-run F9. This targets the
   data-free enabling factor directly, needs no retrain, and is the fastest test
   of whether the drop is recoverable.
2. Answer-format robustness. Either retrain the distill so the MCQ answer surface
   is a terminal single letter (keep the rationale off the answer channel), or
   apply constrained decoding on-device (stop after the first letter, or a
   letter grammar). This removes the verbosity fragility (Finding 2).
3. Bits sweep for confirmation. Score the distill bundle at INT8 or mixed
   precision. If the drop shrinks, it confirms the mechanism is quantization
   error, and it bounds how much precision the deployment needs.

## Verdict for #163

Do not promote distill-v2-rationale on the current INT4 bundle. But the drop is
a fixable format-times-quantization interaction plus a data-free conversion, not
an intrinsic ceiling. Run fix 1 (calibrated INT4) and re-run F9 before closing
the distillation track.
