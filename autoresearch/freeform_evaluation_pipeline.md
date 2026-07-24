# English free-form evaluation/rubric path

The repository now has one paired, dev-only entry point for evaluating a
baseline and a candidate model on the fixed English free-form dev split.
It combines deterministic candidate generation, optional teacher-forced
reference log-likelihood, and same-backend rubric grading with grader-anchor
drift checks.

## Boundaries

- The default input is `eval_freeform/dev.jsonl` and the fixed split manifest
  is checked by item ID. The sealed `eval_freeform/test.jsonl` file is never
  opened.
- Spanish benchmark paths and explicitly sealed paths are rejected by the
  shared guard in `eval_freeform/guards.py`.
- Baseline and candidate use the same selected IDs, scorer config, rubric
  anchors, and judge backend/model. Changing the judge backend invalidates
  historical rubric comparisons.
- Reference-LL is a reported diagnostic. It is not a cross-model primary
  signal and does not become a weighted objective term automatically.
- The `mock` backend is a local schema smoke test only. It is not a quality
  score and must not be used for promotion decisions.

## Smoke test without external calls

This exercises the complete file flow and anchor checks without transmitting
any model output:

```bash
python autoresearch/run_freeform_evaluation.py \
  --baseline-model outputs/merged-model-p6-lr5e5-r2 \
  --candidate-model outputs/merged-model-p6-lr5e5-r2 \
  --label freeform-smoke \
  --python [LOCAL_PATH] \
  --judge-backend mock \
  --items-per-type 1
```

The smoke report is explicitly marked `mock_smoke_only`.

## Authorized rubric pass

OpenRouter is an explicit opt-in because it transmits generated English
free-form answers externally. After authorization, use a fixed label and the
same command shape for the baseline and every candidate comparison:

```bash
python autoresearch/run_freeform_evaluation.py \
  --baseline-model outputs/merged-model-p6-lr5e5-r2 \
  --candidate-model outputs/merged-model-p6-lr5e5-r2 \
  --label freeform-baseline-calibration \
  --python [LOCAL_PATH] \
  --judge-backend openrouter \
  --items-per-type 18 \
  --run-refll
```

The API key is read by the existing judge client from `OPENROUTER_API_KEY` or
can be passed through the process environment. It is never written to the
report. The resulting report records the judge backend, scorer-config hash,
selected IDs, rubric percentage by type, anchor drift, and any reference-LL
diagnostic.

## Promotion policy

The first OpenRouter self-calibration is recorded in
`autoresearch/freeform_rubric_calibration.md`. The rubric remains a secondary
quality guard until a candidate comparison establishes the practical drop
threshold. A candidate still must preserve the English MCQ primary,
contamination boundaries, forgetting, bias, compute, and anchor guards. A
free-form improvement alone cannot promote a candidate that regresses the
MCQ anchor.
