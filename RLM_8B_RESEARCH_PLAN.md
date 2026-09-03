# Plan: RLM on Qwen3-8B research track

Date: 2026-09-03
Status: research probe. This track is SEPARATE from the deployment track
(`LADDER_AND_DISTILLATION_PLAN.md`, Epic F). It is exploratory, not a deployment candidate.

## 0. Framing and honest priors

1. The RLM paper shows gains on LONG-CONTEXT tasks. The current benchmark is SHORT-CONTEXT
   agronomy question/answer. So RLM-on-short-context is an unresearched combination. This plan
   tests it directly, as the user requested.
2. The prior for gain is low for RLM's decomposition mechanism, because short prompts give it
   little to decompose. Any gain most likely comes from tool use, retrieval, or verification, not
   from context splitting. The plan therefore includes a kill criterion, so the track stops early
   if this prior holds.
3. Qwen3-8B is a research vehicle only. It will not run on a Pixel 7. Only a downsized artifact
   (Phase 3) must be Cactus-safe. The 8B phases are not Cactus-constrained.

## 1. Hypotheses

1. H1: a prompted RLM scaffold improves Qwen3-8B on the benchmark, versus prompted Qwen3-8B.
2. H2: fine-tuning Qwen3-8B for RLM adaptability improves both RLM driving and benchmark score.
3. H3: a working 8B method can be downsized to a 0.6B minimal slice (the champion scale).

## 2. Metrics and protocol

1. Primary metric for this track: the generative 98-question ladder
   (`benchmark-workspace/EVALUATION_RESULTS.md` protocol). RLM emits free-form output, so use the
   generative parse, not option-set-normalized log-likelihood.
2. Secondary metric: the `data/val_v3.jsonl` MCQ track, for continuity with the champion program.
3. RLM-adaptability metrics: REPL-block emission rate, valid-tool-call rate, mean decode calls per
   item, and latency (ms).
4. Decode is deterministic: temperature = 0, fixed seed. Apply the contamination wall. Exclude all
   eval items from any training data.

## Phase 0 - Baseline Qwen3-8B (no RLM)

1. Serve Qwen3-8B through an OpenAI-compatible endpoint (vLLM).
2. Score prompted Qwen3-8B on the 98-question ladder and on `val_v3`, using the Epic F harness.
3. Record the baseline with a 95% confidence interval.

## Phase 1 - Apply RLM to Qwen3-8B (prompted, depth 1)

1. Reuse the `rlm_slice/` arms (A bare 2-pass, B RLM LocalREPL, R retrieval, C self-refine),
   pointed at the 8B server.
2. Score each arm against the Phase 0 baseline.
3. Record the RLM-adaptability metrics. A key question: does 8B emit valid REPL blocks, unlike the
   0.6B champion which emitted zero (see `results/I14_GATE_VERDICT.md`)?
4. Kill criterion: if no arm reaches the Phase 0 baseline AND 8B does not drive the REPL, stop.
   RLM-on-short-context is then falsified for this benchmark. Do not fine-tune.
5. Go criterion: proceed to Phase 2 only if 8B drives the REPL AND at least one arm shows a
   non-negative trend worth training toward.

## Phase 2 - Fine-tune Qwen3-8B for RLM adaptability (gated on Phase 1 go)

1. Rung 1 - trajectory SFT: collect RLM trajectories from a strong teacher RLM (for example GPT-5,
   or the released `mit-oasys/rlm-qwen3-30b-a3b-v0.1`) on domain items. Fine-tune Qwen3-8B (LoRA)
   to imitate valid REPL and tool use plus correct answers. Exclude all eval items.
2. Rung 2 (optional, heavier) - RL: use the RLM training harness (`persistent-memory-workspace/rlm/
   training`, verifiers + prime-rl) with a domain reward. Adapt the `oolong` environment to the
   agronomy benchmark distribution.
3. Measure the RLM-adaptability metrics and the benchmark score against Phase 1.
4. Accept if both the benchmark score and the adaptability metrics improve.

## Phase 3 - Downsizing theory to 0.6B (gated on Phase 2 accept)

1. Identify the minimal sub-skill that produced the gain: decomposition, retrieval, or
   verification.
2. Compare two transfer paths:
   - Path A: distill the 8B RLM behavior into the 0.6B champion by trajectory SFT.
   - Path B: extract the single fixed step that helped (for example retrieve-then-answer) that does
     not need model-authored code. Path B is Cactus-safe by construction.
3. Deliverable: a written transfer analysis and a go/no-go for a 0.6B implementation. Do not write
   0.6B code until the theory holds.

## Compute and infrastructure

1. Phase 0 and Phase 1: 8B inference through vLLM. A single GPU is adequate.
2. Phase 2 Rung 1: 8B LoRA SFT. Feasible on the GPU cluster.
3. Phase 2 Rung 2: prime-rl + verifiers + a vLLM inference server + a subprocess REPL. Multi-GPU
   and heavier.

## Risks

1. Short-context prior. The expected gain from decomposition is low.
2. The RLM gain may reduce to retrieval, which was null at 0.6B and is untested at 8B.
3. Teacher-trajectory availability and cost for Phase 2 Rung 1.
4. Downsizing (Phase 3) is the largest leap. Keep it conditional and theoretical first.

## Relation to other tracks

1. Separate from Epic F (deployment distillation). Run them in parallel; do not merge their gates.
2. Keep #121 (Metis) and #122 (Cactus fork) deferred.
