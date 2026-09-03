# Plan: Quantify the size gap (98-question ladder), then distill toward it

Date: 2026-09-03
Companion: `results/I14_GATE_VERDICT.md` (M3 NO-GO), `autoresearch/levers.md` (open levers).

## 0. Context and the RLM-paper finding (why this plan, not more RLM)

1. The RLM paper (`persistent-memory-workspace/docs/pdfs/Recursive Language Models.pdf`)
   uses GPT-5 and Qwen3-Coder-480B-A35B as the main backbones. It also reports "RLM-Qwen3-8B"
   with a median +28% over the base Qwen3-8B on long-context tasks.
2. Two facts from the paper matter here:
   - A prompted Qwen3-8B "struggles to solve long context tasks even in an RLM scaffold".
     The gain needs the root model to be instructed by prompting AND fine-tuning to act
     recursively. The paper calls "training an 8B model to be natively recursive" promising.
   - The gains are on LONG-CONTEXT tasks (input length 2^13 to 2^20 tokens).
3. Conclusion for this project: finetuning a model to drive RLM is real, but only shown at 8B
   and larger, and only for long-context tasks. Our target is a 0.6B model on short-context
   agronomy question/answer, deployed on a Pixel 7 with the Cactus engine. RLM is off-target on
   all three axes (scale, task, deployment). This confirms the M3 NO-GO. The path to "8B quality
   at 0.6B" is knowledge distillation on the domain, not RLM.

## 1. Objective

1. Quantify the size gap. Place the champion `v5clean` on the same 98-question ladder that shows
   qwen3:8b at 90.8% and base 0.6B at 28.6%.
2. Design a distillation lever that aims at that target line, stays Cactus-safe (no graph change),
   and is contamination-guarded.

---

## Part A - Score v5clean on the 98-question ladder

### A.0 Assumptions and constraints

> **AMENDED 2026-09-03 (F1 / #138). The 98-question ladder is LEGACY and does not exist.**
> `benchmark-workspace/EVALUATION_RESULTS.md` (2026-02-25) reports 90.8 / 89.8 / 73.5 / 28.6,
> but the dataset file it scored is gone: `BENCHMARK_PATH` in `evaluate.py` points at a
> missing path, `benchmarks/` is gitignored so the file was never tracked, and the eval
> predates the repo's first commit (2026-03-25). The surviving 210-row graded file contains
> all 98 ids but disagrees with the published answer key on **42/98** and difficulty on
> **33/98**, so the published row is neither reproducible nor extendable and must not be
> quoted as a target line.
>
> **The ladder is now the 210-item
> `benchmarks/natural_fertilizers/natural_fertilizers_benchmark_v1_graded.jsonl`**, where
> `quality/irt/responses/*_natural_fertilizers.jsonl` already holds 9 models whose answer
> keys agree with that file 210/210 (qwen3-8b **0.895**, opus-4-6 0.938, haiku-4-5 0.905,
> llama-3.1-8b 0.805). Target line = the champion-to-qwen3-8b gap. See
> `results/LADDER_98Q_PROTOCOL.md` for the protocol and the evidence.

1. ~~The ladder file is `benchmark-workspace/EVALUATION_RESULTS.md` (Natural Fertilizer MCQ
   Benchmark v1, 98 questions, 2026-02-25).~~ Superseded -- see the amendment above.
2. The protocol is generative answer-parsing with an "ambiguous" bucket, not option-set-normalized
   log-likelihood. Do not compare the two metrics.
3. Decode is deterministic: temperature = 0 (greedy), fixed seed. `evaluate.py` defaults
   `--temperature` to 1.0, so this must be passed explicitly.
4. Never read any sealed split. Do not train on the 98 questions.

### A.1 Locate the harness and dataset
1. Find the exact 98-question file and the generative scorer that produced the table.
2. Record the prompt template, the answer parser, and the "ambiguous" rule in the results file.

### A.2 Contamination check (do this before trusting any score)
1. Compare the 98 questions against `data/train_v3.jsonl`, `data/train_v6.jsonl`, and
   `data/val_v3.jsonl`.
2. Use normalized-text and n-gram overlap, and `source_doc` overlap where present.
3. Record the overlap count. If overlap > 0, mark the affected items and also report a
   contamination-adjusted score that excludes them.

### A.3 Rebuild the ladder on this machine, one protocol
1. Score base `Qwen3-0.6B`. Target near 28.6% to validate the harness locally.
2. Score the champion `outputs/merged-model-p6-lr5e5-r2` (v5clean). This is the new number.
3. If available, re-score `qwen3:8b` (Ollama) under the same harness for a same-run reference.

### A.4 Report
1. Write `results/LADDER_98Q_RESULTS.md`: a table (base 0.6B, champion, qwen3:8b), a 95%
   confidence interval per row (Wilson interval, n = 98), and the "ambiguous" bucket rate.
2. State the target line: the gap from champion to qwen3:8b. Treat n = 98 as a coarse locator,
   not a gate. Near 50% accuracy the interval is about +/-9 pp.

---

## Part B - Distillation lever design

### B.0 Precondition
1. The Part A target line exists.

### B.1 Teacher selection
1. Primary teacher: `qwen3:8b` (local, Ollama, 90.8% on the ladder). Local and cheap.
2. Optional second teacher: `claude-opus` on a small subset, for hard items only.

### B.2 Data design
1. Generate teacher answers over training-domain questions from the QMD agronomy pool and
   `mcq_generation/`. Do NOT use the 98 eval questions.
2. For each item, capture the teacher's final answer. Optionally capture a short teacher rationale
   for a separate think-channel variant.

### B.3 Distillation form (all Cactus-safe SFT, no graph change)
1. Variant 1 - hard-label SFT: the teacher's final answer is the target (bare letter for MCQ,
   free-form answer for question/answer).
2. Variant 2 - rationale-augmented SFT: teacher reasoning in the think channel plus the answer.
   Keep this dense and high-quality; the earlier trace-SFT (#123) failed because the signal was
   thin. Treat Variant 2 as optional and test it against Variant 1.
3. Do not attempt logit/soft-label distillation. The teacher and student use different tokenizers,
   so stay at sequence-level SFT.

### B.4 Contamination wall
1. Exclude all 98 eval questions and every val/dev/test item from teacher-data generation.
2. Hash-check the generated set against those files. Report zero overlap.

### B.5 Quality filter
1. Where a gold answer exists, keep only teacher outputs that match the gold.
2. For open question/answer, filter by a rubric or self-consistency check.

### B.6 Train
1. Single lever = the dataset. Reuse `autoresearch/finetune.py` with a new config
   `autoresearch/configs/phaseNN-distill/`.
2. Base = `unsloth/Qwen3-0.6B`. Use v5clean settled hyperparameters (levers.md Tier 1).

### B.7 Evaluate and gate
1. Score the frozen MCQ primary and the trace track (the existing accept/revert signal).
2. Score the 98-question ladder (the target line).
3. Accept only when the ladder accuracy rises by a meaningful margin AND all guardrails stay clean.
   Watch the known trade: the CROP-auxiliary lever raised token F1 but lowered the MCQ primary
   (levers.md). Distillation may do the same.

### B.8 Deployment check (gated on B.7 pass)
1. Convert the winning checkpoint to Cactus.
2. Score the on-device (quantized) model on the ladder. This is the real deployment number.

## C. Risks
1. Teacher ceiling. qwen3:8b is at ~90.8%, so distilled quality cannot exceed the teacher.
2. Student capacity. A 0.6B model may not absorb 8B knowledge fully.
3. Objective trade. A ladder gain may cost the frozen MCQ primary. Gate on both.
4. Quantization loss. The on-device score may drop below the GPU score.
