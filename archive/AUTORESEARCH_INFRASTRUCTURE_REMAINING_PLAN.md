# Plan: finish autoresearch loop infrastructure after C2 metric-sanity rerun

Date: 2026-07-16

## Current state

- **Phase 2.6 is complete enough for C2/C3 planning:** canonical free-form items, contamination wall, dev/test split, rubric scorer, ref-LL scorer, anchors, and frozen split/config artifacts exist.
- **Phase 2.7 is complete:** `autoresearch/levers.md` is the current action-space source of truth, and #20/#21/#26 are closed.
- **C1 is complete:** `autoresearch/finetune.py`, config loading, checksum manifest, and the 1:1 lever audit are done; chainlink #22 is closed. `C1_HARNESS_PLAN.md` has been moved to `archive/` because it is implemented.
- **C2 is partially complete but blocked:** #33/#34 landed `score_mcq_ll.py`, `score_primary.py`, and frozen objective weights. #49/#35 remain open because the free-form ref-LL component is currently mis-specified.
- **C3/C4 and C5 are not started:** #24 is blocked by #23; #25 is blocked by #24.

## New finding to incorporate

The 2026-07-15 C2 blocker was spurious as an environment claim: the real scorer runs in `conda run -n qwen3-diagnostics`, and both merged model dirs contain `model.safetensors`.

The real blocker is objective design:

- Current ref-LL scores a bare completion prompt (`Question:\n...\n\nReference answer:\n`) with `add_special_tokens=False` and no `tokenizer.apply_chat_template`.
- On the 90-item dev split this ranks **v2-clean above v3-clean** (`-4.370` vs `-4.993`), despite v3-clean's known generation accuracy advantage (73.0% vs 42.55%).
- This comparison is not apples-to-apples: v2-clean mostly learned MCP/conversation responses, while v3-clean learned Qwen chat-template + `<think>...</think>\n{letter}` behavior. The current ref-LL axis rewards being closer to base prose-completion behavior and penalizes the v3 format that actually improves MCQ generation.
- The option-set-normalized MCQ component now points the right way, but the primary only passes because MCQ outweighs the mis-signed free-form term. The free-form term is not acting as a Goodhart guard.
- Scorer code itself is not frozen yet (`eval_freeform/score_refll.py`, `eval_freeform/score_rubric.py`, `autoresearch/score_primary.py`, `autoresearch/score_mcq_ll.py`), so the current manifest does not protect the most gameable surface.

## Critical caveat (added 2026-07-16): the chat-template fix may be necessary but NOT sufficient

A template-rendering probe on `outputs/merged-model-v3-clean` (tokenizer only, no GPU) plus a read
of `eval_freeform/dev.jsonl` surfaced two facts the "just apply the chat template" recommendation
does not account for:

1. **The reference answers carry no `<think>` block.** With `enable_thinking=False`, the Qwen chat
   template injects `<|im_start|>assistant\n<think>\n\n</think>\n\n` before the teacher-forced text;
   with `enable_thinking=True` it ends at `<|im_start|>assistant\n`. v3-clean was SFT'd hard onto
   `<think>...</think>\n{content}` targets, so a reference answer that opens directly with prose is
   still partially off-distribution for v3 even inside the correct template.
2. **~40% of dev references (36/90) are written in third-person instructional meta-register**
   ("the assistant should first connect the crop goal to the site..."), not the first-person direct
   answer both models actually emit. Teacher-forcing a description-of-an-answer is not the same as
   teacher-forcing an answer, and this register is closer to generic prose — which structurally
   favors the less-finetuned v2-clean regardless of template.

Consequence: applying the chat template is the right first move, but it is a **hypothesis, not a
guaranteed sign-flip.** The plan below therefore requires the fix to be *empirically validated*
(does v3 now beat v2 on dev ref-LL under the pinned template?) before ref-LL is trusted even as an
intra-lineage proxy. If the sign does **not** flip after the template fix, that is strong evidence
ref-LL is measuring register/format proximity rather than answer quality, and it should be
**demoted out of the primary objective entirely** (the user's own instinct) and kept, if at all,
only as a reported diagnostic. This must be settled on the *GPU-enabled* host (`qwen3-diagnostics`
with a live 4070 Ti); the sign-flip check cannot be run in a CPU-only session.

## Recommended policy decision

Treat **free-form ref-LL as an intra-lineage proxy only**, not an a-priori cross-approach model comparator.

Concretely:

- Keep ref-LL only if scored under the model's real chat format and use it to compare nearby v3-lineage loop variants that share prompt/template/training conventions.
- Do **not** use raw v2-vs-v3 ref-LL as a hard acceptance criterion, because it mixes different training interfaces and answer surfaces.
- Make rubric generation quality the cross-approach sanity check. Ref-LL can be a cheap per-iteration proxy once calibrated against rubric on v3-lineage variants.
- The loop must not climb any primary objective until this policy is encoded in frozen config and verified.

## Implementation plan

### 1. Replace #49 with a concrete scorer-policy fix

Update chainlink #49 (or add child issues under it) with this task breakdown:

1. **Template-aware ref-LL scorer.**
   - Add a pinned `refll_prompt_mode` to `eval_freeform/scorer_config.json`.
   - Implement at least two explicit modes in `score_refll.py`:
     - `chat_assistant_reference_v1` (default for real scoring): build messages with the item question as a user turn, use `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=<pinned>)` for the prompt prefix, then teacher-force `reference_answer` as the assistant continuation.
     - `legacy_bare_reference_prefix_v1` (only for historical reproduction; never used by gates).
   - Record the mode, `enable_thinking` setting, target token count, and template source in every output JSON.
   - Keep `--mock-model` deterministic and make it include the selected prompt mode in its output.
   - Acceptance is not just "the code applies a template": rerun v2-clean and v3-clean on
     `eval_freeform/dev.jsonl` under the pinned mode. If v3-clean still does not beat v2-clean,
     treat ref-LL as invalid for the primary objective because it is still measuring
     register/format proximity rather than answer quality.

2. **Separate metric semantics in frozen config.**
   - Update `autoresearch/objective_weights.json` so the free-form LL component is labelled as a calibrated **same-lineage proxy**, not a cross-model truth metric.
   - Add a policy field such as `cross_model_refll_comparison: invalid_unless_same_prompt_lineage`.
   - Decide whether the primary for early C3 should be:
     - **Option A, recommended:** MCQ option-set LL as the high-frequency primary plus rubric score as a hard periodic guard; report chat-template ref-LL as diagnostic until it is calibrated on v3-lineage loop variants.
     - Option B: keep the 50/50 MCQ/ref-LL primary after template fix, but require rubric confirmation before C3 accepts it.
   - Given the current evidence, Option A is safer for the first loop because the ref-LL proxy has already failed as a cross-approach Goodhart guard.
   - Make the fallback explicit: if the template-aware rerun remains mis-signed, remove ref-LL
     from the primary entirely instead of tuning weights around it. Keep it only as a labelled
     diagnostic for within-run/model-family analysis.

3. **Freeze scorer code after prompt semantics settle.**
   - Add these code files to `autoresearch/freeze_manifest.py` and regenerate the manifest:
     - `eval_freeform/score_refll.py`
     - `eval_freeform/score_rubric.py`
     - `autoresearch/score_primary.py`
     - `autoresearch/score_mcq_ll.py`
     - any new gate/guardrail config introduced in C3.
   - Keep `eval_freeform/scorer_config.json`, `grader_anchors.jsonl`, split files, and objective weights frozen.

### 2. Rerun C2 sanity with the right evidence

After the scorer-policy fix:

1. Rerun real chat-template ref-LL on the 90-item dev split for v2-clean and v3-clean under `conda run -n qwen3-diagnostics`.
2. Rerun `score_primary.py` with the updated objective policy.
3. Add a new `results/C2_metric_sanity.md` section that explicitly separates:
   - cross-approach sanity: v3-clean should beat v2-clean on real generation accuracy and/or rubric quality;
   - intra-lineage proxy sanity: ref-LL is eligible only for v3-style variants sharing prompt/template lineage;
   - variance/SE estimates for whatever metric C3 will actually climb.
4. If API budget is available, run `score_rubric.py` on generated answers for the same 90 dev items for v2-clean and v3-clean. This requires a small generation artifact first because `score_rubric.py` grades `candidate_answer`, not model paths. Use this as the independent proof that the current legacy ref-LL is the broken proxy.
5. Close #49 only when the report says the C3-climbable objective is no longer pulled toward v2/base-prose behavior. Close #35/#23 only after this rerun passes under the revised acceptance criterion.

### 3. Refresh chainlink plan state

- Leave #14 open until C3/C4 and C5 complete.
- Keep #23 open until C2 is revalidated.
- Keep #24 blocked by #23 and #25 blocked by #24.
- Add subissues under #49 or #23 for:
  - `C2.1: make ref-LL chat-template aware and freeze prompt mode`
  - `C2.2: revise objective policy so ref-LL is intra-lineage proxy / rubric is cross-approach guard`
  - `C2.3: freeze scorer code and rerun metric sanity`
  - optional `C2.4: generate dev candidate answers and run rubric adjudication`
- Archive only completed plan docs. `C1_HARNESS_PLAN.md` is archived now; `PHASE_2.6_2.7_4_PLAN.md` should remain in root until #23/#24/#25 close, then move it to `archive/`.

### 4. Build C3/C4 only after C2 is stable

Once #23 closes, implement #24 with frozen thresholds and dry-run guardrail fixtures:

- Add a frozen `autoresearch/gate_config.json` with:
  - objective direction and minimum delta rule (`> 2 * SE` or two-seed replicate);
  - per-run compute budget;
  - bias threshold;
  - forgetting threshold;
  - rubric guard threshold;
  - grader-anchor drift threshold / pause behavior.
- Add a gate script (for example `autoresearch/gate.py`) that consumes baseline/current metric JSONs and emits `accept|reject|pause` plus reasons.
- Ensure guardrails are independent hard vetoes, not weighted terms the loop can trade off.
- Provide synthetic fixtures/tests for: known-good accept, no-op/noise reject, metric improvement with bias guard reject, metric improvement with rubric drop reject, anchor drift pause.
- Add the gate script and gate config to the frozen manifest before C5.

### 5. Build C5 loop and `program.md`

After #24 closes:

- Implement `autoresearch/run_loop.py` as orchestration only: read `program.md`/journal/levers, execute exactly one `finetune.py` candidate run, run frozen eval/gate, revert or keep, append `journal.jsonl`.
- Write `autoresearch/program.md` with the settled priors from `levers.md`, plus the new metric policy:
  - do not optimize legacy bare-prefix ref-LL;
  - do not interpret v2-vs-v3 ref-LL as quality;
  - prefer smallest lever edits with a stated hypothesis;
  - sealed tests are final-only.
- First end-to-end experiment should remain the position-invariance/permutation lever, but only after the C2 objective no longer rewards base-prose behavior.
- Acceptance: one full dry or smoke iteration writes a reproducible journal row and grep/log evidence shows sealed test files were not read.

## Verification checklist

- `python autoresearch/freeze_manifest.py check` passes after intentional manifest regeneration.
- Intentional edit to any scorer code aborts `finetune.py`/loop before training or eval.
- Chat-template ref-LL output records prompt mode and template use; legacy mode is explicitly labelled non-gating.
- Updated C2 sanity report explains why v2-vs-v3 ref-LL is invalid as a hard comparator and validates the actual gate metric.
- Rubric scorer, if run, uses generated candidate answers and leaves `eval_freeform/test.jsonl` untouched.
- `chainlink issue list -s open` shows only #14, #23/#35/#49 until C2 is fixed; #24/#25 remain blocked until their prerequisites close.

## Assumptions / uncertainties

- The recommended default is to demote ref-LL from the first C3 primary and use it as a calibrated diagnostic until same-lineage v3 variants exist. If you want the loop to keep a 50/50 MCQ/ref-LL primary immediately, require rubric adjudication first.
- Applying chat templates will invalidate all historical free-form ref-LL numbers; keep them only as legacy artifacts.
- A real rubric adjudication requires generated candidate answers for each model; the repo currently has a rubric grader but no dedicated free-form generation wrapper.
