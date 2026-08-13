# Phase 23: GRPO baseline (Tier 4, RL second-stage)

Every prior phase (0-22) is SFT-only. Phase 23 is the first Tier 4 run: online GRPO
on `CopyleftCultivars/Agriculture-Agent-RL-Training-Data` (HF, ~6.19k multi-turn
agentic tool-calling rollout trajectories against a real MCP server, `cultivars-mcp`,
plus PubChem), continuing from the frozen v5clean SFT checkpoint rather than
replacing it — matching the dataset card's own stated intended use ("a GRPO pass on
top of an SFT-trained checkpoint").

This is a design/infrastructure doc written before the run, not a results doc.
**No phase run has happened yet.** See `autoresearch/grpo_stage0_reward_validation.json`
and `autoresearch/grpo_stage1_calibration.json` for what has been measured so far
(reward-function validation and completion-length/wall-clock calibration), and
`autoresearch/levers.md`'s Tier 4 section for the lever-by-lever status.

## Why this dataset needed new infrastructure, not a `prepare_dataset_v3.py` recipe

The dataset is `verifiers`-library-native rollout data (`prompt`/`completion`/`reward`/
`task`/`info`), not SFT pairs — `GRPOTrainer` generates its own completions online and
scores them with a `reward_fn`, it does not train on the dataset's own recorded
`completion`/`reward` columns. Those columns are dropped by
`autoresearch/grpo_dataset.py::load_rollout_prompts`; only `prompt` (+ `task`/`info`
passthrough as reward_fn kwargs) becomes the rollout seed set.

## Reward: static, not live

`autoresearch/grpo_reward.py` reimplements the dataset's own `info.reward_components`
semantics (`tool_use_correctness`, `content_quality`, `citation_presence`,
`ethics_redflags`) without calling `cultivars-mcp`/PubChem live — GRPO needs thousands
of reward evaluations per run, and live tool calls in that critical path would make
iterations non-deterministic and non-reproducible for the `journal.jsonl` audit trail.
Validated in `autoresearch/grpo_stage0_reward_validation.json`: scored through the
reimplemented `combined_reward`, the `agy` source (which self-judged 99.6% of its own
rows as `reward==1.0` in the original export) scores 0.886 — actually *below* `hermes`
(0.894) and only +0.034 above the two-clean-source average, versus the original
0.999-vs-0.819/0.698 gap. `agy` rows are excluded outright from the rollout prompt set
regardless (`autoresearch/grpo_dataset.py::EXCLUDED_GENERATOR_AGENTS`), per
`levers.md`'s FORBIDDEN row.

## Held-out evaluation

`autoresearch/grpo_dataset.py::holdout_bucket()` deterministically assigns each
non-`agy` row to a `train` or `eval` bucket from a hash of `info.record_uuid` (default
10% eval, seed 42) — same seed/fraction on both `train_grpo.py` (`holdout="train"`)
and `autoresearch/score_grpo_primary.py` (`holdout="eval"`) is what keeps the eval
slice genuinely held out, independent of `--max-samples` or dataset ordering.
Confirmed on the real dataset: 522/4964 non-agy rows (10.5%) land in `eval`,
deterministically and reproducibly across separate invocations.

## Primary metric: not the existing MCQ/freeform harness

`autoresearch/objective_weights.json` pins the current primary objective to MCQ
option-set gold-log-likelihood only; `generate_freeform_candidates.py` has no
tool-call generation support at all. Neither can measure what GRPO is training.
`autoresearch/score_grpo_primary.py` is the Tier 4 primary: it generates fresh
completions for the held-out eval prompts with the candidate model, parses
`<tool_call>...</tool_call>` blocks (Qwen3's native chat-template tool-call
convention — confirmed against `outputs/lora-adapter-p6-lr5e5-r2/final/chat_template.jinja`),
and scores them with `combined_reward`. The existing MCQ/freeform harness still runs
against a GRPO candidate, but only as `gate.py`'s forgetting/regression guardrail
(does GRPO damage v5clean's SFT-era competence?), never as the accept/reject signal.

## Compute budget

`gate_config.json`'s `compute_budget_grpo.max_minutes_per_iteration: 90.0`, passed via
`gate.py --budget-key compute_budget_grpo`, separate from the SFT tier's calibrated
28-minute budget (rollout generation is far more expensive per-iteration than SFT
forward/backward passes). Calibrated from a measured 25.92s/step at this phase's exact
settings (`num_generations=8`, `max_completion_len=768`, `max_prompt_len=1024`,
`batch_size=8`), extrapolated to ~86.4 minutes for a 200-step run.

## Run recipe

`autoresearch/configs/phase23-grpo-baseline/p23-grpo-baseline.yaml`, invoked via:

```bash
conda activate qwen3-grpo
python autoresearch/grpo_finetune.py \
  --config autoresearch/configs/phase23-grpo-baseline/p23-grpo-baseline.yaml
```

Chain: `freeze_manifest.py check` -> `train_grpo.py` (continues from
`outputs/lora-adapter-p6-lr5e5-r2/final`) -> `merge_adapter.py` ->
`outputs/merged-model-phase23-grpo-baseline`. Scoring (not yet wired into a single
orchestrated step -- run manually after merge):

```bash
python autoresearch/score_grpo_primary.py \
  --model outputs/merged-model-phase23-grpo-baseline \
  --output results/primary_grpo_phase23-grpo-baseline.json
python autoresearch/score_mcq_ll.py --model outputs/merged-model-phase23-grpo-baseline ...  # forgetting/bias guardrail
python autoresearch/run_freeform_evaluation.py --baseline-model outputs/merged-model-p6-lr5e5-r2 \
  --candidate-model outputs/merged-model-phase23-grpo-baseline ...  # rubric guardrail
python autoresearch/gate.py --budget-key compute_budget_grpo \
  --baseline results/primary_grpo_phase23-baseline-reference.json \
  --candidate results/primary_grpo_phase23-grpo-baseline.json \
  --guardrails <assembled guardrail report> --output results/autoresearch_gate_p23-grpo-baseline.json
```

## Open items before this can actually run as a real (non-smoke) phase

- **No baseline reference for `gate.py`'s `--baseline`.** `score_grpo_primary.py` has
  never been run against the unmodified v5clean checkpoint, so there is no
  `primary_grpo_*_v5clean_reference.json` to diff the candidate against yet. This
  phase's first execution should include a `--model outputs/merged-model-p6-lr5e5-r2`
  baseline scoring pass before the GRPO-trained candidate is compared.
- **Guardrail assembly is manual.** Unlike `finetune.py`'s SFT chain, nothing yet
  automates collecting `score_mcq_ll.py` + `run_freeform_evaluation.py` +
  compute/sealed-test/bias readings into the single guardrails JSON `gate.py` expects
  -- this phase's first run will need that assembled by hand or with a small ad hoc
  script, not a reusable one yet.
- **Reward-component weights are untuned** (`autoresearch/grpo_reward_weights.json`,
  marked PILOT). This baseline run uses the defaults as-is; do not read a reject
  decision here as evidence against GRPO in general without first checking whether
  weight tuning would have changed the outcome.
