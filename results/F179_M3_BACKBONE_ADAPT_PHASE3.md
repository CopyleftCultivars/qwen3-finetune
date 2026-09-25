# F179 / M3 — #232 Phase 3: 210-item ladder scoring of the 1.7B distillation

Parent #232 (backbone-adaptation disambiguation, child of epic #179). Follows
Phase 1 (GO — recipe runs on `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`) and Phase 2
(full 4-epoch LoRA SFT, 1612 steps, loss 2.19→0.83, merged into
`outputs/merged-model-distill-v2-rationale-1.7b`). Scope: **ladder scoring only.**
No retraining, no LPM work, no generation-legibility probe, no quantization.

**Verdict: GO for Phase 4.** The adaptation demonstrably landed: the distilled
1.7B scores **0.8667** on the 210-item ladder vs **0.7286** for base Qwen3-1.7B —
**+13.8 pp**, paired McNemar p = 9.0e-06. That clears the Phase-4 GO bar (a fair,
adapted 1.7B encoder for the LPM retrain). Two honest caveats are recorded in
§4: the lift is about one-third of F8's 0.6B lift, and the adapted 1.7B merely
**matches** (does not beat) the 0.6B distillation's 0.8714.

## 1. Harness (F8's exact invocation, not defaults assumed)

The F8 numbers in `results/F8_DISTILLATION_RESULTS.md` were produced by
`scripts/ladder_score_local.py`, which imports the frozen prompt/parser
(`SYSTEM_PROMPT`, `PROMPT_TEMPLATE`, `extract_answer`) directly from
`benchmark-workspace/evaluate.py` and drives a local HF checkpoint through the
in-process `rlm_slice.model_host.ModelHost`. The exact invocation was:

```bash
cd [LOCAL_PATH]
[LOCAL_PATH] scripts/ladder_score_local.py \
  --model <model> --label <label>
# defaults are the protocol: max-new-tokens 8, thinking OFF, temperature 0, seed 42
```

Benchmark: `benchmark-workspace/benchmarks/natural_fertilizers/
natural_fertilizers_benchmark_v1_graded.jsonl` (210 items; easy n=175,
hard n=33, moderate n=2). `ModelHost` loads 4-bit NF4 (`full_precision=False`,
the default and unchanged since #116), which is what F8 used; the 1.7B models
fit the 12 GB card either way.

**Harness-fidelity check.** Re-running the F8 0.6B-distill row on today's stack
gives **182/210 = 0.8667** vs F8's recorded **183/210 = 0.8714** — a 1-item net
difference across 3 nondeterministic hard items (NF-0025, NF-0111, NF-0162).
That is the run-to-run noise floor of the 4-bit path (~1.4 pp on the hard tier,
~0.5 pp overall). Artifact: `results/ladder210_f8_repro0.6B.{jsonl,_summary.json}`.
All conclusions below are far larger than this floor.

## 2. The 210-item ladder — F8-comparable table

| Model | Accuracy | Wilson 95% | easy (n=175) | hard (n=33) |
| --- | --- | --- | --- | --- |
| qwen3-8b (same harness, target) | 0.8571 | 0.8034–0.8981 | 0.9371 (164/175) | 0.4545 (15/33) |
| **distill-v2-rationale (0.6B, F8)** | **0.8714** | **0.8194–0.9101** | **0.9029** | **0.7273** |
| **distill-v2-rationale (1.7B, this phase)** | **0.8667** | **0.8140–0.9061** | **0.9029 (158/175)** | **0.6667 (22/33)** |
| v5clean (0.6B champion) | 0.7810 | 0.7202–0.8316 | 0.8286 | 0.5455 |
| **base Qwen3-1.7B (this phase)** | **0.7286** | **0.6647–0.7842** | **0.7943 (139/175)** | **0.3939 (13/33)** |
| base Qwen3-0.6B | 0.4286 | 0.3635–0.4962 | 0.4686 | 0.2424 |

(moderate n=2 omitted from the F8-style columns: 1.7B distill 2/2, 1.7B base
1/2, 0.6B distill 1/2.)

Raw outputs committed:
`results/ladder210_distill-v2-rationale-1.7b.jsonl` + `_summary.json`,
`results/ladder210_base1.7B.jsonl` + `_summary.json`.

## 3. Base → distill at 1.7B (the direct analog of F8's row)

| backbone | base | distilled | Δ | paired McNemar |
| --- | --- | --- | --- | --- |
| 0.6B | 0.4286 (90/210) | 0.8714 (183/210) | **+44.3 pp** | — (F8) |
| 1.7B | 0.7286 (153/210) | **0.8667 (182/210)** | **+13.8 pp** | **p = 9.0e-06** (distill-only 7, base-only 36) |

The 1.7B adaptation lift is unambiguous: +29 items, non-overlapping Wilson CIs
([0.814, 0.906] vs [0.665, 0.784]), unpaired two-proportion z = +3.52,
95% CI for the difference [+6.2, +21.4] pp. Hard tier carries the bigger lift
(0.3939 → 0.6667, **+27.3 pp**) than easy (0.7943 → 0.9029, +10.9 pp).

Capacity alone also matters: base 1.7B is **+30.0 pp** over base 0.6B
(0.7286 vs 0.4286, z = +6.23) — which is exactly why the raw margin shrinks.

## 4. The two caveats (do not soften)

1. **The 1.7B lift is ~⅓ of the 0.6B lift, not comparable in raw pp.** +13.8 pp
   vs +44.3 pp. The comparison is not like-for-like: base 1.7B already starts
   30 pp higher, so there is less headroom. Relative to its own base, the 0.6B
   distillation is 2.03× while the 1.7B is 1.19×.
2. **The adapted 1.7B does not beat the 0.6B distillation.** 0.8667 vs 0.8714 is
   Δ = −0.5 pp, paired McNemar **p = 1.00** (1.7B-only 16, 0.6B-only 15),
   diff 95% CI [−6.9, +6.0] pp. The two are statistically identical, and the
   1.7B is not in a clearly better range. It also matches the 8B teacher
   (0.8667 vs 0.8571, p = 0.83).

## 5. Answer to the gate question

- **Did 1.7B distillation beat 1.7B base by a margin comparable to F8's 0.6B
  result?** **No** — +13.8 pp vs +44.3 pp, about one-third; the gap is real and
  is explained by base-1.7B's much higher starting point.
- **Is it in the same range as, or clearly better than, the 0.6B distillation's
  0.8714?** **Same range — statistically identical** (0.8667, Δ −0.5 pp,
  p = 1.00). Not better.

## 6. GO / NO-GO for Phase 4 (LPM retraining)

**GO.** The explicit Phase-4 bar was "1.7B distillation clearly beats 1.7B base
(adaptation demonstrably landed)". It does: +13.8 pp, p = 9.0e-06, non-overlapping
CIs, and the hard tier — the tier F8's verdict hinges on — rises +27.3 pp. The
1.7B encoder is therefore adapted, which removes #231's distilled-0.6B vs
raw-1.7B confound and makes the Phase-4 LPM retrain and the Phase-5 legibility
probe fair.

**Proceed, but scope the expectation:** because the adapted 1.7B matches (rather
than beats) the adapted 0.6B on the ladder, Phase 4/5 should not be sold as a
ladder-quality gain. Its sole job is to test whether an *adapted* 1.7B backbone
surfaces persona-distinctive content where raw 1.7B (and raw/distilled 0.6B) did
not — width-plus-adaptation, now without the confound. If Phase 5 still returns
generic output, the epic can rule out both width and this adaptation recipe, which
is the information it needs.

## 7. Reproduce

```bash
cd [LOCAL_PATH]
PY=[LOCAL_PATH]
$PY scripts/ladder_score_local.py --model outputs/merged-model-distill-v2-rationale-1.7b \
  --label distill-v2-rationale-1.7b
$PY scripts/ladder_score_local.py --model Qwen/Qwen3-1.7B --label base1.7B
# F8 harness-fidelity reproduction:
$PY scripts/ladder_score_local.py --model outputs/merged-model-distill-v2-rationale \
  --label f8_repro0.6B
```

Artifacts: `results/F179_M3_BACKBONE_ADAPT_PHASE3.md`, `..._PHASE3.json`, and the
raw ladder JSONL/summary files listed in §2–§3. Models are under gitignored
`outputs/`.
