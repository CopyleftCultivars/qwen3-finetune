# F179 / M3 — #207 (S5-S6): regression — #185 ladder unchanged, #193 gap stays negligible

Milestone #14 (Epic #179). Both parts pass. Real ARM engine under the M1
qemu/docker harness.

## Bundle resolution

The issue's "distill-calibrated-deploy" does not exist by that name. The #185
baseline (`results/ladder210_m1_p185_reserved_noise_summary.json`, n=210,
correct=182, model=`outputs/r179-m1/model-p185`) points at
`outputs/r179-m1/p185-deploy`, which does exist. Confirmed it is the right
bundle: its `token_embeddings.weights` md5 `b2c8c640c11d4e728d30e17d8a3d3729`
equals `outputs/r179-m1/p185-weights/token_embeddings.weights` (the converted
`model-p185` tensor). Built the suppress variant exactly as #205 did
(`scripts/cactus_set_suppress_ids.py`), verified with `diff -rq` that the ONLY
difference is `components/manifest.json`:

```
cp -r outputs/r179-m1/p185-deploy outputs/r179-m1/p185-suppress-deploy
python scripts/cactus_set_suppress_ids.py outputs/r179-m1/p185-suppress-deploy        # 266 ids
python scripts/cactus_set_suppress_ids.py outputs/r179-m1/p185-suppress-deploy --check
diff -rq outputs/r179-m1/p185-deploy outputs/r179-m1/p185-suppress-deploy  # only manifest.json
```

## Part 1 — 210-item ladder regression: PASS

New committed transport `scripts/cactus_engine_ladder.py` drives the benchmark's
own protocol through `cactus_complete`. The prompt/parser are imported from
`benchmark-workspace/evaluate.py` (`SYSTEM_PROMPT`, `PROMPT_TEMPLATE`,
`extract_answer`) — not duplicated. Transport verified: the engine's
`cactus_render_prompt` for NF-0001 is byte-identical to HF
`apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`
(162 tokens), so the wording/parser match what produced the reference rows.

Greedy, temperature 0, `max_tokens=8`, 8 shards per condition, 210 items each:

| condition | engine | bundle | correct | accuracy | ambiguous |
|---|---|---|---|---|---|
| unpatched | v2.1.0 (vanilla) | p185-deploy | **174/210** | 0.8286 | 0 |
| patched | v2.1.0 + Δ | p185-suppress-deploy | **174/210** | 0.8286 | 0 |

**Per-item diff: 0/210 differ** (`differ_question_ids: []`, `correct_delta: 0`).
The suppression patch + manifest key is a no-op for prompts that never reference
reserved IDs.

Honest note on the absolute number: #185 recorded 182/210 on the **bf16 merged
model** via `ladder_score_local.py` (host GPU), not the CQ4 engine. The engine
gives 174/210 — an 8-item engine/CQ4-vs-bf16 drift (M1 already flagged a similar
harness drift). It is identical across both engines, so it is not a regression
from this patch. The gate is the 0/210 A/B diff, which passes.

## Part 2 — #193 engine leg with suppression: PASS

`scripts/lpm_cq4_engine_leak.py` against `p193-suppress-deploy` with the patched
v2.1.0 engine (the bundle #206's S3 test used): **0 reserved emissions in every
condition** (none / +1 mem / +8 mem × 2 prompts = 6 conditions, 96 tokens;
`reserved_emitted=0/16` each), and coherent text. The tied-head saturation that
motivated #187/#204/#205 is gone, and memory injection still changes output
(the +1/+8-mem continuations differ from the no-mem continuation), so the
reserved-token mechanism is live rather than inert.

Recall-gap conclusion (section 3 of `F179_M3_QUANT_GAP_RESULTS.md`) — verified
structurally, not assumed: the bf16-vs-CQ4 recall@1 measurement (0.250 vs 0.250,
**Δ = 0.000**) is computed by `dynamic_recall_matrix` in
`scripts/lpm_quant_gap.py`, which calls `forward_ntp(model, prompts, tid, tam)`
with **soft-prompt vectors** and normal target token IDs. That path never
references reserved rows, the manifest, or the Cactus engine — it is a pure HF
forward on the dequantized CQ4 weights. Suppression is a sampler-level ban on
reserved IDs, so it cannot affect that measurement. The gap conclusion is
therefore unaffected by suppression being enabled, and **#194 remains correctly
skipped**.

## Verdict

- Ladder: unpatched 174/210 == patched 174/210, **0/210 items differ** → the
  suppression patch is a no-op for ordinary prompts.
- #193 gap: unchanged (Δrecall@1 = 0.000); the engine leg no longer saturates on
  reserved IDs with suppression on; #194 stays skipped.
- No regression; #207 complete.

## Reproduce

```
# 0. suppress bundle for p185 (see Bundle resolution above)
# 1. prompts (host; reuses evaluate.py protocol)
python scripts/cactus_engine_ladder.py prepare --out outputs/r179-m1/s5s6/ladder_prompts.jsonl
# 2. run each condition, sharded (8 containers per condition; repeat --offset 0..7)
docker run --rm --platform linux/arm64 -e CACTUS_DISABLE_CLOUD_HANDOFF=1 \
  -e CACTUS_LIB_PATH=/work/lib.so -v <v2.1.0-tree>:/src/cactus-sdk \
  -v <engine.so>:/work/lib.so:ro -v <bundle>:/work/bundle:ro \
  -v outputs/r179-m1/s5s6/ladder_prompts.jsonl:/work/prompts.jsonl:ro \
  -v outputs/r179-m1/s5s6/runA:/work/out -v scripts/cactus_engine_ladder.py:/work/ladder.py:ro \
  cactus-arm64-convert:built python3 /work/ladder.py run --bundle /work/bundle \
  --prompts /work/prompts.jsonl --out /work/out/A<i>.jsonl --offset <i> --stride 8 --max-tokens 8
# 3. merge shards (dedup by question_id) then compare
python scripts/cactus_engine_ladder.py compare \
  --prompts outputs/r179-m1/s5s6/ladder_prompts.jsonl \
  --a outputs/r179-m1/s5s6/runA_merged.jsonl --b outputs/r179-m1/s5s6/runB_merged.jsonl \
  --a-label unpatched_vanilla --b-label patched_suppress \
  --out results/F179_M3_S5S6_LADDER_DIFF.json
# 4. part 2 engine leak (bundle p193-suppress-deploy, patched .so)
docker run ... python3 scripts/lpm_cq4_engine_leak.py   # -> results/F179_M3_S5S6_ENGINE_LEAK.{txt,json}
```

Machine-readable: `results/F179_M3_S5S6_LADDER_DIFF.json` (summary + 210 per-item
rows), `results/F179_M3_S5S6_ENGINE_LEAK.json` (+ raw `.txt`). Raw per-shard
engine output (gitignored) under `outputs/r179-m1/s5s6/run{A,B}/`.
