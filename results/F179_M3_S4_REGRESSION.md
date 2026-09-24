# F179 / M3 — #227 (S4): disarmed regression (G4) + armed sanity (G5)

Milestone #14 (Epic #179), parent #223, fourth step of its S1–S6 work order.
Verification-only: no engine changes. Runs on engine commit **a82155c**
(#226 S3) in the fork `cactus-sdk-v210-suppress`, branch `v2.1.0-suppress-patch`.

**Both gates PASS. G4-primary: the new engine build is a byte-for-byte no-op on
the #185 210-item ladder for a bundle that never opts in (0/210 differ from
#207's committed baseline). G4-secondary: runtime `cactus_clear_memory_prefix()`
returns the armed bundle to byte-identical disarmed behaviour (12/12). G5: armed
generation emits 0 reserved IDs, no NaN, no empty completions, and differs from
disarmed on 11/12 prompts.**

## 0. Resolving the G4 scope ambiguity (both checks run)

The issue's original text ("`clear_memory_prefix()` → re-run the #185 ladder")
conflates two different bundle families: #207's ladder ran against
**p185-suppress-deploy**, while the armed/disarmed memory-prefix bundles are
built from **p193-suppress-deploy**. Both readings were executed separately, as
the handoff requested; neither is redundant.

| check | bundle | what it proves |
|---|---|---|
| G4 primary | `p185-suppress-deploy` (+ new engine) | new code compiled in is a no-op for bundles with no `memory_prefix_*` keys |
| G4 secondary | `gate_bundles/armed` + runtime `cactus_clear_memory_prefix()` | the runtime clear FFI path returns an armed model to disarmed behaviour |

## 1. G4 primary — #185 210-item ladder, new engine vs #207 baseline

Protocol exactly as #207: `scripts/cactus_engine_ladder.py` (benchmark's own
`evaluate.py` prompt/parser, `cactus_complete`, greedy, temperature 0,
`max_tokens=8`), 8 shards × stride 8. The reference is #207's committed
`runB_merged.jsonl` (the patched `p185-suppress-deploy` run) via
`results/F179_M3_S5S6_LADDER_DIFF.json`; this new run used the same bundle and
the **new** `.so` (a82155c).

| condition | engine | bundle | correct | accuracy | ambiguous |
|---|---|---|---|---|---|
| new engine | a82155c | p185-suppress-deploy | **174/210** | 0.8286 | 0 |
| #207 baseline | v2.1.0+Δ | p185-suppress-deploy | **174/210** | 0.8286 | 0 |

**Per-item diff: 0/210 differ**, `differ_question_ids: []`, `correct_delta: 0`
(`results/F179_M3_S4_LADDER_DIFF.json`). Because `p185-suppress-deploy` has no
`memory_prefix_file`/`memory_prefix_rows` keys, `memory_prefix_rows_` stays 0
and the prepend, row-overwrite, H2 and H6 paths are all skipped — this is the
empirical confirmation that #224–#226's code is inert unless a bundle opts in.

## 2. G4 secondary — runtime `cactus_clear_memory_prefix()` disarm

Take the armed p193 bundle, call `cactus_clear_memory_prefix()` immediately
after `cactus_init`, and compare greedy output to the never-armed
`p193-suppress-deploy` bundle on #214's 12 prompts (session reset before each
generation; single fresh prompt + one generation, per #226's handoff — no
multi-turn reuse). This exercises the runtime FFI path that G4-primary never
touches.

**Result: `armed_cleared == disarmed` ids on 12/12 prompts** — exact
byte-identical token sequences (`results/F179_M3_S4_G5_SANITY.json`,
verdict in `results/F179_M3_S4_G5_VERDICT.json`). The runtime clear is a true
disarm.

## 3. G5 — armed sanity

Same 12 prompts, greedy (`max_tokens=16`), armed
(`outputs/r179-m1/s2a/gate_bundles/armed`) vs disarmed
(`outputs/r179-m1/p193-suppress-deploy`). No `<|mem*|>` special tokens in any
prompt — tier A1's arming is a bundle property, not a prompt reference, so this
is a plain armed-vs-disarmed comparison (#214's old three-condition special-token
structure does not apply).

| i | prompt | reserved armed | reserved disarmed | armed≠disarmed | armed_cleared==disarmed |
|---|---|---|---|---|---|
| 0 | The best fertilizer for tomatoes is | 0 | 0 | ✓ | ✓ |
| 1 | One fact about soil: | 0 | 0 | ✓ | ✓ |
| 2 | How do I increase potassium in my garden? | 0 | 0 | ✓ | ✓ |
| 3 | My tomato leaves are turning yellow… | 0 | 0 | ✗ | ✓ |
| 4 | What is the best cover crop… Tennessee | 0 | 0 | ✓ | ✓ |
| 5 | I want to make fermented plant juice at home | 0 | 0 | ✓ | ✓ |
| 6 | How much water does a young apple tree need | 0 | 0 | ✓ | ✓ |
| 7 | Tell me about Korean natural farming inputs | 0 | 0 | ✓ | ✓ |
| 8 | What can I use to add nitrogen naturally… | 0 | 0 | ✓ | ✓ |
| 9 | Describe how to make biochar | 0 | 0 | ✓ | ✓ |
| 10 | What are good companion plants for corn | 0 | 0 | ✓ | ✓ |
| 11 | Explain crop rotation in one paragraph | 0 | 0 | ✓ | ✓ |

**Aggregate: 0 reserved-ID emissions in both conditions (pool
151670–151935), 0 NaN, 0 empty completions, 11/12 outputs differ → G5 PASS.**
Prompt 3 is byte-identical under both conditions (the prefix did not change that
continuation); the accept floor is ≥10/12.

Per the scope note, the content differences are **not** interpreted here as
evidence that the memory is legible/persona-specific — this is a
fire/no-crash/no-suppression-regression check only. The "does the corrected
object produce real persona content" question is #228 (S5)'s G6.

## 4. Independent re-runs (epic review requirement)

- **G4 primary comparator** (`cactus_engine_ladder.py compare`) re-run against
  its own output → byte-identical (`diff -q` clean).
- **Engine-determinism spot re-run**: 8 ladder items (one per shard offset) were
  re-generated in fresh containers and compared to the primary merged run →
  **0/8 differ** (ids + raw response).
- **G4-secondary/G5 comparator** (`scripts/s4_g5_verdict.py`) re-run against the
  same sanity JSON → byte-identical. The sanity run itself was executed twice and
  produced the same 11/12-differ + 0-reserved pattern.

## 5. Notes

- NaN detection uses a word-boundary regex (`\b(-?nan|-?inf(inity)?)\b`) plus a
  token-id/vocab and U+FFFD check. The first draft used a plain substring test
  that false-positived on "information"; fixed before the final run, and the
  verdict comparator recomputes NaN independently rather than trusting the raw
  field.
- `cactus_embed`/ordinary embedding paths were not touched by #226 and are out
  of scope here.
- No engine commit for this issue; all engine work is a82155c. This issue's
  artifacts are the comparator scripts + result JSONs in qwen3-finetune.

## Reproduce

```bash
QWEN=$(pwd)
FORK=[LOCAL_PATH]
LIB=--  # see below

# G4 primary: 8 shards against p185-suppress-deploy with the new .so
for i in $(seq 0 7); do  # (parallel)
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk -v "$QWEN":"$QWEN":ro \
  -v /tmp/ladder_new:/work/out -v "$QWEN/outputs/r179-m1/p185-suppress-deploy":/work/bundle:ro \
  -v "$QWEN/outputs/r179-m1/s5s6/ladder_prompts.jsonl":/work/prompts.jsonl:ro \
  -v "$QWEN/scripts/cactus_engine_ladder.py":/work/ladder.py:ro \
  -e CACTUS_LIB_PATH=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so -e PYTHONPATH=/src/cactus-sdk/python \
  cactus-arm64-convert:built python3 /work/ladder.py run --bundle /work/bundle --prompts /work/prompts.jsonl \
    --out /work/out/shard$i.jsonl --offset $i --stride 8 --max-tokens 8 --label s$i & done; wait
cat /tmp/ladder_new/shard*.jsonl > /tmp/ladder_new/merged.jsonl
python scripts/cactus_engine_ladder.py compare --prompts outputs/r179-m1/s5s6/ladder_prompts.jsonl \
  --a /tmp/ladder_new/merged.jsonl --b outputs/r179-m1/s5s6/runB_merged.jsonl \
  --a-label new_engine_p185suppress --b-label 207_baseline_patched \
  --out results/F179_M3_S4_LADDER_DIFF.json

# G4 secondary + G5: 12-prompt armed/armed_cleared/disarmed
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk -v "$QWEN":"$QWEN":ro \
  -v /tmp/s4:/work/out -v "$QWEN/scripts":/work/scripts:ro \
  -e CACTUS_LIB_PATH=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so -e PYTHONPATH=/src/cactus-sdk/python \
  cactus-arm64-convert:built python3 /work/scripts/s4_g5_sanity.py \
    --armed "$QWEN/outputs/r179-m1/s2a/gate_bundles/armed" \
    --disarmed "$QWEN/outputs/r179-m1/p193-suppress-deploy" \
    --out /work/out/s4_g5_sanity.json --max-tokens 16
python scripts/s4_g5_verdict.py --sanity /tmp/s4/s4_g5_sanity.json --out results/F179_M3_S4_G5_VERDICT.json
```

Artifacts: `results/F179_M3_S4_LADDER_DIFF.json` (210 per-item),
`results/F179_M3_S4_G5_SANITY.json` (raw per-prompt ids/text),
`results/F179_M3_S4_G5_VERDICT.json` (verdict + per-prompt rows);
scripts `scripts/s4_g5_sanity.py`, `scripts/s4_g5_verdict.py`
(`scripts/cactus_engine_ladder.py` reused unchanged from #207).
