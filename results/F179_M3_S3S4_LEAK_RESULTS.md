# F179 / M3 — #206 (S3-S4): leak A/B + argmax-parity gate

Milestone #14 (Epic #179). **Both gates PASS.** Real ARM engine under the M1
qemu/docker harness; no rebuild (the three `.so` artifacts already existed).

## S4 — MUST-VERIFY argmax parity: PASS (exact)

Control and treatment see **byte-identical bundle weights** (p193-suppress-deploy
is p193-deploy + the manifest key), so only the suppression differs. Greedy
decode, 16 prompts × 12 tokens, compared position-by-position on the host
(`scripts/cactus_engine_s3s4.py parity` + host comparator).

| comparison | what it tests | positions compared | mismatches |
|---|---|---|---|
| A. patched vs vanilla, **p193-deploy** (empty set) | patch is inert when the set is empty | 192/192 full-sequence equality (reserved included) | **0** |
| B. patched+suppress (**baseline+key**) vs vanilla (**baseline**) | denominator-only claim: biased path reproduces full-vocab argmax over the allowed vocab | 189 non-reserved control positions | **0** |
| C. patched vs vanilla on **p193-suppress-deploy** (vanilla has the gap) | active suppression: both suppress token 1 via different code paths | 19 non-reserved control positions | **0** |
| D. patched+suppress vs vanilla on **p193-deploy** (literal control) | the issue's literal framing | 0 comparable (control saturated: 192/192 reserved) | n/a (documented) |

Total: **400 compared positions, 0 mismatches.** Comparison A is exact even on
reserved tokens (empty set → patch inert). B is the load-bearing check: with the
suppress set populated, the dense-bias path picks exactly the same non-reserved
argmax as the unpatched sampler over 189 positions. C confirms both suppression
implementations agree where the gap-affected vanilla still emits a normal token
(token 1). D is vacuous only because the baked bundle saturates the tied head —
every control argmax is reserved — not because parity failed; A/B/C cover the
claim. **The denominator-only model is empirically confirmed.**

## S3 — leak A/B: PASS

Bundle `p193-suppress-deploy` for all three engines; real `cactus_tokenize`/
`cactus_complete` (never `cactus_submit`); empty / +1 mem / +8 mem prompts,
16 greedy tokens each; `CACTUS_DISABLE_CLOUD_HANDOFF=1` (removes a fixed 15 s
handoff wait; does not touch the sampling path).

| condition | engine | generations | tokens | reserved emitted | rate |
|---|---|---|---|---|---|
| C1-only | main-HEAD glibc, no suppression feature | 48 | 768 | **768** | 1.000 |
| v2.1.0, no Δ | vanilla glibc | 48 | 768 | **711** | 0.926 |
| v2.1.0 + Δ | patched glibc | **512** (base) + 48 (ext) | **8708** | **0** | **0.000** |

- C1-only is 16/16 reserved on every generation — reproduces #204's original
  finding, so the harness itself is validated.
- v2.1.0 no Δ leaks **from token 2**: exactly one non-reserved token (token 1,
  where `samp_ctx_active_` is still false and `argmax_logits_at` applies the set)
  before the repetition-penalty disjunct trips `samp_ctx_active_` and the graph
  branch skips the set. Predicted, and observed (711/768 = 768 − 57 ≈ 48 × 1.2).
- v2.1.0 + Δ: **0 reserved over 512 generations, and 0 over the extended
  560-generation / 8708-token run** — above the 8192-token / 266-ID bar, with
  every prompt×memory variant contributing. No tuning, no retries, under 8-bit
  acceptance.

Init time ~1.8–2.0 s per condition.

## Verdict

- Argmax parity: **exact** (0 mismatches / 400 positions).
- Leak: **0/266 over 8708 generated tokens** (> 512×16).

Both gates pass → #207 (regression: #185 ladder unchanged, #193 gap still
negligible) is the next step. No F3/F4 escalation needed.

## Reproduce

```
# S4 parity (per condition; one engine/bundle per container invocation)
docker run --rm --platform linux/arm64 -e CACTUS_LIB_PATH=/work/lib.so \
  -v <v2.1.0-tree>:/src/cactus-sdk -v <engine.so>:/work/lib.so:ro \
  -v <bundle>:/work/bundle:ro -v scripts/cactus_engine_s3s4.py:/work/run.py:ro \
  cactus-arm64-convert:built python3 /work/run.py parity --max-tokens 12 --out /work/out/p.json
# S3 leak (sharded): add --generations 512 --offset i --stride 16 --jsonl /work/out/s<i>.jsonl
```

Raw engine outputs (gitignored, under `outputs/r179-m1/s3s4/`):
`parity_*.json`, `parity_verdict.json`, `leak_patched_s*.jsonl`,
`leak_ext_*.jsonl`, `leak_c1_48.jsonl`, `leak_vanilla_48.jsonl`,
`leak_verdict.json`, `leak_patched_combined.json`.
