# D203 Gate 1 (#211) — S0+T0 baseline recall@1

Milestone #14 (Epic #179), parent #203. Gate: is the cheapest possible
query-independent selector (S0) close enough to #193's 0.250 oracle-fit
reference that #203 can close with **zero** new routing mechanism?

**Answer: no. S0 is at/below chance at #193's own fit settings and well below
the oracle at the issue's default settings. #203 is not closeable by S0; proceed
to Gate 2 (#212, S2a exact-latent selector).**

## What S0+T0 is (and is not)

- **S0** = one fixed slot index per head per persona, chosen once from
  *training-time* information only. For each persona we mean-pool that persona's
  own non-held-out training query encodings (`fit_q`) into a single fixed query,
  run `LPMModule.forward` **once**, and take `attention.argmax(dim=-1)` per head.
  No eval query ever touches selection.
- **T0** = always inject those 8 selected vectors, no per-turn decision.

This is explicitly **not** #193's oracle. #193's `oracle_pack`
(`scripts/lpm_quant_gap.py:oracle_pack`) computes `attn.argmax(dim=-1)` from
`probe_q`, the encoding of each val persona's **own held-out probe prompt**
(`fit_val_slots` lines 139-141), i.e. a different, more informative query for
every eval item. #211 must not reuse it or it would reproduce 0.250 by
construction.

## Method

`fit_val_slots` was extended to also return `fit_q, fit_pid` (the fitting loop
itself is unchanged; `stage_prepare`/`stage_dynamic` call sites updated). The new
`s0_select` mean-pools per persona, runs one forward, argmaxes per head, and
gathers `bake[k, h, top[k, h]]` exactly as `oracle_pack` does, yielding a
`[K, H, d]` tensor. A new `s0` subcommand scores it two ways:

1. **dynamic soft-prompt path** — the 8 selected vectors injected as soft
   prompts, scored with the same `forward_ntp` code as #193's 0.250. This is the
   matched comparison.
2. **reserved-row path** — the same 8 vectors written to reserved embedding rows
   and scored with `recall_matrix` (the literal T0 deployment form).

The `s0` run also scores the **probe_q oracle** on the same fitted module and
settings, as a matched same-fit control (the fair internal reference, since the
script's default fit budget differs from #193's — see below).

## Protocol

- val corpus `data/persona_memory/lpm_corpus_val.jsonl`, checkpoint
  `outputs/lpm-persona-memory-192/checkpoint.pt`, base model
  `outputs/merged-model-distill-v2-rationale` (bf16), encoder 6 layers.
- 24 val personas in two disjoint K=12 slices (`--persona-offset 0`, `12`),
  aggregated. Per-slice chance = 1/12 = 0.0833; nominal 24-way chance = 1/24 =
  0.0417.
- Two fit budgets were run because the issue's "defaults" are **not** #193's:
  - issue defaults `fit_steps=40, fit_max_length=128, max_length=192`
    (`quant_gap.py` parser defaults), and
  - #193's actual dynamic-run settings `fit_steps=20, fit_max_length=96,
    max_length=160` (per `F179_M3_QUANT_GAP_RESULTS.md` §3).

## Results

### Issue-default fit budget (40 / 128 / 192)

| selector | slice0 | slice1 | aggregate (24) | own NTP | own−offdiag |
|---|---|---|---|---|---|
| **S0 dynamic (T0)** | 0.1667 (2/12) | 0.1667 (2/12) | **0.1667 (4/24)** | 3.14 / 4.52 | −0.097 / −0.042 |
| oracle probe_q (matched) | 0.4167 (5/12) | 0.250 (3/12) | 0.3333 (8/24) | 2.31 / 3.27 | −0.256 / −0.054 |
| S0 reserved rows (deploy form) | 0.0833 (1/12) | 0.0833 (1/12) | 0.0833 (2/24) | 12.31 / 11.34 | +0.025 / +0.035 |

### #193's exact fit budget (20 / 96 / 160)

| selector | slice0 | slice1 | aggregate (24) | own NTP | own−offdiag |
|---|---|---|---|---|---|
| **S0 dynamic (T0)** | 0.0000 (0/12) | 0.0833 (1/12) | **0.0417 (1/24)** | 3.15 / 4.52 | −0.060 / +0.004 |
| oracle probe_q (matched) | 0.2500 (3/12) | 0.1667 (2/12) | 0.2083 (5/24) | 2.37 / 3.29 | −0.200 / −0.038 |
| S0 reserved rows (deploy form) | 0.0000 (0/12) | 0.1667 (2/12) | 0.0833 (2/24) | 8.71 / 11.24 | +0.063 / +0.030 |

### Comparison to #193's 0.250

#193 reported **0.250 (6/24)** for the probe_q oracle on the dynamic path at
fit budget 20/96/160. The matched same-budget oracle here reproduces it closely
at **0.2083 (5/24)** (small difference attributable to run/version
nondeterminism on the GPU bf16 path; slice0 3/12 vs 4/12).

Against that reference:

- **Issue defaults:** S0 = 0.1667 vs oracle 0.3333 and #193's 0.250 → S0 is
  50% of the matched oracle and two-thirds of the historical reference. Above
  per-slice chance (0.0833) but far from parity.
- **#193's own fit budget:** S0 = 0.0417 (1/24), i.e. **at/below** the operative
  per-slice chance (0.0833) and exactly the nominal 24-way chance. The matched
  oracle is 0.2083. Here S0 carries essentially no selection signal.

The own−offdiag margin corroborates this: the oracle soft prompt makes the
correct target clearly more likely than other personas' targets (−0.20 to
−0.26 nats), while the S0 prompt's margin is near zero and in one slice positive
(+0.004 nats), i.e. no discriminative benefit.

### The reserved-row (literal T0) path is not a selector test

Both reserved-row rows sit at chance (0.0833) with `own_ntp` ~11–12 nats vs
`no_memory` ~10 nats. That is the known tied-head contamination (#193 §4 / #204):
patching learned high-norm rows into the tied embedding table swamps the output
softmax, so the recall decision is dominated by contamination, not by which slots
were selected. It cannot discriminate selectors and is **not** evidence about S0.

## Decision

S0+T0 does **not** reach parity with the oracle reference under either fit
budget, and at #193's own budget it is at/below chance. Per the issue's own
criterion ("if it is far below, proceed to Gate 2"), this is the justification to
**proceed to #212 (S2a exact-latent selector)**. #203 is **not** closed by this
gate.

## Reproduction

```bash
PY=[LOCAL_PATH]

# issue-default fit budget (40/128/192), two slices
$PY scripts/lpm_quant_gap.py s0 --n-personas 12 --persona-offset 0
$PY scripts/lpm_quant_gap.py s0 --n-personas 12 --persona-offset 12

# #193's exact fit budget (20/96/160), two slices
$PY scripts/lpm_quant_gap.py s0 --n-personas 12 --persona-offset 0  --fit-steps 20 --fit-max-length 96 --max-length 160 --out-tag _m193fit
$PY scripts/lpm_quant_gap.py s0 --n-personas 12 --persona-offset 12 --fit-steps 20 --fit-max-length 96 --max-length 160 --out-tag _m193fit
```

Machine-readable per-slice output:
`results/D203_S0_BASELINE_offset{0,12}[_m193fit].json`. Each JSON also records
the selected S0 slot indices (`s0_top`, `[K,8]`) and per-head attention maxima.
