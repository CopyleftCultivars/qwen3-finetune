# #238 G2 RESULT -- tiny Qwen3.5 adaptation: save/reload/merge PASS, Cactus deployment BLOCKED

Date: 2026-09-30. Machine record: `results/QWEN3_5_2B_G2_TINY_ADAPT.json`.
Raw evidence (gitignored, on this host): `outputs/g2_tiny_adapt_metrics.json`,
`outputs/g2_roundtrip/` (attach/merge/verify/compare + logits),
`outputs/g2_hf_weight_audit.json`, `outputs/g2_cactus_deploy.json`,
`outputs/lora-adapter-g2-qwen35-tiny/`, `outputs/merged-model-g2-qwen35-tiny/`.
Pipeline: `scripts/g2_training_rows.py`, `scripts/g2_benchmark_overlap.py`,
`scripts/g2_qwen35_tiny_adapt.py`, `scripts/g2_qwen35_adapter_roundtrip.py`,
`scripts/g2_hf_weight_audit.py` (incl. `canonicalize_checkpoint`),
`scripts/g2_cactus_deploy_check.py`.
Unit tests: `tests/test_g2_weight_audit.py`, `tests/test_g2_training_rows.py`
(13 passed, CPU-only). `ruff check` clean on all G2 files.

Environment (the #236 retry isolated venv): transformers 5.5.0 + Unsloth
2026.6.9, bf16 LoRA (`load_in_4bit=False`), torch 2.10.0+cu128, RTX 4070 Ti
12 GB. Base: `Qwen/Qwen3.5-2B` @ `15852e8c` (`Qwen3_5ForConditionalGeneration`,
24 layers = 18 Gated-DeltaNet + 6 full attention).

## 1. Module targets derived from the actual model (not the Qwen3 recipe)

Linear inventory of the base model: 285 linears = 90 GDN attention
projections + 24 full-attention projections + 72 MLP projections + 1 tied
head/embed + 98 vision-tower linears. Language leaves:
`in_proj_qkv/z/b/a`, `out_proj` (GDN only), `q/k/v/o_proj` (full-attn only),
`gate/up/down_proj` (MLP).

LoRA (r=16, alpha=16, seed 3407) adapted 186 modules / 372 tensors /
16,819,200 trainable params: all 90 GDN projections, all 24 full-attention
projections, all 72 MLP projections. `gdn_leaves_requested_but_missing: []`.
Intentional omissions: vision tower (text-only adaptation), tied
embeddings/lm_head, all norms/biases, GDN SSM state params (`A_log`,
`conv1d`, `dt_bias`, `norm.weight` -- F32, not LoRA targets), MTP
speculative head (not a LoRA target; restored byte-identical in the merged
file). Unsloth compiled the requested leaves to a PEFT target regex (recorded
in the metrics); every requested leaf is confirmed wrapped.

## 2. Real optimizer steps on non-benchmark examples

8 steps, batch-size 1, lr 2e-4, max-seq-len 512, first 8 usable rows of
`data/train.jsonl`. Overlap vs the frozen 210-item ladder (same 8 rows):
0 exact normalized hits, max 8-gram Jaccard 0.0 (< 0.6 alarm) -> NON_BENCHMARK.
Loss 1.1731 -> 0.9380, all losses finite, all 372 grad tensors finite every
step, nonzero throughout. Adapter delta L2 2.5265, 372/372 LoRA tensors
changed. Wall 13.65 s, peak VRAM 8542.8 MiB. Adapter + tokenizer saved to
`outputs/lora-adapter-g2-qwen35-tiny/`.

## 3. Save / reload / merge (fresh process per stage)

* attach (PEFT reload of saved adapter): 372 LoRA tensors, L2 norm 31.61,
  nonzero; probes generate.
* merge (`merge_and_unload`, merge arithmetic vs `s*B@A`: 186 tensors
  checked, worst rel err 0.0039 <= bf16 eps, within rounding): 186 changed /
  431 unchanged, names and shapes identical to base in-memory.
* verify (plain transformers load of the saved merged dir, no PEFT):
  bitwise-identical logits to the in-memory merged model on all 4 probes
  (worst delta 0.0) with identical greedy text -> disk round-trip PASS.
* compare: PEFT-attach vs merged weights agree top-1 on all probes within
  2 bf16 ulps (greedy 32-token texts differ on near-tie flips -- reported,
  not gated); base-vs-adapted worst logit delta 9.05 (36x the arithmetic
  noise), all 4 greedy texts changed -> adaptation-sensitive PASS.
* tokenizer/config: 10/10 encode cases identical incl. chat renders and
  unicode/punctuation edges (files differ byte-wise from an Unsloth
  processor re-save -- functional equivalence, not byte identity);
  loaded model-defining config fields equal (file diffs are materialized
  defaults: dtype, partial_rotary_factor, transformers_version).

Verdicts: `save_reload_merge_fidelity PASS`, `adaptation_sensitive_probe PASS`.

## 4. HF weight audit of the merged file (raw safetensors bytes)

One real defect found and fixed in this session: the Unsloth+PEFT save path
writes language tensors as
`model.language_model.language_model.language_model.<rest>` and vision
tensors as `model.language_model.visual.<rest>` (transformers remaps both on
load, so HF inference was unaffected, but raw-weight consumers see wrong
names). `canonicalize_checkpoint` (key rewrite only, bytes untouched) renamed
617 keys and restored the 15 `mtp.*` tensors byte-verbatim; the merge stage
now canonicalizes on save (reported in `merge.json`), covered by unit tests.

Post-fix audit vs the base snapshot: 632/632 names identical, all shapes
identical, 186 value-changed = exactly the LoRA-adapted set (no unexpected,
none missing), 36 F32->BF16 GDN tensors inside bf16 rounding
(`dtype_cast_only`), 0 `dtype_changed`, 410 identical. Verdict PASS.

## 5. Cactus deployment: custom deployment BLOCKED (recorded, not assumed)

* Manifest mapping: all 186 adapted HF names resolve in the stock CQ
  conversion manifest, all `converted`/CQ4 -- the converter's name mapping
  covers the adapted set, and adapted deltas would flow through INT4
  quantization with unmeasured effect. Mapping only.
* Changed-weight checksums: raw sha256 of all 186 changed tensors differs
  base-vs-merged (186/186 differing) -- the change is measured at HF level.
* Hosted bundle (pinned G1 artifact, re-inspected): 396 files, 389 flat
  `*.weights`, no `components/manifest.json`, no graph, no runtime plan ->
  weights-only, NOT runnable. There is nothing to deploy adapted weights into.
* Upstream SDK (v2.2.1, read-only source quotes in the deploy JSON):
  `finetuning.md` states converts "produce CQ weights only" and "local
  runtime bundle generation is unavailable while the graph builder is being
  rewritten"; the CLI resolver (`utils.py:19`) matches only `-cqN` archives
  (hosted `-int4`/`-int8` do not match); no `qwen3_5` profile exists in the
  transpiler profiles; the `convert --lora` merge path exists but terminates
  at weights-only plus the binding #380 graph failure.
* Weight replacement was NOT attempted and is recorded as insufficient: the
  hosted layout has no graph/manifest, the CLI cannot fetch it, and the
  engine refuses it (G1 evidence). A changed file is not proof the runtime
  uses it. Proof-of-use criterion (runnable adapted artifact + matched
  HF-vs-Cactus adaptation-sensitive probes) NOT MET.

Exact upstream requirements (ordered): R1 -- Cactus-Compute publishes a
`-cqN` runnable bundle for Qwen3.5-2B (then re-run G1, `cactus convert <base>
--lora <adapter>`, matched probes); R2 (alternative) -- upstream `qwen3_5`
export-patch profile fixing the `_update_linear_attn_mask` torch.export guard
so local graph build succeeds. #380 remains binding for the local-graph route.

Stock benchmarking stands independently on the HF runtime; BLOCKED gates only
the on-device comparison leg. G3 may proceed on HF; G4/G5 wait on R1/R2.
