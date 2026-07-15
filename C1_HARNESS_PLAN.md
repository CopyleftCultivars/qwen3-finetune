# Plan: chainlink #22 — frozen/editable split + checksum manifest + finetune.py

## Context

Chainlink #21 (lever inventory, `autoresearch/levers.md`) and #26 (replay-ratio knob)
are done and committed. #22 is the next unblocked step in the Phase 4 autoresearch
harness: build the security boundary the whole loop depends on — a checksum-guarded
frozen manifest that aborts a run if anything it shouldn't touch gets edited, and
`autoresearch/finetune.py`, the *only* file an autonomous agent is allowed to modify
during the search loop, parameterized over exactly the SEARCH DIMENSION rows in
`levers.md`. This is broken into chainlink subissues sized for separate worker agents
rather than done as one large inline change, organized around minimizing shared-file
conflicts between workers.

Two things don't exist yet and are explicitly out of scope for #22, staying as noted
placeholders in the manifest for later issues to fill in: the sealed free-form/MCQ test
sets and scorer/grader artifacts (belong to #16/Phase 2.6 and #23/C2), and the
guardrail-script freezing question, including whether `../benchmark-eval-workspace/diagnostics/*`
(cross-repo) should be in-scope (belongs to #24/C4, which owns guardrail wiring).

## Design decisions

**`autoresearch/frozen_manifest.json`** — flat JSON: `{generated_at, algorithm: "sha256",
entries: {path: {sha256, kind: "file"|"dir", note}}}`. Directory entries hash every file
inside via a sorted-concatenation digest, so a new file silently added to a "frozen dir"
also trips the check (not just modifications to existing files). Frozen right now:
`data/val_v3.jsonl`, `data/val.jsonl` (the only held-out splits that currently exist —
noted as "dev-split proxy, not yet a sealed C3/C4 test set"), and `autoresearch/levers.md`
itself (finetune.py must not silently drift from the spec it's built against).

**`autoresearch/freeze_manifest.py`** — generator + checker in one file, argparse
subcommands: `generate --i-know-what-im-doing` (walks a `FROZEN_PATHS` list with inline
comments like `# TODO #23: add sealed test set` for future phases to extend) writes the
manifest; `check` recomputes and compares, printing a diff-style mismatch report and
calling `sys.exit(1)` on any mismatch (not a raised exception) — because `finetune.py`
invokes it as a subprocess (`subprocess.run(..., check=True)`) as its literal first step,
so a `CalledProcessError` propagates and kills the run before any GPU work starts.

**`autoresearch/finetune.py` invocation strategy: subprocess, not import.** `train.py`
sets `os.environ["UNSLOTH_RETURN_LOGITS"]="1"` before importing unsloth as its literal
first import — unsloth patches global torch/transformers state at import time in ways
that aren't safely re-enterable in a process that also wants to later invoke
`merge_adapter.py`'s own unsloth init. This also matches existing practice: `train.py`
and `merge_adapter.py` are already two separate process invocations today (per
`train.py`'s own docstring: "Next step: run finetune/merge.py..."). `finetune.py` runs
the checksum check, then chains `subprocess.run([sys.executable, "prepare_dataset_v3.py", ...])`
→ `train.py` → `merge_adapter.py`, translating its own flags into each subprocess's argv.

**`configs/` schema** — YAML, one file per experiment (`configs/<name>.yaml`), flat
key:value map with keys matching `finetune.py`'s flag names (minus `--`), because these
are hand-edited between runs and YAML supports inline comments explaining *why* a lever
was set a value — no existing config-file convention exists anywhere in this repo to
follow instead. CLI flags override YAML values when both are given. `schema_version: 1`
key for forward compatibility.

**`finetune.py`'s exact flag surface** (one per SEARCH DIMENSION row in `levers.md`):
`--lr`, `--lr-scheduler`, `--warmup-ratio`, `--epochs`, `--early-stop-patience`,
`--batch-size`, `--grad-accum`, `--lora-r`, `--lora-alpha`, `--lora-dropout`,
`--target-modules`, `--weight-decay`, `--max-seq-len`, `--packing`,
`--conversation-ratio`, `--target-permutation-seed`, `--sft-format`,
`--synth-item-count`, plus `--run-name` (drives `outputs/{lora-adapter,merged-model}-<run-name>`
naming, matching the existing `-v3-clean` convention) and `--config` (YAML path).

## Chainlink subissue breakdown (children of #22)

Three fully independent workers first (no shared files, no dependencies on each other),
then a convergence task, then two more sequenced ones:

1. **#27 — Frozen manifest system** — new file `autoresearch/freeze_manifest.py` +
   generated `frozen_manifest.json`. Touches only new files. No dependencies.
2. **train.py: remaining hyperparameter flags + eval-loop/early-stopping + packing** —
   adds `--lr-scheduler`, `--warmup-ratio`, `--weight-decay`, `--lora-dropout`,
   `--target-modules` as real flags (currently hardcoded); wires an eval loop
   (`eval_strategy` is currently `"no"`) so early-stopping is possible; implements
   sequence packing in `build_dataset()`. One file (`train.py`), bundled into one task
   specifically to avoid two parallel workers both editing `train.py`'s
   `build_training_args`/`build_parser`/`build_dataset` at once. No dependencies.
3. **prepare_dataset_v3.py: SFT target-format flag + target-permutation-seed flag** —
   adds an output-format alternative to the hardcoded `<think>...\n</think>\n{letter}`
   in `_cyclic_shift_pair`, and a flag consuming the 321-item augmentation seed set from
   `../benchmark-eval-workspace/results/position_spread_breakdown.json` (B3 diagnostic 3)
   to apply extra permutation coverage to just those items. One file, no dependencies
   (runs after #26's already-merged `--conversation-ratio` work, no live conflict).
4. **`finetune.py` skeleton + checksum gate + subprocess wiring** — the integration
   point. Blocked by #1 (needs the checker), #2 (needs train.py's final flag names),
   #3 (needs prepare_dataset_v3.py's final flag names). New file only.
5. **`configs/` dir + YAML loader wired into finetune.py + 1-2 example configs**
   (one reproducing v3-clean's exact known settings as a baseline reference). Blocked
   by #4 (needs finetune.py's flag surface to design the schema against).
6. **Final 1:1 audit against `levers.md` + `--synth-item-count` flag + close out any
   gaps** — verifies every SEARCH DIMENSION row maps to a real `finetune.py` flag,
   updates `levers.md`'s knob column where flag names differ from what was drafted.
   Blocked by #5.

Dependency graph: `{1, 2, 3}` parallel → `4` → `5` → `6`.

## Verification

- `freeze_manifest.py check` exits 0 on a clean tree, exits 1 with a clear diff after
  touching any frozen path (test: `touch data/val_v3.jsonl` or edit one byte, rerun
  check).
- `finetune.py --config configs/v3-clean-baseline.yaml --run-name test-smoke` runs the
  full chain (prepare → train → merge) end-to-end on a tiny smoke config (e.g. 1 epoch,
  tiny data subset) and produces `outputs/lora-adapter-test-smoke/` +
  `outputs/merged-model-test-smoke/`.
- Diff `finetune.py --help`'s flag list against `levers.md`'s SEARCH DIMENSION rows —
  every row has exactly one corresponding flag, nothing extra.
- Edit `autoresearch/levers.md` by one character, rerun `finetune.py` — confirm it
  aborts via the checksum gate before touching the GPU.

## Status

Subissues created under #22 as work proceeds; see `chainlink show 22` for current
children and blocking relationships. No implementation started yet as of this plan's
initial write — subissue creation and dependency wiring only.
