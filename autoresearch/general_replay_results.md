# General-instruction replay results

Phase 19 tested one new lever: token-matched additions from the audited,
English Databricks Dolly 15k source to the complete v5clean training pool.
The QMD/new-MCQ authoring track was not changed or combined. All runs used the
v5clean recipe and were executed serially on the one CUDA GPU.

## Results

| Condition | English primary | English MCQ argmax | General validation token-F1 | General exact | Spanish transfer-dev primary | Spanish transfer-dev argmax |
|---:|---:|---:|---:|---:|---:|---:|
| v5clean / 0% control | -0.7752 | 72.6% | 0.2278 | 0.0% | -1.4337 | 52.3% |
| 5% replay | -1.9999 | 40.7% | 0.3225 | 2.0% | -1.5737 | 43.0% |
| 10% replay | -1.0717 | 59.3% | 0.3314 | 3.9% | **-0.9066** | **64.8%** |
| 20% replay | -1.4494 | 54.0% | **0.3567** | **5.1%** | -1.6632 | 41.4% |

The general-instruction metric improved monotonically with replay ratio, but
the closed-book English MCQ anchor regressed at every tested dose. Spanish
transfer-dev was non-monotone: 10% was its best result, while 5% and 20%
regressed below v5clean. The 10% candidate is therefore an interesting
diagnostic tradeoff, not a valid replacement or promotion candidate.

## Disposition

No candidate clears the English no-regression gate, so v5clean remains the
champion and this source/dose bracket is closed. Reopening the lever requires
a materially different general-instruction source or training recipe; do not
combine this rejected replay source with other levers in the autoresearch
loop. The local validation and Spanish transfer results are reported-only
diagnostics, not promotion gates.

The sealed English and Spanish benchmark tests were not read. Spanish
free-form data and external judging were not used.
