# F2 (#139): contamination check — 210-item ladder vs training/val splits

Date: 2026-09-03. Plan A.2, amended to the 210-item benchmark. Script:
`scripts/ladder_contamination.py`. Raw: `results/ladder_210_contamination.json`.

**Result: CLEAN. Zero contamination. The ladder is safe to score against.**

| Signal | Hits |
| --- | --- |
| Exact normalized-text match | **0 / 210** |
| Near match (8-gram Jaccard >= 0.60) | **0 / 210** |
| `source_document` appearing in training text | 0 / 210 (see caveat) |

Compared against 9,289 rows: `data/train_v3.jsonl` (4400),
`data/train_v6.jsonl` (4400), `data/val_v3.jsonl` (489).

## The detector was validated before its zero was believed

A detector that returns zero on everything is indistinguishable from a broken
one, so it was checked against positive controls:

| Control | Jaccard | Expected |
| --- | --- | --- |
| A `val_v3` row against itself | **1.000** | ~1.0 -- detector fires |
| A ladder question vs. a lightly-edited copy | **0.562** | high -- detector is edit-sensitive |
| **Max over all 210 x 9,289 real pairs** | **0.0130** | -- |

Items above 0.20 similarity: **0**. Items above 0.05: **0**. The observed
maximum (0.0130, NF-0031) sits two orders of magnitude below the 0.60 alarm, so
the conclusion does not depend on where the threshold was set -- any threshold
between 0.02 and 0.60 returns the same answer. That is a stronger claim than
"nothing crossed the line".

## Caveat on the source_document signal

The `source_document` check contributed nothing usable. Ladder rows carry values
like `corpus:Natural Fertilizers Benchmark/animal_manure` -- a synthetic corpus
*label*, not a document path -- so there is no filename that could appear in
training text. All 210 rows are populated, but the field cannot express the
overlap it was meant to detect. The exact and n-gram signals carry the result;
treat the source column above as inconclusive rather than as a third clean pass.

## Consequence

F3 (#140) may proceed and its numbers stand without a contamination adjustment.
No contaminated-item exclusion list is needed, and F4's report should state the
clean result together with the caveat above rather than claiming three
independent clean signals.
