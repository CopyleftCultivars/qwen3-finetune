# L7 (#130) free-form dev track: run invalid, corpus gap found

Date: 2026-09-02. `eval_freeform/dev.jsonl`, 90 items, v5clean, greedy.
Rows in `results/freeform_slice_{base,R}.jsonl`.

## The run does not answer the question, because retrieval never fired

| | base | R (retrieval) |
| --- | --- | --- |
| Mean reference logprob | -3.5802 | -3.5596 |
| Mean answer length | 1674 chars | 1514 chars |
| Mean wall ms | 7909 | 11478 |

Paired delta reads mean +0.0206, t = +3.24 -- and that number should be
discarded. The median delta is exactly **0.0000**, because **64 of 90 items
retrieved zero snippets**:

| Snippets returned | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Items | **64** | 2 | 4 | 1 | 2 | 17 |

79/90 items fell through BM25 to vsearch, and 64 found nothing either way. On
those items the R prompt is byte-identical to base, so the delta is structurally
zero and the apparent t-statistic is driven by the ~22 items where retrieval did
return something. This is a **retrieval failure, not a null result about
retrieval's value on free-form**.

## Root cause: the free-form track and the indexed corpus are disjoint

All 90 dev items carry `source_doc` under
`../qtmd_workspace/markdown_corpus/...`. The QMD collections adopted in
CORPUS_DECISION.md are rooted at `benchmark-workspace/organized/`. **These are
different document sets.** The free-form questions were generated from a corpus
that is not indexed, so retrieval had nothing correct to find -- searching
"Strawberries Growing Selection Guide" returns an unrelated organic gardening
book at score 0.40.

That corpus exists locally and is not indexed:
`[LOCAL_PATH]`, 1586
markdown files across 1854 topic directories.

A secondary factor: free-form questions are eval scaffolding, not user
questions -- "A grower asks for advice related to the source topic X. They want
the answer to combine ... rather than giving a single generic tip." That phrasing
is a poor BM25 query even when the target document *is* indexed, and is the case
the model-authored query (`query_mode="model"`, #133) exists to handle. It was
not used here; this run retrieved on raw question text.

## By type (retained only as a hint, given the above)

| Type | n | mean delta | improved |
| --- | --- | --- | --- |
| quantitative | 18 | +0.0752 | 10/18 |
| multi_hop_factual | 18 | +0.0111 | 8/18 |
| diagnostic_reasoning | 18 | +0.0107 | 1/18 |
| free_form_explanation | 18 | +0.0032 | 2/18 |
| constraint_satisfaction | 18 | +0.0027 | 1/18 |

The two types where retrieval fired most often are also the two that gained
most. That is consistent with retrieval helping when it works, and is the reason
to fix the corpus gap rather than write the free-form arm off.

## Status: #130 is NOT closed

Two things block a verdict:

1. **Corpus gap.** The free-form source corpus is unindexed. Fixing it means
   adding `qtmd_workspace/markdown_corpus` to QMD (1586 files, plus an embedding
   pass) -- a change to the shared global index, and an owner decision.
   Note the contamination is far more acute here than for MCQ: each item's
   `reference_answer` is derived from its `source_doc`, so indexing that corpus
   means retrieval can surface the document the graded answer came from. That is
   the real deployment configuration, but it is close to an answer key for
   measurement purposes.
2. **No rubric grade.** `OPENROUTER_API_KEY` is unset, so `score_rubric.py`
   cannot run. Candidates are generated and sit in the exact shape
   `--input` expects, so grading is a single command once a key is available.
   Reference-LL is a proxy and the scorer config's own divergence alarm warns
   against accepting it in place of the rubric.
