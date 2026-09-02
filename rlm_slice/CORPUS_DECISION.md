# L9 (#132): corpus decision for the RAG arm

Date: 2026-09-02. Satisfies `LOCAL_RLM_SLICE_PLAN.md` sec.3.3.1 and the L9
acceptance criterion ("either a named corpus with a license and provenance note,
or a recorded 'defer' decision").

**Decision: ADOPT** the QMD agronomy collections. The RAG arm (#118) is
buildable, so I13 does not defer.

## The corpus

Ten agronomy collections indexed by QMD (`qmd collection list`), rooted at
`[LOCAL_PATH]`:

| Collection | Files |
| --- | --- |
| Natural_Fertilizers | 143 |
| Soil_Science_and_Microbiology | 115 |
| Organic_Farming | 110 |
| Indigenous_and_Traditional_Knowledge | 73 |
| Agroecology | 69 |
| Sustainable_Water_Management | 65 |
| Regenerative_Agriculture | 59 |
| Integrated_Farming_Systems | 56 |
| Permaculture | 53 |
| cannabis2 | 0 (empty -- exclude) |
| **Total (non-empty)** | **743** |

`miso` (13) and `symbolic-execution` (69) are unrelated and are excluded.
Index: `~/.cache/qmd/index.sqlite`, 825 docs, 328,809 vectors.

## Retrieval mode selection

40 MCQ rows from `data/val_v3.jsonl`, k=5, measured by `rlm_slice/corpus_probe.py`.
"Gold margin" = (fraction of gold-option content words appearing in the
retrieved text) minus (the same, averaged over the three distractors). It asks
whether retrieval surfaces text that *discriminates* the answer, rather than
merely on-topic filler.

| Mode | Dedupe | Items with hits | Gold margin | Margin > 0 | Dup hits / 5 | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| search (BM25) | no | 34/40 | +0.023 | 24% | 2.79 | 1.14 s |
| search (BM25) | **yes** | 34/40 | **+0.068** | **44%** | 0 | **1.21 s** |
| vsearch (vector) | no | 40/40 | +0.045 | 30% | 3.18 | 7.22 s |
| vsearch (vector) | yes | **40/40** | +0.057 | 38% | 0 | 5.65 s |

**Deduplication is the single biggest lever and is nearly free.** The same
source document is indexed under several collections -- `edible-forest-gardens`
appears in Permaculture, Agroecology and Integrated_Farming_Systems under one
identical QMD docid -- so an undeduplicated top-5 averaged only ~2 distinct
documents. Collapsing by docid tripled the gold margin and nearly doubled the
positive fraction at +70 ms.

**Recommended `retrieve()` for L10 (#133):** `qmd search` with over-fetch
(n = 4k) and docid dedupe, falling back to `vsearch` for the ~15% of questions
BM25 misses entirely. That buys vsearch's full coverage at BM25's latency for
the common case.

**`qmd query` is excluded.** It runs LLM query expansion and reranking. Putting
a larger model inside an arm whose premise is a 0.6B on a phone would confound
the comparison against v5clean (sec.2.3) and inflate the sec.3.5.3 latency
estimate's honesty. `qmd search` is documented as "no LLM"; `vsearch` is vector
similarity only. Both are defensible; `query` is not.

## Two caveats the gate must carry

**1. The signal is real but weak.** Gold overlap 0.166 vs distractor 0.098, and
only 44% of items have a positive margin -- meaning for the majority, retrieved
text supports the distractors as much as the gold. This does not predict a
Pass-2 gain. Read against sec.6.4 (the champion already gains `delta_trace` =
+0.167 from its own trace), the RAG arm has to beat a baseline that is already
benefiting from reasoning. Adopting the corpus makes the arm *testable*; it does
not make it likely to clear the sec.3.5 gate.

**2. Provenance is the same tree the benchmark came from.** The collections sit
under `benchmark-workspace/organized/`, and `mcq_generation/` generated MCQs
from these collections by RAG. So retrieval can surface the passage a question
was written from. That is optimistic relative to an unseen question -- but it is
also the real deployment configuration, since the app would ship this corpus.
Both things are true, so sec.3.3.5's "measure the RAG arm separately" is what
keeps it honest, and the gold-margin column above is the check on how much work
retrieval is actually doing. Any headline RAG win should be reported with this
caveat attached, never as a clean generalization result.

**Licensing, unresolved and owner-decided.** The corpus is extracted markdown
from published books (e.g. Holmgren, *Permaculture: Principles and Pathways*;
Jacke & Toensmeier, *Edible Forest Gardens*). Local research evaluation is one
thing; **bundling these files into a shipped Flutter app is a separate licensing
question that must be settled before I15 (#120)**, and this eval does not
settle it. Flagged here so the port does not inherit the assumption silently.
