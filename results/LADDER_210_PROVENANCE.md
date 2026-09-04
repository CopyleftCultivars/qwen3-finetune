# Ladder provenance: is the ladder sealed against the distill training pool? (#161)

Date: 2026-09-04. Precondition for #160's option (B) — the 210-item ladder can
only gate deploy-shaped-lever promotion if it is sealed against what the distill
model trained on (`data/sources/teacher_distill_qwen3-8b.jsonl`, the 871
gold-filtered teacher MCQs that built `train_distill`).

Full per-item data: `results/ladder210_provenance_pairs.json`.

## 1. Source-document (topic) sharing

The ladder's `source_document` is a corpus label
`corpus:Natural Fertilizers Benchmark/<topic>`. The distill source carries the
same taxonomy in `_source_topic`. Joining on the topic:

- **180 / 210 ladder items share a source document (topic) with a distill
  training item.**
- **30 / 210 do not** — these are the two topics absent from the distill pool:
  `green_manures_cover_crops` (15) and `soil_health_microbial_activity` (15).

The 30 non-shared items are the cleanest possible held-out subset: their topics
never appear in the distill training data.

## 2. Question-level overlap (the auditable call)

For every shared-source ladder item, the closest same-topic distill question was
found and scored with the 8-gram Jaccard. Distribution over the 180 shared items:

| closest Jaccard | items |
| --- | --- |
| 0.00 | **178** |
| 0.05–0.10 | 2 |
| ≥ 0.60 (alarm) | **0** |

The only two non-zero pairs, shown in full:

> **NF-0004** (answer C): "What is the general rule-of-thumb for the daily
> consumption rate of compost worms, as stated in the 'Manual of On-Farm
> Vermicomposting and Vermiculture'?"
> vs distill: "In the 'Manual of On-Farm Vermicomposting and Vermiculture,' what
> is the goal of vermicomposting with respect to manure?"

> **NF-0011** (answer B): "What is a dietary characteristic of compost worms, as
> noted in the 'Manual of On-Farm Vermicomposting and Vermiculture'?"
> vs distill: same distill question as above.

Both cite the **same source document** ("Manual of On-Farm Vermicomposting and
Vermiculture") but ask **different questions about different facts** (consumption
rate / dietary characteristic vs. the goal of vermicomposting). Neither is a
restatement. This is "same source, different question" — the acceptable case.

**Verdict: zero question-level leakage.** No ladder question is restated in the
distill training data, even within the same source document.

## 3. The script bug (fixed)

`scripts/ladder_contamination.py` previously reported `source_document hits: 0`
by substring-searching the ladder's corpus label in the instruction/output
splits — a join on a key neither side carries, which silently read as a clean
pass. Fixed:

- `--source-pool <jsonl>` (repeatable) joins ladder `source_document` topics to
  the pool's `topic`/`_source_topic`.
- Without `--source-pool`, and when the splits carry no joinable source field,
  the script **exits non-zero** with `FAILED LOUDLY: the source-document signal
  cannot run … a source check that returns zero on a missing join key is a
  silent no-op.`

## 4. Is the ladder sealed enough to gate promotion?

The promotion criterion is a meaningful rise over the **champion** (v5clean), and
the distillation gain survives the cleanest subset:

| subset | distill | qwen3-8b | v5clean |
| --- | --- | --- | --- |
| shared-source (180) | 0.8889 | 0.8667 | 0.8000 |
| **non-shared (30)** | **0.7667** | **0.8000** | **0.6667** |
| all (210) | 0.8714 | 0.8571 | 0.7810 |

- **vs v5clean (the gate): robust.** The distill beats v5clean by ~+10 pp on the
  non-shared subset (0.767 vs 0.667) and ~+9 pp on the shared subset — the
  promotion criterion holds regardless of provenance.
- **vs the 8B teacher (the headline): not robust.** On the 30 items whose topics
  are absent from training, the teacher still wins (0.800 vs 0.767). The
  "0.8714 beats 0.8571" claim is concentrated on the shared-source items. The
  honest statement is the one already made in #160: the distill **matches** the
  8B at 0.6B; it does not cleanly beat it on source-disjoint items.

## Conclusion

The ladder is **lexically sealed** (0 exact, 0 near, max 8-gram Jaccard 0.091)
and **provenance-mapped** (180/210 topic-shared, 0 question restatements). It is
**sealed enough to gate promotion**, with one calibrated claim: the promotion
criterion (distill vs v5clean) is robust on the source-disjoint subset, but the
"beats the 8B teacher" headline should read as "matches the 8B". No ladder items
need to be excluded for the gate; the +9.0 pp over v5clean stands.
