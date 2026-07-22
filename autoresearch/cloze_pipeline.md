# Cloze data pipeline

This is the first data-pipeline experiment after the Tier-1 and executable
Tier-2 sweeps. The current v5clean assembly is approximately 4,409 flattened
chat pairs plus 1,555 QMD MCQ variants after cyclic-shift expansion. QMD is
benchmark-adjacent, so the cloze stream is deliberately built from the clean
chat source instead of deriving more records from QMD questions.

The [public CROP repository](https://github.com/InternScience/Crop) documents
closed-book/open QA, event extraction, NER, summary, and multi-round dialogue
tasks; it does not document an explicit cloze task in the released training
recipe. This repository therefore treats cloze as a CROP-inspired hypothesis,
not as a claimed reproduction of CROP.

## Two-stage build

```bash
python3 scripts/generate_cloze_from_conversations.py \
  --input data/sources/gemma4/hf_dataset/train.jsonl \
  --output data/sources/cloze_chat_raw.jsonl

python3 scripts/check_cloze_contamination.py \
  --cloze data/sources/cloze_chat_raw.jsonl \
  --benchmark [LOCAL_PATH] \
  --output data/sources/cloze_chat_clean.jsonl \
  --report results/cloze_chat_contamination.json
```

The generator deterministically masks one domain phrase or measured value in a
short assistant sentence. It records source metadata and marks each row
`unverified`; the audit emits only rows marked `passed`. `prepare_dataset_v3.py`
refuses unaudited rows, so accidentally skipping the audit cannot silently add
the new stream to training.

## Training integration

Use an audited source explicitly, optionally with a deterministic cap:

```bash
python3 prepare_dataset_v3.py \
  --conversations data/sources/gemma4/hf_dataset/train.jsonl \
  --qmd-mcq data/sources/qmd_mcq_all_phase3.jsonl \
  --cloze-source data/sources/cloze_chat_clean.jsonl \
  --cloze-count 400 \
  --filter-system-prompts \
  --adversarial-refusals data/sources/adversarial_refusals.jsonl \
  --adversarial-refusal-count 10 \
  --output-dir data --suffix _cloze_probe
```

The cloze target is the exact missing phrase, while MCQ letter accounting is
kept separate so cloze answers ending in `A`, `B`, `C`, or `D` cannot alter the
gold-letter balance diagnostic. The first GPU experiment should hold v5clean's
settings fixed and vary only the audited cloze count/proportion.

The hygiene filter is separate from cloze authoring. It removes persona-only
conversation rows and strips the known composite persona prefix from QMD MCQ
instructions while retaining the underlying question. The reviewed refusal
source is opt-in; its rows are not mixed into the ordinary chat corpus.

The deployment prompt in `autoresearch/system_prompts/natural_farming.txt` is
also kept separate from training records. `autoresearch/compare_system_prompt.py`
reports the same model's domain MCQ score with and without that prompt, so a
prompt benefit is visible without changing the frozen primary gate.

This pipeline is intentionally deterministic and conservative. It is not a
replacement for expert-authored cloze questions; a later data-authoring pass
can add richer blanks once this controlled probe establishes whether the task
family helps the benchmark objective without hurting forgetting or rubric
quality.
