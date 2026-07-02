# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

### Fixed

### Changed
- Phase 1: Data Preparation Pipeline (#1)
- Create system_prompt_short.txt with shortened master prompt (~300 tokens) (#4)
- Phase 3: HuggingFace Spaces Chatbot (#3)
- Inject full master prompt at inference in Gradio app (#22)
- Create spaces/requirements.txt (#21)
- Create spaces/app.py Gradio chatbot with llama-cpp-python (#20)
- Phase 2: Gemma4 SFT Colab Notebook (#2)
- Export GGUF (q4_k_m) and upload to HF Hub (#19)
- Save and push LoRA adapter to HuggingFace Hub (#18)
- Configure TrainingArguments (batch=2, grad_accum=4, lr=2e-4, epochs=3, fp16) (#17)
- Configure SFTTrainer with ShareGPT chat template formatting (#16)
- Load merged dataset from HuggingFace Datasets repo (#15)
- Load Gemma4 E2B with 4-bit QLoRA via Unsloth (#14)
- Write Colab install cell (unsloth, trl, datasets, bitsandbytes, peft) (#13)
- Verify merged_sharegpt.jsonl output (#12)
- Merge all datasets in order and write merged_sharegpt.jsonl (#11)
- Convert syntheticdata-distiset-farming-chemistry to ShareGPT (30 rows) (#10)
- Convert Semisynthetic_Data_Natural_Farming_Fundamentals to ShareGPT (48 rows) (#9)
- Convert KNF-Methods-QnA to ShareGPT (100 rows) (#8)
- Convert Natural-Farming-Real-QandA to ShareGPT (36919 rows) (#7)
- Reconstruct multi-turn threads into ShareGPT format (#6)
- Merge & deduplicate conversation CSVs (#5)
