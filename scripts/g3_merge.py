#!/usr/bin/env python3
"""#239 G3: merge the G3 Qwen3.5 LoRA adapter into a standalone checkpoint.

PEFT merge_and_unload on CPU (bf16 2B fits RAM), then
g2_hf_weight_audit.canonicalize_checkpoint so raw-weight consumers see
canonical tensor names and the 15 mtp.* head tensors are restored
byte-verbatim (same defect class G2 found and fixed).

Run with the isolated transformers-5.5 venv AFTER g3_qwen35_sft.py:
  V=localllm-qwen3.5/retry-evidence/q1-isolated/venv/bin/python
  $V scripts/g3_merge.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter",
                        default="outputs/lora-adapter-g3-qwen35-distill/final")
    parser.add_argument("--base", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output",
                        default="outputs/merged-model-g3-qwen35-distill")
    args = parser.parse_args(argv)

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from g2_hf_weight_audit import canonicalize_checkpoint

    adapter = REPO / args.adapter
    output = REPO / args.output
    assert adapter.exists(), f"adapter not found: {adapter}"

    print(f"loading base {args.base} on CPU ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True)
    print("loading adapter ...", flush=True)
    peft_model = PeftModel.from_pretrained(base, str(adapter))
    print("merging ...", flush=True)
    merged = peft_model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True)
    tok.save_pretrained(str(output))

    base_dir = Path(snapshot_download(args.base))
    report = canonicalize_checkpoint(output, base_dir)
    (output / "merge.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    files = sorted(p.name for p in output.iterdir())
    print(f"merged -> {output}: {files}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
