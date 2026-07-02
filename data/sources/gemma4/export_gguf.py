import os, sys, getpass

HF_USERNAME     = "CopyleftCultivars"
HF_ADAPTER_REPO = f"{HF_USERNAME}/gemma4-natural-farming-lora"
HF_GGUF_REPO    = f"{HF_USERNAME}/gemma4-natural-farming-gguf"
MAX_SEQ_LENGTH  = 1024

# HF auth — use env var in CI, interactive prompt locally
token = os.environ.get("HF_TOKEN") or getpass.getpass("HuggingFace write token: ")

from huggingface_hub import login
login(token=token)
print(f"Logged in. Adapter: {HF_ADAPTER_REPO}")

# Patch subprocess before any unsloth import so llama.cpp errors print live
import subprocess
_orig_run = subprocess.run
def _loud_run(args, **kwargs):
    check = kwargs.pop("check", False)
    kwargs.pop("capture_output", None)
    result = _orig_run(args, capture_output=True, **kwargs)
    if result.returncode != 0:
        decode = lambda b: b.decode(errors="replace") if isinstance(b, bytes) else (b or "")
        print("\n=== llama.cpp STDOUT ===", flush=True)
        print(decode(result.stdout)[-4000:] or "(empty)", flush=True)
        print("=== llama.cpp STDERR ===", flush=True)
        print(decode(result.stderr)[-4000:] or "(empty)", flush=True)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args, result.stdout, result.stderr)
    return result
subprocess.run = _loud_run

import torch
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    import subprocess, os
    print("--- CUDA debug ---", file=sys.stderr)
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}", file=sys.stderr)
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print("nvidia-smi:", r.stdout or r.stderr, file=sys.stderr)
    # Try to surface the real torch CUDA init error
    try:
        torch.cuda.init()
    except Exception as e:
        print(f"torch.cuda.init() error: {e}", file=sys.stderr)
    sys.exit(1)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

print(f"Loading adapter from {HF_ADAPTER_REPO} ...")
model, tokenizer = FastModel.from_pretrained(
    model_name=HF_ADAPTER_REPO,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,   # bfloat16 — avoids bitsandbytes device_map issues,
    dtype=torch.bfloat16, # fits in 12GB VRAM for a 4B model (~8GB)
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma4")
print("Model loaded.")

# Pre-clone llama.cpp to ~/.unsloth/llama.cpp where unsloth expects it.
# Without this, unsloth tries to download it at runtime and fails if the
# Nix shell blocks the internet connectivity check it uses.
import pathlib
llama_cpp_dir = pathlib.Path.home() / ".unsloth" / "llama.cpp"
quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
if not quantize_bin.exists():
    print(f"ERROR: llama-quantize not found at {quantize_bin}", file=sys.stderr)
    print("Run  just build-llama-cpp  in a normal terminal first.", file=sys.stderr)
    sys.exit(1)
print(f"llama-quantize: {quantize_bin}")

# Save the merged model to the workspace before attempting GGUF conversion.
# This lets you upload gemma4-merged/ to Kaggle as a dataset and skip the
# slow merge step entirely if the GGUF conversion needs to be re-run.
MERGED_DIR = pathlib.Path(__file__).parent / "gemma4-merged"
if MERGED_DIR.exists() and any(MERGED_DIR.iterdir()):
    print(f"Merged model already at {MERGED_DIR} — skipping save.")
else:
    print(f"Saving merged model to {MERGED_DIR} ...")
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method="merged_16bit",
    )
    print("Merge saved.")

print("Exporting GGUF → HF Hub ...")
model.push_to_hub_gguf(
    HF_GGUF_REPO,
    tokenizer,
    quantization_method="q4_k_m",
    temporary_location="/tmp",
)
print(f"\nDone — https://huggingface.co/{HF_GGUF_REPO}")
