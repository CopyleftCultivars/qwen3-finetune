import os, sys, torch, pathlib, getpass

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN") or getpass.getpass("HF token: "))

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastModel.from_pretrained(
    "CopyleftCultivars/gemma4-natural-farming-lora",
    max_seq_length=1024,
    load_in_4bit=False,
    dtype=torch.bfloat16,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma4")

out = pathlib.Path(__file__).parent / "gemma4-merged"
print(f"Saving merged model to {out.resolve()} ...")
model.save_pretrained_merged(str(out), tokenizer, save_method="merged_16bit")
print("Done. Upload gemma4-merged/ to Kaggle as a dataset.")
