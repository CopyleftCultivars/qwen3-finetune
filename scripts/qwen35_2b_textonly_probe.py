#!/usr/bin/env python3
"""#236 Phase 0 / Q4: is there a genuine TEXT-ONLY mode for Qwen3.5-2B?

Runs inside the arm64 conversion container (transformers 5.5.4, torch 2.8 CPU).
Tries the text-only causal-LM loader classes against the plain HF checkpoint and
does a short greedy generation on a natural-farming prompt with NO image input.
Reports which loader class works (do not assume it is the same as Qwen3).
"""
import time
import torch
import transformers as T

MODEL = "Qwen/Qwen3.5-2B"
PROMPT = "The best fertilizer for tomatoes is"

print("transformers", T.__version__)
for c in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration", "AutoModelForCausalLM", "AutoModelForMultimodalLM"):
    print("has", c, hasattr(T, c))

tok = T.AutoTokenizer.from_pretrained(MODEL)
text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
print("rendered prompt (repr):", repr(text[:200]))

loaded = None
for cls_name in ("Qwen3_5ForCausalLM", "AutoModelForCausalLM"):
    cls = getattr(T, cls_name)
    try:
        t0 = time.time()
        m = cls.from_pretrained(MODEL, dtype=torch.float32)
        print(f"LOADED via {cls_name} in {time.time()-t0:.0f}s -> {type(m).__name__}")
        loaded = (cls_name, m)
        break
    except Exception as e:
        print(f"LOAD FAILED via {cls_name}: {type(e).__name__}: {str(e)[:400]}")

if loaded is None:
    print("TEXT_ONLY_LOAD: FAILED (no text-only class loaded)")
    raise SystemExit(1)

name, model = loaded
model.eval()
ids = tok(text, return_tensors="pt")
t0 = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
dt = time.time() - t0
gen = out[0][ids["input_ids"].shape[1]:]
print(f"GENERATED {int(gen.shape[0])} tokens in {dt:.0f}s")
print("OUT:", repr(tok.decode(gen, skip_special_tokens=True)))
print("TEXT_ONLY_LOAD: OK via", name)
