#!/usr/bin/env python3
"""
Offline embedding pre-computation — run after build_rag_corpus.py.

Loads the same GGUF that runs on HF Spaces, embeds every passage in
rag_corpus.json, and saves the L2-normalised vectors to
spaces/data/rag_embeddings.npy.

When rag_embeddings.npy is present in the Space, rag.py loads it
directly at startup instead of recomputing (~5-10 min on CPU for
300+ chunks).

Requires: llama-cpp-python, numpy
One-time GGUF download (~3 GB) cached in ~/.cache/huggingface/
"""

import json
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "spaces" / "data"

GGUF_REPO = os.getenv("GGUF_REPO", "CopyleftCultivars/gemma4-natural-farming-gguf")
GGUF_FILE = os.getenv("GGUF_FILE", "gemma4.Q4_K_M.gguf")


def main() -> None:
    corpus_path = DATA / "rag_corpus.json"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} not found — run build_rag_corpus.py first")

    corpus: list[str] = json.loads(corpus_path.read_text())
    print(f"Corpus: {len(corpus)} chunks")

    print(f"Loading {GGUF_REPO} / {GGUF_FILE} for embedding...")
    from llama_cpp import Llama
    llm = Llama.from_pretrained(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        embedding=True,
        n_ctx=512,
        verbose=False,
    )
    print("Model ready. Embedding chunks...")

    vecs = []
    for i, text in enumerate(corpus):
        resp = llm.create_embedding(text)
        v = np.array(resp["data"][0]["embedding"], dtype=np.float32)
        if i == 0:
            print(f"Embedding shape: {v.shape}")
        if v.ndim == 2:
            v = v.mean(axis=0)
        norm = np.linalg.norm(v)
        vecs.append(v / norm if norm > 0 else v)
        if (i + 1) % 50 == 0 or (i + 1) == len(corpus):
            print(f"  [{i+1}/{len(corpus)}]")

    matrix = np.stack(vecs)
    out = DATA / "rag_embeddings.npy"
    np.save(str(out), matrix)
    print(f"\nSaved {matrix.shape} embeddings → {out}")
    print(f"File size: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
