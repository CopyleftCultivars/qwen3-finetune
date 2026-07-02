#!/usr/bin/env python3
"""
Offline RAG corpus builder — run once before deploying to HF Spaces.

Runs curated qmd queries covering the full KNF reasoning space,
deduplicates results, and writes spaces/data/rag_corpus.json as a
plain list[str] (one passage per element, no metadata).

After running this, run scripts/precompute_embeddings.py to produce
spaces/data/rag_embeddings.npy for instantaneous Space startup.
"""

import json
import re
import subprocess
import hashlib
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "spaces" / "data"
OUT.mkdir(parents=True, exist_ok=True)

CURATION_QUERIES = [
    # KNF input preparation
    "how to prepare FAA fish amino acid fermentation steps ratio",
    "FPJ fermented plant juice recipe preparation timing young shoots",
    "OHN oriental herbal nutrient garlic ginger cinnamon preparation",
    "WCA water soluble calcium eggshell vinegar preparation",
    "LAB lactic acid bacteria serum rice wash milk preparation",
    "IMO indigenous microorganism collection forest aerobic",
    "IMO-3 propagation bokashi rice bran beneficial microbes",
    "FFJ fermented fruit juice sugar ratio ripe fruit preparation",
    "WSP water soluble phosphorus bones vinegar preparation",
    "BRV brown rice vinegar fermentation preparation",
    # Application rates and timing
    "KNF application rate foliar spray dilution ratio 1:500 1:1000",
    "natural farming input timing crop growth stage vegetative fruiting",
    "fermented input soil drench foliar feeding schedule frequency",
    "KNF input combination mixing compatible fermented inputs",
    # NPK sourcing from plants
    "high nitrogen plants bioaccumulator natural farming comfrey nettle",
    "phosphorus rich plants organic fertilizer source bone",
    "potassium bioaccumulator banana comfrey natural farming",
    # Plant-specific mineral profiles
    "nettle Urtica dioica nitrogen content mineral composition",
    "comfrey Symphytum officinale potassium mineral analysis",
    "banana Musa spp potassium fruit peel fermentation",
    "seaweed kelp mineral content trace elements farming",
    "dock sorrel Rumex mineral bioaccumulator nitrogen",
    "horsetail Equisetum silica mineral content natural farming",
    # Crop-specific guidance
    "vegetable crop KNF input program nitrogen phosphorus potassium",
    "fruiting crops phosphorus potassium fermented fruit juice",
    "leafy green nitrogen FPJ FAA natural farming",
    "root crops phosphorus potassium natural farming inputs",
    "rice natural farming input schedule transplanting",
    "tomato pepper fruiting KNF fermented inputs",
    # Soil biology and composting
    "soil microorganism activity KNF IMO beneficial bacteria fungi",
    "fermentation beneficial microbes soil health Korean natural farming",
    "composting ratio carbon nitrogen C:N organic matter",
    "soil pH correction natural farming lime calcium",
    "humus formation decomposition soil organic matter",
    "worm castings vermicompost mineral content application",
    # Livestock and broader natural farming
    "natural farming livestock health probiotic fermented feed",
    "seed treatment soaking germination natural farming SES",
    "pest disease prevention natural farming plant extract",
    # Regional and tropical plants
    "tropical plants nitrogen fixing natural farming humid",
    "cover crops green manure nitrogen phosphorus legume",
    "companion planting nutrient cycling agroforestry polyculture",
    # JADAM-specific techniques
    "JADAM natural farming JMS microbial solution preparation",
    "JADAM wetting agent saponin natural soap soapwort",
    "ultra low cost natural farming village indigenous",
    # Fermentation science
    "fermentation sugar brown sugar ratio KNF concentration",
    "aerobic anaerobic fermentation microorganism process",
    "vinegar acetic acid fermentation pH natural farming buffer",
    # Philosophy and principles
    "Cho Han-Kyu natural farming philosophy indigenous microorganisms",
    "Masanobu Fukuoka natural farming do-nothing philosophy",
    "JADAM ultra low cost agriculture fermented inputs Youngsang Cho",
]

TOP_PER_QUERY = 10
MIN_SCORE = 20


def qmd_query(q: str, top_n: int = TOP_PER_QUERY) -> list:
    try:
        r = subprocess.run(["qmd", "query", q], capture_output=True, text=True, timeout=90)
        return _parse_qmd(r.stdout, top_n)
    except Exception as e:
        print(f"  [qmd warning] {e}")
        return []


def _parse_qmd(text: str, top_n: int) -> list:
    results = []
    for block in re.split(r"\n(?=qmd://)", text):
        if not block.startswith("qmd://"):
            continue
        score = re.search(r"Score:\s+(\d+)%", block)
        content = re.search(r"@@[^\n]*\n(.*)", block, re.DOTALL)
        if content:
            score_val = int(score.group(1)) if score else 0
            if score_val >= MIN_SCORE:
                results.append({
                    "score": score_val,
                    "text": content.group(1).strip()[:600],
                })
        if len(results) >= top_n:
            break
    return results


def text_fingerprint(text: str) -> str:
    return hashlib.md5(text[:200].encode()).hexdigest()


if __name__ == "__main__":
    print(f"Running {len(CURATION_QUERIES)} curation queries (top_n={TOP_PER_QUERY}, min_score={MIN_SCORE})...\n")
    all_texts: list[str] = []
    seen_fps: set[str] = set()

    for i, query in enumerate(CURATION_QUERIES, 1):
        print(f"[{i:02d}/{len(CURATION_QUERIES)}] {query[:60]}...", end=" ", flush=True)
        results = qmd_query(query)
        new = 0
        for r in results:
            fp = text_fingerprint(r["text"])
            if fp not in seen_fps:
                seen_fps.add(fp)
                all_texts.append(r["text"])
                new += 1
        print(f"{new} new ({len(results)} total)")

    out = OUT / "rag_corpus.json"
    out.write_text(json.dumps(all_texts, indent=2, ensure_ascii=False))

    total_chars = sum(len(t) for t in all_texts)
    approx_tokens = total_chars // 4
    print(f"\n── Summary ──")
    print(f"  Chunks:         {len(all_texts)}")
    print(f"  Approx tokens:  {approx_tokens:,}")
    print(f"  Output:         {out}")
    print(f"\nNext: run scripts/precompute_embeddings.py to produce rag_embeddings.npy")
