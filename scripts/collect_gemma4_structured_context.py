#!/usr/bin/env python3
"""Convert structured reference tables under data/sources/gemma4/datasets_for_rag/ into
instruction/output training pairs via deterministic (non-LLM) templates, and append them to
data/sources/gemma4/hf_dataset/train.jsonl.

These are RAG-corpus-style structured facts (recipes, crop requirements, composting methods,
plant-by-region lists, bioaccumulator NPK tables, mineral profiles, regenerative-ag scenarios),
not conversation logs — templating avoids LLM cost/hallucination risk and is exact by
construction, matching the pre-existing precedent in crop_mineral_data/plant_mineral_profiles_in_phrases.md.

Explicitly OUT of scope (see MCQ_DIAGNOSTIC_PLAN.md and chainlink for why):
  - CROP-dataset/ (232MB, 210K generic rice/corn QA, English+Chinese) — not natural-farming
    specific; its single/multi-turn QA *structure* is earmarked as a template for a future
    NF-focused, English/Spanish dataset (separate initiative, not built here).
  - CollegeOfAgricultureAndForestry_Agricultural_Crop_Dataset/Crop_recommendation.csv — raw
    NPK/climate -> crop-label ML rows, not natural-farming specific.

Usage:
    python scripts/collect_gemma4_structured_context.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GEMMA4_DIR = Path("data/sources/gemma4")
RAG_DIR = GEMMA4_DIR / "datasets_for_rag"
TRAIN_PATH = GEMMA4_DIR / "hf_dataset" / "train.jsonl"


def _sharegpt_pair(instruction: str, output: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": instruction.strip()},
            {"from": "gpt", "value": output.strip()},
        ]
    }


def _load_json_repairing_missing_commas(path: Path) -> list:
    """Load a top-level JSON array, tolerating malformed punctuation between elements
    (missing commas, or entirely missing `},{` between elements — both seen in these
    source files). Extracts each top-level `{...}` object independently via brace-depth
    matching and parses it on its own, so punctuation between objects can't break parsing
    as long as each individual object's own JSON is well-formed.
    """
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    objects: list[dict] = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


def _bioaccumulators(path: Path) -> list[dict]:
    records = _load_json_repairing_missing_commas(path)
    pairs = []
    for r in records:
        ingredient = r.get("Ingredient", "").strip()
        n, p, k = r.get("N", "--"), r.get("P", "--"), r.get("K", "--")
        if not ingredient or all(v == "--" for v in (n, p, k)):
            continue
        parts = []
        for label, val in (("nitrogen (N)", n), ("phosphorus (P)", p), ("potassium (K)", k)):
            if val != "--":
                parts.append(f"{label} at {val}")
        output = (
            f"{ingredient} is a natural farming bioaccumulator reported to provide "
            f"{', '.join(parts)}."
        )
        pairs.append(_sharegpt_pair(
            f"What NPK (nitrogen-phosphorus-potassium) values does {ingredient} provide as a "
            "natural fertilizer bioaccumulator?",
            output,
        ))
    return pairs


def _recipes(path: Path) -> list[dict]:
    records = _load_json_repairing_missing_commas(path)
    pairs = []
    for r in records:
        name = r.get("Recipe", "").strip()
        ingredients = r.get("Ingredients", [])
        steps = r.get("Steps", [])
        if not name or not steps:
            continue
        output = (
            f"Ingredients: {', '.join(ingredients)}.\n\nSteps:\n"
            + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        )
        pairs.append(_sharegpt_pair(f"How do I make {name} for natural farming?", output))
    return pairs


def _crop_requirements(path: Path) -> list[dict]:
    records = _load_json_repairing_missing_commas(path)
    pairs = []
    for r in records:
        crop = r.get("Crop", "").strip()
        sci = r.get("Scientific Name", "").strip()
        npk = r.get("NPK Requirements", {})
        extra = r.get("Additional Requirements", {})
        if not crop or not npk:
            continue
        npk_str = ", ".join(f"{k}: {v}" for k, v in npk.items())
        extra_str = ", ".join(f"{k}: {v}" for k, v in extra.items())
        output = f"{crop} ({sci}) requires {npk_str}."
        if extra_str:
            output += f" Additional growing conditions: {extra_str}."
        pairs.append(_sharegpt_pair(
            f"What are the NPK and growing requirements for {crop}?", output
        ))
    return pairs


def _composting_methods(path: Path) -> list[dict]:
    records = _load_json_repairing_missing_commas(path)
    pairs = []
    for r in records:
        method = r.get("Method", "").strip()
        ingredients = r.get("Ingredients", [])
        steps = r.get("Steps", [])
        if not method or not steps:
            continue
        output = (
            f"Ingredients: {', '.join(ingredients)}.\n\nSteps:\n"
            + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        )
        pairs.append(_sharegpt_pair(f"How do I do {method} for composting?", output))
    return pairs


def _plants_by_region(path: Path) -> list[dict]:
    records = _load_json_repairing_missing_commas(path)
    pairs = []
    for r in records:
        location = r.get("Location", "").strip()
        plants = r.get("Plants", [])
        if not location or not plants:
            continue
        pairs.append(_sharegpt_pair(
            f"What plants grow locally in {location} that could be used in natural farming?",
            f"Locally growing plants in {location} include: {', '.join(plants)}.",
        ))
    return pairs


def _mineral_profiles(path: Path) -> list[dict]:
    records = _load_json_repairing_missing_commas(path)
    pairs = []
    for r in records:
        common = r.get("common_name", "").strip()
        sci = r.get("scientific_name", "").strip()
        parts = r.get("parts", {})
        if not common or not parts:
            continue
        part_descs = []
        for part_name, part_data in parts.items():
            minerals = part_data.get("minerals", {})
            if not minerals:
                continue
            mineral_strs = []
            for mineral, vals in minerals.items():
                unit = vals.get("unit", "")
                lo, hi = vals.get("min"), vals.get("max")
                if lo is not None and hi is not None:
                    mineral_strs.append(f"{mineral} {lo}-{hi} {unit}")
                elif hi is not None:
                    mineral_strs.append(f"{mineral} up to {hi} {unit}")
                elif lo is not None:
                    mineral_strs.append(f"{mineral} at least {lo} {unit}")
            if mineral_strs:
                part_descs.append(f"the {part_name} contains {', '.join(mineral_strs)}")
        if not part_descs:
            continue
        pairs.append(_sharegpt_pair(
            f"What is the mineral content of {common} ({sci})?",
            f"For {common} ({sci}): " + "; ".join(part_descs) + ".",
        ))
    return pairs


def _regenerative_scenarios(path: Path) -> list[dict]:
    records = _load_json_repairing_missing_commas(path)
    pairs = []
    for r in records:
        scenario = r.get("scenario_description", "").strip()
        actions = r.get("actions", [])
        outcome = r.get("expected_outcome", "").strip()
        if not scenario or not actions:
            continue
        output = "Recommended actions: " + "; ".join(actions) + "."
        if outcome:
            output += f" Expected outcome: {outcome}"
        pairs.append(_sharegpt_pair(
            f"Given this regenerative farming scenario, what should be done? {scenario}",
            output,
        ))
    return pairs


def main() -> None:
    if not TRAIN_PATH.exists():
        print(f"error: {TRAIN_PATH} not found", file=sys.stderr)
        sys.exit(1)

    existing = [json.loads(line) for line in open(TRAIN_PATH, encoding="utf-8") if line.strip()]
    print(f"Existing hf_dataset/train.jsonl: {len(existing)} conversations", file=sys.stderr)

    sources: list[tuple[str, Path, callable]] = [
        ("bioaccumulators", RAG_DIR / "BioaccumulatorsList.json", _bioaccumulators),
        ("recipes", RAG_DIR / "NF_Recipes.json", _recipes),
        ("crop_requirements",
         RAG_DIR / "SemiSynthetic_Crop_Requirements" / "Crop_Requirements.json",
         _crop_requirements),
        ("composting",
         RAG_DIR / "SemiSynthetic_Composting_Knowledge_For_Agriculture"
         / "SemiSynthetic_Composting_Knowledge_For_Agriculture.json",
         _composting_methods),
        ("plants_by_region", RAG_DIR / "Locally_Growing_Plants_by_Region.json", _plants_by_region),
        ("mineral_profiles",
         RAG_DIR / "crop_mineral_data" / "plant_mineral_profiles.json",
         _mineral_profiles),
        ("regenerative_scenarios",
         RAG_DIR / "SemiSynthetic_Data_For_Regenerative_Farming_Agriculture"
         / "SemiSynthetic_Data_For_Regenerative_Farming_Agriculture.json",
         _regenerative_scenarios),
    ]

    all_new: list[dict] = []
    for name, path, loader in sources:
        if not path.exists():
            print(f"  {name}: SKIPPED (not found: {path})", file=sys.stderr)
            continue
        pairs = loader(path)
        print(f"  {name}: {len(pairs)} pairs from {path.name}", file=sys.stderr)
        all_new.extend(pairs)

    existing_human_turns = {
        conv["conversations"][0]["value"]
        for conv in existing
        if conv.get("conversations")
    }
    deduped = [p for p in all_new if p["conversations"][0]["value"] not in existing_human_turns]
    dropped = len(all_new) - len(deduped)
    if dropped:
        print(f"  Dropped {dropped} duplicates of existing conversations", file=sys.stderr)

    # Also dedup within the new batch itself (e.g. two sources producing the same question).
    seen: set[str] = set()
    final: list[dict] = []
    for p in deduped:
        key = p["conversations"][0]["value"]
        if key in seen:
            continue
        seen.add(key)
        final.append(p)

    combined = existing + final
    with open(TRAIN_PATH, "w", encoding="utf-8") as fh:
        for conv in combined:
            fh.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(combined)} conversations to {TRAIN_PATH}", file=sys.stderr)
    print(f"  ({len(existing)} existing + {len(final)} newly added)", file=sys.stderr)


if __name__ == "__main__":
    main()
