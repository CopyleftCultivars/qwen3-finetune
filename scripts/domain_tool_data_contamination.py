"""#173: contamination check for the domain-tool data sources vs sealed evals.

Same exact-match + 8-gram-Jaccard convention as scripts/ladder_contamination.py
(the instruction_hash / load_eval_hash_denylist pattern data/make_synthetic_traces.py
uses), applied to the four files rlm_slice/domain_data.py loads for #174's tools:
plant_mineral_profiles.json, crop_requirements.json, local_plants_by_region.json,
enriched_recipes.json.

This data already trained v5clean (it was ported into gemma4's dataset-building
pipeline before train.jsonl was built, confirmed separately: ~1,900 train.jsonl
lines reference it). The question here is narrower and matters more than a normal
new-data check: does any of it exact-or-near-match text inside the two SEALED
evaluation surfaces themselves (eval_freeform/test.jsonl, the 210-item natural
fertilizers ladder) -- which would be a pre-existing leak, not a new one.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlm_slice.domain_data import (  # noqa: E402
    load_crop_requirements,
    load_enriched_recipes,
    load_local_plants_by_region,
    load_plant_mineral_profiles,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_FREEFORM_TEST = REPO_ROOT / "eval_freeform" / "test.jsonl"
LADDER_210 = Path(os.environ.get("LADDER_210_PATH", "data/ladder210.jsonl"))
NGRAM_N = 8
JACCARD_ALARM = 0.6


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower())


def shingles(text: str, n: int = NGRAM_N) -> set[str]:
    words = norm(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def flatten_text(value: object) -> str:
    """Collect every string leaf in a JSON value into one blob for hashing/shingling."""
    parts: list[str] = []

    def walk(v: object) -> None:
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            for item in v.values():
                walk(item)
        elif isinstance(v, list):
            for item in v:
                walk(item)

    walk(value)
    return " ".join(parts)


def domain_records() -> list[tuple[str, str, str]]:
    """(source_file, record_label, flattened_text) for every domain record."""
    records: list[tuple[str, str, str]] = []
    for row in load_plant_mineral_profiles():
        label = row.get("common_name") or row.get("scientific_name") or "?"
        records.append(("plant_mineral_profiles.json", str(label), flatten_text(row)))
    for row in load_crop_requirements():
        label = row.get("Crop") or row.get("Scientific Name") or "?"
        records.append(("crop_requirements.json", str(label), flatten_text(row)))
    for row in load_local_plants_by_region():
        label = row.get("Location") or "?"
        records.append(("local_plants_by_region.json", str(label), flatten_text(row)))
    for row in load_enriched_recipes():
        label = row.get("recipe_name") or row.get("recipe_abbrev") or "?"
        records.append(("enriched_recipes.json", str(label), flatten_text(row)))
    return records


def eval_texts(path: Path, fields: tuple[str, ...]) -> list[tuple[str, str]]:
    """(item_id, text) pairs pulled from a sealed jsonl using whichever field is present."""
    if not path.exists():
        print(f"  MISSING eval source: {path}")
        return []
    out: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            item_id = str(row.get("id") or row.get("question_id") or "?")
            text = " ".join(str(row.get(f, "")) for f in fields if row.get(f))
            if text.strip():
                out.append((item_id, text))
    return out


def main() -> int:
    records = domain_records()
    print(f"domain records: {len(records)} across "
          f"{len({r[0] for r in records})} source files")

    eval_sources = {
        "eval_freeform/test.jsonl": eval_texts(EVAL_FREEFORM_TEST, ("question", "reference_answer")),
        "ladder_210": eval_texts(LADDER_210, ("question", "A", "B", "C", "D")),
    }
    for name, items in eval_sources.items():
        print(f"  {name}: {len(items)} items")

    eval_norm: dict[str, tuple[str, str]] = {}
    eval_shingles: list[tuple[str, str, set[str]]] = []
    for source_name, items in eval_sources.items():
        for item_id, text in items:
            qn = " ".join(norm(text).split())
            eval_norm.setdefault(qn, (source_name, item_id))
            eval_shingles.append((source_name, item_id, shingles(text)))

    exact: list[tuple[str, str, str]] = []
    near: list[tuple[str, str, str, float]] = []
    for source_file, label, text in records:
        tn = " ".join(norm(text).split())
        if tn and tn in eval_norm:
            src, item_id = eval_norm[tn]
            exact.append((f"{source_file}:{label}", src, item_id))
            continue
        ts = shingles(text)
        best = 0.0
        best_src = ""
        best_id = ""
        for src, item_id, es in eval_shingles:
            j = jaccard(ts, es)
            if j > best:
                best, best_src, best_id = j, src, item_id
        if best >= JACCARD_ALARM:
            near.append((f"{source_file}:{label}", best_src, best_id, round(best, 3)))

    print(f"\nexact matches: {len(exact)}")
    print(f"near matches (J>={JACCARD_ALARM}): {len(near)}")
    for row in exact[:5]:
        print(f"   exact: {row}")
    for row in near[:5]:
        print(f"   near: {row}")

    out = {
        "format": "domain_tool_data_contamination_v1",
        "n_domain_records": len(records),
        "domain_source_files": sorted({r[0] for r in records}),
        "eval_sources": {name: str(EVAL_FREEFORM_TEST if "freeform" in name else LADDER_210)
                          for name in eval_sources},
        "n_eval_items": {name: len(items) for name, items in eval_sources.items()},
        "ngram_n": NGRAM_N,
        "jaccard_alarm": JACCARD_ALARM,
        "n_exact": len(exact),
        "n_near": len(near),
        "exact": exact,
        "near": near,
    }
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "domain_tool_data_contamination.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
