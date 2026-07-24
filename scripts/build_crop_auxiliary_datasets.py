#!/usr/bin/env python3
"""Build four independent, contamination-audited CROP-style task streams.

The task records are deterministic templates over the repository's natural-
farming structured sources.  The generic CROP dataset is deliberately not an
input.  Each task is split by source document, then capped to the same pilot
dose so the first autoresearch comparison changes one task lever at a time:

* summarization
* NER/entity extraction
* structured extraction
* domain open-book QA

The English benchmark is used only as a contamination reference.  No sealed
evaluation file is accepted by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TASKS = (
    "summarization",
    "ner",
    "structured_extraction",
    "open_book_qa",
)
SOURCE_FILES = {
    "recipe": "enriched_recipes.json",
    "crop_requirement": "crop_requirements.json",
    "regional_plants": "local_plants_by_region.json",
    "mineral_profile": "plant_mineral_profiles.json",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
MIN_EXACT_CHARS = 48
MIN_OVERLAP_TOKENS = 6
LEXICAL_THRESHOLD = 0.85


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(TOKEN_RE.findall(text))


def tokens(text: str) -> set[str]:
    return set(normalize(text).split())


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: expected a JSON array of objects")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def ensure_open_benchmark_path(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    if "spanish-benchmark-workspace" in parts:
        raise ValueError("Spanish transfer benchmark paths are evaluation-only")
    if "eval_freeform" in parts and path.name.casefold() == "test.jsonl":
        raise ValueError("sealed free-form test paths are forbidden")
    if "sealed" in path.name.casefold():
        raise ValueError("sealed benchmark paths are forbidden")


def source_id(kind: str, index: int, row: dict[str, Any]) -> str:
    fingerprint = digest_bytes(canonical_json(row).encode("utf-8"))[:12]
    return f"{kind}:{index:04d}:{fingerprint}"


def record_without_unused_passages(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    if kind == "recipe":
        # The qmd_passages field is provenance context, not a training input.
        # Keeping it out of the prompt prevents accidentally replaying source
        # passages whose independence from the benchmark has not been proven.
        value.pop("qmd_passages", None)
    return value


def build_documents(source_root: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for kind, filename in SOURCE_FILES.items():
        path = source_root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        rows = load_json(path)
        file_sha = digest_file(path)
        for index, row in enumerate(rows):
            clean_row = record_without_unused_passages(kind, row)
            documents.append(
                {
                    "source_id": source_id(kind, index, row),
                    "source_kind": kind,
                    "source_path": portable_path(path),
                    "source_sha256": file_sha,
                    "source_record_index": index,
                    "source_record_sha256": digest_bytes(canonical_json(row).encode("utf-8")),
                    "record": clean_row,
                }
            )
    return documents


def _list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def render_record(document: dict[str, Any]) -> str:
    kind = document["source_kind"]
    row = document["record"]
    if kind == "recipe":
        npk = ", ".join(f"{key}={value}" for key, value in _mapping(row.get("npk_provided")).items())
        dilution = "; ".join(f"{key}: {value}" for key, value in _mapping(row.get("dilution_ratio")).items())
        return "\n".join(
            (
                f"Recipe name: {row.get('recipe_name', '')}",
                f"Abbreviation: {row.get('recipe_abbrev', '')}",
                f"Primary nutrient: {row.get('primary_nutrient', '')}",
                f"NPK profile: {npk}",
                f"Suggested bioaccumulators: {', '.join(_list(row.get('suggested_bioaccumulators')))}",
                f"Dilution ratios: {dilution}",
                f"Ingredients: {', '.join(_list(row.get('ingredients')))}",
                f"Steps: {' '.join(_list(row.get('steps')))}",
                f"Uses: {' '.join(_list(row.get('uses')))}",
            )
        )
    if kind == "crop_requirement":
        npk = ", ".join(f"{key}={value}" for key, value in _mapping(row.get("NPK Requirements")).items())
        additional = ", ".join(f"{key}={value}" for key, value in _mapping(row.get("Additional Requirements")).items())
        return (
            f"Crop: {row.get('Crop', '')}; scientific name: {row.get('Scientific Name', '')}; "
            f"stated NPK requirements: {npk}; additional requirements: {additional}."
        )
    if kind == "regional_plants":
        return (
            f"Location: {row.get('Location', '')}; listed plants: {', '.join(_list(row.get('Plants')))}; "
            f"seasonality note: {row.get('seasonality_note', '')}."
        )
    parts: list[str] = []
    for part_name, part_data in _mapping(row.get("parts")).items():
        minerals = ", ".join(
            f"{mineral}={canonical_json(values)}"
            for mineral, values in _mapping(part_data.get("minerals")).items()
        )
        parts.append(f"{part_name}: {minerals}")
    return (
        f"Common name: {row.get('common_name', '')}; scientific name: {row.get('scientific_name', '')}; "
        f"parts and mineral ranges: {'; '.join(parts)}; "
        f"QMD corroborated: {row.get('qmd_corroborated', False)}."
    )


def summary_output(document: dict[str, Any]) -> str:
    kind = document["source_kind"]
    row = document["record"]
    if kind == "recipe":
        npk = ", ".join(f"{key}={value}" for key, value in _mapping(row.get("npk_provided")).items())
        uses = "; ".join(_list(row.get("uses")))
        dilution = "; ".join(f"{key}: {value}" for key, value in _mapping(row.get("dilution_ratio")).items())
        return (
            f"{row.get('recipe_name', '')} ({row.get('recipe_abbrev', '')}) is described as a preparation "
            f"centered on {row.get('primary_nutrient', '')}. Its stated NPK profile is {npk}. "
            f"The listed dilution ratios are {dilution}. Its stated uses are: {uses}."
        )
    if kind == "crop_requirement":
        npk = ", ".join(f"{key}={value}" for key, value in _mapping(row.get("NPK Requirements")).items())
        additional = ", ".join(f"{key}={value}" for key, value in _mapping(row.get("Additional Requirements")).items())
        return (
            f"{row.get('Crop', '')} ({row.get('Scientific Name', '')}) has stated NPK requirements of {npk}. "
            f"Additional listed requirements are {additional}."
        )
    if kind == "regional_plants":
        return (
            f"The {row.get('Location', '')} record lists {', '.join(_list(row.get('Plants')))}. "
            f"Its seasonality note is: {row.get('seasonality_note', '')}."
        )
    part_names = ", ".join(_mapping(row.get("parts")).keys())
    return (
        f"The profile for {row.get('common_name', '')} ({row.get('scientific_name', '')}) covers the parts "
        f"{part_names}. It records mineral ranges for those parts, and QMD corroboration is "
        f"{row.get('qmd_corroborated', False)}."
    )


def add_entity(entities: list[dict[str, str]], text: Any, entity_type: str) -> None:
    if text is None or isinstance(text, (dict, list)):
        return
    value = str(text).strip()
    if value:
        entities.append({"text": value, "type": entity_type})


def ner_output(document: dict[str, Any]) -> str:
    kind = document["source_kind"]
    row = document["record"]
    entities: list[dict[str, str]] = []
    if kind == "recipe":
        add_entity(entities, row.get("recipe_name"), "preparation")
        add_entity(entities, row.get("recipe_abbrev"), "preparation")
        add_entity(entities, row.get("primary_nutrient"), "nutrient")
        for value in _mapping(row.get("npk_provided")):
            add_entity(entities, value, "nutrient")
        for value in _list(row.get("suggested_bioaccumulators")):
            add_entity(entities, value, "organism")
        for value in _list(row.get("ingredients")):
            add_entity(entities, value, "material")
    elif kind == "crop_requirement":
        add_entity(entities, row.get("Crop"), "crop")
        add_entity(entities, row.get("Scientific Name"), "organism")
        for value in _mapping(row.get("NPK Requirements")):
            add_entity(entities, value, "nutrient")
        for key, value in _mapping(row.get("Additional Requirements")).items():
            add_entity(entities, f"{key}={value}", "measurement")
    elif kind == "regional_plants":
        add_entity(entities, row.get("Location"), "location")
        for value in _list(row.get("Plants")):
            add_entity(entities, value, "organism")
    else:
        add_entity(entities, row.get("common_name"), "organism")
        add_entity(entities, row.get("scientific_name"), "organism")
        for part_name, part_data in _mapping(row.get("parts")).items():
            add_entity(entities, part_name, "plant_part")
            for mineral_name, values in _mapping(part_data.get("minerals")).items():
                add_entity(entities, mineral_name, "nutrient")
                add_entity(entities, _mapping(values).get("unit"), "measurement")
    unique = sorted({(item["text"], item["type"]): item for item in entities}.values(), key=lambda item: (item["type"], item["text"]))
    return canonical_json({"entities": unique})


def structured_output(document: dict[str, Any]) -> str:
    kind = document["source_kind"]
    row = document["record"]
    if kind == "recipe":
        value = {
            "record_type": kind,
            "recipe_name": row.get("recipe_name"),
            "recipe_abbrev": row.get("recipe_abbrev"),
            "primary_nutrient": row.get("primary_nutrient"),
            "npk_provided": row.get("npk_provided", {}),
            "suggested_bioaccumulators": row.get("suggested_bioaccumulators", []),
            "dilution_ratio": row.get("dilution_ratio", {}),
            "ingredients": row.get("ingredients", []),
            "steps": row.get("steps", []),
            "uses": row.get("uses", []),
        }
    elif kind == "crop_requirement":
        value = {
            "record_type": kind,
            "crop": row.get("Crop"),
            "scientific_name": row.get("Scientific Name"),
            "npk_requirements": row.get("NPK Requirements", {}),
            "additional_requirements": row.get("Additional Requirements", {}),
        }
    elif kind == "regional_plants":
        value = {
            "record_type": kind,
            "location": row.get("Location"),
            "plants": row.get("Plants", []),
            "seasonality_note": row.get("seasonality_note", ""),
        }
    else:
        value = {
            "record_type": kind,
            "scientific_name": row.get("scientific_name"),
            "common_name": row.get("common_name"),
            "parts": row.get("parts", {}),
            "qmd_corroborated": row.get("qmd_corroborated", False),
        }
    return canonical_json(value)


def open_book_variants(document: dict[str, Any]) -> list[tuple[str, str, str]]:
    kind = document["source_kind"]
    row = document["record"]
    if kind == "recipe":
        return [
            (
                "primary_nutrient",
                f"What primary nutrient is stated for {row.get('recipe_name', '')}?",
                str(row.get("primary_nutrient", "")),
            ),
            (
                "foliar_dilution",
                f"What foliar dilution ratio is stated for {row.get('recipe_name', '')}?",
                str(_mapping(row.get("dilution_ratio")).get("foliar", "")),
            ),
        ]
    if kind == "crop_requirement":
        return [
            (
                "scientific_name",
                f"What scientific name is listed for {row.get('Crop', '')}?",
                str(row.get("Scientific Name", "")),
            ),
            (
                "ph_requirement",
                f"What pH range is listed for {row.get('Crop', '')}?",
                str(_mapping(row.get("Additional Requirements")).get("pH", "")),
            ),
        ]
    if kind == "regional_plants":
        return [
            (
                "plants",
                f"Which plants are listed for {row.get('Location', '')}?",
                ", ".join(_list(row.get("Plants"))),
            ),
            (
                "seasonality",
                f"What seasonality note is given for {row.get('Location', '')}?",
                str(row.get("seasonality_note", "")),
            ),
        ]
    first_part = next(iter(_mapping(row.get("parts"))), "")
    minerals = _mapping(_mapping(row.get("parts")).get(first_part, {})).get("minerals", {})
    return [
        (
            "part_minerals",
            f"Which minerals are listed for the {first_part} part of {row.get('common_name', '')}?",
            ", ".join(str(value) for value in _mapping(minerals)),
        ),
        (
            "qmd_corroborated",
            f"Is the mineral profile for {row.get('common_name', '')} marked as QMD corroborated?",
            str(row.get("qmd_corroborated", False)),
        ),
    ]


def make_item(task: str, document: dict[str, Any], variant: str | None = None) -> dict[str, Any]:
    context = render_record(document)
    if task == "summarization":
        instruction = (
            "Summarize the following natural-farming reference record in two to four concise sentences. "
            "Preserve named entities, quantities, ranges, and cautions. Do not add facts.\n\n" + context
        )
        output = summary_output(document)
    elif task == "ner":
        instruction = (
            "Extract every named entity in the following natural-farming reference record. "
            "Return JSON only with an entities array; each item must have text and one of these types: "
            "organism, preparation, nutrient, location, material, measurement, plant_part, crop. "
            "Do not invent entities.\n\n" + context
        )
        output = ner_output(document)
    elif task == "structured_extraction":
        instruction = (
            "Convert the following natural-farming reference record into its canonical structured JSON form. "
            "Return JSON only, preserve every supplied value, and do not add facts.\n\n" + context
        )
        output = structured_output(document)
    elif task == "open_book_qa":
        if variant is None:
            raise ValueError("open-book items require a variant")
        variants = {key: (question, answer) for key, question, answer in open_book_variants(document)}
        question, output = variants[variant]
        instruction = (
            "Answer the question using only the supplied natural-farming reference context. "
            "If the context does not contain the answer, say that it is not stated.\n\n"
            f"Reference context:\n{context}\n\nQuestion: {question}"
        )
    else:
        raise ValueError(f"unknown task: {task}")

    row_id = f"{task}:{document['source_id']}:{variant or 'default'}"
    return {
        "id": row_id,
        "instruction": instruction,
        "output": output,
        "language": "en",
        "auxiliary_task": task,
        "source_id": document["source_id"],
        "source_kind": document["source_kind"],
        "provenance": {
            "source_path": document["source_path"],
            "source_sha256": document["source_sha256"],
            "source_record_index": document["source_record_index"],
            "source_record_sha256": document["source_record_sha256"],
            "authoring_method": "deterministic_template_v1",
            "source_document_independence": "not_claimed",
        },
    }


def build_candidates(task: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        if task == "open_book_qa":
            rows.extend(make_item(task, document, variant) for variant, _, _ in open_book_variants(document))
        else:
            rows.append(make_item(task, document))
    return rows


def benchmark_texts(path: Path) -> list[tuple[str, str, str, set[str]]]:
    texts: list[tuple[str, str, str, set[str]]] = []
    for index, row in enumerate(load_jsonl(path)):
        for field in ("question", "A", "B", "C", "D"):
            value = str(row.get(field, "")).strip()
            normalized = normalize(value)
            if normalized:
                texts.append((str(row.get("question_id", index)), field, normalized, set(normalized.split())))
    return texts


def audit_candidates(
    rows: list[dict[str, Any]],
    benchmark: list[tuple[str, str, str, set[str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    max_overlap: dict[str, Any] = {"score": 0.0}
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        reason: str | None = None
        match: dict[str, Any] = {}
        if row["id"] in seen_ids:
            reason = "duplicate_item_id"
        elif row.get("language") != "en":
            reason = "unexpected_language"
        elif row.get("auxiliary_task") not in TASKS:
            reason = "unexpected_task"
        else:
            for field in ("instruction", "output"):
                value = str(row.get(field, ""))
                normalized = normalize(value)
                if len(normalized) >= MIN_EXACT_CHARS:
                    for question_id, benchmark_field, benchmark_text, benchmark_tokens in benchmark:
                        if len(benchmark_text) >= MIN_EXACT_CHARS and (
                            benchmark_text in normalized or normalized in benchmark_text
                        ):
                            reason = "exact_normalized_benchmark_substring"
                            match = {"benchmark_question_id": question_id, "benchmark_field": benchmark_field, "source_field": field}
                            break
                        source_tokens = set(normalized.split())
                        score = jaccard(source_tokens, benchmark_tokens)
                        if score > max_overlap.get("score", 0.0):
                            max_overlap = {"score": round(score, 6), "source_id": row["source_id"], "source_field": field, "benchmark_question_id": question_id, "benchmark_field": benchmark_field}
                        if len(source_tokens & benchmark_tokens) >= MIN_OVERLAP_TOKENS and score >= LEXICAL_THRESHOLD:
                            reason = "high_lexical_benchmark_overlap"
                            match = {"benchmark_question_id": question_id, "benchmark_field": benchmark_field, "source_field": field, "score": round(score, 6)}
                            break
                if reason:
                    break
        seen_ids.add(row["id"])
        if reason:
            rejected.append({"source_index": index, "id": row.get("id"), "source_id": row.get("source_id"), "reason": reason, **match})
            continue
        clean_row = dict(row)
        clean_row["_contamination_status"] = "passed"
        clean_row["_contamination_audit"] = "auxiliary_benchmark_contamination_audit_v1"
        clean_row["_benchmark_question_text_overlap"] = "not_detected"
        clean.append(clean_row)
    report = {
        "format": "auxiliary_benchmark_contamination_audit_v1",
        "candidate_count": len(rows),
        "clean_count": len(clean),
        "rejected_count": len(rejected),
        "exact_or_substring_overlap_count": sum(item["reason"] == "exact_normalized_benchmark_substring" for item in rejected),
        "high_lexical_overlap_count": sum(item["reason"] == "high_lexical_benchmark_overlap" for item in rejected),
        "lexical_overlap_threshold": LEXICAL_THRESHOLD,
        "minimum_overlap_tokens": MIN_OVERLAP_TOKENS,
        "maximum_token_jaccard": max_overlap,
        "rejected": rejected,
        "passed": not rejected,
        "sealed_test_paths_read": [],
        "source_document_independence": "not_claimed",
        "notes": [
            "The audit covers generated instruction and output text against English MCQ stems/options.",
            "It does not establish semantic or source-document independence.",
            "The generic CROP dataset is not an input; sources are repository natural-farming records.",
        ],
    }
    return clean, report


def stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def split_and_cap(rows: list[dict[str, Any]], train_count: int, validation_count: int, split: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_ids = sorted({str(row["source_id"]) for row in rows})
    ranked_sources = sorted(source_ids, key=lambda value: stable_rank(value, seed))
    validation_source_count = max(1, int(round(len(source_ids) * (1.0 - split))))
    validation_ids = set(ranked_sources[:validation_source_count])
    train_ids = set(source_ids) - validation_ids
    train_rows = [row for row in rows if row["source_id"] in train_ids]
    validation_rows = [row for row in rows if row["source_id"] in validation_ids]
    train_rows.sort(key=lambda row: stable_rank(row["id"], seed))
    validation_rows.sort(key=lambda row: stable_rank(row["id"], seed))
    if len(train_rows) < train_count or len(validation_rows) < validation_count:
        raise ValueError(
            f"pilot dose unavailable: train={len(train_rows)} validation={len(validation_rows)} "
            f"requested={train_count}/{validation_count}"
        )
    train = train_rows[:train_count]
    validation = validation_rows[:validation_count]
    if {row["source_id"] for row in train} & {row["source_id"] for row in validation}:
        raise AssertionError("source-document split overlap")
    return train, validation, {
        "candidate_source_count": len(source_ids),
        "train_source_count": len(train_ids),
        "validation_source_count": len(validation_ids),
        "train_candidate_count": len(train_rows),
        "validation_candidate_count": len(validation_rows),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return digest_file(path)


def build_task(task: str, documents: list[dict[str, Any]], benchmark: list[tuple[str, str, set[str]]], output_root: Path, train_count: int, validation_count: int, split: float, seed: int, benchmark_path: Path) -> dict[str, Any]:
    candidates = build_candidates(task, documents)
    clean, contamination = audit_candidates(candidates, benchmark)
    if not clean:
        raise ValueError(f"{task}: contamination audit removed every candidate")
    train, validation, split_report = split_and_cap(clean, train_count, validation_count, split, seed)
    task_root = output_root / task
    train_path = task_root / "train.jsonl"
    validation_path = task_root / "validation.jsonl"
    train_sha = write_jsonl(train_path, train)
    validation_sha = write_jsonl(validation_path, validation)
    manifest = {
        "format": "crop_style_auxiliary_dataset_manifest_v1",
        "task": task,
        "language": "en",
        "pilot_dose": {"train_count": train_count, "validation_count": validation_count, "split_fraction": split, "seed": seed},
        "source_files": [
            {"path": path, "sha256": sha, "source_kinds": sorted(kinds)}
            for path, (sha, kinds) in sorted(
                {
                    document["source_path"]: (
                        document["source_sha256"],
                        {
                            item["source_kind"]
                            for item in documents
                            if item["source_path"] == document["source_path"]
                        },
                    )
                    for document in documents
                }.items()
            )
        ],
        "candidate_count": len(candidates),
        "clean_candidate_count": len(clean),
        "contamination": {"benchmark_path": str(benchmark_path.resolve()), "benchmark_sha256": digest_file(benchmark_path), **contamination},
        "split": split_report,
        "train": {"path": str(train_path.resolve()), "sha256": train_sha, "count": len(train), "source_ids": sorted({row["source_id"] for row in train})},
        "validation": {"path": str(validation_path.resolve()), "sha256": validation_sha, "count": len(validation), "source_ids": sorted({row["source_id"] for row in validation})},
        "train_validation_source_overlap": sorted({row["source_id"] for row in train} & {row["source_id"] for row in validation}),
        "training_allowed": not contamination["rejected"],
        "sealed_test_paths_read": [],
        "notes": [
            "This stream is intentionally independent from the other three auxiliary task levers.",
            "Combine streams only after single-task runs have been measured.",
            "The deterministic labels require a later quality review before treating the task as a promoted lever.",
        ],
    }
    (task_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "task": task,
        "candidate_count": len(candidates),
        "clean_candidate_count": len(clean),
        "train_count": len(train),
        "validation_count": len(validation),
        "rejected_count": contamination["rejected_count"],
        "train_validation_source_overlap": manifest["train_validation_source_overlap"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/sources/gemma4/spaces/data"))
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/auxiliary"))
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=TASKS, action="append", dest="tasks")
    args = parser.parse_args()
    if args.train_count < 1 or args.validation_count < 1:
        parser.error("pilot counts must be positive")
    if not 0.0 < args.split < 1.0:
        parser.error("--split must be between 0 and 1")
    if not args.benchmark.exists():
        parser.error(f"benchmark not found: {args.benchmark}")
    ensure_open_benchmark_path(args.benchmark)
    selected_tasks = tuple(args.tasks or TASKS)
    documents = build_documents(args.source_root.resolve())
    benchmark = benchmark_texts(args.benchmark.resolve())
    summaries = [
        build_task(
            task,
            documents,
            benchmark,
            args.output_root.resolve(),
            args.train_count,
            args.validation_count,
            args.split,
            args.seed,
            args.benchmark.resolve(),
        )
        for task in selected_tasks
    ]
    root_manifest = {
        "format": "crop_style_auxiliary_root_manifest_v1",
        "tasks": list(selected_tasks),
        "source_root": str(args.source_root.resolve()),
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": digest_file(args.benchmark.resolve()),
        "source_document_count": len(documents),
        "pilot_dose": {"train_count": args.train_count, "validation_count": args.validation_count, "split_fraction": args.split, "seed": args.seed},
        "sealed_test_paths_read": [],
        "summaries": summaries,
    }
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    (args.output_root.resolve() / "manifest.json").write_text(json.dumps(root_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
