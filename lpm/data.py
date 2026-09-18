"""Corpus loading and the is_first_chunk decision for LPM training (#190).

#191's corpus tags every record with ``is_first_chunk`` (t=0, where
``prompt_text == target_text`` by construction per LPM Alg. 1) and ``source`` /
``cross_chunk``. This module is the single place that applies the #190 decision:
by default first-chunk records are dropped from the NTP objective, because their
prompt *is* their target and training on them only teaches the module to copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from lpm.config import LPMConfig

DEFAULT_CORPUS = Path("data/persona_memory/lpm_corpus_train.jsonl")


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def filter_records(
    records: list[dict[str, Any]],
    config: LPMConfig | None = None,
) -> list[dict[str, Any]]:
    """Apply the #190 copy-shortcut decision (+ keep only usable rows).

    ``config.include_first_chunk=False`` (default) drops t=0 records. It also
    drops malformed rows (empty prompt/target) that would poison the loss.
    """
    config = config or LPMConfig()
    kept: list[dict[str, Any]] = []
    for record in records:
        if not str(record.get("prompt_text", "")).strip():
            continue
        if not str(record.get("target_text", "")).strip():
            continue
        if record.get("is_first_chunk") and not config.include_first_chunk:
            continue
        kept.append(record)
    return kept


@dataclass
class PersonaIndex:
    persona_to_id: dict[str, int]

    @classmethod
    def build(cls, records: Iterable[dict[str, Any]]) -> "PersonaIndex":
        personas: dict[str, int] = {}
        for record in records:
            pid = str(record["persona_id"])
            if pid not in personas:
                personas[pid] = len(personas)
        return cls(persona_to_id=personas)

    def __len__(self) -> int:
        return len(self.persona_to_id)

    def id_of(self, persona_id: str) -> int:
        return self.persona_to_id[persona_id]


def group_by_persona(
    records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["persona_id"]), []).append(record)
    for group in groups.values():
        group.sort(key=lambda r: r["chunk_index"])
    return groups


def select_subset(
    records: list[dict[str, Any]],
    config: LPMConfig | None = None,
    *,
    n_personas: int | None = None,
    records_per_persona: int | None = None,
    min_chunks_per_persona: int = 2,
) -> list[dict[str, Any]]:
    """Deterministic, persona-aware subset for smoke tests and probes."""
    groups = group_by_persona(filter_records(records, config))
    chosen = [pid for pid, group in groups.items() if len(group) >= min_chunks_per_persona]
    chosen.sort()
    if n_personas is not None:
        chosen = chosen[:n_personas]
    subset: list[dict[str, Any]] = []
    for pid in chosen:
        group = groups[pid]
        if records_per_persona is not None:
            group = group[:records_per_persona]
        subset.extend(group)
    return subset


def tokenize_pairs(
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    max_length: int = 256,
) -> dict[str, Any]:
    """Tokenize prompt/target texts for the NTP objective.

    Returns ``input_ids``/``attention_mask`` for prompts and labels for targets,
    plus ``persona_ids`` (index into the caller's PersonaIndex-built mapping is
    done separately) and ``weights``/``cross_chunk``/``source`` metadata.
    """
    prompts = [r["prompt_text"] for r in records]
    targets = [r["target_text"] for r in records]
    prompt_enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    target_enc = tokenizer(
        targets, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    return {
        "prompt_input_ids": prompt_enc["input_ids"],
        "prompt_attention_mask": prompt_enc["attention_mask"],
        "target_input_ids": target_enc["input_ids"],
        "target_attention_mask": target_enc["attention_mask"],
        "records": records,
    }
