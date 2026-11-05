#!/usr/bin/env python3
"""Shared plumbing for the #191 persona-memory corpus build.

#191 builds an LPM (Latent Personal Memory, arXiv 2606.20911) training corpus
from the real Gemma4-era conversation logs at
import os
$PERSONA_CONVERSATIONS_DIR plus
targeted synthetic cross-session augmentation. Four scripts share this module:

  scripts/build_persona_memory_corpus.py  -- real logs -> held-out multi-session
                                             probes + single-session seed material
  scripts/persona_memory_contamination.py -- #139-style check vs the ladder /
                                             val_v3 / eval_freeform (dev only)
  data/make_persona_memory_corpus.py      -- synthetic multi-session generator
  scripts/chunk_persona_memory_corpus.py  -- previous-chunk NTP chunking

Two conventions from #191 are load-bearing and live here so all four scripts
agree on them:

  * Bad turns (``evaluation == "bad"``) are dropped BEFORE any structural
    analysis (session-gap splitting, chunking), so a dropped turn neither
    creates nor hides a session boundary.
  * A "real session gap" is >12h between consecutive turns under one
    ``conversation_id``. Conversations with one are the genuine longitudinal
    data and are held out of training entirely (and out of synthetic
    augmentation) so #196 has a clean probe set.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONVERSATIONS_DIR = Path(os.environ.get("PERSONA_CONVERSATIONS_DIR", "data/persona_memory/conversations"))

LADDER_210 = Path(os.environ.get("LADDER_210_PATH", "data/ladder210.jsonl"))

SESSION_GAP_HOURS = 12.0

# The sealed test split. This module (and every script importing it) never
# opens it -- see data/make_synthetic_traces.py's documented deviation for the
# same policy, and eval_freeform/README-style policy: touched once at final
# evaluation only.
SEALED_EVAL_FREEFORM_TEST = REPO_ROOT / "eval_freeform" / "test.jsonl"

NGRAM_N = 8
JACCARD_ALARM = 0.6


@dataclass
class Turn:
    """One message row from a Gemma4 conversation CSV, bad-filtered upstream."""

    message_id: str
    role: str
    text: str
    created_at: datetime
    evaluation: str = ""

    @property
    def is_human(self) -> bool:
        return self.role == "human"

    def to_json(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "from": self.role,
            "text": self.text,
            "createdAt": self.created_at.isoformat(),
            "evaluation": self.evaluation,
        }


@dataclass
class Conversation:
    conversation_id: str
    turns: list[Turn] = field(default_factory=list)
    session_count: int = 1
    max_gap_hours: float = 0.0

    @property
    def is_multisession(self) -> bool:
        return self.session_count > 1

    @property
    def span_hours(self) -> float:
        if len(self.turns) < 2:
            return 0.0
        return (self.turns[-1].created_at - self.turns[0].created_at).total_seconds() / 3600.0

    def persona_id(self, prefix: str = "gemma4") -> str:
        return f"{prefix}:{self.conversation_id}"


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_conversation_csv(path: Path) -> tuple[str, list[Turn]]:
    """Read one conversation file. ``evaluation == "bad"`` turns are dropped here."""
    conversation_id = path.stem
    turns: list[Turn] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            evaluation = (row.get("evaluation") or "").strip()
            if evaluation == "bad":
                continue
            created = (row.get("createdAt") or "").strip()
            if not created:
                continue
            turns.append(
                Turn(
                    message_id=(row.get("message_id") or "").strip(),
                    role=(row.get("from") or "").strip(),
                    text=(row.get("text") or "").strip(),
                    created_at=parse_timestamp(created),
                    evaluation=evaluation,
                )
            )
    turns.sort(key=lambda t: t.created_at)
    return conversation_id, turns


def load_conversations(
    conversations_dir: Path = DEFAULT_CONVERSATIONS_DIR,
    *,
    min_turns: int = 1,
) -> list[Conversation]:
    """Load every ``*.csv`` under ``conversations_dir``, bad turns already dropped.

    The caller is expected to point at the real ``conversations/`` directory;
    the sibling ``conversations (1)/`` is a strict subset and must not be
    passed (it would double-count nothing unique, but wastes time and muddies
    provenance). ``min_turns`` drops conversations with fewer surviving turns.
    """
    if not conversations_dir.is_dir():
        raise FileNotFoundError(f"conversations directory not found: {conversations_dir}")
    conversations: list[Conversation] = []
    for path in sorted(conversations_dir.glob("*.csv")):
        cid, turns = read_conversation_csv(path)
        if len(turns) < min_turns:
            continue
        conversations.append(Conversation(conversation_id=cid, turns=turns))
    return conversations


def split_sessions(turns: list[Turn], gap_hours: float = SESSION_GAP_HOURS) -> list[list[Turn]]:
    """Split a time-sorted turn list wherever consecutive turns are >``gap_hours`` apart."""
    if not turns:
        return []
    sessions: list[list[Turn]] = [[turns[0]]]
    for previous, current in zip(turns, turns[1:]):
        if (current.created_at - previous.created_at) > timedelta(hours=gap_hours):
            sessions.append([])
        sessions[-1].append(current)
    return sessions


def annotate_sessions(conversation: Conversation, gap_hours: float = SESSION_GAP_HOURS) -> Conversation:
    """Fill ``session_count`` / ``max_gap_hours`` from the >gap_hours rule."""
    sessions = split_sessions(conversation.turns, gap_hours)
    conversation.session_count = len(sessions)
    if len(conversation.turns) >= 2:
        gaps = [
            (b.created_at - a.created_at).total_seconds() / 3600.0
            for a, b in zip(conversation.turns, conversation.turns[1:])
        ]
        conversation.max_gap_hours = round(max(gaps), 3)
    return conversation


def session_to_json(session: list[Turn], session_index: int) -> dict[str, Any]:
    return {
        "session_index": session_index,
        "start": session[0].created_at.isoformat(),
        "end": session[-1].created_at.isoformat(),
        "n_turns": len(session),
        "turns": [turn.to_json() for turn in session],
    }


def render_session_text(session: list[Turn]) -> str:
    """Deterministic plain-text rendering used as the NTP chunk target/prompt."""
    lines: list[str] = []
    for turn in session:
        speaker = "User" if turn.is_human else "Assistant"
        lines.append(f"{speaker}: {turn.text}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Text normalization / hashing -- identical primitives to
# scripts/ladder_contamination.py and data/make_synthetic_traces.py so a row is
# judged the same way everywhere. (#139 convention.)
# --------------------------------------------------------------------------- #

def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower())


def normalized_flat(text: str) -> str:
    return " ".join(norm(text).split())


def instruction_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def shingles(text: str, n: int = NGRAM_N) -> set[str]:
    words = norm(text).split()
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(needle: set[str], haystack: set[str]) -> float:
    """Fraction of ``needle`` shingles present in ``haystack``.

    Jaccard is symmetric and dilutes a short eval question quoted inside a long
    real turn; containment catches that case (eval item fully present in a
    training turn) without flagging ordinary topical overlap.
    """
    if not needle or not haystack:
        return 0.0
    return len(needle & haystack) / len(needle)


def flatten_text(value: Any) -> str:
    """Collect every string leaf of a JSON value into one blob."""
    parts: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)
    return " ".join(parts)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_domain_reference() -> dict[str, list[dict[str, Any]]]:
    """The #172 domain sources, for grounding generated facts.

    These are the already-ported, valid files under
    ``data/sources/gemma4/spaces/data/`` (the sibling repo's
    ``bioaccumulators_sample.json`` is a 29-byte "Invalid username or password."
    error stub and its ``NF_Recipes.json`` / ``USDA_Plants_Database.json`` are
    malformed JSON, so they are deliberately not read here).
    """
    data_dir = REPO_ROOT / "data" / "sources" / "gemma4" / "spaces" / "data"
    return {
        "plant_mineral_profiles": json.loads(
            (data_dir / "plant_mineral_profiles.json").read_text(encoding="utf-8")
        ),
        "crop_requirements": json.loads(
            (data_dir / "crop_requirements.json").read_text(encoding="utf-8")
        ),
        "local_plants_by_region": json.loads(
            (data_dir / "local_plants_by_region.json").read_text(encoding="utf-8")
        ),
        "enriched_recipes": json.loads(
            (data_dir / "enriched_recipes.json").read_text(encoding="utf-8")
        ),
    }


def chunk_session(
    session: list[Turn],
    *,
    chunk_turns: int = 2,
    max_chunk_chars: int = 1200,
) -> list[str]:
    """Split one session into fixed-size chunks (dialogue analog of LPM's
    fixed-size context chunks).

    A chunk closes when it reaches ``chunk_turns`` turns OR adding the next turn
    would exceed ``max_chunk_chars``. Order is preserved; concatenating the
    returned chunks reconstructs the session text turn-for-turn (modulo the
    per-chunk join), which is what the chunker asserts.
    """
    chunks: list[str] = []
    current: list[Turn] = []
    current_chars = 0
    for turn in session:
        turn_chars = len(turn.text)
        would_exceed = current and (current_chars + turn_chars > max_chunk_chars)
        if current and (len(current) >= chunk_turns or would_exceed):
            chunks.append(render_session_text(current))
            current = []
            current_chars = 0
        current.append(turn)
        current_chars += turn_chars
    if current:
        chunks.append(render_session_text(current))
    return chunks
