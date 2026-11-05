from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from persona_memory_common import (  # noqa: E402
    Turn,
    chunk_session,
    read_conversation_csv,
    split_sessions,
)
from persona_memory_contamination import (  # noqa: E402
    ContaminationIndex,
    EvalItem,
    candidate_text,
    check_rows,
)
from scripts.chunk_persona_memory_corpus import build_records, split_train_val  # noqa: E402


def _turn(role: str, text: str, day: int = 0, hour: int = 0) -> Turn:
    return Turn(
        message_id=f"{role}-{day}-{hour}",
        role=role,
        text=text,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=day, hours=hour),
    )


class RealLogParsingTests(unittest.TestCase):
    def test_bad_turns_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conv123.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "agent_id", "conversation_id", "message_id", "text",
                        "from", "sources", "evaluation", "read", "createdAt",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "agent_id": "a", "conversation_id": "conv123", "message_id": "m1",
                    "text": "keep me", "from": "human", "sources": "null",
                    "evaluation": "", "read": "false", "createdAt": "2024-01-01T00:00:00+00:00",
                })
                writer.writerow({
                    "agent_id": "a", "conversation_id": "conv123", "message_id": "m2",
                    "text": "drop me", "from": "agent", "sources": "null",
                    "evaluation": "bad", "read": "false", "createdAt": "2024-01-01T00:01:00+00:00",
                })
            cid, turns = read_conversation_csv(path)
            self.assertEqual(cid, "conv123")
            self.assertEqual([t.text for t in turns], ["keep me"])

    def test_session_gap_split_at_12h(self):
        turns = [
            _turn("human", "a", day=0, hour=0),
            _turn("agent", "b", day=0, hour=1),
            _turn("human", "c", day=1, hour=0),  # 23h later -> new session
            _turn("human", "d", day=1, hour=2),
        ]
        sessions = split_sessions(turns)
        self.assertEqual([len(s) for s in sessions], [2, 2])

    def test_under_12h_stays_one_session(self):
        turns = [_turn("human", "a", hour=0), _turn("agent", "b", hour=11)]
        self.assertEqual(len(split_sessions(turns)), 1)


class ChunkingTests(unittest.TestCase):
    def test_chunk_session_respects_turn_budget_and_keeps_text(self):
        turns = [_turn("human", f"turn {i}") for i in range(5)]
        chunks = chunk_session(turns, chunk_turns=2, max_chunk_chars=10_000)
        self.assertEqual(len(chunks), 3)
        for turn in turns:
            self.assertIn(turn.text, "\n\n".join(chunks))

    def test_chunk_closes_on_char_budget(self):
        turns = [_turn("human", "x" * 40), _turn("human", "y" * 40)]
        chunks = chunk_session(turns, chunk_turns=10, max_chunk_chars=50)
        self.assertEqual(len(chunks), 2)


class LeakageGuardTests(unittest.TestCase):
    def _row(self, persona: str = "gemma4:abc") -> dict:
        return {
            "persona_id": persona,
            "sessions": [
                {"session_index": 0, "turns": [
                    {"from": "human", "text": "I farm corn in Oklahoma."},
                    {"from": "assistant", "text": "Noted."},
                ]},
                {"session_index": 1, "turns": [
                    {"from": "human", "text": "What fertilizer suits my crop?"},
                    {"from": "assistant", "text": "Use compost tea."},
                ]},
            ],
        }

    def test_prompt_never_reads_the_target_chunk(self):
        records = build_records(
            [self._row()], source_label="synthetic_multisession",
            chunk_turns=2, max_chunk_chars=1200,
        )
        self.assertGreaterEqual(len(records), 2)
        for record in records:
            self.assertLessEqual(record["prompt_source_index"], record["chunk_index"])
            if record["chunk_index"] > 0:
                self.assertNotEqual(record["prompt_text"], record["target_text"])
        # the cross-chunk record's prompt must equal the immediately previous target
        cross = [r for r in records if r["cross_chunk"]]
        self.assertTrue(cross)
        first = records[0]
        self.assertTrue(first["is_first_chunk"])
        self.assertEqual(first["prompt_chunk_id"], first["target_chunk_id"])

    def test_train_val_split_is_persona_disjoint(self):
        rows = [self._row(f"gemma4:{i}") for i in range(20)]
        records = build_records(
            rows, source_label="real_single_session", chunk_turns=2, max_chunk_chars=1200
        )
        train, val = split_train_val(records, 0.25)
        self.assertEqual({r["persona_id"] for r in train} & {r["persona_id"] for r in val}, set())


class ContaminationTests(unittest.TestCase):
    def test_quoted_eval_question_is_flagged(self):
        question = "According to the excerpt, what is a primary purpose of using animal manures to improve soil?"
        index = ContaminationIndex([
            EvalItem(source="ladder_210", item_id="NF-0001", text=question)
        ])
        candidate = (
            "User: I read that manures matter.\n"
            f"Assistant: {question}\n"
        )
        hits = index.classify(candidate)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["eval_source"], "ladder_210")

    def test_unrelated_text_is_clean(self):
        index = ContaminationIndex([
            EvalItem(source="ladder_210", item_id="NF-0001", text="what is a primary purpose of animal manures here")
        ])
        self.assertEqual(index.classify("I grow corn in Oklahoma and use compost tea."), [])

    def test_check_rows_partitions_and_annotates(self):
        index = ContaminationIndex([
            EvalItem(source="val_v3", item_id="v1", text="is creeping thyme native to washington state")
        ])
        rows = [
            {"conversation_id": "clean", "sessions": [{"turns": [{"text": "what eats aphids"}]}]},
            {"conversation_id": "dirty", "sessions": [{"turns": [
                {"text": "question: is creeping thyme native to washington state please tell me"}
            ]}]},
        ]
        clean, dirty = check_rows(rows, index)
        self.assertEqual([r["conversation_id"] for r in clean], ["clean"])
        self.assertEqual([r["conversation_id"] for r in dirty], ["dirty"])
        self.assertEqual(dirty[0]["contamination"]["label"], "dirty")

    def test_candidate_text_handles_session_rows(self):
        row = {"sessions": [{"turns": [{"text": "hello"}, {"text": "world"}]}]}
        self.assertEqual(candidate_text(row), "hello world")


class DialogueValidationTests(unittest.TestCase):
    def test_parse_valid_dialogue(self):
        from data.make_persona_memory_corpus import parse_dialogue

        raw = json.dumps({
            "sessions": [
                {"days_later": 0, "turns": [{"from": "human", "text": "hi"}, {"from": "assistant", "text": "hello"}]},
                {"days_later": 30, "turns": [{"from": "human", "text": "recall?"}, {"from": "assistant", "text": "yes"}]},
            ],
            "facts": [{"fact": "grows corn", "stated_session": 0, "recalled_session": 1, "changed": False, "changed_to": None}],
        })
        payload = parse_dialogue(raw)
        self.assertEqual(len(payload["sessions"]), 2)

    def test_parse_rejects_single_session(self):
        from data.make_persona_memory_corpus import parse_dialogue

        with self.assertRaises(ValueError):
            parse_dialogue(json.dumps({"sessions": [{"days_later": 0, "turns": [{"from": "human", "text": "hi"}]}], "facts": []}))

    def test_build_briefs_grounds_in_domain(self):
        from data.make_persona_memory_corpus import build_briefs

        domain = {
            "local_plants_by_region": [{"Location": "Oklahoma", "Plants": ["Comfrey"]}],
            "crop_requirements": [{"Crop": "Corn", "Scientific Name": "Zea mays", "NPK Requirements": {}, "Additional Requirements": {}}],
            "plant_mineral_profiles": [{"common_name": "Comfrey"}],
            "enriched_recipes": [{"recipe_name": "FPJ", "recipe_abbrev": "FPJ"}],
        }
        seeds = [{"conversation_id": "c1", "text": "I have a farm in Oklahoma"}]
        briefs = build_briefs(seeds, domain, 5, seed=1, evolution_fraction=0.4)
        self.assertEqual(len(briefs), 5)
        for brief in briefs:
            self.assertEqual(brief["region"], "Oklahoma")
            self.assertEqual(brief["crop"], "Corn")
            self.assertTrue(brief["voice_examples"])


if __name__ == "__main__":
    unittest.main()
