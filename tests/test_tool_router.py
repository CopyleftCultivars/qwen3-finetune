from __future__ import annotations

import math
import unittest

from rlm_slice import tool_router


class TokenizeAndBM25Tests(unittest.TestCase):
    """chainlink #175. These mirror cactus-engine/src/rag.cpp's tokenize_words
    and compute_bm25_score exactly, so the constants/behavior asserted here
    are the port's fidelity check, not arbitrary choices."""

    def test_tokenize_drops_short_and_nonalnum_tokens(self):
        # "a" and "to" are <=2 chars and dropped, matching the C++ `length > 2` gate.
        self.assertEqual(tool_router.tokenize_words("Add a cup to the mix!"), ["add", "cup", "the", "mix"])

    def test_tool_to_text_is_name_space_description(self):
        tool = {"name": "foo", "description": "does a thing"}
        self.assertEqual(tool_router.tool_to_text(tool), "foo does a thing")

    def test_bm25_score_is_zero_for_no_overlap(self):
        score = tool_router.compute_bm25_score(
            query_words=["nitrogen", "deficiency"],
            doc_content="regional plant lookup for bioaccumulators",
            avg_doc_len=5.0,
            doc_freqs={"regional": 1, "plant": 1, "lookup": 1, "bioaccumulators": 1},
            total_docs=2,
        )
        self.assertEqual(score, 0.0)

    def test_bm25_score_is_positive_for_overlap(self):
        score = tool_router.compute_bm25_score(
            query_words=["nitrogen", "ratio"],
            doc_content="nitrogen ratio calculator for bioaccumulator plants",
            avg_doc_len=6.0,
            doc_freqs={"nitrogen": 1, "ratio": 1},
            total_docs=2,
        )
        self.assertGreater(score, 0.0)


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(tool_router.cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(tool_router.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_mismatched_lengths_score_zero(self):
        self.assertEqual(tool_router.cosine_similarity([1.0], [1.0, 0.0]), 0.0)


class RRFFusionTests(unittest.TestCase):
    def test_item_ranked_first_by_both_signals_scores_highest(self):
        emb_ranked = [(0.9, 0), (0.1, 1)]
        bm25_ranked = [(5.0, 0), (0.0, 1)]
        scored = tool_router.compute_rrf_scores(emb_ranked, bm25_ranked, num_items=2)
        self.assertEqual(scored[0][1], 0)

    def test_scores_use_the_documented_rrf_constants(self):
        emb_ranked = [(0.9, 0), (0.1, 1)]
        bm25_ranked = [(5.0, 0), (0.0, 1)]
        scored = dict((idx, score) for score, idx in tool_router.compute_rrf_scores(emb_ranked, bm25_ranked, 2))
        expected_0 = tool_router.RRF_EMB_WEIGHT / (tool_router.RRF_K + 1) + tool_router.RRF_BM25_WEIGHT / (
            tool_router.RRF_K + 1
        )
        self.assertAlmostEqual(scored[0], expected_0)


def _stub_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words one-hot-ish stub: no torch/model download needed."""
    vocab = ["bioaccumulator", "ratio", "region", "recipe", "retrieval", "ferment"]

    def vec(text: str) -> list[float]:
        words = set(tool_router.tokenize_words(text))
        return [1.0 if v in words else 0.0 for v in vocab]

    return [vec(t) for t in texts]


class SelectRelevantToolsTests(unittest.TestCase):
    """chainlink #175: the query-based router, with a deterministic stub
    embedder so this test doesn't need sentence-transformers/torch."""

    TOOLS = [
        {"name": "bioaccumulator_ratio", "description": "Compare crop nutrient ratio against bioaccumulator plant"},
        {"name": "regional_recipe_lookup", "description": "Region and recipe lookup for bioaccumulator plants"},
    ]

    def test_empty_tool_list_returns_no_scores(self):
        result = tool_router.select_relevant_tools("anything", [], embed_fn=_stub_embed_fn)
        self.assertEqual(result.scores, [])

    def test_every_tool_is_scored(self):
        result = tool_router.select_relevant_tools("bioaccumulator ratio question", self.TOOLS, embed_fn=_stub_embed_fn)
        self.assertEqual({s.name for s in result.scores}, {"bioaccumulator_ratio", "regional_recipe_lookup"})

    def test_selected_respects_top_k(self):
        result = tool_router.select_relevant_tools("bioaccumulator ratio question", self.TOOLS, embed_fn=_stub_embed_fn)
        self.assertEqual(len(result.selected(top_k=1)), 1)

    def test_selected_respects_min_score(self):
        result = tool_router.select_relevant_tools("bioaccumulator ratio question", self.TOOLS, embed_fn=_stub_embed_fn)
        self.assertEqual(result.selected(min_score=10_000.0), [])


if __name__ == "__main__":
    unittest.main()
