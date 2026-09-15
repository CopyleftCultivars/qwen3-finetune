from __future__ import annotations

import unittest
from pathlib import Path

from rlm_slice.retrieval import NO_HITS_MESSAGE, RetrievalResult, Snippet

REPO_ROOT = Path(__file__).resolve().parents[1]


class RetrievalMissFeedbackTests(unittest.TestCase):
    """chainlink #171: a retrieval miss must reach the model as a signal,
    not silence -- as_context() must never return an empty string when
    retrieval was actually attempted."""

    def test_empty_snippets_render_as_informative_miss_message(self):
        result = RetrievalResult(
            query="a query with no matches",
            snippets=[],
            mode_used="search",
            latency_ms=12.3,
        )
        self.assertEqual(result.as_context(), NO_HITS_MESSAGE)
        self.assertNotEqual(result.as_context(), "")

    def test_real_snippets_still_render_normally_not_the_miss_message(self):
        result = RetrievalResult(
            query="a query with matches",
            snippets=[
                Snippet(docid="d1", file="qmd://coll/d1", title="Doc One", text="relevant text", score=1.0),
            ],
            mode_used="search",
            latency_ms=12.3,
        )
        context = result.as_context()
        self.assertNotEqual(context, NO_HITS_MESSAGE)
        self.assertIn("Doc One", context)
        self.assertIn("relevant text", context)

    def test_snippets_with_only_blank_text_still_fall_back_to_miss_message(self):
        # A snippet can survive dedupe but render empty after stripping/
        # truncation; that must still surface as a miss, not a blank context.
        result = RetrievalResult(
            query="q",
            snippets=[Snippet(docid="d1", file="qmd://coll/d1", title="T", text="   ", score=1.0)],
            mode_used="search",
            latency_ms=1.0,
        )
        self.assertEqual(result.as_context(), NO_HITS_MESSAGE)


if __name__ == "__main__":
    unittest.main()
