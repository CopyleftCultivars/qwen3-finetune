from __future__ import annotations

import unittest

from rlm_slice import multi_tool_context as mtc
from rlm_slice.retrieval import RetrievalResult


def _stub_embed_fn(texts: list[str]) -> list[list[float]]:
    """Same deterministic stub style as test_tool_router.py."""
    vocab = ["bioaccumulator", "ratio", "region", "recipe", "corn", "comfrey", "mekong"]

    def vec(text: str) -> list[float]:
        words = set(w for w in text.lower().replace(",", " ").split())
        return [1.0 if any(v in w for w in words) else 0.0 for v in vocab]

    return [vec(t) for t in texts]


def _stub_retrieve_fn(query: str, k: int) -> RetrievalResult:
    return RetrievalResult(query=query, snippets=[], mode_used="search", latency_ms=0.0)


class ArgExtractionTests(unittest.TestCase):
    """chainlink #175: heuristic argument extraction over the closed domain vocab."""

    def test_bioaccumulator_ratio_extracts_both_slots_when_present(self):
        args = mtc.extract_args("bioaccumulator_ratio", "How much Comfrey should I use for Corn?")
        self.assertEqual(args, {"crop": "Corn", "candidate_plant": "Comfrey"})

    def test_bioaccumulator_ratio_returns_none_when_a_slot_is_missing(self):
        args = mtc.extract_args("bioaccumulator_ratio", "How much Comfrey should I use?")
        self.assertIsNone(args)

    def test_regional_recipe_lookup_always_has_a_problem_fallback(self):
        args = mtc.extract_args("regional_recipe_lookup", "What grows in the Mekong Delta?")
        self.assertEqual(args["region"], "Mekong Delta")
        self.assertEqual(args["problem"], "What grows in the Mekong Delta?")

    def test_unknown_tool_name_returns_none(self):
        self.assertIsNone(mtc.extract_args("not_a_real_tool", "anything"))


class BuildMultiToolContextTests(unittest.TestCase):
    """Stubs both retrieval and embedding so this exercises only the
    controller-owned routing/extraction/execution logic, not qmd/torch."""

    def test_bioaccumulator_query_selects_and_executes_the_ratio_tool(self):
        ctx = mtc.build_multi_tool_context(
            "What ratio of Comfrey should I add for Corn's nitrogen needs?",
            embed_fn=_stub_embed_fn,
            retrieve_fn=_stub_retrieve_fn,
        )
        by_name = {c.name: c for c in ctx.tool_calls}
        self.assertIn("bioaccumulator_ratio", by_name)
        record = by_name["bioaccumulator_ratio"]
        self.assertTrue(record.args_available)
        if record.selected_by_router:
            self.assertTrue(record.executed)
            self.assertIsNotNone(record.result)

    def test_off_topic_query_still_records_every_tool_even_if_none_executed(self):
        ctx = mtc.build_multi_tool_context(
            "What is the boiling point of water?",
            embed_fn=_stub_embed_fn,
            retrieve_fn=_stub_retrieve_fn,
        )
        self.assertEqual(len(ctx.tool_calls), 2)

    def test_selected_but_missing_args_is_recorded_not_silently_dropped(self):
        # "Comfrey" alone gives regional_recipe_lookup nothing structured but the
        # problem fallback keeps it runnable; bioaccumulator_ratio has no crop,
        # so if the router selects it anyway, args_available must be False.
        ctx = mtc.build_multi_tool_context(
            "Tell me about Comfrey as a bioaccumulator",
            top_k=2,  # force both tools to be "selected" regardless of score
            embed_fn=_stub_embed_fn,
            retrieve_fn=_stub_retrieve_fn,
        )
        by_name = {c.name: c for c in ctx.tool_calls}
        ratio_record = by_name["bioaccumulator_ratio"]
        self.assertTrue(ratio_record.selected_by_router)
        self.assertFalse(ratio_record.args_available)
        self.assertFalse(ratio_record.executed)

    def test_as_context_includes_retrieval_and_executed_tool_results_only(self):
        ctx = mtc.build_multi_tool_context(
            "What ratio of Comfrey should I add for Corn's nitrogen needs?",
            top_k=2,
            embed_fn=_stub_embed_fn,
            retrieve_fn=_stub_retrieve_fn,
        )
        text = ctx.as_context()
        executed_names = [c.name for c in ctx.tool_calls if c.executed]
        for name in executed_names:
            self.assertIn(f"[{name}]", text)

    def test_as_dict_is_json_serializable_for_the_per_question_audit_record(self):
        import json

        ctx = mtc.build_multi_tool_context(
            "What grows in the Mekong Delta?",
            embed_fn=_stub_embed_fn,
            retrieve_fn=_stub_retrieve_fn,
        )
        json.dumps(ctx.as_dict())  # must not raise


if __name__ == "__main__":
    unittest.main()
