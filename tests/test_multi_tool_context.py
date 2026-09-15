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


class HasRealContentTests(unittest.TestCase):
    """chainlink #175 v2: viability is decided by running the (cheap, local)
    tool and checking its actual output, not by a router score threshold --
    see multi_tool_context.py's module docstring for why a score-based
    threshold was tried and rejected (the on-topic/off-topic score
    distributions from the first sweep didn't separate cleanly)."""

    def test_error_result_is_never_real_content(self):
        self.assertFalse(mtc._has_real_content("bioaccumulator_ratio", {"error": "no match"}))

    def test_bioaccumulator_ratio_with_ratios_is_real_content(self):
        result = {"ratios_ppm_per_requirement_unit": {"phosphorus": 1.2}}
        self.assertTrue(mtc._has_real_content("bioaccumulator_ratio", result))

    def test_bioaccumulator_ratio_with_empty_ratios_is_not_real_content(self):
        result = {"ratios_ppm_per_requirement_unit": {}, "not_comparable": ["nitrogen", "phosphorus", "potassium"]}
        self.assertFalse(mtc._has_real_content("bioaccumulator_ratio", result))

    def test_regional_recipe_lookup_with_any_nonempty_field_is_real_content(self):
        self.assertTrue(mtc._has_real_content("regional_recipe_lookup", {"region_matches": [{"Location": "x"}]}))
        self.assertTrue(mtc._has_real_content("regional_recipe_lookup", {"regions_growing_crop": ["x"]}))
        self.assertTrue(mtc._has_real_content("regional_recipe_lookup", {"matching_recipes": [{"recipe_name": "x"}]}))

    def test_regional_recipe_lookup_with_all_empty_fields_is_not_real_content(self):
        result = {"region_matches": [], "regions_growing_crop": [], "matching_recipes": []}
        self.assertFalse(mtc._has_real_content("regional_recipe_lookup", result))


class BuildMultiToolContextTests(unittest.TestCase):
    """Stubs both retrieval and embedding so this exercises only the
    controller-owned extraction/execution/viability/routing logic, not
    qmd/torch."""

    def test_bioaccumulator_query_is_viable_and_selected(self):
        ctx = mtc.build_multi_tool_context(
            "What ratio of Comfrey should I add for Corn's nitrogen needs?",
            embed_fn=_stub_embed_fn,
            retrieve_fn=_stub_retrieve_fn,
        )
        by_name = {c.name: c for c in ctx.tool_calls}
        record = by_name["bioaccumulator_ratio"]
        self.assertTrue(record.args_available)
        self.assertTrue(record.executed)
        self.assertTrue(record.viable)
        self.assertTrue(record.selected_by_router)
        self.assertIsNotNone(record.result)

    def test_off_topic_query_still_records_every_tool_even_if_none_selected(self):
        ctx = mtc.build_multi_tool_context(
            "What is the boiling point of water?",
            embed_fn=_stub_embed_fn,
            retrieve_fn=_stub_retrieve_fn,
        )
        self.assertEqual(len(ctx.tool_calls), 2)
        self.assertFalse(any(c.selected_by_router for c in ctx.tool_calls))

    def test_missing_args_means_never_viable_or_selected_even_with_top_k_2(self):
        # No crop named -> bioaccumulator_ratio can never be viable/selected,
        # regardless of top_k -- this is the "real select-nothing" behavior
        # the first #175 sweep's forced top-1 pick did not have.
        ctx = mtc.build_multi_tool_context(
            "Tell me about Comfrey as a bioaccumulator",
            top_k=2,
            embed_fn=_stub_embed_fn,
            retrieve_fn=_stub_retrieve_fn,
        )
        by_name = {c.name: c for c in ctx.tool_calls}
        ratio_record = by_name["bioaccumulator_ratio"]
        self.assertFalse(ratio_record.args_available)
        self.assertFalse(ratio_record.executed)
        self.assertFalse(ratio_record.viable)
        self.assertFalse(ratio_record.selected_by_router)

    def test_executed_but_empty_result_is_not_viable_or_selected(self):
        # regional_recipe_lookup always has a `problem` fallback (args_available
        # is trivially true), so it always executes -- but an empty result must
        # not be selected/injected just because it ran.
        ctx = mtc.build_multi_tool_context(
            "What is the boiling point of water?",
            top_k=2,
            embed_fn=_stub_embed_fn,
            retrieve_fn=_stub_retrieve_fn,
        )
        by_name = {c.name: c for c in ctx.tool_calls}
        recipe_record = by_name["regional_recipe_lookup"]
        self.assertTrue(recipe_record.args_available)
        self.assertTrue(recipe_record.executed)
        self.assertFalse(recipe_record.viable)
        self.assertFalse(recipe_record.selected_by_router)

    def test_as_context_includes_retrieval_and_selected_tool_results_only(self):
        ctx = mtc.build_multi_tool_context(
            "What ratio of Comfrey should I add for Corn's nitrogen needs?",
            top_k=2,
            embed_fn=_stub_embed_fn,
            retrieve_fn=_stub_retrieve_fn,
        )
        text = ctx.as_context()
        selected_names = [c.name for c in ctx.tool_calls if c.selected_by_router]
        not_selected_names = [c.name for c in ctx.tool_calls if not c.selected_by_router]
        self.assertTrue(selected_names)  # this query should select at least one tool
        for name in selected_names:
            self.assertIn(f"[{name}]", text)
        for name in not_selected_names:
            self.assertNotIn(f"[{name}]", text)

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
