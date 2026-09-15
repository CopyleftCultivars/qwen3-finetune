from __future__ import annotations

import unittest

from rlm_slice import domain_tools


class BioaccumulatorRatioTests(unittest.TestCase):
    """chainlink #174."""

    def test_known_crop_and_plant_produce_comparable_ratios(self):
        result = domain_tools.bioaccumulator_ratio("Corn", "Okra")
        self.assertNotIn("error", result)
        self.assertEqual(result["crop"], "Corn")
        self.assertIn("phosphorus", result["ratios_ppm_per_requirement_unit"])
        self.assertIn("potassium", result["ratios_ppm_per_requirement_unit"])
        self.assertIn("nitrogen", result["not_comparable"])

    def test_unknown_crop_returns_explicit_error_not_an_exception(self):
        result = domain_tools.bioaccumulator_ratio("Not A Real Crop Xyz", "Okra")
        self.assertIn("error", result)

    def test_unknown_plant_returns_explicit_error_not_an_exception(self):
        result = domain_tools.bioaccumulator_ratio("Corn", "Not A Real Plant Xyz")
        self.assertIn("error", result)

    def test_case_insensitive_and_substring_matching(self):
        result = domain_tools.bioaccumulator_ratio("corn", "okra")
        self.assertNotIn("error", result)


class RegionalRecipeLookupTests(unittest.TestCase):
    """chainlink #174. use_qmd=False everywhere so tests don't depend on the
    `qmd` binary being installed -- retrieve() already degrades gracefully
    without it, but the point of these tests is the structured-lookup logic."""

    def test_requires_at_least_one_argument(self):
        result = domain_tools.regional_recipe_lookup(use_qmd=False)
        self.assertIn("error", result)

    def test_region_substring_match(self):
        result = domain_tools.regional_recipe_lookup(region="Mekong", use_qmd=False)
        self.assertNotIn("error", result)
        self.assertTrue(any("Mekong" in row["Location"] for row in result["region_matches"]))

    def test_crop_to_regions_reverse_lookup(self):
        result = domain_tools.regional_recipe_lookup(crop="Rice", use_qmd=False)
        self.assertNotIn("error", result)
        self.assertIsInstance(result["regions_growing_crop"], list)

    def test_recipe_matching_by_bioaccumulator_name(self):
        result = domain_tools.regional_recipe_lookup(crop="Comfrey", use_qmd=False)
        self.assertNotIn("error", result)
        self.assertTrue(len(result["matching_recipes"]) > 0)
        self.assertTrue(
            any("Comfrey" in b for r in result["matching_recipes"] for b in r.get("suggested_bioaccumulators", []))
        )

    def test_problem_nutrient_hint_matches_primary_nutrient(self):
        result = domain_tools.regional_recipe_lookup(problem="nitrogen deficiency", use_qmd=False)
        self.assertNotIn("error", result)
        self.assertTrue(any(r.get("primary_nutrient") == "N" for r in result["matching_recipes"]))

    def test_qmd_skipped_message_when_use_qmd_false(self):
        result = domain_tools.regional_recipe_lookup(crop="Rice", use_qmd=False)
        self.assertEqual(result["qmd_context"], "qmd lookup skipped (use_qmd=False)")


class ToolDefinitionShapeTests(unittest.TestCase):
    """The Cactus-shaped definitions must match cactus-sdk's ToolFunction
    (name/description/parameters) since select_relevant_tools scores
    name + " " + description verbatim -- chainlink #172's on-device subissue
    (#177) reuses these without redefinition."""

    def test_each_definition_has_name_description_parameters(self):
        for tool in domain_tools.TOOL_DEFINITIONS:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("parameters", tool)
            self.assertIsInstance(tool["description"], str)
            self.assertGreater(len(tool["description"]), 0)

    def test_registry_names_match_definition_names(self):
        definition_names = {t["name"] for t in domain_tools.TOOL_DEFINITIONS}
        self.assertEqual(definition_names, set(domain_tools.TOOL_REGISTRY.keys()))

    def test_domain_tools_matches_agronomy_tools_shape(self):
        tools = domain_tools.domain_tools()
        self.assertTrue(all(callable(fn) for fn in tools.values()))


if __name__ == "__main__":
    unittest.main()
