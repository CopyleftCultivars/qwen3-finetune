from __future__ import annotations

import unittest

from rlm_slice import domain_data


class DomainDataLoaderTests(unittest.TestCase):
    """chainlink #173: the four domain sources load, and the superseded
    bioaccumulator list is not one of them."""

    def test_plant_mineral_profiles_is_the_only_bioaccumulator_source(self):
        rows = domain_data.load_plant_mineral_profiles()
        self.assertEqual(len(rows), 660)
        self.assertTrue(all("scientific_name" in r or "common_name" in r for r in rows))

    def test_crop_requirements_loads(self):
        rows = domain_data.load_crop_requirements()
        self.assertEqual(len(rows), 97)

    def test_local_plants_by_region_loads(self):
        rows = domain_data.load_local_plants_by_region()
        self.assertEqual(len(rows), 127)
        self.assertTrue(all("Location" in r and "Plants" in r for r in rows))

    def test_enriched_recipes_loads(self):
        rows = domain_data.load_enriched_recipes()
        self.assertGreater(len(rows), 0)
        self.assertTrue(all("recipe_name" in r for r in rows))

    def test_archived_bioaccumulator_list_is_not_loaded_by_this_module(self):
        self.assertFalse(hasattr(domain_data, "load_bioaccumulators_list"))
        self.assertFalse(hasattr(domain_data, "BIOACCUMULATORS_LIST"))


if __name__ == "__main__":
    unittest.main()
