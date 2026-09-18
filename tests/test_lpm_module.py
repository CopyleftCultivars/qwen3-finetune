from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lpm.config import (  # noqa: E402
    LPMConfig,
    RESERVED_ID_COUNT,
    entropy_target,
    reserved_id_map,
    slot_to_reserved_id,
)
from lpm.data import PersonaIndex, filter_records, group_by_persona, select_subset  # noqa: E402

try:
    import torch  # noqa: E402

    from lpm.memory import LPMModule, magnitude_penalty, project_to_ball  # noqa: E402

    HAVE_TORCH = True
except Exception:  # noqa: BLE001 -- broken/missing torch must skip, not fail
    HAVE_TORCH = False


def _record(persona: str, chunk: int, first: bool) -> dict:
    return {
        "persona_id": persona,
        "chunk_index": chunk,
        "is_first_chunk": first,
        "cross_chunk": not first,
        "source": "synthetic_multisession",
        "prompt_text": f"{persona} prompt {chunk}",
        "target_text": f"{persona} target {chunk}",
    }


class ConfigTests(unittest.TestCase):
    def test_persona_mem_config_is_256_slots(self):
        config = LPMConfig()
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.n_slots, 32)
        self.assertEqual(config.total_slots, 256)
        self.assertEqual(config.total_slots * config.d_mem, 65536)

    def test_entropy_target_is_0_4_ln_k(self):
        self.assertAlmostEqual(entropy_target(32), 0.4 * math.log(32), places=9)

    def test_reserved_id_map_is_bijective_into_pool(self):
        mapping = reserved_id_map()
        self.assertEqual(len(mapping), 256)
        self.assertEqual(len(set(mapping)), 256)
        self.assertEqual(min(mapping), 151670)
        self.assertEqual(max(mapping), 151925)
        self.assertLessEqual(max(mapping), 151670 + RESERVED_ID_COUNT - 1)
        # 10 IDs of headroom remain in the 266-ID pool.
        self.assertEqual(151670 + RESERVED_ID_COUNT - 1 - max(mapping), 10)
        self.assertEqual(mapping[151670], (0, 0))
        self.assertEqual(mapping[151701], (0, 31))
        self.assertEqual(mapping[151702], (1, 0))

    def test_slot_to_reserved_id_boundaries(self):
        self.assertEqual(slot_to_reserved_id(0, 0), 151670)
        self.assertEqual(slot_to_reserved_id(7, 31), 151925)
        with self.assertRaises(ValueError):
            slot_to_reserved_id(9, 0)  # flat 288 > 265 (266-ID pool)

    def test_validate_rejects_oversized_slot_pool(self):
        with self.assertRaises(ValueError):
            LPMConfig(n_heads=9, n_slots=32).validate()


class FirstChunkDecisionTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            _record("p1", 0, True),
            _record("p1", 1, False),
            _record("p2", 0, True),
        ]

    def test_default_drops_first_chunk(self):
        kept = filter_records(self.records)
        self.assertEqual([r["chunk_index"] for r in kept], [1])

    def test_toggle_keeps_first_chunk(self):
        kept = filter_records(self.records, LPMConfig(include_first_chunk=True))
        self.assertEqual(len(kept), 3)

    def test_select_subset_needs_two_cross_chunks(self):
        # p1 has one cross-chunk record -> excluded at min_chunks_per_persona=2.
        subset = select_subset(self.records, LPMConfig(), min_chunks_per_persona=2)
        self.assertEqual(subset, [])

    def test_persona_index_and_grouping(self):
        index = PersonaIndex.build(self.records)
        self.assertEqual(len(index), 2)
        self.assertEqual(index.id_of("p1"), 0)
        groups = group_by_persona(self.records)
        self.assertEqual([r["chunk_index"] for r in groups["p1"]], [0, 1])


@unittest.skipUnless(HAVE_TORCH, "torch unavailable in this env")
class ModuleTests(unittest.TestCase):
    def _module(self, n_personas: int = 4) -> "LPMModule":
        return LPMModule(n_personas, LPMConfig())

    def test_forward_shapes_and_attention_normalization(self):
        config = LPMConfig()
        module = self._module()
        persona_ids = torch.tensor([0, 1, 2, 3])
        query = torch.randn(4, config.d_model)
        out = module(persona_ids, query)
        self.assertEqual(tuple(out["prompts"].shape), (4, config.n_heads, config.d_model))
        self.assertEqual(tuple(out["attention"].shape), (4, config.n_heads, config.n_slots))
        self.assertEqual(tuple(out["bake_vectors"].shape), (4, config.n_heads, config.n_slots, config.d_model))
        self.assertTrue(torch.allclose(out["attention"].sum(-1), torch.ones(4, config.n_heads), atol=1e-5))
        self.assertTrue(torch.isfinite(out["prompts"]).all())
        self.assertGreaterEqual(float(out["entropy_penalty"].detach()), 0.0)
        self.assertGreaterEqual(float(out["mag_penalty"].detach()), 0.0)

    def test_slot_parameter_budget(self):
        module = self._module(n_personas=10)
        self.assertEqual(module.slot_memory.slots.shape, (10, 8, 32, 256))
        self.assertEqual(module.slot_memory.slots.numel(), 10 * 65536)

    def test_gradients_reach_slots_and_projector(self):
        config = LPMConfig()
        module = self._module()
        out = module(torch.tensor([0, 1]), torch.randn(2, config.d_model))
        loss = out["prompts"].pow(2).mean() + module.loss_terms(out)["entropy_penalty"]
        loss.backward()
        self.assertIsNotNone(module.slot_memory.slots.grad)
        self.assertGreater(float(module.slot_memory.slots.grad.norm()), 0.0)
        self.assertGreater(float(module.projection.w_v[0].weight.grad.norm()), 0.0)

    def test_bake_vectors_match_projector(self):
        module = self._module()
        ids = torch.tensor([3])
        bake = module.bake_vectors(ids)
        manual = module.projection.project_slots(module.slot_memory(ids))
        self.assertTrue(torch.allclose(bake, manual, atol=1e-6))

    def test_project_to_ball_bounds_norm_and_is_differentiable(self):
        vectors = (torch.randn(4, 8, 32, 1024) * 10.0).requires_grad_(True)
        bounded = project_to_ball(vectors, 4.0)
        self.assertLessEqual(float(bounded.norm(dim=-1).amax()), 4.0 + 1e-4)
        self.assertGreater(float(bounded.norm(dim=-1).max()), 3.0)
        bounded.sum().backward()
        self.assertIsNotNone(vectors.grad)

    def test_max_mode_penalty_tracks_worst_vector_not_mean(self):
        # One severe vector among many mild ones: mean mode hides it, max mode does not.
        vectors = torch.full((1, 1, 128, 8), 0.1)
        vectors[0, 0, 0, 0] = 10.0  # single severe vector
        max_pen = float(magnitude_penalty(vectors, 4.0, mode="max"))
        mean_pen = float(magnitude_penalty(vectors, 4.0, mode="mean"))
        self.assertGreater(max_pen, 30.0)
        self.assertLess(mean_pen, 0.5)
        self.assertEqual(float(magnitude_penalty(vectors, 4.0, mode="none")), 0.0)

    def test_uniform_attention_meets_entropy_target(self):
        # A uniform distribution over 32 slots has entropy ln(32) > 0.4*ln(32),
        # so the hinge regularizer is inactive (penalty 0).
        uniform = torch.full((2, 8, 32), 1.0 / 32)
        entropy = -(uniform * torch.log(uniform)).sum(-1)
        self.assertAlmostEqual(float(entropy.mean()), math.log(32), places=5)
        target = entropy_target(32)
        penalty = torch.relu(target - entropy).mean()
        self.assertEqual(float(penalty), 0.0)

    def test_magnitude_penalty_only_above_threshold(self):
        below = torch.full((4, 8, 32, 1024), 0.001)
        self.assertEqual(float(magnitude_penalty(below, 4.0)), 0.0)
        above = torch.zeros(1, 1, 1, 1024)
        above[..., 0] = 8.0
        self.assertAlmostEqual(float(magnitude_penalty(above, 4.0)), 16.0, places=4)


if __name__ == "__main__":
    unittest.main()
