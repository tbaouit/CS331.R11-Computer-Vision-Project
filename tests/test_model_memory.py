"""CPU regression tests for sequential GPU stage residency and single-init SAM caching."""
from types import SimpleNamespace
import unittest

from src.models.memory_manager import FoundationModels
from src.pipelines.backend import AmodalBackend


class FakeModel:
    def __init__(self, name, events, resident, fail=False):
        self.name, self.events, self.resident = name, events, resident
        self.fail = fail

    def to(self, device):
        self.events.append((self.name, device))
        if device == "cuda":
            if self.resident:
                raise AssertionError("Two models would occupy CUDA simultaneously")
            self.resident.add(self.name)
            if self.fail:
                raise RuntimeError("synthetic CUDA OOM during transfer")
        else:
            self.resident.discard(self.name)
        return self


class ModelMemoryTests(unittest.TestCase):
    def make_registry(self, fail=None):
        registry = FoundationModels("stabilityai/stable-diffusion-2-inpainting", "ViT-B/32")
        events, resident, loads = [], set(), []

        def load(name):
            self.assertFalse(resident, "Previous stage must be offloaded before loading next weights")
            loads.append(name)
            return FakeModel(name, events, resident, name == fail)

        registry._load = load
        registry._clear_cache = lambda: None
        return registry, events, resident, loads

    def test_lazy_load_never_overlaps_gpu_models_and_reuses_sam_once(self):
        registry, events, resident, loads = self.make_registry()
        self.assertEqual(loads, [])
        sam_first = registry.sam
        self.assertIs(registry.sam, sam_first)
        registry.ram_plus
        registry.openai_clip
        registry.gdino
        self.assertIs(registry.sam, sam_first)  # SAM ViT-H is reused across iterations, never reloaded!
        registry.instaorder
        registry.pipe
        registry.release_gpu()
        self.assertEqual(loads, ["sam", "ram_plus", "openai_clip", "gdino", "instaorder", "inpainting"])
        self.assertFalse(resident)
        self.assertIsNone(registry._active)
        self.assertEqual(events[:3], [("sam", "cuda"), ("sam", "cpu"), ("ram_plus", "cuda")])

    def test_failed_activation_returns_partially_moved_model_to_cpu(self):
        registry, events, resident, _ = self.make_registry(fail="gdino")
        registry.ram_plus
        with self.assertRaisesRegex(RuntimeError, "synthetic CUDA OOM"):
            registry.gdino
        self.assertFalse(resident)
        self.assertIsNone(registry._active)
        self.assertEqual(events[-2:], [("gdino", "cuda"), ("gdino", "cpu")])

    def test_backend_releases_last_model_on_success_and_error_in_both_modes(self):
        for mode in ("baseline", "proposed"):
            for fail in (False, True):
                with self.subTest(mode=mode, fail=fail):
                    registry, _, resident, _ = self.make_registry()
                    backend = AmodalBackend.__new__(AmodalBackend)
                    backend.models = registry
                    backend.mode = mode
                    backend.config, backend.job = {"text_query_policy": "dataset_object_name"}, None

                    def complete(**kwargs):
                        registry.pipe
                        if fail:
                            raise RuntimeError("synthetic generation failure")
                        return {"finished": True}

                    backend.completer = SimpleNamespace(complete=complete)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "synthetic generation failure"):
                            backend.complete(None, None, "chair", occluded_mask=1)
                    else:
                        self.assertEqual(backend.complete(None, None, "chair", occluded_mask=1), {"finished": True})
                    self.assertFalse(resident)
                    self.assertIsNone(registry._active)


if __name__ == "__main__":
    unittest.main()
