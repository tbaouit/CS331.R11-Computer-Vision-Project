"""CPU regression tests for stage residency and inference-only EMA handling."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.models.registry import FoundationModels
from src.models.pix2gestalt import apply_inference_ema, Pix2GestaltModel
from src.models.qwen_vl import QwenVLModel
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
        registry = FoundationModels("sd2-id", "clip-id")
        events, resident, loads = [], set(), []
        def load(name):
            self.assertFalse(resident, "Previous stage must be offloaded before loading next weights")
            loads.append(name)
            return FakeModel(name, events, resident, name == fail)
        registry._load = load
        registry._clear_cache = lambda: None
        return registry, events, resident, loads

    def test_lazy_load_and_critic_loop_never_overlap_gpu_models(self):
        registry, events, resident, loads = self.make_registry()
        self.assertEqual(loads, [])
        # Both pipelines follow this order, including returning to Qwen as critic.
        first = registry.shape_predictor
        self.assertIs(registry.shape_predictor, first)
        registry.vlm
        registry.clip_model
        registry.pipe
        registry.vlm
        registry.pipe
        registry.vlm
        self.assertIs(registry.shape_predictor, first)  # next sample reuses weights
        registry.release_gpu()
        self.assertEqual(loads, ["shape", "vlm", "clip", "sd2"])
        self.assertFalse(resident)
        self.assertIsNone(registry._active)
        self.assertEqual(events[:3], [("shape", "cuda"), ("shape", "cpu"), ("vlm", "cuda")])

    def test_failed_activation_returns_partially_moved_model_to_cpu(self):
        registry, events, resident, _ = self.make_registry(fail="vlm")
        registry.shape_predictor
        with self.assertRaisesRegex(RuntimeError, "synthetic CUDA OOM"):
            registry.vlm
        self.assertFalse(resident)
        self.assertIsNone(registry._active)
        self.assertEqual(events[-2:], [("vlm", "cuda"), ("vlm", "cpu")])

    def test_ema_weights_are_copied_before_freezing_and_backup_is_removed(self):
        class TinyDiffusion:
            def __init__(self):
                self.use_ema = True
                self.model = SimpleNamespace(weights=np.array([1., 2.]), requires_grad=True)
                self.model_ema = SimpleNamespace(copy_to=self.copy_ema)
            def copy_ema(self, target):
                if not target.requires_grad:
                    raise AssertionError("Upstream EMA cannot copy frozen parameters")
                target.weights[:] = [3., 5.]
            def requires_grad_(self, flag):
                self.model.requires_grad = flag
        model = TinyDiffusion()
        apply_inference_ema(model)
        np.testing.assert_array_equal(model.model.weights, [3., 5.])
        self.assertFalse(model.use_ema)
        self.assertFalse(model.model.requires_grad)
        self.assertFalse(hasattr(model, "model_ema"))
        apply_inference_ema(model)  # no second EMA copy or backup
        np.testing.assert_array_equal(model.model.weights, [3., 5.])

    def test_wrappers_move_parameters_and_input_device_together(self):
        for cls in (Pix2GestaltModel, QwenVLModel):
            with self.subTest(wrapper=cls.__name__):
                wrapper = cls.__new__(cls)
                events, resident = [], set()
                wrapper.model = FakeModel("model", events, resident)
                wrapper.device = "cpu"
                self.assertIs(wrapper.to("cuda"), wrapper)
                self.assertEqual(wrapper.device, "cuda")
                wrapper.to("cpu")
                self.assertEqual(wrapper.device, "cpu")
                self.assertFalse(resident)

    def test_backend_releases_last_model_on_success_and_error_in_both_modes(self):
        for mode in ("improved", "oracle_mask"):
            for fail in (False, True):
                with self.subTest(mode=mode, fail=fail):
                    registry, _, resident, _ = self.make_registry()
                    backend = AmodalBackend.__new__(AmodalBackend)
                    backend.models, backend.mode = registry, mode
                    backend.config, backend.job = {}, None
                    def complete(**kwargs):
                        registry.vlm
                        if fail:
                            raise RuntimeError("synthetic generation failure")
                        return {"finished": True}
                    backend.completer = SimpleNamespace(complete=complete)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "synthetic generation failure"):
                            backend.complete(None, None, "object", occluded_mask=1 if mode == "oracle_mask" else None)
                    else:
                        self.assertEqual(backend.complete(None, None, "object", occluded_mask=1 if mode == "oracle_mask" else None), {"finished": True})
                    self.assertFalse(resident)
                    self.assertIsNone(registry._active)


if __name__ == "__main__":
    unittest.main()
