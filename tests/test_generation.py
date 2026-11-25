"""CPU regressions for Baseline (SD2 Direct) and Proposed (Open-World Amodal SD2) pipelines."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.models.openworld_models import validate_openworld_proposed_config
from src.pipelines.baseline import Baseline, build_baseline_prompt, validate_baseline_config
from src.pipelines.proposed import Proposed

ROOT = Path(__file__).resolve().parents[1]


def baseline_config():
    return json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))


def proposed_config():
    return json.loads((ROOT / "configs/proposed.json").read_text(encoding="utf-8"))


class GenerationTests(unittest.TestCase):
    def test_sd2_baseline_uses_class_label_prompt_and_single_pass_inpainting(self):
        from src.evaluation.runner import _uses_occluded_mask_input

        cfg = baseline_config()
        validate_baseline_config(cfg)
        self.assertTrue(_uses_occluded_mask_input(cfg))
        self.assertEqual(cfg["inpainting_model_id"], "sd2-community/stable-diffusion-2-inpainting")
        self.assertEqual(cfg["text_query_policy"], "dataset_object_name")
        self.assertEqual(
            build_baseline_prompt("teddy bear"),
            "a photo of a teddy bear, complete object on a plain white background, natural colors, realistic texture",
        )
        self.assertEqual(
            build_baseline_prompt(""),
            "a photo of a object, complete object on a plain white background, natural colors, realistic texture",
        )

        image = np.full((48, 64, 3), 140, np.uint8)
        vis = np.zeros((48, 64), bool)
        vis[10:30, 10:20] = True
        hidden = np.zeros_like(vis)
        hidden[10:30, 20:30] = True
        models = SimpleNamespace(device="cpu", pipe=object())
        with patch("src.pipelines.baseline.run_inpaint_roi_pass", return_value=np.full_like(image, 77)) as call:
            result = Baseline(models, cfg).complete(image, vis, hidden, label="wooden chair")
        call.assert_called_once()
        self.assertEqual(
            call.call_args.kwargs["prompt"],
            "a photo of a wooden chair, complete object on a plain white background, natural colors, realistic texture",
        )
        self.assertTrue(np.all(result["inpainted_rgba"][..., :3] == 77))
        self.assertIn("SD2 Direct", result["_trace"]["backend"])
        self.assertEqual(result["critique_history"], [])

    def test_refiner_returns_entire_decoded_patch_and_seam_uses_output(self):
        from src.eval.metrics import compute_appearance_coherence_metrics
        from src.stages.inpaint_refiner import refine_exact_mask

        canvas = np.zeros((40, 48, 3), np.uint8)
        visible = np.zeros(canvas.shape[:2], bool)
        visible[8:30, 10:20] = True
        hidden = np.zeros_like(visible)
        hidden[8:30, 20:30] = True
        decoded = np.full_like(canvas, 180)
        with patch("src.stages.inpaint_refiner.run_inpaint_roi_pass", return_value=decoded):
            actual = refine_exact_mask(None, "cpu", canvas, visible, hidden, "", "", 0.24, 6, 28, 42)
        np.testing.assert_array_equal(actual, decoded)
        stats = compute_appearance_coherence_metrics(visible, hidden, actual)
        self.assertEqual(stats, {"seam_mae": 0.0, "color_hist_sim": 1.0})

    def test_non_sd2_backbone_is_rejected_for_baseline_and_proposed(self):
        for cfg, validator in ((baseline_config(), validate_baseline_config), (proposed_config(), validate_openworld_proposed_config)):
            cfg["inpainting_model_id"] = "stable-diffusion-v1-5/stable-diffusion-inpainting"
            with self.assertRaisesRegex(ValueError, "SD2 Inpainting"):
                validator(cfg)

    def test_openworld_proposed_uses_visible_mask_directly_and_stops_at_max_iter_3(self):
        """Verify 4 Open-World Amodal modules run without LISA/LaMa, SAM is reused, and loop stops at max_iter=3."""
        cfg = proposed_config()
        cfg["max_iter"] = 3
        h, w = 48, 64
        image = np.full((h, w, 3), [90, 120, 150], np.uint8)
        vis = np.zeros((h, w), bool)
        vis[12:34, 10:26] = True
        occ = np.zeros((h, w), bool)
        occ[12:34, 26:42] = True

        sam_calls = []
        sd2_calls = []

        class FakeTensor:
            def __init__(self, arr):
                self.arr = np.asarray(arr, dtype=np.float32)

            def size(self, dim=0):
                return self.arr.shape[dim]

            def clone(self):
                return FakeTensor(self.arr.copy())

            def cpu(self):
                return self

            def float(self):
                return self

        class FakeRAM:
            def infer_tags(self, img_pil):
                return ["table", "chair", "background"]

        class FakeCLIP:
            def select_prompt(self, masked_pil, classes, text_query=""):
                return text_query or "table"

        class FakeGDINO:
            def detect(self, img_pil, caption, box_thresh=0.35, text_thresh=0.35):
                return FakeTensor([[0.5, 0.5, 0.4, 0.4]]), ["table(0.88)", "occluder(0.79)"]

        class FakeSAM:
            def segment_boxes(self, img_pil, boxes_filt):
                sam_calls.append(img_pil.size)
                pw, ph = img_pil.size
                m_target = np.zeros((ph, pw), dtype=bool)
                m_target[ph // 4 : 3 * ph // 4, pw // 4 : 3 * pw // 4] = True
                m_occ = np.zeros((ph, pw), dtype=bool)
                m_occ[ph // 4 : 3 * ph // 4, pw // 2 : 7 * pw // 8] = True
                return np.array(img_pil), np.stack([m_target, m_occ], axis=0)

        class FakeInstaOrder:
            def analyze_masks(self, img_np, masks, mask_id):
                target_m = masks[mask_id] > 0
                mh, mw = target_m.shape[:2]
                occ_m = np.zeros((mh, mw), dtype=np.uint8)
                occ_m[:, mw // 2 :] = 1
                occ_m[target_m] = 0
                return [occ_m]

        class FakePipe:
            def __call__(self, **kwargs):
                sd2_calls.append(kwargs["prompt"])
                out_img = Image.new("RGB", (512, 512), (110, 130, 160))
                return SimpleNamespace(images=[out_img])

        fake_torch = types.ModuleType("torch")
        fake_torch.Generator = lambda device="cpu": SimpleNamespace(manual_seed=lambda s: s)
        fake_torch.inference_mode = contextlib.nullcontext
        fake_torch.cat = lambda tensors, dim=0: tensors[0]

        models = SimpleNamespace(
            device="cpu",
            ram_plus=FakeRAM(),
            openai_clip=FakeCLIP(),
            gdino=FakeGDINO(),
            sam=FakeSAM(),
            instaorder=FakeInstaOrder(),
            pipe=FakePipe(),
        )
        completer = Proposed(models, cfg)
        old_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            with tempfile.TemporaryDirectory() as temp:
                job = Path(temp)
                result = completer.complete(image, vis, text_query="wooden table", job_dir=job, occluded_mask=occ)
                self.assertEqual(len(sd2_calls), 3)
                self.assertEqual(result["_trace"]["synthesis_attempts"], 3)
                self.assertEqual(result["_trace"]["selected_class"], "wooden table")
                self.assertEqual(len(sam_calls), 4)
                self.assertEqual(result["inpainted_rgba"].shape, (h, w, 4))
                self.assertTrue((job / "masked_image.png").is_file())
                self.assertTrue((job / "prompt.txt").is_file())
                self.assertTrue((job / "amodal_completion_crop.png").is_file())
                self.assertTrue((job / "amodal_segmentation.png").is_file())
        finally:
            if old_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch


if __name__ == "__main__":
    unittest.main()
