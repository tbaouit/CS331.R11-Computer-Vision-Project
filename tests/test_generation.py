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

    def test_auto_visible_mask_mode_fallback_debug_artifacts_and_empty_graceful_handling(self):
        """Verify --visible-mask-source auto predicts visible_mask via GDINO+SAM, logs visible_iou, and handles empty predictions."""
        from scripts.evaluate_runs import evaluate_profile
        from src.evaluation.runner import run_one

        cfg = proposed_config()
        cfg["visible_mask_source"] = "auto"
        cfg["max_iter"] = 1
        h, w = 48, 64
        image = np.full((h, w, 3), [100, 130, 160], np.uint8)
        gt_vis = np.zeros((h, w), bool)
        gt_vis[10:30, 12:32] = True
        gt_occ = np.zeros((h, w), bool)
        gt_occ[10:30, 32:48] = True
        gt_amodal = gt_vis | gt_occ

        class FakeTensor:
            def __init__(self, arr):
                self.arr = np.asarray(arr, dtype=np.float32)

            def size(self, dim=0):
                return self.arr.shape[dim]

            def __getitem__(self, item):
                return FakeTensor(self.arr[item])

            def clone(self):
                return FakeTensor(self.arr.copy())

            def cpu(self):
                return self

            def float(self):
                return self

            def tolist(self):
                return self.arr.tolist()

        gdino_thresholds = []

        class FakeGDINO:
            def __init__(self, empty=False):
                self.empty = empty

            def detect(self, img_pil, caption, box_thresh=0.35, text_thresh=0.35):
                gdino_thresholds.append(box_thresh)
                if self.empty or box_thresh > 0.20:
                    return FakeTensor(np.zeros((0, 4), dtype=np.float32)), []
                return (
                    FakeTensor([[0.25, 0.25, 0.2, 0.2], [0.4, 0.45, 0.35, 0.4]]),
                    ["chair(0.21)", "chair(0.64)"],
                )

        class FakeSAM:
            def segment_boxes(self, img_pil, boxes_filt):
                pw, ph = img_pil.size
                n = boxes_filt.size(0)
                pred_m = np.zeros((n, ph, pw), dtype=bool)
                pred_m[:, 12:32, 14:34] = True
                return np.array(img_pil), pred_m

        class FakeRAM:
            def infer_tags(self, img_pil):
                return ["chair", "table"]

        class FakeCLIP:
            def select_prompt(self, masked_pil, classes, text_query=""):
                return text_query or "chair"

        class FakeInstaOrder:
            def analyze_masks(self, img_np, masks, mask_id):
                target_m = masks[mask_id] > 0
                mh, mw = target_m.shape[:2]
                occ_m = np.zeros((mh, mw), dtype=np.uint8)
                occ_m[12:32, 34:48] = 1
                occ_m[target_m] = 0
                return [occ_m]

        class FakePipe:
            def __call__(self, **kwargs):
                return SimpleNamespace(images=[Image.new("RGB", (512, 512), (120, 140, 170))])

        fake_torch = types.ModuleType("torch")
        fake_torch.Generator = lambda device="cpu": SimpleNamespace(manual_seed=lambda s: s)
        fake_torch.inference_mode = contextlib.nullcontext
        fake_torch.cat = lambda tensors, dim=0: tensors[0]

        old_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for name, arr in (
                    ("image.png", image),
                    ("visible.png", gt_vis.astype(np.uint8) * 255),
                    ("hidden.png", gt_occ.astype(np.uint8) * 255),
                    ("amodal.png", gt_amodal.astype(np.uint8) * 255),
                ):
                    Image.fromarray(arr).save(root / name)
                record = dict(
                    sample_id="dev_auto_1",
                    split="dev",
                    name="chair",
                    hidden_ratio=0.4,
                    image="image.png",
                    visible_mask="visible.png",
                    gt_hidden="hidden.png",
                    gt_amodal="amodal.png",
                )

                received_visible_args = []
                received_occluded_args = []

                class AutoBackendWrapper:
                    def __init__(self, empty=False):
                        models = SimpleNamespace(
                            device="cpu",
                            ram_plus=FakeRAM(),
                            openai_clip=FakeCLIP(),
                            gdino=FakeGDINO(empty=empty),
                            sam=FakeSAM(),
                            instaorder=FakeInstaOrder(),
                            pipe=FakePipe(),
                        )
                        self.completer = Proposed(models, cfg)
                        self.job = None

                    def begin_sample(self, job):
                        self.job = Path(job)

                    def complete(self, img, visible, label, occluded_mask=None):
                        received_visible_args.append(visible)
                        received_occluded_args.append(occluded_mask)
                        return self.completer.complete(
                            image=img,
                            visible_mask=visible,
                            text_query=label,
                            job_dir=self.job,
                            occluded_mask=occluded_mask,
                        )

                profile_dir = root / "runs" / "proposed_auto" / "1234567890abcdef"
                res, skipped = run_one(record, root, profile_dir, cfg, AutoBackendWrapper(empty=False), "1234567890abcdef")
                self.assertFalse(skipped)
                self.assertIsNone(received_visible_args[0])
                self.assertIsNone(received_occluded_args[0])
                self.assertIn(0.15, gdino_thresholds)
                self.assertFalse(res["trace"]["visible_pred_empty"])
                self.assertGreater(res["metrics"]["visible_iou"], 0.5)
                job_dir = next(profile_dir.glob("dev_auto_1/*"))
                self.assertTrue((job_dir / "pred_visible.png").is_file())
                self.assertTrue((job_dir / "auto_seg_debug.jpg").is_file())
                self.assertTrue((job_dir / "auto_segmentation.json").is_file())

                eval_dir, rows = evaluate_profile(
                    profile_dir,
                    [record],
                    device="cpu",
                    evaluate=lambda *a, **kw: {"CLIP_score": 25.0, "SSIM": 0.95},
                )
                agg = json.loads((eval_dir / "aggregate.json").read_text(encoding="utf-8"))
                self.assertIn("visible_iou", agg["overall"]["metrics"])
                self.assertAlmostEqual(
                    agg["overall"]["metrics"]["visible_iou"]["mean"],
                    res["metrics"]["visible_iou"],
                )

                # Verify empty detection after fallback logs visible_pred_empty=True and does not crash
                empty_profile = root / "runs" / "proposed_auto" / "fedcba0987654321"
                res_empty, _ = run_one(record, root, empty_profile, cfg, AutoBackendWrapper(empty=True), "fedcba0987654321")
                self.assertTrue(res_empty["trace"]["visible_pred_empty"])
                self.assertEqual(res_empty["metrics"]["visible_iou"], 0.0)
        finally:
            if old_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch


if __name__ == "__main__":
    unittest.main()
