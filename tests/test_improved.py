import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.scamodal_improved import (
    adaptive_seam_blend,
    build_appearance_prior,
    build_vlm_visual_prompts,
    compose_cropped_for_critic,
    compute_square_roi,
    extract_square_roi,
    fuse_pix2gestalt_predictions,
    lock_low_frequency_to_prior,
    parse_structured_vlm_reasoning,
    paste_square_roi,
)

ROOT = Path(__file__).resolve().parents[1]


class ImprovedPipelineTests(unittest.TestCase):
    def test_roi_crop_small_border_and_full_frame(self):
        # 1. Small interior object -> zoomed window inside frame (no padding)
        vis_small = np.zeros((480, 640), dtype=bool)
        vis_small[200:240, 300:330] = True
        roi_small = compute_square_roi(vis_small, (480, 640), context_scale=3.8, min_crop=256)
        self.assertFalse(roi_small["is_full_frame"])
        self.assertEqual(roi_small["size"], 256)
        self.assertGreaterEqual(roi_small["y0"], 0)
        self.assertGreaterEqual(roi_small["x0"], 0)
        self.assertLessEqual(roi_small["y0"] + roi_small["size"], 480)
        self.assertLessEqual(roi_small["x0"] + roi_small["size"], 640)

        # Exact round-trip on constant-resolution ROI
        grid_y, grid_x = np.meshgrid(np.arange(480, dtype=np.uint8) % 251, np.arange(640, dtype=np.uint8) % 251, indexing="ij")
        img = np.dstack([grid_y, grid_x, np.full((480, 640), 99, dtype=np.uint8)])
        patch = extract_square_roi(img, roi_small, pad_value=0)
        self.assertEqual(patch.shape, (256, 256, 3))
        blank = np.zeros_like(img)
        restored = paste_square_roi(blank, patch, roi_small, resize_mode="nearest")
        np.testing.assert_array_equal(restored[vis_small], img[vis_small])

        # 2. Border-touching small object -> clamped inside frame
        vis_border = np.zeros((480, 640), dtype=bool)
        vis_border[10:60, 0:45] = True
        roi_border = compute_square_roi(vis_border, (480, 640), context_scale=3.8, min_crop=256)
        self.assertEqual(roi_border["x0"], 0)
        self.assertEqual(roi_border["y0"], 0)

        # 3. Large object -> full-frame centered square padding and exact un-pad
        vis_large = np.zeros((480, 640), dtype=bool)
        vis_large[30:450, 40:380] = True
        roi_large = compute_square_roi(vis_large, (480, 640), context_scale=3.8, min_crop=256)
        self.assertTrue(roi_large["is_full_frame"])
        self.assertEqual(roi_large["size"], 640)
        self.assertEqual(roi_large["y0"], -80)
        self.assertEqual(roi_large["x0"], 0)
        large_patch = extract_square_roi(img, roi_large, pad_value=127)
        restored_large = paste_square_roi(np.zeros_like(img), large_patch, roi_large, resize_mode="nearest")
        np.testing.assert_array_equal(restored_large, img)

    def test_fuse_pix2gestalt_suppresses_halo_outliers_and_keeps_interior_slot(self):
        # Work on 256x256 native ROI so 256->256 geometry is crisp and transparent.
        h, w = 256, 256
        image = np.full((h, w, 3), 240, dtype=np.uint8)
        vis = np.zeros((h, w), dtype=bool)
        # U-shaped suit chest [80:180, 70:170] with an interior microphone cutout [110:180, 112:128]
        vis[80:180, 70:170] = True
        vis[110:180, 112:128] = False
        image[vis] = np.array([65, 50, 40], dtype=np.uint8)
        roi = {"y0": 0, "x0": 0, "size": 256, "is_full_frame": True}

        rgb_good = np.full((256, 256, 3), 255, dtype=np.uint8)
        rgb_good[70:180, 70:170] = np.array([68, 52, 42], dtype=np.uint8)

        alpha_1 = np.zeros((256, 256), dtype=np.uint8)
        alpha_1[70:180, 70:170] = 240
        # Soft exterior CarveKit feathering halo ("cat-ears" above object: alpha=55 < 122)
        alpha_1[45:70, 75:165] = 55
        # Interior microphone cutout with partial CarveKit alpha=45 (>= 0.12*255)
        alpha_1[110:180, 112:128] = 45
        # Disconnected stray island far away
        alpha_1[10:28, 10:28] = 250

        alpha_2 = alpha_1.copy()

        # Sample 3: runaway outlier hallucination covering almost the entire canvas
        rgb_outlier = np.full((256, 256, 3), 180, dtype=np.uint8)
        alpha_outlier = np.full((256, 256), 250, dtype=np.uint8)

        fused = fuse_pix2gestalt_predictions(
            samples_rgb_256=[rgb_good, rgb_good, rgb_outlier],
            alphas_256=[alpha_1, alpha_2, alpha_outlier],
            image=image,
            visible_mask=vis,
            roi=roi,
            alpha_threshold=0.48,
        )
        amodal = fused["amodal_mask"]
        # 1. Visible is 100% preserved
        self.assertTrue(np.all(amodal[vis]))
        # 2. Top hidden extension [70:80, 80:160] (alpha=240) is included
        self.assertGreater(amodal[72:78, 80:160].mean(), 0.95)
        # 3. Soft exterior "cat-ear" halo [45:65, 75:165] (alpha=55) is completely rejected
        self.assertEqual(int(amodal[45:65, 75:165].sum()), 0)
        # 4. Disconnected island [10:28, 10:28] is completely removed
        self.assertEqual(int(amodal[10:28, 10:28].sum()), 0)
        # 5. Interior microphone slot [115:175, 114:126] (enclosed on left, right, up) is recovered
        self.assertGreater(amodal[115:175, 114:126].mean(), 0.90)
        # 6. Runaway sample index 2 was excluded
        self.assertNotIn(2, fused["diagnostics"]["kept_indices"])

    def test_appearance_prior_handles_dark_dropout_and_preserves_white_objects(self):
        # Case A: Dark brown suit with pure white dropout hole inside missing_mask
        h, w = 128, 128
        img_dark = np.full((h, w, 3), 20, dtype=np.uint8)
        vis = np.zeros((h, w), dtype=bool)
        amodal = np.zeros((h, w), dtype=bool)
        amodal[25:105, 30:100] = True
        vis[25:105, 30:100] = True
        vis[55:95, 55:75] = False  # hidden microphone hole
        img_dark[vis] = np.array([72, 54, 42], dtype=np.uint8)

        # Shifted Pix2Gestalt RGB (+18 brighter) with pure white [254, 254, 254] dropout in hole
        p2g_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        p2g_rgb[amodal] = np.array([90, 72, 60], dtype=np.uint8)
        p2g_rgb[55:95, 55:75] = np.array([254, 254, 254], dtype=np.uint8)
        p2g_alpha = np.zeros((h, w), dtype=np.float32)
        p2g_alpha[amodal] = 0.95

        prior_dark = build_appearance_prior(img_dark, vis, amodal, p2g_rgb, p2g_alpha)
        self.assertGreater(prior_dark["dropout_fraction"], 0.85)
        hole_mean_luma = prior_dark["seed_canvas"][60:90, 58:72].mean()
        self.assertLess(hole_mean_luma, 100.0)  # Filled with dark brown tones, NOT 254 white or 127 gray

        # Case B: Genuinely white styrofoam container -> NOT flagged as dropout
        img_white = np.full((h, w, 3), 200, dtype=np.uint8)
        img_white[vis] = np.array([232, 234, 238], dtype=np.uint8)
        p2g_white = np.full((h, w, 3), 255, dtype=np.uint8)
        p2g_white[amodal] = np.array([228, 230, 235], dtype=np.uint8)
        prior_white = build_appearance_prior(img_white, vis, amodal, p2g_white, p2g_alpha)
        self.assertEqual(prior_white["dropout_fraction"], 0.0)
        self.assertGreater(prior_white["seed_canvas"][60:90, 58:72].mean(), 215.0)

    def test_vlm_prompts_and_structured_parser(self):
        img = np.full((200, 300, 3), 120, dtype=np.uint8)
        vis = np.zeros((200, 300), dtype=bool)
        vis[80:130, 110:150] = True
        amodal = vis.copy()
        amodal[80:150, 110:165] = True

        ctx, iso, panel = build_vlm_visual_prompts(img, vis, amodal, target_size=224)
        self.assertEqual(ctx.shape, (224, 224, 3))
        self.assertEqual(iso.shape, (224, 224, 3))
        self.assertEqual(panel.shape[0], 224)
        # Corner of isolated_rgb is pure white [255, 255, 255]
        np.testing.assert_array_equal(iso[0, 0], np.array([255, 255, 255], dtype=np.uint8))

        raw = (
            "**OBJECT:** brown plush teddy bear\n"
            "**OCCLUDED_PARTS:** Lower torso and right leg hidden behind white plush toy.\n"
            "**PROMPT:** brown plush teddy bear sitting upright, soft warm brown fur, plain white background"
        )
        parsed = parse_structured_vlm_reasoning(raw)
        self.assertEqual(parsed["object_name"], "brown plush teddy bear")
        self.assertIn("soft warm brown fur", parsed["sd_prompt"])

        fallback = parse_structured_vlm_reasoning("The target object is a green glass jar with metal lid.", "jar")
        self.assertEqual(fallback["object_name"], "jar")
        self.assertIn("green glass jar", fallback["sd_prompt"])

    def test_occluder_censoring_and_frequency_color_lock_prevent_food_hallucination(self):
        # 1. Verify bright red ketchup occluder inside container hole is completely censored in context_rgb
        img = np.full((160, 160, 3), 230, dtype=np.uint8)
        amodal = np.zeros((160, 160), dtype=bool)
        amodal[20:140, 20:140] = True
        vis = amodal.copy()
        vis[45:115, 45:115] = False  # interior hole where sandwich sits
        img[45:115, 45:115] = np.array([245, 25, 20], dtype=np.uint8)  # bright red ketchup occluder

        ctx, _, _ = build_vlm_visual_prompts(img, vis, amodal, target_size=160)
        center_rgb = ctx[80, 80].astype(float)
        # Red channel is no longer dominant (replaced by neutral slate censor stripes)
        self.assertLess(abs(center_rgb[0] - center_rgb[1]), 15.0)

        # 2. Verify lock_low_frequency_to_prior blocks SD2 from painting dark purple/red food over white container
        seed_rgb = np.full((160, 160, 3), 224, dtype=np.uint8)  # clean white styrofoam from Pix2Gestalt
        sd_hallucinated = seed_rgb.copy()
        sd_hallucinated[45:115, 45:115] = np.array([85, 65, 95], dtype=np.uint8)  # purple onions hallucination
        locked = lock_low_frequency_to_prior(seed_rgb, sd_hallucinated, amodal & (~vis))
        self.assertGreater(locked[55:105, 55:105].mean(), 212.0)

    def test_adaptive_seam_blend_and_critic_crop(self):
        img = np.full((120, 120, 3), 80, dtype=np.uint8)
        synth = np.full((120, 120, 3), 160, dtype=np.uint8)
        vis = np.zeros((120, 120), dtype=bool)
        vis[40:75, 40:75] = True
        amodal = vis.copy()
        amodal[40:90, 40:75] = True

        blended = adaptive_seam_blend(img, synth, vis, amodal)
        exact_frac = float(np.all(blended[vis] == img[vis], axis=1).mean())
        self.assertGreater(exact_frac, 0.90)
        self.assertEqual(int(blended[~amodal].sum()), 0)

        critic_crop = compose_cropped_for_critic(blended, amodal, target_size=256)
        self.assertEqual(critic_crop.shape, (256, 256, 3))
        np.testing.assert_array_equal(critic_crop[0, 0], np.array([255, 255, 255], dtype=np.uint8))

    def test_compare_runs_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / "smoke"
            (smoke / "s1").mkdir(parents=True)
            Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(smoke / "s1/gt.png")
            manifest = [{"sample_id": "s1", "name": "jar", "gt_amodal": "s1/gt.png"}]
            manifest_path = smoke / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            scamodal_root = root / "scamodal"
            for prof, mode, aiou, hiou in [
                ("1111111111111111", "upstream_baseline", 0.65, 0.40),
                ("2222222222222222", "improved", 0.88, 0.74),
            ]:
                jdir = scamodal_root / prof / "s1" / "abcdef123456"
                jdir.mkdir(parents=True)
                for fn in ("input.png", "visible.png", "amodal.png"):
                    Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(jdir / fn)
                Image.fromarray(np.full((32, 32, 4), 200, dtype=np.uint8)).save(jdir / "completion.png")
                res = {
                    "status": "completed",
                    "sample_id": "s1",
                    "final_score": 8.0,
                    "config": {"pipeline_mode": mode},
                    "trace": {"backend": mode},
                    "metrics": {
                        "amodal_iou": aiou,
                        "hidden_iou": hiou,
                        "visible_mae": 0.02,
                        "visible_exact_fraction": 0.98,
                    },
                }
                (jdir / "result.json").write_text(json.dumps(res), encoding="utf-8")

            grid_out = scamodal_root / "comparison_grid.jpg"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/compare_runs.py"),
                    "--manifest",
                    str(manifest_path),
                    "--scamodal-root",
                    str(scamodal_root),
                    "--output-grid",
                    str(grid_out),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("+0.2300", proc.stdout)
            self.assertTrue(grid_out.is_file())
            self.assertTrue(grid_out.with_suffix(".json").is_file())

    def test_improved_completer_best_of_n_and_trace_artifacts(self):
        import types
        from src.scamodal_improved import ImprovedAmodalCompleter

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )

        class FakeBase:
            device = "cpu"
            _pipe = types.SimpleNamespace(tokenizer=None)

            @staticmethod
            def _finalize_rgba(rgb, mask, h, w):
                out = np.zeros((h, w, 4), dtype=np.uint8)
                out[:, :, :3] = rgb
                out[:, :, 3] = mask.astype(np.uint8) * 255
                return out

        completer = ImprovedAmodalCompleter(
            FakeBase(),
            config={"seed": 42, "enable_critic": True, "critique_threshold": 7.0, "max_critic_iter": 2},
        )

        h, w = 80, 100
        img = np.full((h, w, 3), 60, dtype=np.uint8)
        vis = np.zeros((h, w), dtype=bool)
        vis[20:50, 25:55] = True
        amodal = vis.copy()
        amodal[20:68, 25:55] = True
        p2g_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        p2g_rgb[amodal] = 62
        p2g_alpha = np.zeros((h, w), dtype=np.float32)
        p2g_alpha[amodal] = 0.96

        def fake_shape(image_in, vis_in, job_dir):
            if job_dir is not None:
                Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(job_dir / "pix2gestalt_rgb_256.png")
                Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(job_dir / "carvekit_alpha_256.png")
                Image.fromarray(amodal.astype(np.uint8) * 255).save(job_dir / "shape_mask_original.png")
            return {
                "amodal_mask": amodal,
                "best_idx": 0,
                "p2g_rgb_full": p2g_rgb,
                "p2g_alpha_full": p2g_alpha,
                "diagnostics": {"kept_indices": [0]},
            }

        completer._predict_shape_and_prior = fake_shape
        completer._reason_target_aware = lambda ctx, iso, user_query="": {
            "object_name": "black suitcase",
            "occluded_parts": "bottom wheels",
            "sd_prompt": "black upright suitcase on white background",
            "raw": "OBJECT: black suitcase\nPROMPT: black upright suitcase on white background",
        }
        completer._verify_prompt_clip = lambda iso, parsed, user_query="": (
            parsed["sd_prompt"],
            {"verified": True, "chosen": parsed["sd_prompt"]},
        )

        # Iter 0 returns clean dark pixel 64 (Critic scores 6.5); Iter 1 returns degraded pixel 190 (Critic scores 4.0).
        call_idx = {"n": 0}
        def fake_refine(**kwargs):
            val = 64 if call_idx["n"] == 0 else 190
            call_idx["n"] += 1
            out = kwargs["seed_canvas"].copy()
            out[amodal & (~vis)] = val
            return out

        scores = [
            {"score": 6.5, "structural": 7.0, "texture": 6.5, "context": 6.0, "feedback": "slight seam", "raw": ""},
            {"score": 4.0, "structural": 4.0, "texture": 4.0, "context": 4.0, "feedback": "worse color", "raw": ""},
        ]
        completer._refine_roi_sd2 = fake_refine
        completer._critique_target_aware = lambda **kwargs: scores.pop(0)

        old_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            with tempfile.TemporaryDirectory() as tmp:
                job = Path(tmp)
                out = completer.complete(img, vis, text_query="", job_dir=job)
                # Best-of-N keeps Iter 0 (score=6.5, pixel=64) instead of Iter 1 (score=4.0, pixel=190)!
                self.assertEqual(out["final_score"], 6.5)
                self.assertEqual(int(out["inpainted_rgba"][60, 40, 0]), 64)
                for expected_file in (
                    "outpainting.json",
                    "pix2gestalt_rgb_256.png",
                    "carvekit_alpha_256.png",
                    "shape_mask_original.png",
                    "prior_rgb.png",
                    "vlm_visual_prompt.jpg",
                    "reasoning.txt",
                    "prompt_selection.json",
                    "prompt_01.txt",
                    "prompt_02.txt",
                    "inpainting_01.png",
                    "inpainting_02.png",
                    "critic_input_01.png",
                    "critic_input_02.png",
                    "critique_history.json",
                ):
                    self.assertTrue((job / expected_file).is_file(), expected_file)
        finally:
            if old_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch


if __name__ == "__main__":
    unittest.main()
