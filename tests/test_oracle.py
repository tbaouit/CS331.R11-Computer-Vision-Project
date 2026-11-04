import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np
from PIL import Image

from src.evaluation.metrics import compute_appearance_coherence_metrics
from src.evaluation.runner import run_one
from src.pipelines.oracle_completer import OracleMaskAmodalCompleter
from src.stages.appearance_prior import build_oracle_appearance_seed
from src.stages.vlm_guidance import build_oracle_vlm_prompts

ROOT = Path(__file__).resolve().parents[1]


class OracleMaskPipelineTests(unittest.TestCase):
    def test_experiment_passes_occluded_mask_only_in_oracle_mode(self):
        class SpyOracleBackend:
            def __init__(self):
                self.received_occluded = None

            def begin_sample(self, job):
                self.job = job

            def complete(self, image, visible, label, occluded_mask=None):
                self.received_occluded = occluded_mask
                amodal = visible | (occluded_mask if occluded_mask is not None else False)
                rgb = np.full_like(image, 210)
                rgb[visible] = image[visible]
                rgba = np.dstack([rgb, amodal.astype(np.uint8) * 255])
                return {
                    "inpainted_rgba": rgba,
                    "amodal_mask": amodal,
                    "critique_history": [{"score": 9.0}],
                    "final_score": 9.0,
                    "_trace": {"pad_info": {}, "backend": "SCAmodal Oracle-Mask"},
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = np.full((24, 30, 3), 180, dtype=np.uint8)
            vis = np.zeros((24, 30), dtype=bool)
            vis[6:14, 8:18] = True
            gt_amodal = vis.copy()
            gt_amodal[6:20, 8:22] = True
            gt_hidden = gt_amodal & (~vis)

            Image.fromarray(img).save(root / "image.png")
            Image.fromarray(vis.astype(np.uint8) * 255).save(root / "visible.png")
            Image.fromarray(gt_amodal.astype(np.uint8) * 255).save(root / "gt_amodal.png")
            Image.fromarray(gt_hidden.astype(np.uint8) * 255).save(root / "gt_hidden.png")

            record = {
                "sample_id": "train_oracle_1",
                "split": "train",
                "image": "image.png",
                "visible_mask": "visible.png",
                "gt_amodal": "gt_amodal.png",
                "gt_hidden": "gt_hidden.png",
                "name": "box",
            }
            spy = SpyOracleBackend()
            cfg = {"pipeline_mode": "oracle_mask", "critique_threshold": 7, "appearance_metrics": False}
            res, skipped = run_one(record, root, root / "out", cfg, spy, "prof_oracle")
            self.assertFalse(skipped)
            self.assertIsNotNone(spy.received_occluded)
            np.testing.assert_array_equal(spy.received_occluded, gt_hidden)
            self.assertEqual(res["metrics"]["amodal_iou"], 1.0)
            self.assertEqual(res["metrics"]["hidden_iou"], 1.0)
            job_dir = next((root / "out/train_oracle_1").iterdir())
            self.assertTrue((job_dir / "occluded_input.png").is_file())

    def test_oracle_seed_suppresses_container_shadow_and_fills_missed_foundation(self):
        # 1. White takeout container with dark sandwich shadow smudge in Pix2Gestalt RGB
        h, w = 128, 128
        img = np.full((h, w, 3), 225, dtype=np.uint8)
        amodal = np.zeros((h, w), dtype=bool)
        amodal[20:108, 20:108] = True
        vis = amodal.copy()
        vis[42:86, 42:86] = False
        occ = amodal & (~vis)
        img[vis] = np.array([224, 226, 230], dtype=np.uint8)
        img[occ] = np.array([75, 45, 30], dtype=np.uint8)  # dark brown sandwich in raw input

        p2g_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        p2g_rgb[amodal] = np.array([222, 224, 228], dtype=np.uint8)
        p2g_rgb[50:78, 50:78] = np.array([125, 120, 118], dtype=np.uint8)  # residual dark smudge
        p2g_alpha = np.zeros((h, w), dtype=np.float32)
        p2g_alpha[amodal] = 0.95

        seed_info = build_oracle_appearance_seed(img, vis, occ, p2g_rgb, p2g_alpha)
        self.assertGreater(seed_info["shadow_suppressed_fraction"], 0.25)
        center_luma = seed_info["seed_canvas"][55:73, 55:73].mean()
        self.assertGreater(center_luma, 205.0)  # Clean white styrofoam, shadow smudge suppressed!

        # 2. Dark brick building whose bottom foundation strip `occ` was missed by Pix2Gestalt (`alpha = 0`)
        img_bldg = np.full((h, w, 3), 180, dtype=np.uint8)
        vis_bldg = np.zeros((h, w), dtype=bool)
        vis_bldg[40:85, 35:80] = True
        occ_bldg = np.zeros((h, w), dtype=bool)
        occ_bldg[85:98, 35:80] = True  # foundation hidden behind sand dune
        img_bldg[vis_bldg] = np.array([68, 56, 52], dtype=np.uint8)

        p2g_bldg = np.full((h, w, 3), 255, dtype=np.uint8)
        p2g_bldg[vis_bldg] = np.array([70, 58, 54], dtype=np.uint8)
        p2g_alpha_bldg = np.zeros((h, w), dtype=np.float32)
        p2g_alpha_bldg[vis_bldg] = 0.96  # Pix2Gestalt completely missed `occ_bldg`!

        seed_bldg = build_oracle_appearance_seed(img_bldg, vis_bldg, occ_bldg, p2g_bldg, p2g_alpha_bldg)
        self.assertGreater(seed_bldg["unfilled_fraction"], 0.95)
        foundation_luma = seed_bldg["seed_canvas"][occ_bldg].mean()
        self.assertLess(foundation_luma, 85.0)  # Propagated dark brick color, NOT 255 white!

    def test_oracle_vlm_blueprint_prompts_and_coherence_metrics(self):
        img = np.full((160, 200, 3), 90, dtype=np.uint8)
        vis = np.zeros((160, 200), dtype=bool)
        vis[50:90, 60:95] = True
        occ = np.zeros((160, 200), dtype=bool)
        occ[62:74, 95:145] = True  # exact tennis racquet handle silhouette
        img[vis] = np.array([50, 120, 70], dtype=np.uint8)

        ctx, bp, iso, panel = build_oracle_vlm_prompts(img, vis, occ, target_size=192)
        self.assertEqual(ctx.shape, (192, 192, 3))
        self.assertEqual(bp.shape, (192, 192, 3))
        self.assertEqual(iso.shape, (192, 192, 3))
        self.assertEqual(panel.shape[0], 192)
        np.testing.assert_array_equal(iso[0, 0], np.array([255, 255, 255], dtype=np.uint8))

        good_comp = img.copy()
        good_comp[occ] = np.array([52, 118, 72], dtype=np.uint8)
        bad_comp = img.copy()
        bad_comp[occ] = np.array([220, 30, 30], dtype=np.uint8)

        m_good = compute_appearance_coherence_metrics(img, vis, occ, good_comp)
        m_bad = compute_appearance_coherence_metrics(img, vis, occ, bad_comp)
        self.assertLess(m_good["seam_mae"], 5.0)
        self.assertGreater(m_bad["seam_mae"], 80.0)
        self.assertGreater(m_good["color_hist_sim"], m_bad["color_hist_sim"])

    def test_oracle_completer_end_to_end_and_three_way_compare_cli(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)

        class FakeBase:
            device = "cpu"
            _pipe = types.SimpleNamespace(tokenizer=None)

            @staticmethod
            def _finalize_rgba(rgb, mask, h, w):
                out = np.zeros((h, w, 4), dtype=np.uint8)
                out[:, :, :3] = rgb
                out[:, :, 3] = mask.astype(np.uint8) * 255
                return out

        completer = OracleMaskAmodalCompleter(
            FakeBase(),
            config={"seed": 42, "enable_critic": True, "critique_threshold": 8.5, "max_critic_iter": 2},
        )

        h, w = 64, 80
        img = np.full((h, w, 3), 45, dtype=np.uint8)
        vis = np.zeros((h, w), dtype=bool)
        vis[15:40, 20:45] = True
        occ = np.zeros((h, w), dtype=bool)
        occ[40:56, 20:45] = True
        amodal = vis | occ

        def fake_prior(image_in, vis_in, amodal_in, job_dir):
            if job_dir is not None:
                Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(job_dir / "pix2gestalt_rgb_256.png")
                Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(job_dir / "carvekit_alpha_256.png")
                Image.fromarray(amodal_in.astype(np.uint8) * 255).save(job_dir / "shape_mask_original.png")
            p2g_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
            p2g_rgb[amodal_in] = 48
            p2g_alpha = np.zeros((h, w), dtype=np.float32)
            p2g_alpha[amodal_in] = 0.95
            return {"p2g_rgb_full": p2g_rgb, "p2g_alpha_full": p2g_alpha, "diagnostics": {"mask_source": "input_occluded_mask"}}

        completer._synthesize_amodal_prior = fake_prior
        completer._reason_oracle_aware = lambda ctx, bp, user_query="": {
            "object_name": "black luggage",
            "occluded_parts": "bottom body",
            "sd_prompt": "black upright suitcase on white background",
            "raw": "OBJECT: black luggage\nOCCLUDED_PARTS: bottom body\nPROMPT: black upright suitcase on white background",
        }
        completer._verify_prompt_clip = lambda iso, parsed, user_query="": (
            parsed["sd_prompt"],
            {"verified": True, "chosen": parsed["sd_prompt"]},
        )

        idx = {"n": 0}
        def fake_refine(**kwargs):
            val = 46 if idx["n"] == 0 else 175
            idx["n"] += 1
            out = kwargs["seed_canvas"].copy()
            out[occ] = val
            return out

        scores = [
            {"score": 8.0, "structural": 8.0, "texture": 8.0, "context": 8.0, "feedback": "minor seam", "raw": ""},
            {"score": 5.0, "structural": 5.0, "texture": 5.0, "context": 5.0, "feedback": "too bright", "raw": ""},
        ]
        completer._refine_exact_mask_sd2 = fake_refine
        completer._critique_target_aware = lambda **kwargs: scores.pop(0)

        old_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            with tempfile.TemporaryDirectory() as tmp:
                job = Path(tmp) / "job"
                job.mkdir()
                out = completer.complete(img, vis, occ, text_query="", job_dir=job)
                np.testing.assert_array_equal(out["amodal_mask"], amodal)
                self.assertEqual(out["final_score"], 8.0)
                self.assertEqual(int(out["inpainted_rgba"][48, 30, 0]), 46)

                # Also verify 3-way compare_runs.py CLI (Baseline -> Improved -> Oracle-Mask)
                root = Path(tmp) / "cmp"
                (root / "smoke/s1").mkdir(parents=True)
                Image.fromarray(amodal.astype(np.uint8) * 255).save(root / "smoke/s1/gt_amodal.png")
                Image.fromarray(occ.astype(np.uint8) * 255).save(root / "smoke/s1/gt_hidden.png")
                manifest_path = root / "smoke/manifest.json"
                manifest_path.write_text(
                    json.dumps([{"sample_id": "s1", "name": "luggage", "gt_amodal": "s1/gt_amodal.png", "gt_hidden": "s1/gt_hidden.png"}]),
                    encoding="utf-8",
                )
                scamodal_root = root / "scamodal"
                for prof, mode, aiou, hiou, fill_val in [
                    ("1111111111111111", "upstream_baseline", 0.68, 0.27, 150),
                    ("2222222222222222", "improved", 0.77, 0.40, 58),
                    ("3333333333333333", "oracle_mask", 1.00, 1.00, 46),
                ]:
                    jdir = scamodal_root / prof / "s1" / "abcdef123456"
                    jdir.mkdir(parents=True)
                    Image.fromarray(img).save(jdir / "input.png")
                    Image.fromarray(vis.astype(np.uint8) * 255).save(jdir / "visible.png")
                    Image.fromarray(amodal.astype(np.uint8) * 255).save(jdir / "amodal.png")
                    comp = np.zeros((h, w, 4), dtype=np.uint8)
                    comp[:, :, :3] = img
                    comp[occ, :3] = fill_val
                    comp[:, :, 3] = amodal.astype(np.uint8) * 255
                    Image.fromarray(comp).save(jdir / "completion.png")
                    (jdir / "result.json").write_text(
                        json.dumps(
                            {
                                "status": "completed",
                                "sample_id": "s1",
                                "final_score": 8.5,
                                "config": {"pipeline_mode": mode},
                                "trace": {"backend": mode},
                                "metrics": {
                                    "amodal_iou": aiou,
                                    "hidden_iou": hiou,
                                    "visible_mae": 0.0,
                                    "visible_exact_fraction": 1.0,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )

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
                self.assertIn("3-Way: Baseline -> Improved -> Oracle-Mask", proc.stdout)
                self.assertIn("0.6800->0.7700->1.0000", proc.stdout)
                self.assertTrue(grid_out.is_file())
                report = json.loads(grid_out.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertEqual(report["mode"], "three_way")
                self.assertLess(report["mean_oracle_mask"]["seam_mae"], report["mean_baseline"]["seam_mae"])
        finally:
            if old_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch


if __name__ == "__main__":
    unittest.main()

