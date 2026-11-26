import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.eval import appearance_metrics as metrics
from src.evaluation.reporting import aggregate
from src.evaluation.runner import run_one, sha
from src.evaluation.validation import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_runs_cli", ROOT / "scripts/evaluate_runs.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class AppearanceEvaluationTests(unittest.TestCase):
    def arrays(self):
        image = np.full((40, 48, 3), 80, np.uint8)
        visible = np.zeros(image.shape[:2], bool)
        visible[10:20, 12:22] = True
        output = np.full_like(image, 190)
        output[visible] = image[visible]
        return image, visible, output

    def test_author_protocol_masks_both_images_but_clip_sees_full_output(self):
        image, visible, output = self.arrays()
        with patch.object(metrics, "clip_score", return_value=24) as clip, \
             patch.object(metrics, "openclip_score", return_value=0.35) as oclip, \
             patch.object(metrics, "lpips_distance", return_value=0) as lpips, \
             patch.object(metrics, "vgg_feature_similarity", return_value=0.92) as vgg, \
             patch.object(metrics, "dino_similarity", return_value=0.81) as dino, \
             patch.object(metrics, "ssim_score", return_value=1) as ssim:
            result = metrics.compute_visible_appearance_metrics(image, output, visible, "teddy bear")
        self.assertEqual(clip.call_args.args[1:], ("teddy bear", "cpu"))
        self.assertEqual(oclip.call_args.args[1:], ("teddy bear", "cpu"))
        # DINO is called exactly ONCE and shared by Feature_Similarity and DINO_similarity
        dino.assert_called_once()
        # Full-object paper protocol: visible input on white (255) vs full completed amodal output on white (255)
        for call in (lpips.call_args, vgg.call_args, ssim.call_args):
            comp_arg, vis_arg = call.args[:2]
            self.assertEqual(comp_arg.shape, vis_arg.shape)
            self.assertGreaterEqual(min(comp_arg.shape[:2]), 32)
            self.assertTrue(np.any(vis_arg == 255))
        dino_vis, dino_full = dino.call_args.args[:2]
        self.assertEqual(dino_vis.shape, dino_full.shape)
        self.assertTrue(np.any(dino_vis == 255))
        self.assertEqual(
            result,
            dict(
                CLIP_score=24,
                LPIPS=0,
                Feature_Similarity=0.81,
                SSIM=1,
                OpenCLIP_score=0.35,
                PSNR=metrics.psnr_score(lpips.call_args.args[0], lpips.call_args.args[1]),
                VGG_similarity=0.92,
                DINO_similarity=0.81,
            ),
        )

    def test_psnr_uses_full_masked_canvas_like_author(self):
        a = np.zeros((40, 40, 3), np.uint8)
        b = a.copy()
        b[:20, :20] = 255
        self.assertAlmostEqual(metrics.psnr_score(a, b), 10 * np.log10(4))

    def test_empty_mask_or_different_canvas_rejected_before_models(self):
        image, visible, output = self.arrays()
        with self.assertRaisesRegex(ValueError, "empty"):
            metrics.compute_visible_appearance_metrics(image, output, visible & False, "object")
        with self.assertRaisesRegex(ValueError, "canvas"):
            metrics.compute_visible_appearance_metrics(image, output[:-1], visible, "object")

    def fixture(self, root):
        image, visible, _ = self.arrays()
        amodal = visible.copy()
        amodal[8:24, 10:26] = True
        hidden = amodal & ~visible
        for name, array in (("image", image), ("visible", visible.astype(np.uint8) * 255),
                            ("amodal", amodal.astype(np.uint8) * 255), ("hidden", hidden.astype(np.uint8) * 255)):
            Image.fromarray(array).save(root / f"{name}.png")
        record = dict(sample_id="dev_1", split="dev", image="image.png", visible_mask="visible.png",
                      gt_amodal="amodal.png", gt_hidden="hidden.png", name="object",
                      hidden_ratio=float(hidden.sum() / amodal.sum()))

        class Backend:
            calls = 0
            def begin_sample(self, job):
                pass
            def complete(self, image, visible, label):
                self.calls += 1
                return dict(inpainted_rgba=np.dstack([image, amodal.astype(np.uint8) * 255]),
                            amodal_mask=amodal, final_score=8)

        backend = Backend()
        profile = root / "runs/profile"
        run_one(record, root, profile, dict(critique_threshold=7), backend, "profile")
        return record, profile, backend

    def test_offline_eval_resume_reports_and_changed_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record, profile, backend = self.fixture(root)
            calls = []
            def fake(*args, **kwargs):
                calls.append(1)
                return dict(CLIP_score=25., LPIPS=0., Feature_Similarity=1., SSIM=1., PSNR=100.)
            destination, rows = evaluator.evaluate_profile(profile, [record], "cpu", evaluate=fake)
            evaluator.evaluate_profile(profile, [record], "cpu", evaluate=fake)
            self.assertEqual(len(calls), 1)
            self.assertEqual(backend.calls, 1)
            self.assertTrue((destination / "metrics.csv").is_file())
            report = json.loads((destination / "aggregate.json").read_text())
            self.assertEqual(report["overall"]["metrics"]["CLIP_score"]["mean"], 25)
            job = next(profile.glob("*/*/completion.png")).parent
            before = sha(job / "result.json")
            Image.fromarray(np.zeros((40, 48, 3), np.uint8)).save(job / "completion.png")
            _, failed = evaluator.evaluate_profile(profile, [record], "cpu", True, fake)
            self.assertEqual(failed[0]["status"], "failed")
            self.assertIn("checksum", failed[0]["error"])
            self.assertEqual(sha(job / "result.json"), before)

    def test_missing_failed_scores_not_counted_as_zero(self):
        result = aggregate([dict(status="completed", metrics={"LPIPS": .2}),
                            dict(status="failed"), dict(status="missing")])
        self.assertEqual(result["metrics"]["LPIPS"]["mean"], .2)
        self.assertEqual(result["metrics"]["LPIPS"]["n"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["missing"], 1)

    def test_different_subsets_keep_separate_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            record, profile, _ = self.fixture(Path(temporary))
            def fake(*args, **kwargs):
                return dict(CLIP_score=25., LPIPS=0., Feature_Similarity=1., SSIM=1., PSNR=100.)
            first, _ = evaluator.evaluate_profile(profile, [record], "cpu", evaluate=fake)
            before = (first / "aggregate.json").read_bytes()
            second, _ = evaluator.evaluate_profile(profile, [record, dict(record, sample_id="dev_2")], "cpu", evaluate=fake)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "aggregate.json").read_bytes(), before)
            report = json.loads((second / "aggregate.json").read_text())
            self.assertEqual(report["overall"]["missing"], 1)

    def test_preflight_catches_wrong_hidden_and_nonbinary_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record, _, _ = self.fixture(root)
            self.assertEqual(validate_manifest([record], root)["invalid"], 0)
            Image.fromarray(np.zeros((40, 48), np.uint8)).save(root / "hidden.png")
            report = validate_manifest([record], root)
            self.assertIn("Hidden mask differs", report["samples"][0]["error"])
            Image.fromarray(np.full((40, 48), 120, np.uint8)).save(root / "visible.png")
            report = validate_manifest([record], root)
            self.assertIn("not binary", report["samples"][0]["error"])

    def test_backup_merges_new_metric_cache_without_overwriting_local_results(self):
        import zipfile
        from src.evaluation.backups import auto_restore_drive_backups
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs, backup = root / "runs", root / "backup"
            profile_id = "a" * 16
            dest = runs / profile_id / "sample/signature"
            dest.mkdir(parents=True)
            (dest / "result.json").write_text("local")
            backup.mkdir()
            with zipfile.ZipFile(backup / f"amodal_{profile_id}_20261105T000000Z.zip", "w") as bundle:
                bundle.writestr("sample/signature/result.json", "old backup")
                bundle.writestr("sample/signature/appearance_id.json", "cached metrics")
            restored = auto_restore_drive_backups(runs, backup, profiles={profile_id})
            self.assertEqual(restored, [runs / profile_id])
            self.assertEqual((dest / "result.json").read_text(), "local")
            self.assertTrue((dest / "appearance_id.json").is_file())

    def test_seeded_export_is_repeatable_and_contact_sheet_is_bounded(self):
        from scripts.prepare_data import export_samples
        from src.data.cocoa_loader import Target
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.fromarray(np.full((40, 48, 3), 80, np.uint8)).save(root / "source.png")
            region = dict(name="object", isStuff=0, occlude_rate=.5,
                          segmentation=[8, 8, 32, 8, 32, 32, 8, 32],
                          visible_mask=[8, 8, 20, 8, 20, 32, 8, 32])
            targets = [Target("dev", index, index, 0, root / "source.png", 48, 40, region)
                       for index in range(30)]
            class Dataset:
                split = "dev"
                def targets(self):
                    return iter(targets)
            cfg = dict(limit=25, one_object_per_image=True, min_visible_pixels=1,
                       min_amodal_pixels=1, min_hidden_ratio=.1, max_hidden_ratio=.9,
                       sampling="seeded_shuffle", seed=42, preview_limit=2)
            outputs = []
            for name in ("a", "b"):
                folder = root / name
                folder.mkdir()
                rows = export_samples(Dataset(), cfg, folder)
                outputs.append([r["sample_id"] for r in rows])
                with Image.open(folder / "contact_sheet.jpg") as sheet:
                    self.assertEqual(sheet.height, 560)
                self.assertEqual(len(rows), 25)
            self.assertEqual(outputs[0], outputs[1])
            self.assertNotEqual(outputs[0], [t.sample_id for t in targets[:25]])

    def test_reeval_full_object_paper_protocol_unmasked_white_bg_and_subset_breakdown(self):
        from scripts.reeval_full_object_metrics import prepare_paper_image_pair, reevaluate_full_object_protocol

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record1, base_profile, _ = self.fixture(root)
            record2 = dict(record1, sample_id="dev_light", hidden_ratio=0.05)
            run_one(record2, root, base_profile, dict(critique_threshold=7), self.fixture(root)[2], "profile")

            prop_profile = root / "runs/proposed_profile"
            _, _, backend2 = self.fixture(root)
            run_one(record1, root, prop_profile, dict(critique_threshold=7), backend2, "prop")
            run_one(record2, root, prop_profile, dict(critique_threshold=7), backend2, "prop")

            job_dir = next(prop_profile.glob("dev_1/*/result.json")).parent
            vis_crop, comp_crop, vis_full, comp_full = prepare_paper_image_pair(job_dir)
            self.assertEqual(vis_crop.shape, comp_crop.shape)
            self.assertGreaterEqual(min(vis_crop.shape[:2]), 32)
            # Outside the amodal mask must be plain white (255), NOT black (0)
            self.assertTrue(np.all(vis_full[0, 0] == 255))
            self.assertTrue(np.all(comp_full[0, 0] == 255))

            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps([record1, record2]), encoding="utf-8")
            out_json = root / "comparisons/full_820_paper_protocol_metrics.json"

            def fake_eval_batch(pairs, device="cpu", batch_size=32):
                return [
                    {
                        "LPIPS": 0.12,
                        "Feature_Similarity": 0.89,
                        "VGG_similarity": 0.78,
                        "SSIM": 0.84,
                        "CLIP_score": 26.4,
                        "DINO_similarity": 0.75,
                    }
                    for _ in pairs
                ]

            with patch("scripts.reeval_full_object_metrics.evaluate_batch_pairs", side_effect=fake_eval_batch):
                report = reevaluate_full_object_protocol(
                    manifest_path=manifest_path,
                    baseline_dir=base_profile,
                    proposed_dir=prop_profile,
                    output_json=out_json,
                    device="cpu",
                    min_occlusion=0.15,
                )
            self.assertTrue(out_json.is_file())
            self.assertTrue(out_json.with_suffix(".csv").is_file())
            self.assertEqual(report["evaluated_samples"]["proposed"], 2)
            self.assertEqual(report["occluded_subset_count"]["proposed"], 1)
            self.assertAlmostEqual(report["overall"]["proposed"]["LPIPS"]["mean"], 0.12)
            self.assertAlmostEqual(report["occluded_subset_ge_15pct"]["proposed"]["SSIM"]["mean"], 0.84)


if __name__ == "__main__":
    unittest.main()
