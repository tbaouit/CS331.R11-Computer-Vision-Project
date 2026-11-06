import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.evaluation import appearance_metrics as metrics
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
             patch.object(metrics, "lpips_distance", return_value=0) as lpips, \
             patch.object(metrics, "clip_feature_similarity", return_value=1) as feature, \
             patch.object(metrics, "ssim_score", return_value=1) as ssim:
            result = metrics.compute_visible_appearance_metrics(image, output, visible, "teddy bear")
        np.testing.assert_array_equal(clip.call_args.args[0], output)
        self.assertEqual(clip.call_args.args[1:], ("teddy bear", "cpu"))
        for call in (lpips.call_args, feature.call_args, ssim.call_args):
            a, b = call.args[:2]
            np.testing.assert_array_equal(a, b)
            self.assertFalse(a[~visible].any())
        self.assertEqual(result, dict(CLIP_score=24, LPIPS=0, Feature_Similarity=1, SSIM=1, PSNR=100))

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
        from src.evaluation.comparison import auto_restore_drive_backups
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
        from src.data.cocoa import Target
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


if __name__ == "__main__":
    unittest.main()
