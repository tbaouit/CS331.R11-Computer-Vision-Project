import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.compare_baseline import compare_profiles
from scripts.evaluate_runs import evaluate_profile
from src.evaluation.runner import run_one, sha
from src.pipelines.backend import AmodalBackend
from src.pipelines.sd2_direct_completer import DEFAULT_PROMPT, SD2DirectCompleter, prepare_direct_inputs
from src.stages.sd2_refiner import _run_sd2_roi_pass, refine_exact_mask_sd2_oracle

ROOT = Path(__file__).resolve().parents[1]


class OnlySD2Models:
    device = "cuda"

    def __init__(self, **kwargs):
        self.pipe = object()
        self.releases = 0

    def release_gpu(self):
        self.releases += 1

    def __getattr__(self, name):
        raise AssertionError(f"Baseline must not access another model: {name}")


class SD2DirectTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/sd2_direct.json").read_text())
        self.image = np.full((100, 160, 3), 230, np.uint8)
        self.visible = np.zeros((100, 160), bool)
        self.hidden = self.visible.copy()
        # Object touches the image edge: crop must pad, not stretch or wrap around.
        self.visible[20:80, :30] = True
        self.hidden[20:80, 30:60] = True
        self.image[self.visible] = (50, 90, 120)
        self.image[self.hidden] = (250, 1, 7)  # Occluder; must never enter SD2.

    def test_removes_occluder_and_matches_main_sd2_crop_and_mask(self):
        canvas, vis, occ, amodal, roi, region = prepare_direct_inputs(self.image, self.visible, self.hidden)
        np.testing.assert_array_equal(canvas[vis], self.image[vis])
        self.assertTrue(np.all(canvas[occ] == 127))
        self.assertTrue(np.all(canvas[~amodal] == 255))
        self.assertLess(roi["x0"], 0)
        with patch("src.stages.sd2_refiner._run_sd2_roi_pass", return_value=canvas) as call:
            refine_exact_mask_sd2_oracle(object(), "cuda", canvas, vis, occ, occ,
                                        "prompt", "negative", 1., 7., 28, 42, 255.)
        self.assertEqual(call.call_args.kwargs["roi"], roi)
        np.testing.assert_array_equal(call.call_args.kwargs["inpaint_region"], region)

    def test_single_pass_uses_only_sd2_and_composites_exact_mask(self):
        generated = np.full_like(self.image, (18, 99, 170))
        with tempfile.TemporaryDirectory() as temp, \
             patch("src.pipelines.sd2_direct_completer._run_sd2_roi_pass", return_value=generated) as call:
            result = SD2DirectCompleter(OnlySD2Models(), self.config).complete(
                self.image, self.visible, self.hidden, Path(temp))
            settings = json.loads((Path(temp) / "sd2_settings.json").read_text())
            self.assertFalse(settings["category_annotation_used"])
            self.assertTrue((Path(temp) / "baseline_input.png").is_file())
        call.assert_called_once()
        self.assertEqual(call.call_args.kwargs["prompt"], DEFAULT_PROMPT)
        self.assertEqual(call.call_args.kwargs["strength"], 1.)
        self.assertEqual(call.call_args.kwargs["num_inference_steps"], 28)
        out = result["inpainted_rgba"]
        np.testing.assert_array_equal(out[:, :, 3] > 0, self.visible | self.hidden)
        np.testing.assert_array_equal(out[:, :, :3][self.hidden], generated[self.hidden])
        self.assertLessEqual(np.abs(out[:, :, :3][self.visible].astype(int) - self.image[self.visible]).max(), 1)
        self.assertFalse(out[:, :, :3][~(self.visible | self.hidden)].any())
        self.assertIsNone(result["final_score"])
        self.assertEqual(result["critique_history"], [])

    def test_dispatch_ignores_annotation_label_and_releases_gpu(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, {"torch": fake_torch}), \
             patch("src.models.registry.FoundationModels", OnlySD2Models), \
             patch("src.pipelines.sd2_direct_completer._run_sd2_roi_pass", return_value=self.image) as call:
            backend = AmodalBackend(self.config)
            backend.complete(self.image, self.visible, "secret category", self.hidden)
            self.assertNotIn("secret category", call.call_args.kwargs["prompt"])
            self.assertEqual(backend.models.releases, 1)
            with self.assertRaisesRegex(ValueError, "hidden mask"):
                backend.complete(self.image, self.visible, "secret category")
            self.assertEqual(backend.models.releases, 2)

    def test_empty_hidden_skips_generation_and_invalid_inputs_fail(self):
        with patch("src.pipelines.sd2_direct_completer._run_sd2_roi_pass") as call:
            output = SD2DirectCompleter(OnlySD2Models(), self.config).complete(
                self.image, self.visible, np.zeros_like(self.hidden))
            call.assert_not_called()
        np.testing.assert_array_equal(output["inpainted_rgba"][:, :, :3][self.visible], self.image[self.visible])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            prepare_direct_inputs(self.image, self.visible, self.visible)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            prepare_direct_inputs(self.image, self.visible, self.hidden[:-1])
        with self.assertRaisesRegex(ValueError, "no critic"):
            SD2DirectCompleter(OnlySD2Models(), dict(self.config, enable_critic=True))

    def test_roi_pass_uses_white_padding_and_binary_nearest_mask(self):
        canvas, _, _, _, roi, mask = prepare_direct_inputs(self.image, self.visible, self.hidden)
        captured = {}

        class Generator:
            def __init__(self, device):
                captured["device"] = device
            def manual_seed(self, seed):
                captured["seed"] = seed
                return self

        def pipe(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(images=[Image.new("RGB", (512, 512), (22, 33, 44))])

        fake_torch = types.SimpleNamespace(Generator=Generator, inference_mode=contextlib.nullcontext)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            out = _run_sd2_roi_pass(pipe, "cuda", canvas, mask, roi, "prompt", "negative", 1., 7., 28, 42)
        self.assertEqual(captured["image"].size, (512, 512))
        self.assertEqual(captured["image"].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(set(np.unique(captured["mask_image"])), {0, 255})
        self.assertEqual(captured["seed"], 42)
        self.assertEqual(out.shape, self.image.shape)
        self.assertTrue(np.all(out[self.hidden] == (22, 33, 44)))
        self.assertTrue(np.all(out[:, 150:] == canvas[:, 150:]))

    def prepare_records(self, root):
        for name, array in (("image", self.image), ("visible", self.visible.astype(np.uint8) * 255),
                            ("hidden", self.hidden.astype(np.uint8) * 255),
                            ("amodal", (self.hidden | self.visible).astype(np.uint8) * 255)):
            Image.fromarray(array).save(root / f"{name}.png")
        return [dict(sample_id=f"dev_{i}", split="dev", image="image.png", visible_mask="visible.png",
                     gt_amodal="amodal.png", gt_hidden="hidden.png", name="object", hidden_ratio=.5)
                for i in (1, 2)]

    def make_pair(self, root):
        records = self.prepare_records(root)
        profiles = [root / "runs" / name for name in ("direct", "main")]
        for profile, mode, clip in zip(profiles, ("sd2_direct", "oracle_mask"), (24., 25.)):
            config = dict(self.config, pipeline_mode=mode, enable_critic=mode == "oracle_mask")
            completer = SD2DirectCompleter(OnlySD2Models(), self.config)

            class Backend:
                def begin_sample(self, job):
                    self.job = job
                def complete(self, image, visible, label, occluded_mask=None):
                    return completer.complete(image, visible, occluded_mask, self.job)

            with patch("src.pipelines.sd2_direct_completer._run_sd2_roi_pass", return_value=self.image):
                backend = Backend()
                for record in records:
                    _, skipped = run_one(record, root, profile, config, backend, profile.name)
                    self.assertFalse(skipped)
                    _, skipped = run_one(record, root, profile, config, backend, profile.name)
                    self.assertTrue(skipped)
            evaluate_profile(profile, records, "cpu", evaluate=lambda *a, **kw: dict(
                CLIP_score=clip, LPIPS=0., Feature_Similarity=1., SSIM=1., PSNR=100.))
        return records, profiles

    def test_paired_reports_and_cli_preview_without_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records, profiles = self.make_pair(root)
            summary = compare_profiles(records, *profiles, root / "comparison", preview_count=1)
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["metrics"]["CLIP_score_delta"]["mean"], 1.)
            self.assertEqual(summary["metrics"]["CLIP_score_delta"]["n"], 2)
            with Image.open(root / "comparison" / summary["previews"][0]) as image:
                self.assertEqual(image.size, (1200, 280))
            self.assertTrue((root / "comparison/paired_metrics.csv").is_file())
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(records))
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/compare_baseline.py"),
                "--manifest", str(manifest), "--baseline", "direct", "--main-profile", "main",
                "--runs-root", str(root / "runs"), "--output-dir", str(root / "cli"), "--preview-count", "0"],
                capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_comparison_rejects_protocol_mismatch_and_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records, profiles = self.make_pair(root)
            protocol = next(profiles[1].glob("evaluation/*/*/protocol.json"))
            original = protocol.read_text()
            protocol.write_text("{}")
            with self.assertRaisesRegex(ValueError, "protocols/versions"):
                compare_profiles(records, *profiles, root / "comparison", 0)
            protocol.write_text(original)
            report = protocol.parent / "per_sample.json"
            rows = json.loads(report.read_text())
            rows[1]["status"] = "missing"
            report.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                compare_profiles(records, *profiles, root / "comparison", 0)
            self.assertEqual(json.loads((root / "comparison/coverage.json").read_text())["main"]["missing"], 1)

    def test_comparison_rejects_artifact_tampering_and_wrong_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records, profiles = self.make_pair(root)
            with self.assertRaisesRegex(ValueError, "exact manifest"):
                compare_profiles(records[:1], *profiles, root / "comparison", 0)
            saved = next(profiles[0].glob("*/*/occluded_input.png"))
            Image.fromarray(np.zeros_like(self.hidden, dtype=np.uint8)).save(saved)
            with self.assertRaisesRegex(ValueError, "checksum"):
                compare_profiles(records, *profiles, root / "comparison", 0)

    def test_baseline_preflight_cli_requires_no_critic_or_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = self.prepare_records(root)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(records))
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/run_experiment.py"),
                "--manifest", str(manifest), "--config", str(ROOT / "configs/sd2_direct.json"),
                "--limit", "2", "--validate-only"], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("All selected samples valid", proc.stdout)

    def test_pair_rejects_changed_input_even_with_valid_own_checksums(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records, profiles = self.make_pair(root)
            saved = next(profiles[1].glob("*/*/input.png"))
            changed = self.image.copy()
            changed[0, 0] = 0
            Image.fromarray(changed).save(saved)
            result_path = saved.parent / "result.json"
            result = json.loads(result_path.read_text())
            result["artifact_sha256"]["input.png"] = sha(saved)
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "Stale evaluation"):
                compare_profiles(records, *profiles, root / "comparison", 0)
            evaluate_profile(profiles[1], records, "cpu", evaluate=lambda *a, **kw: dict(
                CLIP_score=25., LPIPS=0., Feature_Similarity=1., SSIM=1., PSNR=100.))
            with self.assertRaisesRegex(ValueError, "Input RGB or masks differ"):
                compare_profiles(records, *profiles, root / "comparison", 0)


if __name__ == "__main__":
    unittest.main()
