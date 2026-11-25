"""CPU regressions for Baseline (SD1.5 Direct) and Proposed (Open-World Amodal SD2) pipelines."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.evaluation.critic_logs import update_critic_log
from src.models.openworld_models import validate_openworld_proposed_config
from src.models.sd15_ip_adapter import blend_latents, ddim_previous_timestep
from src.pipelines.proposed import Proposed, critic_panels
from src.vlm.micro_attributes import parse_micro_attributes, run_micro_reasoning
from src.vlm.strict_critic import parse_strict_critic, run_strict_critic

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/proposed.json").read_text(encoding="utf-8"))


def judgment(score=8, decision="PASS"):
    return json.dumps(
        dict(
            color_match=score,
            structure_continuity=score,
            final_decision=decision,
            reason="Texture assessment",
            repair_instruction="continuous wood grain",
        )
    )


class GenerationTests(unittest.TestCase):
    def test_sd15_baseline_uses_only_inpainting_and_correct_metadata(self):
        from src.evaluation.runner import _uses_occluded_mask_input
        from src.pipelines.baseline import Baseline

        cfg = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
        self.assertTrue(_uses_occluded_mask_input(cfg))
        image = np.full((48, 64, 3), 140, np.uint8)
        vis = np.zeros((48, 64), bool)
        vis[10:30, 10:20] = True
        hidden = np.zeros_like(vis)
        hidden[10:30, 20:30] = True
        models = SimpleNamespace(device="cpu", pipe=object())
        with patch("src.pipelines.baseline.run_inpaint_roi_pass", return_value=np.full_like(image, 77)) as call:
            result = Baseline(models, cfg).complete(image, vis, hidden)
        call.assert_called_once()
        self.assertTrue(np.all(result["inpainted_rgba"][..., :3] == 77))
        self.assertIn("SD1.5 Direct", result["_trace"]["backend"])
        self.assertEqual(result["critique_history"], [])
        self.assertEqual(result["_trace"]["refinement_scheduler"], "DDIMScheduler")

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

    def test_non_sd2_backbone_is_rejected_for_openworld_proposed(self):
        cfg = config()
        cfg["inpainting_model_id"] = "unsupported-model"
        with self.assertRaisesRegex(ValueError, "SD2 Inpainting"):
            validate_openworld_proposed_config(cfg)

    def test_openworld_proposed_uses_visible_mask_directly_and_stops_at_max_iter_3(self):
        """Verify 4 Open-World Amodal modules run without LISA/LaMa, SAM is reused, and loop stops at max_iter=3."""
        import types
        import sys

        cfg = config()
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
                # Report the right-hand non-target region as occluder so it overlaps first_sd_occ_mask across iterations
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
                # Loop must execute exactly max_iter=3 iterations (not 4 as in the original `>` bug)
                self.assertEqual(len(sd2_calls), 3)
                self.assertEqual(result["_trace"]["synthesis_attempts"], 3)
                self.assertEqual(result["_trace"]["selected_class"], "wooden table")
                # Same pre-initialized FakeSAM instance was reused across initial scene + 3 iterations (4 calls total)
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

    def test_micro_json_and_format_repair_do_not_invent_attributes(self):
        raw = json.dumps(
            dict(
                object_class="table",
                primary_color="dark brown",
                secondary_colors=["black"],
                material_and_texture="glossy wood grain",
                sd_prompt="a wooden table",
            )
        )
        self.assertEqual(parse_micro_attributes("```json\n" + raw + "\n```")["primary_color"], "dark brown")
        with patch("src.vlm.json_response.generate_json_response", side_effect=["bad", raw]) as generate:
            answer = run_micro_reasoning(None, None, None)
        self.assertTrue(answer["valid_json"])
        self.assertIn("glossy wood grain", answer["sd_prompt"])
        self.assertEqual(generate.call_count, 2)

    def test_strict_critic_requires_both_scores_and_explicit_pass(self):
        self.assertTrue(parse_strict_critic(judgment())["accepted"])
        for raw in (judgment(5), judgment(9, "FAIL")):
            self.assertFalse(parse_strict_critic(raw)["accepted"])

    def test_latent_blend_keeps_editable_values_and_ddim_uses_post_step_level(self):
        generated = np.full((1, 4, 2, 2), 9.0)
        reference = np.full_like(generated, 2.0)
        mask = np.array([[[[0.0, 1.0], [1.0, 0.0]]]])
        expected = np.array([[[[2.0, 9.0], [9.0, 2.0]]]]).repeat(4, axis=1)
        np.testing.assert_array_equal(blend_latents(generated, reference, mask), expected)
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000), num_inference_steps=50)
        self.assertEqual(ddim_previous_timestep(scheduler, 180), 160)
        self.assertLess(ddim_previous_timestep(scheduler, 0), 0)

    def test_critic_images_share_roi_and_isolate_only_target_pixels(self):
        image = np.full((48, 64, 3), [90, 120, 150], np.uint8)
        vis = np.zeros((48, 64), bool)
        vis[10:30, 8:25] = True
        occ = np.zeros_like(vis)
        occ[10:30, 25:40] = True
        panels = critic_panels(image, vis, vis | occ, np.full_like(image, 100), size=64)
        self.assertTrue(all(p.shape == (64, 64, 3) for p in panels))
        self.assertTrue(np.any(np.all(panels[3] == [40, 130, 255], axis=-1)))
        self.assertTrue(np.all(panels[1][0, 0] == 255))

    def test_batch_critic_log_resume_does_not_duplicate_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            result = dict(
                config={"pipeline_mode": "proposed"},
                sample_id="dev1",
                signature="sig",
                critique_history=[dict(attempt=1, retries=0, final_decision="FAIL")],
                trace=dict(selected_attempt=1, outcome="exhausted_retries"),
            )
            update_critic_log(temp, result)
            update_critic_log(temp, result)
            rows = [json.loads(line) for line in (Path(temp) / "critic_logs.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["selected"])


if __name__ == "__main__":
    unittest.main()
