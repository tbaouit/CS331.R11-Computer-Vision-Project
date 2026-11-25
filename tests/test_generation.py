"""CPU regressions for Proposed schemas, retry decisions, data flow and batch logging."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.vlm.micro_attributes import parse_micro_attributes, run_micro_reasoning
from src.vlm.strict_critic import parse_strict_critic, run_strict_critic
from src.models.sd15_ip_adapter import validate_proposed_config, blend_latents, ddim_previous_timestep
from src.pipelines.proposed import Proposed
from src.pipelines.proposed import critic_panels
from src.evaluation.critic_logs import update_critic_log

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/proposed.json").read_text())


def judgment(score=8, decision="PASS"):
    return json.dumps(dict(color_match=score, structure_continuity=score,
                           final_decision=decision, reason="Texture assessment", repair_instruction="continuous wood grain"))


class GenerationTests(unittest.TestCase):
    def test_sd15_baseline_uses_only_inpainting_and_correct_metadata(self):
        from src.pipelines.baseline import Baseline
        from src.evaluation.runner import _uses_occluded_mask_input
        cfg = json.loads((ROOT / "configs/baseline.json").read_text())
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
        from src.stages.inpaint_refiner import refine_exact_mask
        from src.eval.metrics import compute_appearance_coherence_metrics
        canvas = np.zeros((40, 48, 3), np.uint8)
        visible = np.zeros(canvas.shape[:2], bool)
        visible[8:30, 10:20] = True
        hidden = np.zeros_like(visible)
        hidden[8:30, 20:30] = True
        decoded = np.full_like(canvas, 180)
        with patch("src.stages.inpaint_refiner.run_inpaint_roi_pass", return_value=decoded):
            actual = refine_exact_mask(None, "cpu", canvas, visible, hidden, "", "", .24, 6, 28, 42)
        np.testing.assert_array_equal(actual, decoded)
        stats = compute_appearance_coherence_metrics(visible, hidden, actual)
        self.assertEqual(stats, {"seam_mae": 0.0, "color_hist_sim": 1.0})

    def test_non_sd15_backbone_is_rejected(self):
        cfg = config()
        cfg["inpainting_model_id"] = "unsupported-model"
        with self.assertRaisesRegex(ValueError, "SD1.5"):
            validate_proposed_config(cfg)

    def test_micro_json_and_format_repair_do_not_invent_attributes(self):
        raw = json.dumps(dict(object_class="table", primary_color="dark brown", secondary_colors=["black"],
                              material_and_texture="glossy wood grain", sd_prompt="a wooden table"))
        self.assertEqual(parse_micro_attributes("```json\n" + raw + "\n```")["primary_color"], "dark brown")
        with patch("src.vlm.json_response.generate_json_response", side_effect=["bad", raw]) as generate:
            answer = run_micro_reasoning(None, None, None)
        self.assertTrue(answer["valid_json"])
        self.assertIn("glossy wood grain", answer["sd_prompt"])
        self.assertEqual(generate.call_count, 2)
        with patch("src.vlm.json_response.generate_json_response", return_value="invalid"):
            fallback = run_micro_reasoning(None, None, None)
        self.assertFalse(fallback["valid_json"])
        self.assertEqual(fallback["primary_color"], "unknown")
        self.assertEqual(len(fallback["responses"]), 2)

    def test_strict_critic_requires_both_scores_and_explicit_pass(self):
        self.assertTrue(parse_strict_critic(judgment())["accepted"])
        for raw in (judgment(5), judgment(9, "FAIL")):
            self.assertFalse(parse_strict_critic(raw)["accepted"])
        data = json.loads(judgment())
        data["structure_continuity"] = 3
        self.assertFalse(parse_strict_critic(json.dumps(data))["accepted"])
        for bad in (True, "8", float("nan"), float("inf"), -1, 11):
            data["structure_continuity"] = bad
            with self.assertRaises(ValueError):
                parse_strict_critic(json.dumps(data))
        with patch("src.vlm.json_response.generate_json_response", return_value="PASS"):
            failed = run_strict_critic(None, [], "table", 7)
        self.assertFalse(failed["accepted"])
        self.assertIsNone(failed["score"])
        self.assertFalse(failed["valid_json"])

    def make_completer(self, cfg=None):
        cfg = cfg or config()
        models = SimpleNamespace(device="cpu", pipe=SimpleNamespace(tokenizer=None))
        completer = Proposed(models, cfg)
        image = np.full((48, 64, 3), [90, 120, 150], np.uint8)
        vis = np.zeros((48, 64), bool)
        vis[10:30, 8:25] = True
        occ = np.zeros_like(vis)
        occ[10:30, 25:40] = True
        context = np.full((384, 384, 3), 200, np.uint8)
        completer._reason = lambda *a: (dict(object_name="table", raw="json", sd_prompt="brown wood",
                                             valid_json=True), context, context, context)
        completer._verify_prompt = lambda *a, **kw: ("table", {"verified": True})
        prior = dict(seed_canvas=image.copy(), unfilled_fraction=.5, ref_p90_luma=150)
        completer._prior = lambda *a: (vis | occ, {"diagnostics": {}}, prior)
        calls = []
        def refine(*args):
            calls.append(args)
            return np.full_like(image, 40 + len(calls) * 30), {"latent_steps": []}
        completer._refine = refine
        return completer, image, vis, occ, calls

    def test_retry_stops_on_pass_and_keeps_decoded_rgb_and_exact_mask(self):
        completer, image, vis, occ, calls = self.make_completer()
        responses = [parse_strict_critic(judgment(5)), parse_strict_critic(judgment(8)), parse_strict_critic(judgment(9))]
        completer._critique = lambda *a: responses.pop(0)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            result = completer.complete(image, vis, occluded_mask=occ, job_dir=Path(temp))
            logs = [json.loads(line) for line in (Path(temp) / "critic_logs.jsonl").read_text().splitlines()]
            self.assertEqual(len(logs), 2)
            self.assertEqual(logs[1]["retries"], 1)
        self.assertEqual([args[-1] for args in calls], [42, 1051])
        self.assertIn("continuous wood grain", calls[1][4])
        self.assertEqual(result["_trace"]["selected_attempt"], 2)
        self.assertTrue(result["critic_accepted"])
        self.assertTrue(np.all(result["inpainted_rgba"][..., :3] == 100))
        np.testing.assert_array_equal(result["inpainted_rgba"][:, :, 3] > 0, vis | occ)

    def test_exhausted_retries_returns_best_rejection_without_claiming_pass(self):
        completer, image, vis, occ, calls = self.make_completer()
        responses = [parse_strict_critic(judgment(9, "FAIL")), parse_strict_critic(judgment(4)),
                     dict(score=None, accepted=False, final_decision="FAIL", reason="Malformed", repair_instruction="")]
        completer._critique = lambda *a: responses.pop(0)
        with contextlib.redirect_stdout(io.StringIO()):
            result = completer.complete(image, vis, occluded_mask=occ)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["_trace"]["selected_attempt"], 1)
        self.assertEqual(result["final_score"], 9)
        self.assertFalse(result["critic_accepted"])  # high scores plus explicit FAIL remain FAIL
        self.assertEqual(result["_trace"]["outcome"], "exhausted_retries")

    def test_hidden_mask_is_required(self):
        completer, image, vis, _, _ = self.make_completer()
        with self.assertRaisesRegex(ValueError, "hidden mask"):
            completer.complete(image, vis)

    def test_latent_blend_keeps_editable_values_and_ddim_uses_post_step_level(self):
        generated = np.full((1, 4, 2, 2), 9.)
        reference = np.full_like(generated, 2.)
        mask = np.array([[[[0., 1.], [1., 0.]]]])
        expected = np.array([[[[2., 9.], [9., 2.]]]]).repeat(4, axis=1)
        np.testing.assert_array_equal(blend_latents(generated, reference, mask), expected)
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000), num_inference_steps=50)
        self.assertEqual(ddim_previous_timestep(scheduler, 180), 160)
        self.assertLess(ddim_previous_timestep(scheduler, 0), 0)

    def test_critic_images_share_roi_and_isolate_only_target_pixels(self):
        _, image, vis, occ, _ = self.make_completer()
        panels = critic_panels(image, vis, vis | occ, np.full_like(image, 100), size=64)
        self.assertTrue(all(p.shape == (64, 64, 3) for p in panels))
        self.assertTrue(np.any(np.all(panels[3] == [40, 130, 255], axis=-1)))
        self.assertTrue(np.all(panels[1][0, 0] == 255))

    def test_batch_critic_log_resume_does_not_duplicate_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            result = dict(config={"pipeline_mode": "proposed"}, sample_id="dev1", signature="sig",
                          critique_history=[dict(attempt=1, retries=0, final_decision="FAIL")],
                          trace=dict(selected_attempt=1, outcome="exhausted_retries"))
            update_critic_log(temp, result)
            update_critic_log(temp, result)
            rows = [json.loads(line) for line in (Path(temp) / "critic_logs.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["selected"])


if __name__ == "__main__":
    unittest.main()
