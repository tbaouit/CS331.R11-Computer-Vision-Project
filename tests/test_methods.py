"""Exercise the new experiment contract and report without model downloads."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.build_notebook import CELLS
from scripts.compare_versions import compare_versions, check_configs
from scripts.evaluate_runs import evaluate_profile
from src.evaluation.runner import run_one
from src.models.openworld_models import validate_openworld_proposed_config
from src.pipelines.backend import AmodalBackend
from src.pipelines.baseline import Baseline
from src.pipelines.proposed import Proposed
from src.vlm.micro_attributes import parse_micro_attributes
from src.vlm.strict_critic import parse_strict_critic

ROOT = Path(__file__).resolve().parents[1]
def configs():
    return {role: json.loads((ROOT / "configs" / f"{role}.json").read_text())
            for role in ("baseline", "proposed")}


class VersionTests(unittest.TestCase):
    def test_config_dispatch_and_required_modules(self):
        cfgs = configs()
        check_configs(cfgs)
        cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        for role, cls in (("baseline", Baseline), ("proposed", Proposed)):
            if role != "baseline":
                validate_openworld_proposed_config(cfgs[role])
            def new_models(**kwargs):
                return SimpleNamespace(device=kwargs["device"])
            with patch.dict("sys.modules", {"torch": cuda}), patch("src.pipelines.backend.FoundationModels", side_effect=new_models):
                backend = AmodalBackend(cfgs[role])
                self.assertIs(type(backend.completer), cls)
        cfgs["proposed"]["inpainting_model_id"] = "invalid-model"
        with self.assertRaises(ValueError):
            check_configs(cfgs)

    def test_v3_schema_and_pass_gate(self):
        attrs = dict(object_class="chair", primary_color="red", secondary_colors=["gold"],
                     material_and_texture="brocade", sd_prompt="ornate upholstered chair")
        self.assertEqual(parse_micro_attributes(json.dumps(attrs)), attrs)
        attrs["secondary_colors"] = "gold"
        with self.assertRaises(ValueError):
            parse_micro_attributes(json.dumps(attrs))
        judgment = dict(color_match=9, structure_continuity=6, final_decision="PASS",
                        reason="broken edge", repair_instruction="continue the edge")
        self.assertEqual(parse_strict_critic(json.dumps(judgment))["final_decision"], "REJECT")
        judgment["structure_continuity"] = 8
        self.assertTrue(parse_strict_critic(json.dumps(judgment))["accepted"])

    def test_generated_notebook_has_seven_current_executable_cells(self):
        doc = json.loads((ROOT / "notebooks/Amodal_Completion_Colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(doc["cells"]), 7)
        self.assertEqual(["".join(c["source"]) for c in doc["cells"]], CELLS)
        for source in CELLS:
            compile(source, "<notebook>", "exec")

    def test_report_requires_verified_artifacts_and_counts_critic_rejects(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            rgb = np.full((32, 40, 3), 123, np.uint8)
            visible = np.zeros(rgb.shape[:2], bool)
            visible[3:25, 3:15] = True
            hidden = np.zeros_like(visible)
            hidden[3:25, 15:29] = True
            amodal = visible | hidden
            for name, mask in (("visible", visible), ("hidden", hidden), ("amodal", amodal)):
                Image.fromarray(mask.astype(np.uint8)*255).save(root / f"{name}.png")
            Image.fromarray(rgb).save(root / "rgb.png")
            record = dict(sample_id="dev_test", split="dev", name="chair", hidden_ratio=.5,
                          image="rgb.png", visible_mask="visible.png",
                          gt_amodal="amodal.png", gt_hidden="hidden.png")
            profiles = {}
            for key, config in configs().items():
                profile = profiles[key] = root / key
                history = [] if key == "baseline" else [
                    {"final_decision": "REJECT", "valid_json": False},
                    {"final_decision": "PASS", "valid_json": True}]
                class Backend:
                    def begin_sample(self, job):
                        pass
                    def complete(self, *args, **kwargs):
                        return dict(inpainted_rgba=np.dstack([rgb, amodal.astype(np.uint8)*255]),
                                    amodal_mask=amodal, critique_history=history,
                                    critic_accepted=bool(history))
                run_one(record, root, profile, config, Backend(), key)
                evaluate_profile(profile, [record], "cpu", evaluate=lambda *a, **kw: {"CLIP_score": 22.})
            output = root / "report"
            report = compare_versions([record], profiles, output)
            self.assertEqual(report["critic"]["baseline"]["sample_pass_rate"], None)
            self.assertEqual(report["critic"]["proposed"]["attempt_reject_rate"], .5)
            self.assertEqual(report["critic"]["proposed"]["sample_pass_rate"], 1.)
            self.assertEqual(report["critic"]["proposed"]["invalid_json"], 1)
            self.assertTrue((output / "summary.csv").exists())
            with Image.open(output / report["previews"][0]) as grid:
                self.assertEqual(grid.width, 5 * 236)
            next(profiles["proposed"].glob("*/*/completion.png")).write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                compare_versions([record], profiles, output)
