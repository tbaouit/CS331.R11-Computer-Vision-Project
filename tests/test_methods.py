"""Exercise the SD2 Baseline vs. Open-World Amodal Proposed contract and report without model downloads."""
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
from scripts.compare_versions import check_configs, compare_versions
from scripts.evaluate_runs import evaluate_profile
from src.evaluation.runner import run_one
from src.models.openworld_models import validate_openworld_proposed_config
from src.pipelines.backend import AmodalBackend
from src.pipelines.baseline import Baseline, validate_baseline_config
from src.pipelines.proposed import Proposed

ROOT = Path(__file__).resolve().parents[1]


def configs():
    return {
        role: json.loads((ROOT / "configs" / f"{role}.json").read_text(encoding="utf-8"))
        for role in ("baseline", "proposed")
    }


class VersionTests(unittest.TestCase):
    def test_config_dispatch_and_unified_sd2_inputs(self):
        cfgs = configs()
        check_configs(cfgs)
        # Confirm both baseline and proposed share SD2 Inpainting and dataset_object_name input policy
        self.assertEqual(cfgs["baseline"]["inpainting_model_id"], cfgs["proposed"]["inpainting_model_id"])
        self.assertEqual(cfgs["baseline"]["text_query_policy"], "dataset_object_name")
        self.assertEqual(cfgs["proposed"]["text_query_policy"], "dataset_object_name")

        cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        for role, cls in (("baseline", Baseline), ("proposed", Proposed)):
            if role == "baseline":
                validate_baseline_config(cfgs[role])
            else:
                validate_openworld_proposed_config(cfgs[role])

            def new_models(**kwargs):
                return SimpleNamespace(device=kwargs["device"])

            with patch.dict("sys.modules", {"torch": cuda}), patch(
                "src.pipelines.backend.FoundationModels", side_effect=new_models
            ):
                backend = AmodalBackend(cfgs[role])
                self.assertIs(type(backend.completer), cls)

        cfgs["proposed"]["inpainting_model_id"] = "invalid-model"
        with self.assertRaises(ValueError):
            check_configs(cfgs)

    def test_generated_notebook_has_seven_current_executable_cells(self):
        doc = json.loads((ROOT / "notebooks/Amodal_Completion_Colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(doc["cells"]), 7)
        self.assertEqual(["".join(c["source"]) for c in doc["cells"]], CELLS)
        for source in CELLS:
            compile(source, "<notebook>", "exec")

    def test_report_requires_verified_artifacts(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            rgb = np.full((32, 40, 3), 123, np.uint8)
            visible = np.zeros(rgb.shape[:2], bool)
            visible[3:25, 3:15] = True
            hidden = np.zeros_like(visible)
            hidden[3:25, 15:29] = True
            amodal = visible | hidden
            for name, mask in (("visible", visible), ("hidden", hidden), ("amodal", amodal)):
                Image.fromarray(mask.astype(np.uint8) * 255).save(root / f"{name}.png")
            Image.fromarray(rgb).save(root / "rgb.png")
            record = dict(
                sample_id="dev_test",
                split="dev",
                name="chair",
                hidden_ratio=0.5,
                image="rgb.png",
                visible_mask="visible.png",
                gt_amodal="amodal.png",
                gt_hidden="hidden.png",
            )
            profiles = {}
            for key, config in configs().items():
                profile = profiles[key] = root / key

                class Backend:
                    def begin_sample(self, job):
                        pass

                    def complete(self, *args, **kwargs):
                        return dict(
                            inpainted_rgba=np.dstack([rgb, amodal.astype(np.uint8) * 255]),
                            amodal_mask=amodal,
                            critique_history=[],
                            critic_accepted=None,
                        )

                run_one(record, root, profile, config, Backend(), key)
                evaluate_profile(profile, [record], "cpu", evaluate=lambda *a, **kw: {"CLIP_score": 22.0})
            output = root / "report"
            report = compare_versions([record], profiles, output)
            self.assertIsNone(report["critic"]["baseline"]["sample_pass_rate"])
            self.assertIsNone(report["critic"]["proposed"]["sample_pass_rate"])
            self.assertTrue((output / "summary.csv").exists())
            with Image.open(output / report["previews"][0]) as grid:
                self.assertEqual(grid.width, 5 * 236)
            next(profiles["proposed"].glob("*/*/completion.png")).write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                compare_versions([record], profiles, output)


if __name__ == "__main__":
    unittest.main()
