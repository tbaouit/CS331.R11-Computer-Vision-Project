"""Exercise generation/evaluation hand-off without a Colab or GPU dependency."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.build_notebook import CELLS
from scripts.run_experiment import mark_profile_state


class DriveSessionTests(unittest.TestCase):
    def test_reject_changed_profile_or_manifest_without_overwriting_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profiles" / "baseline.json"
            state = {"profile": "a" * 16, "requested": 400, "manifest_sha256": "b" * 64}
            mark_profile_state(path, state, "running")
            original = path.read_bytes()
            for change in ({"profile": "f" * 16}, {"manifest_sha256": "c" * 64}):
                with self.assertRaisesRegex(ValueError, "Resume profile or manifest changed"):
                    mark_profile_state(path, dict(state, **change), "running")
                self.assertEqual(path.read_bytes(), original)
            mark_profile_state(path, state, "completed")
            self.assertEqual(json.loads(path.read_bytes()), dict(state, status="completed"))

    def test_fresh_evaluation_session_uses_drive_and_requires_all_profiles(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            experiment = root / "Drive" / "sd15"
            batch = experiment / "batch_full"
            records = [{"sample_id": f"dev_{n:03}", "split": "dev"} for n in range(50)]
            runs = batch / "runs"
            runs.mkdir(parents=True)
            manifest = batch / "manifest.json"
            manifest.write_text(json.dumps(records), encoding="utf-8")
            digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            roles = dict(baseline="a" * 16, proposed="c" * 16)
            calls = []

            def sh(*args):
                calls.append(args)
                if "scripts/compare_versions.py" in args:
                    target = args[args.index("--output") + 1]
                    target.mkdir(parents=True)
                    (target / "comparison.json").write_text(json.dumps(dict(
                        metrics={role: {} for role in roles}, critic={role: {} for role in roles},
                        fid=None, previews=[])), encoding="utf-8")

            env = dict(EXPERIMENT=experiment, PROJECT=root / "fresh_runtime", PYTHON="python",
                       EVALUATION_ONLY=True, sh=sh, Path=Path, json=json, re=re, os=__import__("os"),
                       shutil=shutil)
            exec(CELLS[3], env)
            self.assertEqual(calls, [])  # No dataset prep, no generation, no local state required.
            for role, profile in roles.items():
                (runs / role / profile).mkdir(parents=True)
                mark_profile_state(batch / "profiles" / f"{role}.json",
                                   dict(profile=profile, requested=50, manifest_sha256=digest),
                                   "running" if role == "proposed" else "completed")
            with patch.dict(sys.modules, {"IPython.display": SimpleNamespace(display=lambda _: None)}):
                with self.assertRaisesRegex(RuntimeError, "Chưa có kết quả hoàn chỉnh cho: proposed"):
                    exec(CELLS[6], env)
                self.assertEqual(calls, [])
                mark_profile_state(batch / "profiles/proposed.json",
                                   dict(profile=roles["proposed"], requested=50, manifest_sha256=digest),
                                   "completed")
                exec(CELLS[6], env)
            self.assertEqual(len(calls), 3)
            for args, (role, profile) in zip(calls[:2], roles.items()):
                self.assertEqual(args[1], "scripts/evaluate_runs.py")
                self.assertEqual(args[args.index("--profile") + 1], profile)
                self.assertEqual(args[args.index("--runs-root") + 1], runs / role)
                self.assertEqual(args[args.index("--manifest") + 1], manifest)
            self.assertEqual(len(list((batch / "comparisons").glob("*.zip"))), 1)
