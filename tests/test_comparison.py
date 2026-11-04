"""Comparison must not pick up a user's mounted Drive during local tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from src.evaluation import comparison


class ComparisonIsolationTests(unittest.TestCase):
    def test_no_backup_directory_does_not_probe_drive(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Path, "is_dir", side_effect=AssertionError("Unexpected directory probe")):
                self.assertEqual(comparison.auto_restore_drive_backups(Path(tmp)), [])

    def test_local_discovery_never_imports_external_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            profile = runs / ("1" * 16)
            job = profile / "s1" / "abcdef123456"
            job.mkdir(parents=True)
            (job / "result.json").write_text(json.dumps({
                "status": "completed", "sample_id": "s1",
                "config": {"pipeline_mode": "improved"},
            }), encoding="utf-8")
            with patch.object(comparison, "auto_restore_drive_backups",
                              side_effect=AssertionError("Mounted Drive must not be read implicitly")):
                profiles = comparison.discover_profiles_by_mode(runs)
            self.assertEqual(profiles["improved"], [profile])
            self.assertEqual(profiles["baseline"], [])
            self.assertEqual(profiles["oracle_mask"], [])

    def test_explicit_backup_directory_still_restores_latest_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup = root / "drive-results"
            backup.mkdir()
            profile = "a" * 16
            for stamp, sid in [("20261103T100000Z", "old"), ("20261104T100000Z", "new")]:
                with zipfile.ZipFile(backup / f"amodal_{profile}_{stamp}.zip", "w") as bundle:
                    bundle.writestr(f"{sid}/abcdef123456/result.json", json.dumps({
                        "status": "completed", "sample_id": sid,
                        "config": {"pipeline_mode": "oracle_mask"},
                    }))
            runs = root / "runs"
            profiles = comparison.discover_profiles_by_mode(runs, backup)
            self.assertEqual(profiles["oracle_mask"], [runs / profile])
            _, _, results = comparison.load_profile_results(runs / profile)
            self.assertEqual(set(results), {"new"})
            self.assertEqual(comparison.auto_restore_drive_backups(runs, backup), [])


if __name__ == "__main__":
    unittest.main()
