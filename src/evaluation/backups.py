# Phạm vi: Cả hai phiên bản: khôi phục artefact từ ZIP của từng profile để có thể tiếp tục sau khi
# Colab ngắt.
"""Restore missing experiment artifacts from explicit Drive backups."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Optional
import zipfile

# Backup archives are named `<prefix>_<16-hex profile>_<UTC stamp>.zip` (see scripts/run_experiment.py).
BACKUP_NAME_RE = re.compile(r"^[A-Za-z]+_(?P<profile>[0-9a-f]{16})_(?P<stamp>\d{8}T\d{6}Z)\.zip$")


# Chọn ZIP mới nhất của mỗi profile được chỉ định, chỉ chép artefact còn thiếu.
def auto_restore_drive_backups(runs_root: Path, extra_backup_dir: Optional[Path] = None,
                             profiles: Optional[set[str]] = None) -> list[Path]:
    """Restore backups only from the directory explicitly requested by the caller.

    A mounted Drive must not contaminate local runs or temporary test fixtures.
    The Colab comparison cell already passes its Drive directory explicitly.
    """
    if extra_backup_dir is None:
        return []
    search_dirs = [extra_backup_dir]

    latest_zip_by_profile: dict[str, Path] = {}
    for bdir in search_dirs:
        if not bdir.is_dir():
            continue
        for zpath in sorted(bdir.glob("*.zip"), key=lambda p: p.name):
            match = BACKUP_NAME_RE.match(zpath.name)
            if match is None:
                continue
            profile = match.group("profile")
            if profiles is not None and profile not in profiles:
                continue
            prev = latest_zip_by_profile.get(profile)
            if prev is None or match.group("stamp") >= BACKUP_NAME_RE.match(prev.name).group("stamp"):
                latest_zip_by_profile[profile] = zpath

    restored: list[Path] = []
    for profile, zpath in sorted(latest_zip_by_profile.items()):
        dest = runs_root / profile
        try:
            with zipfile.ZipFile(zpath) as bundle:
                zip_result_count = sum(1 for item in bundle.namelist() if item.endswith("/result.json"))
                missing = []
                for item in bundle.infolist():
                    if not (dest / item.filename).resolve().is_relative_to(dest.resolve()):
                        raise ValueError(f"Unsafe archive path in {zpath}: {item.filename}")
                    if not item.is_dir() and not (dest / item.filename).is_file():
                        missing.append(item)
                if not missing:
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                # Merge missing artifacts, including evaluation sidecars even when
                # generation sample counts match. Never overwrite local progress.
                for item in missing:
                    bundle.extract(item, dest)
                restored.append(dest)
                print(f"[Auto-Restore] Restored profile {profile} ({zip_result_count} samples) from {zpath}")
        except Exception as exc:
            print(f"[Auto-Restore] Skipping backup {zpath}: {exc}")
    return restored
