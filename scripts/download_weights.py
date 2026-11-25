# Phạm vi: Ô 3: tải/cache trọng số cho Baseline (SD1.5) và Proposed (Open-World Amodal: RAM++,
# GroundingDINO, SAM ViT-H, InstaOrder, SD2 Inpainting). Không tải LISA-13B hay LaMa.
"""Download and verify checkpoints for Baseline (SD1.5) and Proposed (Open-World Amodal SD2)."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "ckpt"

DRIVE_CKPT_CANDIDATES = (
    Path("/content/drive/MyDrive/CS331/ckpt"),
    Path("/content/drive/MyDrive/amodal_project/ckpt"),
    Path("/content/drive/MyDrive/CS331"),
    Path("/content/drive/MyDrive/amodal_project"),
)

SAM_VIT_H_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
INSTAORDER_GDRIVE_ID = os.environ.get("INSTAORDER_GDRIVE_ID", "1O_ab_bcA1_InstaOrderNet_od")


def _copy_from_drive_if_present(filename: str, target: Path) -> bool:
    for folder in DRIVE_CKPT_CANDIDATES:
        candidate = folder / filename
        if candidate.is_file() and candidate.stat().st_size > 1024 * 1024:
            print(f"[Weights] Copying {filename} from Google Drive ({candidate})...", flush=True)
            shutil.copy2(candidate, target)
            return True
    return False


def _download_url(url: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    print(f"[Weights] Downloading {url} -> {target} ...", flush=True)
    urllib.request.urlretrieve(url, partial)
    os.replace(partial, target)


def main():
    from huggingface_hub import hf_hub_download, snapshot_download

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. RAM++ Swin-L checkpoint
    ram_target = CKPT_DIR / "ram_plus_swin_large_14m.pth"
    if not ram_target.is_file() and not _copy_from_drive_if_present(ram_target.name, ram_target):
        print("[Weights] Fetching RAM++ Swin-L checkpoint from HuggingFace...", flush=True)
        cached = hf_hub_download(
            repo_id="xinyu1205/recognize-anything-plus-model",
            filename="ram_plus_swin_large_14m.pth",
        )
        shutil.copy2(Path(cached).resolve(), ram_target)
    print(f"[Weights] RAM++ checkpoint ready: {ram_target} ({ram_target.stat().st_size} bytes)", flush=True)

    # 2. GroundingDINO Swin-T checkpoint
    gdino_target = CKPT_DIR / "groundingdino_swint_ogc.pth"
    if not gdino_target.is_file() and not _copy_from_drive_if_present(gdino_target.name, gdino_target):
        print("[Weights] Fetching GroundingDINO Swin-T checkpoint from HuggingFace...", flush=True)
        cached = hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO",
            filename="groundingdino_swint_ogc.pth",
        )
        shutil.copy2(Path(cached).resolve(), gdino_target)
    print(f"[Weights] GroundingDINO checkpoint ready: {gdino_target} ({gdino_target.stat().st_size} bytes)", flush=True)

    # 3. SAM ViT-H checkpoint
    sam_target = CKPT_DIR / "sam_vit_h_4b8939.pth"
    if not sam_target.is_file() and not _copy_from_drive_if_present(sam_target.name, sam_target):
        _download_url(SAM_VIT_H_URL, sam_target)
    print(f"[Weights] SAM ViT-H checkpoint ready: {sam_target} ({sam_target.stat().st_size} bytes)", flush=True)

    # 4. InstaOrder checkpoint (InstaOrder_InstaOrderNet_od.pth.tar)
    insta_target = CKPT_DIR / "InstaOrder_InstaOrderNet_od.pth.tar"
    if not insta_target.is_file() and not _copy_from_drive_if_present(insta_target.name, insta_target):
        insta_url = os.environ.get("INSTAORDER_CKPT_URL", "").strip()
        if insta_url:
            _download_url(insta_url, insta_target)
        else:
            print(
                f"[Weights] Note: place {insta_target.name} in {CKPT_DIR} or Google Drive MyDrive/CS331/ckpt/ "
                "(or set INSTAORDER_CKPT_URL).",
                flush=True,
            )

    # 5. Pre-cache shared SD2 Inpainting pipeline (used by both Baseline and Proposed)
    sd_ignores = ["*.onnx", "*.xml", "*.msgpack", "*.ckpt", "*.safetensors.index.json"]
    sd_includes = [
        "model_index.json",
        "unet/*",
        "vae/*",
        "text_encoder/*",
        "tokenizer/*",
        "scheduler/*",
        "feature_extractor/*",
    ]
    repo = "sd2-community/stable-diffusion-2-inpainting"
    print(f"[Weights] Caching shared HuggingFace pipeline {repo} ...", flush=True)
    try:
        snapshot_download(repo, allow_patterns=sd_includes, ignore_patterns=sd_ignores, max_workers=2)
    except Exception as exc:
        print(f"[Weights] Warning caching {repo}: {exc}", flush=True)

    print("[Weights] Checkpoint preparation complete (SD2 shared; no SD1.5, LISA-13B, or LaMa).", flush=True)


if __name__ == "__main__":
    main()
