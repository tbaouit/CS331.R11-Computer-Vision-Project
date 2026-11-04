"""Pix2Gestalt latent-diffusion loader + CarveKit matting interface (strict: fail loudly, never fall back)."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PIX2GESTALT_DIR = ROOT / "pix2gestalt"
DEFAULT_CKPT = ROOT / "ckpt" / "epoch=000005.ckpt"
DEFAULT_CONFIG = PIX2GESTALT_DIR / "pix2gestalt" / "configs" / "sd-finetune-pix2gestalt-c_concat-256.yaml"
VENDOR_PATHS = [
    ROOT / "vendor" / "taming-transformers",
    ROOT / "vendor" / "CLIP",
    PIX2GESTALT_DIR,
    PIX2GESTALT_DIR / "pix2gestalt",
]


def ensure_vendor_paths() -> None:
    """Expose the cloned Pix2Gestalt / taming-transformers / CLIP repos on `sys.path`."""
    for path in VENDOR_PATHS:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_run_pix2gestalt():
    """Return Pix2Gestalt's official `run_pix2gestalt(model, device, input_im, visible_mask, n_samples, ...)`."""
    ensure_vendor_paths()
    from inference import run_pix2gestalt

    return run_pix2gestalt


class Pix2GestaltModel:
    """Holds the Pix2Gestalt LDM (`model`) and the CarveKit background-matting callable (`carvekit_interface`)."""

    def __init__(self, ckpt_path: Path = DEFAULT_CKPT, config_path: Path = DEFAULT_CONFIG, device: str = "cuda"):
        import torch

        ensure_vendor_paths()
        from omegaconf import OmegaConf
        from ldm.util import create_carvekit_interface, instantiate_from_config

        ckpt_path, config_path = Path(ckpt_path), Path(config_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Pix2Gestalt checkpoint not found: {ckpt_path} (run scripts/download_weights.py)")
        if not config_path.is_file():
            raise FileNotFoundError(f"Pix2Gestalt config not found: {config_path} (run scripts/setup_colab.py)")

        print(f"[Models] Loading Pix2Gestalt checkpoint from {ckpt_path}...", flush=True)
        self.device = device
        model = instantiate_from_config(OmegaConf.load(str(config_path)).model)
        # Lightning 1.x checkpoints pickle callback objects, so `weights_only=False` is required.
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        self.model = model.to(device).eval()
        self.carvekit_interface = create_carvekit_interface()
        print("[Models] Pix2Gestalt + CarveKit ready.", flush=True)

