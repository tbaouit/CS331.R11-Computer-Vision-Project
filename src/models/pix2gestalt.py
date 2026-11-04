"""Pix2Gestalt latent-diffusion loader + CarveKit matting interface (strict: fail loudly, never fall back)."""
from __future__ import annotations

from pathlib import Path
import gc
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


def apply_inference_ema(model):
    """Keep the weights used by upstream ema_scope without GPU backup copies.

    Copy before freezing: upstream LitEma.copy_to uses requires_grad to select
    parameters. Inference never updates or restores the training weights.
    """
    if getattr(model, "use_ema", False):
        model.model_ema.copy_to(model.model)
        model.use_ema = False
        del model.model_ema
    model.requires_grad_(False)
    return model


class Pix2GestaltModel:
    """Holds the Pix2Gestalt LDM (`model`) and the CarveKit background-matting callable (`carvekit_interface`)."""

    def __init__(self, ckpt_path: Path = DEFAULT_CKPT, config_path: Path = DEFAULT_CONFIG, device: str = "cuda"):
        import torch

        ensure_vendor_paths()
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        from carvekit.api.high import HiInterface

        ckpt_path, config_path = Path(ckpt_path), Path(config_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Pix2Gestalt checkpoint not found: {ckpt_path} (run scripts/download_weights.py)")
        if not config_path.is_file():
            raise FileNotFoundError(f"Pix2Gestalt config not found: {config_path} (run scripts/setup_colab.py)")

        print(f"[Models] Loading Pix2Gestalt checkpoint from {ckpt_path}...", flush=True)
        self.device = device
        model = instantiate_from_config(OmegaConf.load(str(config_path)).model)
        # Lightning 1.x checkpoints pickle callback objects, so `weights_only=False` is required.
        try:
            state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False, mmap=True)
        except RuntimeError as error:
            if "mmap" not in str(error):
                raise
            state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        essential = [key for key in missing if key.startswith(("model.", "model_ema.", "first_stage_model.", "cc_projection."))]
        if essential:
            raise RuntimeError(f"Incompatible Pix2Gestalt checkpoint; missing keys: {essential[:10]}")
        del state_dict
        apply_inference_ema(model)
        gc.collect()
        self.model = model.to(device).eval()
        # Match upstream matting parameters; process samples singly on CPU so
        # matting weights and activations do not compete with diffusion on GPU.
        self.carvekit_interface = HiInterface(
            object_type="object", batch_size_seg=1, batch_size_matting=1,
            device="cpu", seg_mask_size=640, matting_mask_size=2048,
            trimap_prob_threshold=231, trimap_dilation=30,
            trimap_erosion_iters=5, fp16=False,
        )
        print("[Models] Pix2Gestalt + CarveKit ready.", flush=True)


    def to(self, device):
        self.model.to(device)
        self.device = str(device)
        return self
