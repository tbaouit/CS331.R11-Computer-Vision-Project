"""Foundation-model loaders (Pix2Gestalt + CarveKit, Qwen3-VL, Stable Diffusion 2 Inpainting, CLIP)."""
from src.models.pix2gestalt import Pix2GestaltModel, ensure_vendor_paths, load_run_pix2gestalt
from src.models.qwen_vl import DEFAULT_VLM_ID, QwenVLModel
from src.models.registry import FoundationModels

__all__ = [
    "DEFAULT_VLM_ID",
    "FoundationModels",
    "Pix2GestaltModel",
    "QwenVLModel",
    "ensure_vendor_paths",
    "load_run_pix2gestalt",
]

