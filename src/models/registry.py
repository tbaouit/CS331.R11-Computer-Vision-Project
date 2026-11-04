"""Load every foundation model exactly once and share it across all samples of a batch."""
from __future__ import annotations

from src.models.pix2gestalt import Pix2GestaltModel
from src.models.qwen_vl import DEFAULT_VLM_ID, QwenVLModel


class FoundationModels:
    """Shared container: `shape_predictor` (Pix2Gestalt+CarveKit), `vlm` (Qwen3-VL), `pipe` (SD2 inpaint), CLIP."""

    def __init__(
        self,
        inpainting_model_id: str,
        clip_model_id: str,
        vlm_model_id: str = DEFAULT_VLM_ID,
        device: str = "cuda",
    ):
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.shape_predictor = Pix2GestaltModel(device=device)
        self.vlm = QwenVLModel(model_id=vlm_model_id, device=device)

        print(f"[Models] Loading inpainting pipeline {inpainting_model_id}...", flush=True)
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpainting_model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

        print(f"[Models] Loading CLIP {clip_model_id}...", flush=True)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=dtype).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        print("[Models] All foundation models loaded.", flush=True)

