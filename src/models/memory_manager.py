# Phạm vi: Cả hai phiên bản dùng bộ quản lý model này. Baseline chỉ lấy SD2; proposed dùng 4 mô-đun
# lõi của Open-World Amodal (RAM++, CLIP ViT-B/32, GroundingDINO + SAM ViT-H nạp 1 lần, InstaOrder, SD2).
"""Cache models in host RAM and keep only the active generation stage on CUDA (<15GB VRAM)."""
from __future__ import annotations

import gc

from src.models.openworld_models import (
    GroundingDINOModel,
    InstaOrderModel,
    OpenAICLIPPromptSelector,
    RAMPlusModel,
    SAMViTHModel,
)


class FoundationModels:
    """Lazy, single-GPU model registry shared by Baseline (SD2) and Proposed (Open-World Amodal SD2).

    Accessing a model property activates that stage and offloads its predecessor to CPU RAM.
    SAM ViT-H is built once and cached across all iterations and samples.
    """

    def __init__(
        self,
        inpainting_model_id: str,
        clip_model_id: str = "ViT-B/32",
        device: str = "cuda",
        pipeline_config: dict | None = None,
    ):
        self.device = device
        self.inpainting_model_id = inpainting_model_id
        self.clip_model_id = clip_model_id
        self.pipeline_config = pipeline_config or {}
        self._models = {}
        self._active = None
        print("[Models] Sequential offload enabled: one stage on GPU; cached weights in CPU RAM.", flush=True)

    @staticmethod
    def _clear_cache():
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_gpu(self):
        if self._active is not None:
            try:
                from diffusers.utils import logging as diffusers_logging

                prev_verbosity = diffusers_logging.get_verbosity()
                diffusers_logging.set_verbosity_error()
                try:
                    self._models[self._active].to("cpu")
                finally:
                    diffusers_logging.set_verbosity(prev_verbosity)
            except Exception:
                self._models[self._active].to("cpu")
            self._active = None
            self._clear_cache()

    def _load(self, name: str):
        import torch

        cfg = self.pipeline_config
        if name == "ram_plus":
            return RAMPlusModel(
                ckpt_path=cfg.get("ram_ckpt", "ckpt/ram_plus_swin_large_14m.pth"),
                image_size=int(cfg.get("ram_image_size", 384)),
                threshold=float(cfg.get("ram_threshold", 0.6)),
                device="cpu",
            )
        if name == "openai_clip":
            return OpenAICLIPPromptSelector(
                model_id=cfg.get("clip_model_id", self.clip_model_id or "ViT-B/32"),
                device="cpu",
            )
        if name == "gdino":
            return GroundingDINOModel(
                config_path=cfg.get(
                    "gdino_config",
                    "vendor/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                ),
                ckpt_path=cfg.get("gdino_ckpt", "ckpt/groundingdino_swint_ogc.pth"),
                device="cpu",
            )
        if name == "sam":
            return SAMViTHModel(
                ckpt_path=cfg.get("sam_ckpt", "ckpt/sam_vit_h_4b8939.pth"),
                device="cpu",
            )
        if name == "instaorder":
            return InstaOrderModel(
                ckpt_path=cfg.get("instaorder_ckpt", "ckpt/InstaOrder_InstaOrderNet_od.pth.tar"),
                device="cpu",
            )
        if name == "inpainting":
            from diffusers import StableDiffusionInpaintPipeline

            print(f"[Models] Loading SD2 inpainting pipeline {self.inpainting_model_id} (FP16)...", flush=True)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.inpainting_model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipe.set_progress_bar_config(disable=True)
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            return pipe
        raise ValueError(f"Unknown model stage: {name}")

    def _activate(self, name: str):
        if self._active == name:
            return self._models[name]
        # Offload BEFORE loading or moving the next model to avoid transient OOM.
        self.release_gpu()
        if name not in self._models:
            self._models[name] = self._load(name)
        model = self._models[name]
        try:
            model.to(self.device)
        except Exception:
            model.to("cpu")
            self._clear_cache()
            raise
        self._active = name
        print(f"[Models] {name} ready on {self.device}.", flush=True)
        return model

    @property
    def ram_plus(self) -> RAMPlusModel:
        return self._activate("ram_plus")

    @property
    def openai_clip(self) -> OpenAICLIPPromptSelector:
        return self._activate("openai_clip")

    @property
    def gdino(self) -> GroundingDINOModel:
        return self._activate("gdino")

    @property
    def sam(self) -> SAMViTHModel:
        return self._activate("sam")

    @property
    def instaorder(self) -> InstaOrderModel:
        return self._activate("instaorder")

    @property
    def pipe(self):
        return self._activate("inpainting")
