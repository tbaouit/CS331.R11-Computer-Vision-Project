# Phạm vi: Cả hai phiên bản dùng bộ quản lý model này. Baseline chỉ lấy SD1.5; proposed dùng 4 mô-đun
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
from src.models.pix2gestalt_prior import Pix2GestaltModel
from src.models.qwen_vl import DEFAULT_VLM_ID, QwenVLModel


# Giữ model đã nạp trong RAM và chỉ đưa stage đang dùng lên GPU.
class FoundationModels:
    """Lazy, single-GPU model registry shared by both completion pipelines.

    Accessing a model property activates that stage and offloads its predecessor.
    SAM ViT-H is built once and cached across all iterations and samples.
    """

    def __init__(
        self,
        inpainting_model_id: str,
        clip_model_id: str = "openai/clip-vit-base-patch32",
        vlm_model_id: str = DEFAULT_VLM_ID,
        device: str = "cuda",
        advanced_config: dict | None = None,
        refinement_scheduler: str | None = None,
    ):
        self.device = device
        self.inpainting_model_id = inpainting_model_id
        self.clip_model_id = clip_model_id
        self.vlm_model_id = vlm_model_id
        self.advanced_config = advanced_config
        self.refinement_scheduler = refinement_scheduler
        self._models = {}
        self._active = None
        self._clip_processor = None
        print("[Models] Sequential offload enabled: one stage on GPU; cached weights in CPU RAM.", flush=True)

    @staticmethod
    def _clear_cache():
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_gpu(self):
        if self._active is not None:
            self._models[self._active].to("cpu")
            self._active = None
            self._clear_cache()

    def _load(self, name: str):
        import torch

        cfg = self.advanced_config or {}
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
        if name == "shape":
            return Pix2GestaltModel(device="cpu")
        if name == "vlm":
            return QwenVLModel(model_id=self.vlm_model_id, device="cpu")
        if name == "inpainting":
            from diffusers import StableDiffusionInpaintPipeline

            print(f"[Models] Loading inpainting pipeline {self.inpainting_model_id} (FP16)...", flush=True)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.inpainting_model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipe.set_progress_bar_config(disable=True)
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            if self.advanced_config is not None and self.advanced_config.get("use_ip_adapter"):
                from src.models.sd15_ip_adapter import configure_proposed_pipe

                configure_proposed_pipe(pipe, self.advanced_config)
            elif self.refinement_scheduler == "ddim":
                from src.models.sd15_ip_adapter import configure_ddim

                configure_ddim(pipe)
            return pipe
        if name == "clip":
            from transformers import CLIPModel, CLIPProcessor

            print(f"[Models] Loading CLIP {self.clip_model_id}...", flush=True)
            model = CLIPModel.from_pretrained(self.clip_model_id, torch_dtype=torch.float16).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
            return model
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
    def shape_predictor(self):
        return self._activate("shape")

    @property
    def vlm(self):
        return self._activate("vlm")

    @property
    def pipe(self):
        return self._activate("inpainting")

    @property
    def clip_model(self):
        return self._activate("clip")

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            from transformers import CLIPProcessor

            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
        return self._clip_processor
