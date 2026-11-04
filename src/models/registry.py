"""Cache models in host RAM and keep only the active generation stage on CUDA."""
from __future__ import annotations

import gc

from src.models.pix2gestalt import Pix2GestaltModel
from src.models.qwen_vl import DEFAULT_VLM_ID, QwenVLModel


class FoundationModels:
    """Lazy, single-GPU model registry shared by both completion pipelines.

    Accessing a model property activates that stage and offloads its predecessor.
    Callers must finish inference before requesting a different model. Processors
    and tokenizers stay on CPU. Loaded weights are reused across samples.
    """

    def __init__(self, inpainting_model_id: str, clip_model_id: str,
                 vlm_model_id: str = DEFAULT_VLM_ID, device: str = "cuda"):
        self.device = device
        self.inpainting_model_id = inpainting_model_id
        self.clip_model_id = clip_model_id
        self.vlm_model_id = vlm_model_id
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

    def _load(self, name):
        import torch
        if name == "shape":
            return Pix2GestaltModel(device="cpu")
        if name == "vlm":
            return QwenVLModel(model_id=self.vlm_model_id, device="cpu")
        if name == "sd2":
            from diffusers import StableDiffusionInpaintPipeline
            print(f"[Models] Loading inpainting pipeline {self.inpainting_model_id}...", flush=True)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.inpainting_model_id, torch_dtype=torch.float16, safety_checker=None,
            )
            pipe.set_progress_bar_config(disable=True)
            pipe.enable_vae_slicing()
            return pipe
        if name == "clip":
            from transformers import CLIPModel, CLIPProcessor
            print(f"[Models] Loading CLIP {self.clip_model_id}...", flush=True)
            model = CLIPModel.from_pretrained(self.clip_model_id, torch_dtype=torch.float16).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
            return model
        raise ValueError(f"Unknown model stage: {name}")

    def _activate(self, name):
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
            # A failed .to() can leave some layers on CUDA.
            model.to("cpu")
            self._clear_cache()
            raise
        self._active = name
        print(f"[Models] {name} ready on {self.device}.", flush=True)
        return model

    @property
    def shape_predictor(self):
        return self._activate("shape")

    @property
    def vlm(self):
        return self._activate("vlm")

    @property
    def pipe(self):
        return self._activate("sd2")

    @property
    def clip_model(self):
        return self._activate("clip")

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            from transformers import CLIPProcessor
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
        return self._clip_processor
