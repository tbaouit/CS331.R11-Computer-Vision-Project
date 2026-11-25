# Phạm vi: Cả hai phiên bản dùng bộ quản lý model này. Baseline chỉ lấy SD1.5; proposed thêm prior, Qwen và
# CLIP.
"""Cache models in host RAM and keep only the active generation stage on CUDA."""
from __future__ import annotations

import gc

from src.models.pix2gestalt_prior import Pix2GestaltModel
from src.models.qwen_vl import DEFAULT_VLM_ID, QwenVLModel


# Giữ model đã nạp trong RAM và chỉ đưa stage đang dùng lên GPU.
class FoundationModels:
    """Lazy, single-GPU model registry shared by both completion pipelines.

    Accessing a model property activates that stage and offloads its predecessor.
    Callers must finish inference before requesting a different model. Processors
    and tokenizers stay on CPU. Loaded weights are reused across samples.
    """

    # Lưu cấu hình và tạo cache rỗng; chưa tải checkpoint tại đây.
    def __init__(self, inpainting_model_id: str, clip_model_id: str,
                 vlm_model_id: str = DEFAULT_VLM_ID, device: str = "cuda", advanced_config=None,
                 refinement_scheduler=None):
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

    # Dọn bộ nhớ chưa còn tham chiếu sau khi đổi stage.
    @staticmethod
    def _clear_cache():
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Đưa model đang hoạt động về CPU để GPU đủ chỗ cho stage kế.
    def release_gpu(self):
        if self._active is not None:
            self._models[self._active].to("cpu")
            self._active = None
            self._clear_cache()

    # Khởi tạo model theo tên stage; cấu hình DDIM và adapter được gắn tại lúc tạo pipeline.
    def _load(self, name):
        import torch
        if name == "shape":
            return Pix2GestaltModel(device="cpu")
        if name == "vlm":
            return QwenVLModel(model_id=self.vlm_model_id, device="cpu")
        if name == "inpainting":
            from diffusers import StableDiffusionInpaintPipeline
            print(f"[Models] Loading inpainting pipeline {self.inpainting_model_id}...", flush=True)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.inpainting_model_id, torch_dtype=torch.float16, safety_checker=None,
            )
            pipe.set_progress_bar_config(disable=True)
            pipe.enable_vae_slicing()
            if self.advanced_config is not None:
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

    # Tái dùng model đã nạp hoặc tạo khi cần; chuyển model cũ về CPU trước.
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

    # Trả Pix2Gestalt cho bước tạo prior của proposed.
    @property
    def shape_predictor(self):
        return self._activate("shape")

    # Trả Qwen dùng cho bước suy luận và critic ở proposed.
    @property
    def vlm(self):
        return self._activate("vlm")

    # Trả SD1.5 inpainting dùng cho cả hai phiên bản.
    @property
    def pipe(self):
        return self._activate("inpainting")

    # Trả encoder kiểm tra prompt của proposed.
    @property
    def clip_model(self):
        return self._activate("clip")

    # Chuẩn hóa crop và prompt cho CLIP; processor được dùng lại.
    @property
    def clip_processor(self):
        if self._clip_processor is None:
            from transformers import CLIPProcessor
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
        return self._clip_processor
