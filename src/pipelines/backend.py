# Phạm vi: Cả hai phiên bản: kiểm tra config, chọn lớp thực thi và giải phóng GPU sau mỗi mẫu.
"""Entry point for the two SD1.5 methods."""
from pathlib import Path

from PIL import Image

from src.models.memory_manager import FoundationModels
from src.models.sd15_ip_adapter import validate_proposed_config
from src.pipelines.baseline import Baseline, validate_sd15_direct


# Chặn backbone hoặc bộ cờ không đúng vai trò baseline và proposed trước khi dùng CUDA.
def validate_experiment_config(config):
    mode = config.get("pipeline_mode")
    if config.get("use_occluded_mask_input") is not True:
        raise ValueError("All versions require the given hidden mask")
    if config.get("text_query_policy", "empty") != "empty":
        raise ValueError("Ground-truth category prompts are disabled")
    if mode == "baseline":
        validate_sd15_direct(config)
    elif mode == "proposed":
        validate_proposed_config(config)
        for flag in ("use_micro_attributes", "use_ip_adapter", "use_latent_blending", "use_strict_critic", "enable_critic"):
            if config.get(flag) is not True:
                raise ValueError(f"Proposed requires {flag}")
    else:
        raise ValueError("Choose baseline or proposed")
    return mode


# Cổng gọi chung để runner không phải biết chi tiết từng pipeline.
class AmodalBackend:
    # Tạo memory manager và chọn đúng lớp thực thi từ số phiên bản.
    def __init__(self, config):
        import torch

        mode = validate_experiment_config(config)
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This configuration requires CUDA. Use the Colab notebook.")
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.Resampling.LANCZOS
        self.models = FoundationModels(
            inpainting_model_id=config["inpainting_model_id"],
            clip_model_id=config.get("clip_model_id", "openai/clip-vit-base-patch32"),
            vlm_model_id=config.get("vlm_model_id", "Qwen/Qwen3-VL-4B-Instruct"),
            device="cuda",
            advanced_config=config if mode == "proposed" else None,
            refinement_scheduler="ddim",
        )
        self.job = None
        if mode == "baseline":
            self.completer = Baseline(self.models, config)
        else:
            from src.pipelines.proposed import Proposed
            self.completer = Proposed(self.models, config)

    # Chuyển thư mục artefact hiện tại cho pipeline trước khi sinh ảnh.
    def begin_sample(self, job):
        self.job = Path(job)

    # Truyền ảnh/mask vào pipeline và luôn trả model về CPU, kể cả khi mẫu lỗi.
    def complete(self, image, visible, label, occluded_mask=None):
        try:
            return self.completer.complete(image=image, visible_mask=visible,
                                           occluded_mask=occluded_mask, job_dir=self.job)
        finally:
            self.models.release_gpu()
