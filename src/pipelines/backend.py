# Phạm vi: Cả hai phiên bản: kiểm tra config, chọn lớp thực thi và giải phóng GPU sau mỗi mẫu.
"""Entry point for the Baseline (SD1.5 Direct) and Proposed (Open-World Amodal SD2) methods."""
from pathlib import Path

from PIL import Image

from src.models.memory_manager import FoundationModels
from src.models.openworld_models import validate_openworld_proposed_config
from src.models.sd15_ip_adapter import SD15_INPAINTING, validate_proposed_config
from src.pipelines.baseline import Baseline, validate_sd15_direct


def validate_experiment_config(config):
    mode = config.get("pipeline_mode")
    if config.get("use_occluded_mask_input") is not True:
        raise ValueError("All versions require the given mask configuration flag")
    if mode == "baseline":
        if config.get("text_query_policy", "empty") != "empty":
            raise ValueError("Baseline disables ground-truth category prompts")
        validate_sd15_direct(config)
    elif mode == "proposed":
        # Support both Open-World Amodal SD2 config and legacy SD1.5 IP-Adapter config if passed in unit tests
        if config.get("inpainting_model_id") == SD15_INPAINTING and "use_ip_adapter" in config:
            validate_proposed_config(config)
            for flag in ("use_micro_attributes", "use_ip_adapter", "use_latent_blending", "use_strict_critic", "enable_critic"):
                if config.get(flag) is not True:
                    raise ValueError(f"Proposed SD1.5 requires {flag}")
        else:
            validate_openworld_proposed_config(config)
    else:
        raise ValueError("Choose baseline or proposed")
    return mode


class AmodalBackend:
    def __init__(self, config):
        import torch

        mode = validate_experiment_config(config)
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This configuration requires CUDA. Use the Colab notebook.")
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.Resampling.LANCZOS
        scheduler = "ddim" if mode == "baseline" or config.get("refinement_scheduler") == "ddim" else None
        self.models = FoundationModels(
            inpainting_model_id=config["inpainting_model_id"],
            clip_model_id=config.get("clip_model_id", "openai/clip-vit-base-patch32"),
            vlm_model_id=config.get("vlm_model_id", "Qwen/Qwen3-VL-4B-Instruct"),
            device="cuda",
            advanced_config=config if mode == "proposed" else None,
            refinement_scheduler=scheduler,
        )
        self.config = config
        self.mode = mode
        self.job = None
        if mode == "baseline":
            self.completer = Baseline(self.models, config)
        else:
            from src.pipelines.proposed import Proposed

            self.completer = Proposed(self.models, config)

    def begin_sample(self, job):
        self.job = Path(job)

    def complete(self, image, visible, label, occluded_mask=None):
        try:
            if self.mode == "baseline":
                return self.completer.complete(
                    image=image,
                    visible_mask=visible,
                    occluded_mask=occluded_mask,
                    job_dir=self.job,
                )
            policy = self.config.get("text_query_policy", "dataset_object_name")
            query = (label or "main object") if policy == "dataset_object_name" else ""
            return self.completer.complete(
                image=image,
                visible_mask=visible,
                text_query=query,
                occluded_mask=occluded_mask,
                job_dir=self.job,
            )
        finally:
            self.models.release_gpu()
