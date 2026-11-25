# Phạm vi: Cả hai phiên bản: kiểm tra config, chọn lớp thực thi và giải phóng GPU sau mỗi mẫu.
"""Entry point for Baseline (SD2 Direct) and Proposed (Open-World Amodal SD2)."""
from pathlib import Path

from PIL import Image

from src.models.memory_manager import FoundationModels
from src.models.openworld_models import validate_openworld_proposed_config
from src.pipelines.baseline import Baseline, validate_baseline_config


def validate_experiment_config(config: dict) -> str:
    mode = config.get("pipeline_mode")
    if config.get("use_occluded_mask_input") is not True:
        raise ValueError("Both baseline and proposed require use_occluded_mask_input=true")
    if mode == "baseline":
        validate_baseline_config(config)
    elif mode == "proposed":
        validate_openworld_proposed_config(config)
    else:
        raise ValueError("Choose pipeline_mode='baseline' or 'proposed'")
    return mode


class AmodalBackend:
    def __init__(self, config: dict):
        import torch

        mode = validate_experiment_config(config)
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This configuration requires CUDA. Use the Colab notebook.")
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.Resampling.LANCZOS
        self.models = FoundationModels(
            inpainting_model_id=config["inpainting_model_id"],
            clip_model_id=config.get("clip_model_id", "ViT-B/32"),
            device="cuda",
            pipeline_config=config,
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
            policy = self.config.get("text_query_policy", "dataset_object_name")
            query = (label or "object") if policy == "dataset_object_name" else ""
            if self.mode == "baseline":
                return self.completer.complete(
                    image=image,
                    visible_mask=visible,
                    occluded_mask=occluded_mask,
                    label=query,
                    job_dir=self.job,
                )
            return self.completer.complete(
                image=image,
                visible_mask=visible,
                text_query=query,
                occluded_mask=occluded_mask,
                job_dir=self.job,
            )
        finally:
            self.models.release_gpu()
