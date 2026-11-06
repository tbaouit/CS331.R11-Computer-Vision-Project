"""Lazy foundation-model dispatcher for blind, mask-given, and SD2 Direct modes."""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

os.environ["MPLBACKEND"] = "Agg"

PIPELINE_MODES = ("improved", "oracle_mask", "sd2_direct")
MODE_ALIASES = {"gt_mask": "oracle_mask"}
TEXT_QUERY_POLICIES = ("empty", "dataset_object_name")


class AmodalBackend:
    def __init__(self, config):
        import torch

        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.Resampling.LANCZOS
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This configuration requires CUDA. Use the Colab notebook.")
        raw_mode = config.get("pipeline_mode", "improved")
        mode = MODE_ALIASES.get(raw_mode, raw_mode)
        if mode not in PIPELINE_MODES:
            raise ValueError(f"Unknown pipeline_mode: {raw_mode} (expected one of {PIPELINE_MODES})")

        from src.models.registry import FoundationModels

        self.models = FoundationModels(
            inpainting_model_id=config["inpainting_model_id"],
            clip_model_id=config.get("clip_model_id", "openai/clip-vit-base-patch32"),
            vlm_model_id=config.get("vlm_model_id", "Qwen/Qwen3-VL-4B-Instruct"),
            device="cuda",
        )
        self.config = config
        self.mode = mode
        self.job = None
        if self.mode == "improved":
            from src.pipelines.improved_completer import ImprovedAmodalCompleter

            self.completer = ImprovedAmodalCompleter(self.models, self.config)
        elif self.mode == "sd2_direct":
            from src.pipelines.sd2_direct_completer import SD2DirectCompleter

            self.completer = SD2DirectCompleter(self.models, self.config)
        else:
            from src.pipelines.oracle_completer import OracleMaskAmodalCompleter

            self.completer = OracleMaskAmodalCompleter(self.models, self.config)

    def begin_sample(self, job):
        self.job = Path(job)

    def complete(self, image, visible, label, occluded_mask=None):
        policy = self.config.get("text_query_policy", "empty")
        if policy not in TEXT_QUERY_POLICIES:
            raise ValueError(f"Unknown text_query_policy: {policy}")
        query = label if policy == "dataset_object_name" else ""
        try:
            if self.mode == "sd2_direct":
                return self.completer.complete(image=image, visible_mask=visible,
                                               occluded_mask=occluded_mask, job_dir=self.job)
            if self.mode == "oracle_mask":
                if occluded_mask is None:
                    raise ValueError("pipeline_mode='oracle_mask' requires occluded_mask input.")
                return self.completer.complete(
                    image=image,
                    visible_mask=visible,
                    occluded_mask=occluded_mask,
                    text_query=query,
                    job_dir=self.job,
                )
            if occluded_mask is not None:
                raise ValueError(f"Blind pipeline_mode='{self.mode}' must not receive occluded_mask.")
            return self.completer.complete(image=image, visible_mask=visible, text_query=query, job_dir=self.job)
        finally:
            self.models.release_gpu()
