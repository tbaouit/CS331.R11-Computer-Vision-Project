"""Shared foundation-model loader and mode dispatcher (`upstream_baseline`, `improved`, `oracle_mask`)."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from PIL import Image

from src.pipelines.baseline_tracer import install_baseline_trace_hooks

os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[2]


class SCAmodalBackend:
    def __init__(self, config):
        import torch

        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.Resampling.LANCZOS
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This full SCAmodal configuration requires CUDA. Use the Colab notebook.")
        raw_mode = config.get("pipeline_mode", "upstream_baseline")
        mode = "oracle_mask" if raw_mode == "gt_mask" else raw_mode
        if mode not in ("upstream_baseline", "improved", "oracle_mask"):
            raise ValueError(f"Unknown pipeline_mode: {raw_mode}")
        for extra in [
            "vendor/taming-transformers",
            "vendor/CLIP",
            "pix2gestalt",
            "pix2gestalt/pix2gestalt",
            "runtime/SCAmodal",
        ]:
            sys.path.insert(0, str(ROOT / extra))
        from amodal_completer import AmodalCompleter

        self.cls = AmodalCompleter
        self.model = AmodalCompleter(
            inpainting_model_id=config["inpainting_model_id"],
            clip_model_id=config["clip_model_id"],
            device="cuda",
        )
        self.config = config
        self.mode = mode
        self.job = None
        self.pad_info = {}
        self.attempt = 0
        self.critiques = []
        self.improved = None
        self.oracle = None
        if self.mode == "improved":
            from src.pipelines.improved_completer import ImprovedAmodalCompleter

            self.improved = ImprovedAmodalCompleter(self.model, self.config)
        elif self.mode == "oracle_mask":
            from src.pipelines.oracle_completer import OracleMaskAmodalCompleter

            self.oracle = OracleMaskAmodalCompleter(self.model, self.config)
        else:
            install_baseline_trace_hooks(self)

    def begin_sample(self, job):
        self.job = Path(job)
        self.pad_info, self.attempt, self.critiques = {}, 0, []
        os.environ["SCAMODAL_RUN_DIR"] = str(job)

    def complete(self, image, visible, label, occluded_mask=None):
        policy = self.config.get("text_query_policy", "upstream_default")
        if policy not in ("upstream_default", "dataset_object_name"):
            raise ValueError("Unknown text_query_policy")
        query = label if policy == "dataset_object_name" else ""
        if self.mode == "oracle_mask":
            if occluded_mask is None:
                raise ValueError("pipeline_mode='oracle_mask' requires occluded_mask input.")
            return self.oracle.complete(
                image=image,
                visible_mask=visible,
                occluded_mask=occluded_mask,
                text_query=query,
                job_dir=self.job,
            )
        if occluded_mask is not None:
            raise ValueError(f"Blind pipeline_mode='{self.mode}' must not receive occluded_mask.")
        if self.mode == "improved":
            return self.improved.complete(image=image, visible_mask=visible, text_query=query, job_dir=self.job)
        output = self.model.complete(
            image=image,
            visible_mask=visible,
            all_masks=[],
            text_query=query,
            enable_critic=True,
            critique_threshold=self.config["critique_threshold"],
            max_critic_iter=self.config["max_critic_iter"],
            mask_only=False,
        )
        output["_trace"] = dict(
            pad_info=self.pad_info,
            synthesis_attempts=self.attempt,
            critique_history=self.critiques,
            backend="Pix2Gestalt (strict)",
        )
        return output
