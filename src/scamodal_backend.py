"""Instrument the author's runtime and dispatch between baseline and improved modes."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image

os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]


class SCAmodalBackend:
    def __init__(self, config):
        import torch
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.Resampling.LANCZOS
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This full SCAmodal configuration requires CUDA. Use the Colab notebook.")
        mode = config.get("pipeline_mode", "upstream_baseline")
        if mode not in ("upstream_baseline", "improved"):
            raise ValueError(f"Unknown pipeline_mode: {mode}")
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
        if self.mode == "improved":
            from src.scamodal_improved import ImprovedAmodalCompleter
            self.improved = ImprovedAmodalCompleter(self.model, self.config)
        else:
            self.improved = None
            self.install_trace_hooks()

    def install_trace_hooks(self):
        vlm = self.cls._vlm
        reason = vlm.reason_occlusion
        def traced_reason(image, visible_mask=None):
            text = reason(image, visible_mask)
            (self.job / "reasoning.txt").write_text(text, encoding="utf-8")
            return text
        vlm.reason_occlusion = traced_reason
        critique = vlm.critique
        def traced_critique(completed_image_np, original_image_np=None):
            Image.fromarray(completed_image_np).save(self.job / f"critic_input_{self.attempt:02d}.png")
            result = critique(completed_image_np, original_image_np)
            self.critiques.append(result)
            (self.job / "critique_history.json").write_text(json.dumps(self.critiques, indent=2), encoding="utf-8")
            return result
        vlm.critique = traced_critique
        select_prompt = self.model._select_prompt
        def traced_select(*args, **kwargs):
            prompt, verification = select_prompt(*args, **kwargs)
            (self.job / "prompt_selection.json").write_text(
                json.dumps(dict(prompt=prompt, clip_verification=verification), indent=2), encoding="utf-8"
            )
            return prompt, verification
        self.model._select_prompt = traced_select
        detect = self.model._detect_outpainting
        def traced_detect(*args, **kwargs):
            self.pad_info = {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in detect(*args, **kwargs).items()
            }
            (self.job / "outpainting.json").write_text(json.dumps(self.pad_info, indent=2), encoding="utf-8")
            return self.pad_info
        self.model._detect_outpainting = traced_detect
        inpaint = self.model._inpaint_step
        def traced_inpaint(*args, **kwargs):
            self.attempt += 1
            prompt = args[2] if len(args) > 2 else kwargs["prompt"]
            (self.job / f"prompt_{self.attempt:02d}.txt").write_text(prompt, encoding="utf-8")
            result = inpaint(*args, **kwargs)
            Image.fromarray(result[0]).save(self.job / f"inpainting_{self.attempt:02d}.png")
            return result
        self.model._inpaint_step = traced_inpaint
        shape = self.cls._shape_predictor.predict_full_shape
        def traced_shape(image, visible):
            result = shape(image, visible)
            Image.fromarray(result.astype(np.uint8) * 255).save(self.job / "shape_mask_original.png")
            return result
        self.cls._shape_predictor.predict_full_shape = traced_shape

    def begin_sample(self, job):
        self.job = Path(job)
        self.pad_info, self.attempt, self.critiques = {}, 0, []
        os.environ["SCAMODAL_RUN_DIR"] = str(job)

    def complete(self, image, visible, label):
        # This call receives no amodal/hidden ground truth.
        policy = self.config.get("text_query_policy", "upstream_default")
        if policy not in ("upstream_default", "dataset_object_name"):
            raise ValueError("Unknown text_query_policy")
        query = label if policy == "dataset_object_name" else ""
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
