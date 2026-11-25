# Phạm vi: Chỉ Baseline: SD2 Inpainting một lượt trên occluded_mask với tên lớp COCO-A (dataset_object_name).
"""Single-pass SD2 Inpainting baseline with class-name prompt and direct decoder output."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.openworld_models import ALLOWED_SD2_MODELS
from src.stages.inpaint_refiner import run_inpaint_roi_pass
from src.vision.compositing import finalize_rgba
from src.vision.image_ops import binary_dilate, binary_erode
from src.vision.roi import mask_bbox

DEFAULT_NEGATIVE_PROMPT = (
    "gray background, flat gray patch, gray fill, occluded, cut off, cropped, "
    "blurry, low quality, distorted, deformed, extra limbs, floating head, "
    "text, watermark, artifacts"
)


def build_baseline_prompt(label: str = "") -> str:
    """Build a clean class-conditioned prompt matching `proposed` input information."""
    clean_label = (label or "").strip() or "object"
    return f"a photo of a {clean_label}, complete object on a plain white background, natural colors, realistic texture"


def validate_baseline_config(config: dict) -> None:
    """Validate that `baseline` uses SD2 Inpainting (matching `proposed`) in single-pass mode."""
    if config.get("pipeline_mode") != "baseline":
        raise ValueError("Expected pipeline_mode='baseline'")
    model_id = config.get("inpainting_model_id")
    if model_id not in ALLOWED_SD2_MODELS:
        raise ValueError(f"Baseline requires SD2 Inpainting ({ALLOWED_SD2_MODELS}), got {model_id!r}")
    policy = config.get("text_query_policy", "dataset_object_name")
    if policy not in ("dataset_object_name", "empty"):
        raise ValueError("text_query_policy must be 'dataset_object_name' or 'empty'")
    if config.get("enable_critic"):
        raise ValueError("Baseline runs single-pass without a critic")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")


def prepare_direct_inputs(image: np.ndarray, visible_mask: np.ndarray, occluded_mask: np.ndarray):
    """Build white-background canvas with observed visible pixels and neutral-gray hidden mask."""
    vis = np.asarray(visible_mask, dtype=bool)
    occ = np.asarray(occluded_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Direct inpainting requires uint8 HxWx3 RGB")
    if vis.shape != image.shape[:2] or occ.shape != vis.shape or not vis.any():
        raise ValueError("Invalid/empty visible mask or mismatched hidden mask")
    if (vis & occ).any():
        raise ValueError("Visible and hidden masks must be disjoint")
    amodal = vis | occ
    # Never expose occluder pixels to the baseline, even when strength < 1.
    canvas = np.full_like(image, 255)
    canvas[vis] = image[vis]
    canvas[occ] = 127
    y1, x1, y2, x2 = mask_bbox(amodal)
    side = max(128, int(round(max(y2 - y1 + 1, x2 - x1 + 1) * 1.28)))
    roi = dict(
        y0=((y1 + y2) // 2) - side // 2,
        x0=((x1 + x2) // 2) - side // 2,
        size=side,
        is_full_frame=False,
    )
    inpaint_region = binary_dilate(occ, 2) & binary_dilate(amodal, 2) & ~binary_erode(vis, 2)
    if not inpaint_region.any():
        inpaint_region = occ.copy()
    return canvas, vis, occ, amodal, roi, inpaint_region


class Baseline:
    """Single-pass Stable Diffusion 2 Inpainting baseline on the given `occluded_mask`."""

    baseline_name = "SD2 Direct"
    settings_filename = "baseline_settings.json"
    timing_key = "sd2_seconds"

    def __init__(self, models, config: dict):
        validate_baseline_config(config)
        self.models = models
        self.config = config
        self.strength = float(config.get("strength", 1.0))
        self.guidance = float(config.get("guidance_scale", 7.5))
        self.steps = int(config.get("num_inference_steps", 50))
        if not 0.16 <= self.strength <= 1.0 or self.steps < 1 or self.guidance < 0:
            raise ValueError("Invalid direct inpainting settings")

    def complete(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        occluded_mask: np.ndarray,
        label: str = "",
        job_dir: Path | None = None,
    ) -> dict:
        if occluded_mask is None:
            raise ValueError(f"{self.baseline_name} requires an input hidden mask")
        started = time.perf_counter()
        canvas, vis, occ, amodal, roi, inpaint_region = prepare_direct_inputs(image, visible_mask, occluded_mask)
        policy = self.config.get("text_query_policy", "dataset_object_name")
        active_label = label if policy == "dataset_object_name" else ""
        prompt = build_baseline_prompt(active_label)
        negative = self.config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        seed = int(self.config.get("seed", 42))
        settings = dict(
            prompt=prompt,
            negative_prompt=negative,
            label=active_label or "object",
            text_query_policy=policy,
            seed=seed,
            strength=self.strength,
            guidance_scale=self.guidance,
            num_inference_steps=self.steps,
            resolution=int(self.config.get("sd_target_size", 512)),
            roi=roi,
            category_annotation_used=(policy == "dataset_object_name"),
        )
        if job_dir is not None:
            job = Path(job_dir)
            Image.fromarray(canvas).save(job / "baseline_input.png")
            Image.fromarray(inpaint_region.astype(np.uint8) * 255).save(job / "inpainting_mask.png")
            (job / "prompt_01.txt").write_text(prompt, encoding="utf-8")
            (job / self.settings_filename).write_text(json.dumps(settings, indent=2), encoding="utf-8")

        load_started = time.perf_counter()
        pipe = self.models.pipe  # The only model this baseline activates (SD2 Inpainting FP16).
        model_load_seconds = time.perf_counter() - load_started
        print(
            f"[{self.baseline_name}] One pass, ROI={roi['size']}, steps={self.steps}, prompt='{prompt}'",
            flush=True,
        )
        inference_started = time.perf_counter()
        completed = run_inpaint_roi_pass(
            pipe=pipe,
            device=self.models.device,
            seed_canvas=canvas,
            inpaint_region=inpaint_region,
            roi=roi,
            prompt=prompt,
            negative_prompt=negative,
            strength=self.strength,
            guidance_scale=self.guidance,
            num_inference_steps=self.steps,
            seed=seed,
        )
        inference_seconds = time.perf_counter() - inference_started
        rgba = finalize_rgba(completed, amodal, *image.shape[:2])
        if job_dir is not None:
            Image.fromarray(completed).save(job / "inpainting_01.png")
        return dict(
            inpainted_rgba=rgba,
            amodal_mask=amodal,
            final_score=None,
            critique_history=[],
            _trace=dict(
                backend=f"{self.baseline_name} (single-pass SD2, input hidden mask + class label)",
                pad_info={},
                synthesis_attempts=1,
                settings=settings,
                model_load_seconds=round(model_load_seconds, 3),
                **{self.timing_key: round(inference_seconds, 3)},
                total_seconds=round(time.perf_counter() - started, 3),
            ),
        )
