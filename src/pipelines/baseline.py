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
        visible_mask: np.ndarray | None,
        occluded_mask: np.ndarray | None,
        label: str = "",
        job_dir: Path | None = None,
    ) -> dict:
        started = time.perf_counter()
        policy = self.config.get("text_query_policy", "dataset_object_name")
        active_label = label if policy == "dataset_object_name" else ""
        auto_mode = (self.config.get("visible_mask_source", "gt") == "auto") or (visible_mask is None)
        if not auto_mode and occluded_mask is None:
            raise ValueError(f"{self.baseline_name} requires an input hidden mask")
        auto_meta = None
        zero_gt_auto = False
        if auto_mode:
            from src.stages.openworld_completion import predict_auto_visible_mask

            auto_seg = predict_auto_visible_mask(self.models, Image.fromarray(image), active_label or label, self.config)
            visible_mask = np.asarray(auto_seg["pred_visible_mask"], dtype=bool)
            auto_meta = {k: v for k, v in auto_seg.items() if k != "pred_visible_mask"}
            if job_dir is not None:
                Path(job_dir).mkdir(parents=True, exist_ok=True)
                (Path(job_dir) / "auto_segmentation.json").write_text(
                    json.dumps(auto_meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            if auto_seg["visible_pred_empty"]:
                h, w = image.shape[:2]
                empty_mask = np.zeros((h, w), dtype=bool)
                rgba = finalize_rgba(np.full_like(image, 255), empty_mask, h, w)
                pad = dict(needs_outpaint=False, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0)
                trace = dict(
                    pad_info=pad,
                    pipeline_mode="baseline",
                    visible_mask_source="auto",
                    visible_pred_empty=True,
                    auto_segmentation=auto_meta,
                    backend=self.baseline_name,
                    seconds=round(time.perf_counter() - started, 3),
                )
                return dict(
                    inpainted_rgba=rgba,
                    amodal_mask=empty_mask,
                    pred_visible_mask=empty_mask,
                    visible_pred_empty=True,
                    auto_segmentation=auto_meta,
                    final_score=None,
                    critic_accepted=None,
                    critique_history=[],
                    _trace=trace,
                )
            if occluded_mask is None:
                # 100% Zero-GT auto mode: build an expanded bounding-box inpaint region around pred_visible_mask
                zero_gt_auto = True
                h_img, w_img = image.shape[:2]
                y1, x1, y2, x2 = mask_bbox(visible_mask)
                bh = max(1, y2 - y1 + 1)
                bw = max(1, x2 - x1 + 1)
                pad_y = max(16, int(round(bh * 0.25)))
                pad_x = max(16, int(round(bw * 0.25)))
                ey1, ey2 = max(0, y1 - pad_y), min(h_img, y2 + 1 + pad_y)
                ex1, ex2 = max(0, x1 - pad_x), min(w_img, x2 + 1 + pad_x)
                roi_occ = np.zeros((h_img, w_img), dtype=bool)
                roi_occ[ey1:ey2, ex1:ex2] = True
                occluded_mask = roi_occ & ~visible_mask
                if not occluded_mask.any():
                    eroded = binary_erode(visible_mask, 4)
                    if eroded.any() and not np.array_equal(eroded, visible_mask):
                        occluded_mask = visible_mask & ~eroded
                        visible_mask = eroded
                    else:
                        occluded_mask = np.zeros_like(visible_mask)
                        occluded_mask[0, 0] = True
                        visible_mask = visible_mask.copy()
                        visible_mask[0, 0] = False
            else:
                occluded_mask = np.asarray(occluded_mask, dtype=bool) & ~visible_mask
        canvas, vis, occ, amodal, roi, inpaint_region = prepare_direct_inputs(image, visible_mask, occluded_mask)
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
        if zero_gt_auto:
            # Segment the completed amodal object with GroundingDINO + SAM so Baseline uses zero GT masks end-to-end
            from src.stages.openworld_completion import filter_out_amodal_segmentation, segment_with_cached_sam

            _, seg_masks, _, _ = segment_with_cached_sam(
                self.models,
                Image.fromarray(completed),
                classes=[active_label or "object"],
                target_class=active_label or "object",
                box_thresh=float(self.config.get("auto_visible_fallback_thresh", 0.15)),
                text_thresh=float(self.config.get("auto_visible_fallback_thresh", 0.15)),
            )
            if seg_masks is not None and len(seg_masks) > 0:
                _, best_seg = filter_out_amodal_segmentation(vis.astype(np.uint8), seg_masks)
                amodal = (best_seg > 0) | vis
            else:
                amodal = vis.copy()
            completed = np.where(amodal[:, :, None], completed, 255).astype(np.uint8)
        rgba = finalize_rgba(completed, amodal, *image.shape[:2])
        if job_dir is not None:
            Image.fromarray(completed).save(job / "inpainting_01.png")
        return dict(
            inpainted_rgba=rgba,
            amodal_mask=amodal,
            pred_visible_mask=vis,
            visible_pred_empty=False,
            auto_segmentation=auto_meta,
            final_score=None,
            critique_history=[],
            _trace=dict(
                backend=f"{self.baseline_name} (single-pass SD2, input hidden mask + class label)",
                visible_mask_source="auto" if auto_mode else "gt",
                visible_pred_empty=False,
                auto_segmentation=auto_meta,
                pad_info={},
                synthesis_attempts=1,
                settings=settings,
                model_load_seconds=round(model_load_seconds, 3),
                **{self.timing_key: round(inference_seconds, 3)},
                total_seconds=round(time.perf_counter() - started, 3),
            ),
        )
