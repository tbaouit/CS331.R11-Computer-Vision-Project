"""Dedicated Mask-Guided (Oracle-Mask) Amodal Appearance Completion pipeline.

When the occluded region mask (`occluded_mask` / `invisible_mask`) is provided
directly as part of the task input alongside `image` and `visible_mask`, the
whole-object geometry `amodal_mask = visible_mask | occluded_mask` is known
exactly (`Amodal IoU = 1.0`, `Hidden IoU = 1.0`).

`OracleMaskAmodalCompleter` exploits this exact geometry end-to-end instead of
predicting segmentation masks:
1. Exact-Target Square ROI (`compute_square_roi` centered on `amodal_mask` at
   `1.65x` scale) + single-pass Pix2Gestalt (`n_samples=1`, `ddim_steps=75`) used
   strictly as a 3D whole-object RGB texture/shading reference.
2. Multi-Cue Boundary + Prior Seed (`build_oracle_appearance_seed`) combining
   occluder-free Telea boundary continuation from `image[visible_mask]` with
   color-harmonized Pix2Gestalt RGB, suppressing residual foreground-occluder
   shadow smudges on enclosed light surfaces (e.g. food shadows inside white
   containers/dishes) and filling any sub-regions missed by Pix2Gestalt (e.g.
   thin handles or building foundations).
3. Silhouette-Blueprint Visual Prompting (`build_oracle_vlm_prompts`) showing
   Qwen3-VL both the stripe-censored scene context and an exact light-blue
   silhouette blueprint of the occluded part relative to `visible_mask`.
4. High-density `512x512` Exact-Mask SD2 Refinement with dual-zone frequency
   color locking, `1-2 px` adaptive seam blending, and Best-of-N Critic scoring.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import re
import time
from typing import Optional

import numpy as np
from PIL import Image

from src.scamodal_improved import (
    DEFAULT_NEGATIVE_PROMPT,
    _binary_dilate,
    _binary_erode,
    _gaussian_blur_rgb,
    _inpaint_holes,
    _mask_outline,
    _nearest_foreground_fill,
    _resize_array,
    _truncate_tokens_clean,
    adaptive_seam_blend,
    compose_cropped_for_critic,
    compute_square_roi,
    extract_square_roi,
    lock_low_frequency_to_prior,
    mask_bbox,
    parse_structured_vlm_reasoning,
    paste_square_roi,
)


def compute_appearance_coherence_metrics(
    image: np.ndarray,
    visible_mask: np.ndarray,
    occluded_mask: np.ndarray,
    completed_rgb: np.ndarray,
) -> dict[str, Optional[float]]:
    """Compute reference-free boundary continuity & color consistency on `(vis, occ)`."""
    vis = np.asarray(visible_mask, dtype=bool)
    occ = np.asarray(occluded_mask, dtype=bool) & (~vis)
    if not vis.any() or not occ.any():
        return {"seam_mae": 0.0, "color_hist_sim": 1.0}

    # 1. Local Seam Continuity MAE: compare 2-px occluded ring right across the boundary
    #    against each pixel's nearest visible neighbor in `image[vis]`.
    occ_seam_ring = occ & _binary_dilate(vis, 2)
    if occ_seam_ring.any():
        vis_canvas = np.zeros_like(image, dtype=np.uint8)
        vis_canvas[vis] = image[vis]
        nearest_vis_rgb = _nearest_foreground_fill(vis_canvas, vis)
        seam_mae = float(
            np.abs(completed_rgb[occ_seam_ring].astype(np.float32) - nearest_vis_rgb[occ_seam_ring].astype(np.float32)).mean()
        )
    else:
        seam_mae = 0.0

    # 2. Per-channel 16-bin normalized histogram intersection between visible & reconstructed occluded RGB.
    vis_px = image[vis]
    occ_px = completed_rgb[occ]
    intersections = []
    bins = np.linspace(0.0, 256.0, 17)
    for c in range(3):
        h_vis, _ = np.histogram(vis_px[:, c], bins=bins)
        h_occ, _ = np.histogram(occ_px[:, c], bins=bins)
        p_vis = h_vis.astype(np.float64) / max(float(h_vis.sum()), 1.0)
        p_occ = h_occ.astype(np.float64) / max(float(h_occ.sum()), 1.0)
        intersections.append(float(np.minimum(p_vis, p_occ).sum()))
    color_hist_sim = float(np.mean(intersections))

    return {
        "seam_mae": round(seam_mae, 3),
        "color_hist_sim": round(color_hist_sim, 4),
    }


def build_oracle_appearance_seed(
    image: np.ndarray,
    visible_mask: np.ndarray,
    occluded_mask: np.ndarray,
    p2g_rgb_full: Optional[np.ndarray] = None,
    p2g_alpha_full: Optional[np.ndarray] = None,
) -> dict:
    """Construct an edge-decontaminated appearance seed tailored to exact `occluded_mask`."""
    vis = np.asarray(visible_mask, dtype=bool)
    occ = np.asarray(occluded_mask, dtype=bool) & (~vis)
    amodal = vis | occ

    if not occ.any():
        canvas = np.full_like(image, 255, dtype=np.uint8)
        canvas[vis] = image[vis]
        return {
            "seed_canvas": canvas,
            "telea_seed": canvas.copy(),
            "unfilled_mask": np.zeros_like(vis, dtype=bool),
            "unfilled_fraction": 0.0,
            "shadow_suppressed_fraction": 0.0,
            "color_gain": [1.0, 1.0, 1.0],
            "color_bias": [0.0, 0.0, 0.0],
            "ref_p90_luma": 255.0,
        }

    # 1. Occluder-free boundary continuation seed (`telea_seed`) propagated purely from `image[vis]`.
    vis_only = np.full_like(image, 255, dtype=np.uint8)
    vis_only[vis] = image[vis]
    nearest_vis = _nearest_foreground_fill(vis_only, vis)
    telea_seed = _inpaint_holes(nearest_vis, ~vis, radius=9)

    img_f = image.astype(np.float32)
    near_occ_vis = vis & _binary_dilate(occ, 14)
    ref_mask = near_occ_vis if near_occ_vis.sum() >= 24 else vis
    ref_luma = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    vis_p90 = float(np.percentile(ref_luma[vis], 90))
    local_p90 = float(np.percentile(ref_luma[ref_mask], 90))
    ref_p90_luma = min(vis_p90, local_p90 + 15.0)

    if p2g_rgb_full is None or p2g_alpha_full is None:
        seed = np.full_like(image, 255, dtype=np.uint8)
        seed[vis] = image[vis]
        seed[occ] = telea_seed[occ]
        collar = _binary_dilate(amodal, 6)
        extrapolated = _nearest_foreground_fill(seed, amodal)
        seed[collar & (~amodal)] = extrapolated[collar & (~amodal)]
        seed[~collar] = 255
        return {
            "seed_canvas": seed,
            "telea_seed": telea_seed,
            "unfilled_mask": occ.copy(),
            "unfilled_fraction": 1.0,
            "shadow_suppressed_fraction": 0.0,
            "color_gain": [1.0, 1.0, 1.0],
            "color_bias": [0.0, 0.0, 0.0],
            "ref_p90_luma": round(ref_p90_luma, 2),
        }

    p2g_f = p2g_rgb_full.astype(np.float32)
    p2g_min = p2g_f.min(axis=-1)
    p2g_chroma = p2g_f.max(axis=-1) - p2g_min
    is_studio_white = (p2g_min >= 240.0) & (p2g_chroma <= 14.0)
    low_alpha = p2g_alpha_full < 0.52

    if ref_p90_luma < 212.0:
        unfilled_mask = occ & (low_alpha | is_studio_white)
    else:
        unfilled_mask = occ & (p2g_alpha_full < 0.32)

    valid_p2g = (p2g_alpha_full >= 0.55) & (~(is_studio_white if ref_p90_luma < 212.0 else False))
    valid_p2g_core = _binary_erode(valid_p2g & amodal, 2)

    # 2. Affine per-channel color harmonization anchored on `vis & valid_p2g_core`.
    anchor = _binary_erode(vis, 1) & valid_p2g_core
    if anchor.sum() < 25:
        anchor = vis & valid_p2g
    gain = np.ones(3, dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)
    if anchor.sum() >= 20:
        for c in range(3):
            src_vals = p2g_f[:, :, c][anchor]
            dst_vals = img_f[:, :, c][anchor]
            src_mean, dst_mean = float(src_vals.mean()), float(dst_vals.mean())
            src_std, dst_std = float(src_vals.std()), float(dst_vals.std())
            g = np.clip(dst_std / max(src_std, 8.0), 0.80, 1.25) if src_std > 3.0 else 1.0
            b = np.clip(dst_mean - g * src_mean, -28.0, 28.0)
            gain[c] = float(g)
            bias[c] = float(b)

    harmonized_f = np.clip(p2g_f * gain[None, None, :] + bias[None, None, :], 0.0, 255.0)
    telea_f = telea_seed.astype(np.float32)

    # 3. Enclosed Homogeneous Light-Surface Shadow Suppressor:
    #    When an occluder (e.g. a dark sandwich inside a white takeout container) sat inside
    #    an enclosed, uniform light-colored object, Pix2Gestalt can retain a dark tonal shadow
    #    where the occluder was. Detect uniform light bordering surfaces and anchor dark smudges
    #    smoothly back to `telea_seed`.
    has_left = np.maximum.accumulate(vis, axis=1)
    has_right = np.maximum.accumulate(vis[:, ::-1], axis=1)[:, ::-1]
    has_up = np.maximum.accumulate(vis, axis=0)
    has_down = np.maximum.accumulate(vis[::-1, :], axis=0)[::-1]
    enclosure = (
        has_left.astype(np.uint8)
        + has_right.astype(np.uint8)
        + has_up.astype(np.uint8)
        + has_down.astype(np.uint8)
    )
    enclosure_frac = float(((enclosure >= 3) & occ).sum() / max(int(occ.sum()), 1))

    border_luma = ref_luma[ref_mask]
    border_chroma = (img_f.max(axis=-1) - img_f.min(axis=-1))[ref_mask]
    border_med_luma = float(np.median(border_luma))
    border_std_luma = float(np.std(border_luma))
    border_p75_chroma = float(np.percentile(border_chroma, 75))

    shadow_suppressed_frac = 0.0
    if (
        enclosure_frac >= 0.50
        and border_med_luma >= 185.0
        and border_p75_chroma <= 32.0
        and border_std_luma <= 38.0
    ):
        harm_luma = 0.299 * harmonized_f[:, :, 0] + 0.587 * harmonized_f[:, :, 1] + 0.114 * harmonized_f[:, :, 2]
        telea_luma = 0.299 * telea_f[:, :, 0] + 0.587 * telea_f[:, :, 1] + 0.114 * telea_f[:, :, 2]
        shadow_mask = occ & (~unfilled_mask) & (
            (harm_luma < telea_luma - 20.0) | (harm_luma < border_med_luma - 30.0)
        )
        if shadow_mask.any():
            shadow_suppressed_frac = float(shadow_mask.sum() / max(int(occ.sum()), 1))
            soft_weight = _gaussian_blur_rgb(
                np.repeat((shadow_mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2), sigma=3.0
            )[:, :, 0:1] / 255.0
            blend_w = np.clip(soft_weight * 0.88, 0.0, 0.88)
            harmonized_f = harmonized_f * (1.0 - blend_w) + telea_f * blend_w

    # 4. Smooth 3-px boundary ramp from `image[vis]` (`telea_seed`) into `harmonized_f` inside `occ`.
    valid_occ_core = occ & valid_p2g_core & (~unfilled_mask)
    prior_blend_f = harmonized_f.copy()
    for step, w_telea in ((1, 0.65), (2, 0.40), (3, 0.20)):
        ring = valid_occ_core & _binary_dilate(vis, step)
        if step > 1:
            ring = ring & (~_binary_dilate(vis, step - 1))
        if ring.any():
            prior_blend_f[ring] = w_telea * telea_f[ring] + (1.0 - w_telea) * harmonized_f[ring]

    prior_blend = np.clip(prior_blend_f, 0.0, 255.0).astype(np.uint8)

    # 5. Assemble seed canvas: exact `image[vis]`, fused prior in `valid_occ_core`,
    #    Telea continuation in `unfilled_mask` & thin outer rim, and 6-px decontaminated collar.
    seed = np.full_like(image, 255, dtype=np.uint8)
    seed[valid_occ_core] = prior_blend[valid_occ_core]
    seed[vis] = image[vis]

    known_good = vis | valid_occ_core
    remaining_occ = occ & (~known_good)
    if remaining_occ.any():
        # Propagate smoothly from `vis | valid_occ_core` across any unfilled occluded regions
        # (e.g. lighthouse foundation behind sand dune or thin tennis racquet shaft).
        base_for_holes = _nearest_foreground_fill(seed, known_good)
        inpainted_holes = _inpaint_holes(base_for_holes, remaining_occ, radius=7)
        seed[remaining_occ] = inpainted_holes[remaining_occ]
        known_good = known_good | remaining_occ

    collar = _binary_dilate(amodal, 6)
    extrapolated = _nearest_foreground_fill(seed, known_good)
    seed[collar & (~amodal)] = extrapolated[collar & (~amodal)]
    seed[~collar] = 255

    unfilled_fraction = float(unfilled_mask.sum() / max(int(occ.sum()), 1))
    return {
        "seed_canvas": seed,
        "telea_seed": telea_seed,
        "unfilled_mask": unfilled_mask,
        "unfilled_fraction": round(unfilled_fraction, 4),
        "shadow_suppressed_fraction": round(shadow_suppressed_frac, 4),
        "color_gain": [round(float(x), 4) for x in gain],
        "color_bias": [round(float(x), 3) for x in bias],
        "ref_p90_luma": round(ref_p90_luma, 2),
    }


def build_oracle_vlm_prompts(
    image: np.ndarray,
    visible_mask: np.ndarray,
    occluded_mask: np.ndarray,
    target_size: int = 384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact-occlusion visual prompts for Qwen3-VL + clean isolated crop for CLIP.

    Returns:
      - `context_rgb`: zoomed scene crop where `visible_mask` is inside BRIGHT GREEN,
        the exact `occluded_mask` is censored with opaque slate stripes inside YELLOW,
        and background context is dimmed to 35%.
      - `blueprint_rgb`: whole-object square crop on pure white showing `image[visible_mask]`
        alongside a soft sky-blue blueprint silhouette of the exact `occluded_mask` so
        Qwen3-VL sees the precise shape and location of the missing object part.
      - `isolated_vis_rgb`: pure `image[visible_mask]` on white for unbiased CLIP scoring.
      - `panel_rgb`: 3-panel diagnostic strip (`context | blueprint | isolated_vis`).
    """
    vis = np.asarray(visible_mask, dtype=bool)
    occ = np.asarray(occluded_mask, dtype=bool) & (~vis)
    amodal = vis | occ

    # Panel 1: Censored scene context
    highlighted = (image.astype(np.float32) * 0.35).astype(np.uint8)
    if occ.any():
        yy, xx = np.indices(vis.shape)
        stripe = ((yy + xx) // 8) % 2 == 0
        censor = np.where(
            stripe[:, :, None],
            np.array([112, 116, 122], dtype=np.uint8),
            np.array([138, 142, 148], dtype=np.uint8),
        )
        highlighted[occ] = censor[occ]
    highlighted[vis] = image[vis]
    highlighted[_mask_outline(occ, thickness=2)] = np.array([255, 215, 0], dtype=np.uint8)
    highlighted[_mask_outline(vis, thickness=2)] = np.array([0, 255, 64], dtype=np.uint8)

    ctx_roi = compute_square_roi(amodal, image.shape, context_scale=2.0, min_crop=192, full_frame_threshold=0.92)
    ctx_patch = extract_square_roi(highlighted, ctx_roi, pad_value=48)
    context_rgb = _resize_array(ctx_patch, (target_size, target_size), mode="linear")

    # Panel 2: Exact whole-object silhouette blueprint on white
    ay1, ax1, ay2, ax2 = mask_bbox(amodal)
    ah, aw = ay2 - ay1 + 1, ax2 - ax1 + 1
    bp_side = max(64, int(round(max(ah, aw) * 1.24)))
    bp_roi = {
        "y0": ((ay1 + ay2) // 2) - bp_side // 2,
        "x0": ((ax1 + ax2) // 2) - bp_side // 2,
        "size": bp_side,
        "is_full_frame": False,
    }
    blueprint_scene = np.full_like(image, 255, dtype=np.uint8)
    if occ.any():
        blueprint_scene[occ] = np.array([224, 236, 252], dtype=np.uint8)
        occ_border = occ & (~_binary_erode(occ, 2))
        blueprint_scene[occ_border] = np.array([75, 135, 225], dtype=np.uint8)
    blueprint_scene[vis] = image[vis]
    bp_patch = extract_square_roi(blueprint_scene, bp_roi, pad_value=255)
    blueprint_rgb = _resize_array(bp_patch, (target_size, target_size), mode="linear")

    # Panel 3: Pure isolated visible pixels for CLIP verification
    vy1, vx1, vy2, vx2 = mask_bbox(vis)
    vh, vw = vy2 - vy1 + 1, vx2 - vx1 + 1
    iso_side = max(64, int(round(max(vh, vw) * 1.22)))
    iso_roi = {
        "y0": ((vy1 + vy2) // 2) - iso_side // 2,
        "x0": ((vx1 + vx2) // 2) - iso_side // 2,
        "size": iso_side,
        "is_full_frame": False,
    }
    white_vis = np.full_like(image, 255, dtype=np.uint8)
    white_vis[vis] = image[vis]
    iso_patch = extract_square_roi(white_vis, iso_roi, pad_value=255)
    isolated_vis_rgb = _resize_array(iso_patch, (target_size, target_size), mode="linear")

    sep = np.full((target_size, 6, 3), 220, dtype=np.uint8)
    panel_rgb = np.concatenate([context_rgb, sep, blueprint_rgb, sep, isolated_vis_rgb], axis=1)
    return context_rgb, blueprint_rgb, isolated_vis_rgb, panel_rgb


class OracleMaskAmodalCompleter:
    """Dedicated Mask-Guided Amodal Appearance Completer (`pipeline_mode = 'oracle_mask'`)."""

    def __init__(self, base_completer, config: Optional[dict] = None):
        self.base = base_completer
        self.cls = base_completer.__class__
        self.device = base_completer.device
        self.config = config or {}

    def _synthesize_amodal_prior(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        amodal_mask: np.ndarray,
        job_dir: Optional[Path],
    ) -> dict:
        """Run 1 fast Pix2Gestalt pass on an `amodal_mask`-centered ROI for 3D RGB reference."""
        import sys
        import torch

        predictor = self.cls._shape_predictor
        if predictor is None or predictor.model is None:
            raise RuntimeError("Strict SCAmodal: Pix2Gestalt model is not loaded.")

        sys.path.insert(0, os.path.join(os.getcwd(), "pix2gestalt", "pix2gestalt"))
        from inference import run_pix2gestalt

        roi_scale = float(self.config.get("oracle_roi_scale", 1.65))
        roi = compute_square_roi(
            amodal_mask,
            image.shape,
            context_scale=roi_scale,
            min_crop=176,
            full_frame_threshold=0.88,
        )

        roi_img = extract_square_roi(image, roi, pad_value=(127, 127, 127))
        roi_vis = extract_square_roi(visible_mask.astype(np.uint8) * 255, roi, pad_value=0)

        resized_img = _resize_array(roi_img, (256, 256), mode="area")
        resized_vis = _resize_array(roi_vis, (256, 256), mode="nearest")
        if not (resized_vis > 127).any() and roi_vis.any():
            ys, xs = np.where(roi_vis > 127)
            cy = int(round(ys.mean() * 255.0 / max(roi_vis.shape[0] - 1, 1)))
            cx = int(round(xs.mean() * 255.0 / max(roi_vis.shape[1] - 1, 1)))
            resized_vis[np.clip(cy, 0, 255), np.clip(cx, 0, 255)] = 255
        resized_vis = ((resized_vis > 127).astype(np.uint8)) * 255
        rgb_vis_256 = np.repeat(resized_vis[:, :, None], 3, axis=2)

        ddim_steps = int(self.config.get("pix2gestalt_ddim_steps", 75))
        p2g_kwargs = {
            "model": predictor.model,
            "device": predictor.device,
            "input_im": resized_img,
            "visible_mask": rgb_vis_256,
            "n_samples": 1,
        }
        try:
            if "ddim_steps" in inspect.signature(run_pix2gestalt).parameters:
                p2g_kwargs["ddim_steps"] = ddim_steps
        except Exception:
            pass

        print(
            f"[Oracle-Prior] Pix2Gestalt Amodal-ROI={roi['size']}x{roi['size']} "
            f"(full_frame={roi['is_full_frame']}, n_samples=1, ddim_steps={ddim_steps})..."
        )
        with torch.inference_mode():
            samples_rgb_256 = run_pix2gestalt(**p2g_kwargs)

        rgb_256 = samples_rgb_256[0]
        rgba_mat = predictor.carvekit_interface([Image.fromarray(rgb_256)])[0]
        alpha_256 = np.asarray(rgba_mat)[:, :, 3]

        blank_rgb = np.full_like(image, 255, dtype=np.uint8)
        blank_alpha = np.zeros(image.shape[:2], dtype=np.float32)
        p2g_rgb_full = paste_square_roi(blank_rgb, rgb_256, roi, resize_mode="linear")
        p2g_alpha_full = np.clip(
            paste_square_roi(blank_alpha, alpha_256.astype(np.float32) / 255.0, roi, resize_mode="linear"),
            0.0,
            1.0,
        )

        if job_dir is not None:
            Image.fromarray(rgb_256).save(job_dir / "pix2gestalt_rgb_256.png")
            Image.fromarray(alpha_256).save(job_dir / "carvekit_alpha_256.png")
            Image.fromarray(amodal_mask.astype(np.uint8) * 255).save(job_dir / "shape_mask_original.png")

        return {
            "p2g_rgb_full": p2g_rgb_full,
            "p2g_alpha_full": p2g_alpha_full,
            "diagnostics": {
                "mask_source": "input_occluded_mask",
                "roi": roi,
                "ddim_steps": ddim_steps,
            },
        }

    def _reason_oracle_aware(
        self,
        context_rgb: np.ndarray,
        blueprint_rgb: np.ndarray,
        user_query: str = "",
    ) -> dict[str, str]:
        import torch
        from qwen_vl_utils import process_vision_info

        vlm = self.cls._vlm
        if vlm is None or vlm.model is None:
            raise RuntimeError("Strict SCAmodal: Qwen3-VL model is not loaded.")

        hint_line = (
            f"Optional category hint: '{user_query.strip()}'.\n"
            if user_query and user_query.strip()
            else ""
        )
        instruction = (
            "You are guiding exact-mask amodal appearance completion for a partially occluded object.\n"
            "- Image 1 shows the scene where ONLY the target object's visible pixels are inside a BRIGHT GREEN "
            "outline, and the exact occluded region to restore is censored by diagonal gray stripes inside YELLOW.\n"
            "- Image 2 shows the target object isolated on pure white, with the exact missing silhouette marked in "
            "LIGHT SKY-BLUE next to the visible object pixels.\n"
            f"{hint_line}"
            "CRITICAL RULES:\n"
            "1. Identify ONLY the target object shown in Image 2 (never describe any removed foreground occluder).\n"
            "2. Use the exact LIGHT SKY-BLUE silhouette in Image 2 to see which anatomical/structural part is missing.\n"
            "3. Write PROMPT as a concise (12-26 words) studio photo description of the COMPLETE, UNOCCLUDED target "
            "object on a plain white background, matching the exact colors, materials, and lighting in Image 2.\n"
            "4. If the target object is a container, takeout box, tray, bowl, plate, or dish whose interior is "
            "marked in light blue, describe it as EMPTY and clean inside (NEVER imagine food or items inside it).\n"
            "5. Do NOT mention masks, stripes, outlines, or light-blue regions in PROMPT.\n\n"
            "Respond in this exact 3-line format:\n"
            "OBJECT: <short noun phrase naming the target object>\n"
            "OCCLUDED_PARTS: <one short sentence naming the exact missing part shown in light blue>\n"
            "PROMPT: <concise description of the complete unoccluded target object on a white background>"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(context_rgb)},
                    {"type": "image", "image": Image.fromarray(blueprint_rgb)},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = vlm.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(vlm.device)

        with torch.no_grad():
            generated_ids = vlm.model.generate(
                **inputs, max_new_tokens=140, do_sample=False, temperature=None, top_p=None, top_k=None
            )

        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw_output = vlm.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return parse_structured_vlm_reasoning(raw_output, fallback_query=user_query)

    def _verify_prompt_clip(
        self,
        isolated_vis_rgb: np.ndarray,
        parsed_vlm: dict[str, str],
        user_query: str = "",
    ) -> tuple[str, dict]:
        import torch

        clip_model = self.cls._clip_model
        clip_proc = self.cls._clip_processor
        clip_tok = clip_proc.tokenizer

        qwen_prompt = _truncate_tokens_clean(parsed_vlm["sd_prompt"], clip_tok, max_tokens=65)
        obj_phrase = _truncate_tokens_clean(
            f"a photo of {parsed_vlm['object_name']} isolated on a white background",
            clip_tok,
            max_tokens=40,
        )
        fallbacks = [
            "a person",
            "a piece of luggage or bag",
            "a container or tray",
            "a ceramic bowl or dish",
            "a glass jar or bottle",
            "a stuffed toy or teddy bear",
            "sports equipment or racket",
            "a building or lighthouse",
            "a vehicle",
            "furniture",
            "an animal",
            "an object on a white background",
        ]
        if user_query and user_query.strip():
            fallbacks.insert(0, f"a photo of {user_query.strip()}")

        candidates = [qwen_prompt, obj_phrase] + fallbacks
        text_inputs = clip_tok(
            candidates,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        if text_inputs["input_ids"].shape[1] > 77:
            text_inputs = {k: v[:, :77] for k, v in text_inputs.items()}

        crop_pil = Image.fromarray(isolated_vis_rgb)
        image_inputs = clip_proc.image_processor([crop_pil] * len(candidates), return_tensors="pt")
        clip_inputs = {
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device),
            "pixel_values": image_inputs["pixel_values"].to(self.device),
        }

        with torch.no_grad():
            outputs = clip_model(**clip_inputs)
            sims = (outputs.image_embeds[0:1] @ outputs.text_embeds.T).squeeze(0).detach().cpu().tolist()

        qwen_score = float(sims[0])
        obj_score = float(sims[1])
        fb_scores = sims[2:]
        best_fb_idx = max(range(len(fb_scores)), key=lambda i: fb_scores[i])
        best_fb = candidates[2 + best_fb_idx]
        best_fb_score = float(fb_scores[best_fb_idx])

        best_qwen_score = max(qwen_score, obj_score)
        best_qwen_text = qwen_prompt if qwen_score >= obj_score - 0.01 else obj_phrase
        verified = (best_qwen_score >= 0.18) and (best_qwen_score >= best_fb_score - 0.03)
        chosen = best_qwen_text if verified else best_fb
        status = "ACCEPTED" if verified else "REJECTED"
        print(
            f"[CLIP Verify] Qwen {status} "
            f"(prompt={qwen_score:.3f}, obj={obj_score:.3f}, best_fb='{best_fb}'={best_fb_score:.3f})"
        )
        return chosen, {
            "verified": bool(verified),
            "object_name": parsed_vlm["object_name"],
            "qwen_clip_score": round(qwen_score, 4),
            "object_clip_score": round(obj_score, 4),
            "best_fallback": best_fb,
            "best_fallback_score": round(best_fb_score, 4),
            "chosen": chosen,
        }

    def _refine_exact_mask_sd2(
        self,
        seed_canvas: np.ndarray,
        visible_mask: np.ndarray,
        occluded_mask: np.ndarray,
        unfilled_mask: np.ndarray,
        prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        ref_p90_luma: float,
    ) -> np.ndarray:
        """High-density 512x512 ROI SD2 refinement inside exact `occluded_mask`."""
        import torch

        pipe = self.cls._pipe
        vis = np.asarray(visible_mask, dtype=bool)
        occ = np.asarray(occluded_mask, dtype=bool) & (~vis)
        amodal = vis | occ
        unfilled = np.asarray(unfilled_mask, dtype=bool) & occ
        if not occ.any():
            return seed_canvas.copy()

        ay1, ax1, ay2, ax2 = mask_bbox(amodal)
        ah, aw = ay2 - ay1 + 1, ax2 - ax1 + 1
        roi_side = max(128, int(round(max(ah, aw) * 1.28)))
        roi = {
            "y0": ((ay1 + ay2) // 2) - roi_side // 2,
            "x0": ((ax1 + ax2) // 2) - roi_side // 2,
            "size": roi_side,
            "is_full_frame": False,
        }

        protected_vis = _binary_erode(vis, 2)
        inpaint_region = _binary_dilate(occ, 2) & _binary_dilate(amodal, 2) & (~protected_vis)
        if not inpaint_region.any():
            inpaint_region = occ.copy()

        roi_img = extract_square_roi(seed_canvas, roi, pad_value=255)
        roi_mask = extract_square_roi(inpaint_region.astype(np.uint8) * 255, roi, pad_value=0)

        pil_img_512 = Image.fromarray(roi_img).resize((512, 512), Image.Resampling.LANCZOS)
        pil_mask_512 = Image.fromarray(roi_mask).resize((512, 512), Image.Resampling.NEAREST)

        eff_strength = float(np.clip(strength, 0.16, 1.0))
        eff_steps = max(int(num_inference_steps), int(np.ceil(10.0 / eff_strength)))

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.inference_mode():
            out_pil_512 = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_img_512,
                mask_image=pil_mask_512,
                strength=eff_strength,
                num_inference_steps=eff_steps,
                guidance_scale=float(guidance_scale),
                height=512,
                width=512,
                generator=generator,
            ).images[0]

        out_roi_np = np.asarray(out_pil_512.resize((roi_side, roi_side), Image.Resampling.LANCZOS))
        synthesized_full = paste_square_roi(seed_canvas, out_roi_np, roi, resize_mode="linear")

        out = seed_canvas.copy()
        out[occ] = synthesized_full[occ]

        # Dual-Zone Frequency Lock:
        # - Zone A (`valid_prior_zone`): Pix2Gestalt already produced valid structure & color ->
        #   tightly lock low-frequency color (`max_low_shift=10.0`) and transfer crisp SD2 detail.
        # - Zone B (`unfilled_zone`): Pix2Gestalt missed this sub-region (filled by `telea_seed`) ->
        #   allow richer SD2 structural/texture synthesis (`max_low_shift=26.0`, `high_gain=0.90`)
        #   while still preventing alien object hallucinations!
        unfilled_zone = _binary_dilate(unfilled, 3) & occ
        valid_prior_zone = occ & (~unfilled_zone)
        if valid_prior_zone.any():
            out = lock_low_frequency_to_prior(
                seed_rgb=seed_canvas,
                sd_rgb=out,
                lock_mask=valid_prior_zone,
                max_low_shift=10.0,
                high_gain=0.58,
            )
        if unfilled_zone.any():
            out = lock_low_frequency_to_prior(
                seed_rgb=seed_canvas,
                sd_rgb=out,
                lock_mask=unfilled_zone,
                max_low_shift=26.0,
                high_gain=0.90,
            )

        if ref_p90_luma < 212.0:
            outer_rim = occ & (~_binary_erode(amodal, 4))
            out_f = out.astype(np.float32)
            white_bleed = outer_rim & (out_f.min(axis=-1) >= 232.0) & ((out_f.max(axis=-1) - out_f.min(axis=-1)) <= 16.0)
            if white_bleed.any():
                out[white_bleed] = seed_canvas[white_bleed]

        return out

    def _critique_target_aware(
        self,
        critic_completed_rgb: np.ndarray,
        context_highlight_rgb: np.ndarray,
        object_name: str,
    ) -> dict:
        import torch
        from qwen_vl_utils import process_vision_info

        vlm = self.cls._vlm
        if vlm is None or vlm.model is None:
            raise RuntimeError("Strict SCAmodal: Qwen3-VL model is not loaded.")

        instruction = (
            f"You are a strict quality critic for mask-guided amodal appearance completion of a '{object_name}'.\n"
            "- Image 1 shows the reconstructed complete target object isolated on a white background.\n"
            "- Image 2 shows the scene where the target's visible region is inside BRIGHT GREEN and the removed "
            "foreground occluder is covered by a diagonal striped censor mask inside YELLOW.\n\n"
            "Evaluate ONLY the reconstructed target object in Image 1 on a 0-10 integer scale:\n"
            "1. STRUCTURAL: internal anatomical/structural continuity across the reconstructed region.\n"
            "2. TEXTURE: seamless color, material, and lighting transition between visible and restored parts (penalize seams, gray smudges, or hallucinated items).\n"
            "3. CONTEXT: semantic fidelity to the BRIGHT GREEN target object in Image 2 (note: if the target is a container/tray/dish, a clean EMPTY interior is 10/10 correct).\n\n"
            "Respond EXACTLY in this 4-line format:\n"
            "STRUCTURAL: <int>\n"
            "TEXTURE: <int>\n"
            "CONTEXT: <int>\n"
            "FEEDBACK: <one short actionable fix for texture/color continuity, or 'good' if seamless>"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(critic_completed_rgb)},
                    {"type": "image", "image": Image.fromarray(context_highlight_rgb)},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = vlm.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(vlm.device)

        with torch.no_grad():
            generated_ids = vlm.model.generate(
                **inputs, max_new_tokens=120, do_sample=False, temperature=None, top_p=None, top_k=None
            )

        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = vlm.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        cleaned = re.sub(r"\*+", "", output_text)

        def _parse_score(label: str) -> float:
            match = re.search(rf"{label}\s*:\s*([\d.]+)", cleaned, re.IGNORECASE)
            if not match:
                raise ValueError(f"Critic response missing {label}: {output_text}")
            val = float(match.group(1))
            if not 0.0 <= val <= 10.0:
                raise ValueError(f"Critic score out of range: {label}={val}")
            return val

        structural = _parse_score("STRUCTURAL")
        texture = _parse_score("TEXTURE")
        context = _parse_score("CONTEXT")
        fb_match = re.search(r"FEEDBACK\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE | re.DOTALL)
        feedback = fb_match.group(1).strip() if fb_match else ""

        return {
            "score": (structural + texture + context) / 3.0,
            "structural": structural,
            "texture": texture,
            "context": context,
            "feedback": feedback,
            "raw": output_text,
        }

    def complete(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        occluded_mask: np.ndarray,
        text_query: str = "",
        job_dir: Optional[Path] = None,
    ) -> dict:
        import torch

        if occluded_mask is None:
            raise ValueError("OracleMaskAmodalCompleter requires an explicit `occluded_mask` input.")

        t0 = time.perf_counter()
        h, w = image.shape[:2]
        vis = np.asarray(visible_mask, dtype=bool)
        occ = np.asarray(occluded_mask, dtype=bool) & (~vis)
        if occ.shape != vis.shape:
            raise ValueError(f"occluded_mask shape {occ.shape} != visible_mask shape {vis.shape}")
        amodal_mask = vis | occ

        pad_info = {
            "needs_outpaint": False,
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
        }
        if job_dir is not None:
            (job_dir / "outpainting.json").write_text(json.dumps(pad_info, indent=2), encoding="utf-8")

        if not occ.any():
            rgba = self.base._finalize_rgba(image, amodal_mask, h, w)
            if job_dir is not None:
                (job_dir / "reasoning.txt").write_text("Empty occluded_mask input; nothing to restore.", encoding="utf-8")
                (job_dir / "critique_history.json").write_text("[]", encoding="utf-8")
            return {
                "input_image": image,
                "visible_mask": vis,
                "amodal_mask": amodal_mask,
                "inpainted_rgba": rgba,
                "vlm_reasoning": "Empty occluded_mask input; nothing to restore.",
                "critique_history": [],
                "final_score": None,
                "_trace": {
                    "pad_info": pad_info,
                    "synthesis_attempts": 0,
                    "critique_history": [],
                    "backend": "SCAmodal Oracle-Mask (Exact Input Occluded Mask)",
                },
            }

        # 1. Single-pass 3D whole-object appearance prior on exact amodal ROI.
        t_prior0 = time.perf_counter()
        prior_raw = self._synthesize_amodal_prior(image, vis, amodal_mask, job_dir)
        prior_info = build_oracle_appearance_seed(
            image=image,
            visible_mask=vis,
            occluded_mask=occ,
            p2g_rgb_full=prior_raw["p2g_rgb_full"],
            p2g_alpha_full=prior_raw["p2g_alpha_full"],
        )
        t_prior = time.perf_counter() - t_prior0

        seed_canvas = prior_info["seed_canvas"]
        unfilled_mask = prior_info["unfilled_mask"]
        unfilled_frac = float(prior_info["unfilled_fraction"])
        if job_dir is not None:
            Image.fromarray(seed_canvas).save(job_dir / "prior_rgb.png")

        # 2. Exact-blueprint Qwen3-VL reasoning + unbiased CLIP verification.
        t_vlm0 = time.perf_counter()
        print("[Oracle-Branch A] Exact-blueprint semantic reasoning (Qwen3-VL)...")
        context_rgb, blueprint_rgb, isolated_vis_rgb, panel_rgb = build_oracle_vlm_prompts(
            image, vis, occ, target_size=384
        )
        if job_dir is not None:
            Image.fromarray(panel_rgb).save(job_dir / "vlm_visual_prompt.jpg")

        parsed_vlm = self._reason_oracle_aware(context_rgb, blueprint_rgb, user_query=text_query)
        print(
            f"[Oracle VLM] OBJECT='{parsed_vlm['object_name']}' | "
            f"MISSING='{parsed_vlm['occluded_parts']}' | PROMPT='{parsed_vlm['sd_prompt']}'"
        )
        if job_dir is not None:
            (job_dir / "reasoning.txt").write_text(parsed_vlm["raw"], encoding="utf-8")

        base_prompt, clip_info = self._verify_prompt_clip(isolated_vis_rgb, parsed_vlm, user_query=text_query)
        t_vlm = time.perf_counter() - t_vlm0
        if job_dir is not None:
            (job_dir / "prompt_selection.json").write_text(
                json.dumps({"prompt": base_prompt, "parsed_vlm": parsed_vlm, "clip_verification": clip_info}, indent=2),
                encoding="utf-8",
            )

        # 3. Exact-mask 512x512 ROI SD2 refinement + Best-of-N Critic selection.
        t_syn0 = time.perf_counter()
        sd_tok = self.cls._pipe.tokenizer
        prompt = _truncate_tokens_clean(
            f"{base_prompt}, studio object photography on plain white background, sharp texture, natural colors",
            sd_tok,
            max_tokens=70,
        )
        obj_lower = f"{parsed_vlm['object_name']} {base_prompt}".lower()
        extra_neg = (
            ", food, sandwich, meat, onions, toppings, filled container"
            if any(k in obj_lower for k in ("container", "tray", "takeout", "box", "bowl", "plate", "dish"))
            else ""
        )
        neg_prompt = _truncate_tokens_clean(DEFAULT_NEGATIVE_PROMPT + extra_neg, sd_tok, max_tokens=70)

        base_strength = (
            float(self.config.get("unfilled_refine_strength", 0.62))
            if unfilled_frac >= 0.25
            else float(self.config.get("prior_refine_strength", 0.24))
        )
        strength_step = 0.12 if unfilled_frac >= 0.25 else 0.08
        guidance_scale = 6.0 if unfilled_frac < 0.25 else 7.0
        num_steps = 28
        base_seed = int(self.config.get("seed", 42))
        enable_critic = bool(self.config.get("enable_critic", True))
        critique_threshold = float(self.config.get("critique_threshold", 7.0))
        max_critic_iter = int(self.config.get("max_critic_iter", 2))
        n_iters = max_critic_iter if enable_critic else 1

        critique_history: list[dict] = []
        best_rgba: Optional[np.ndarray] = None
        best_rgb: Optional[np.ndarray] = None
        best_score: float = -1.0
        attempts = 0

        for crit_iter in range(n_iters):
            attempts += 1
            iter_strength = min(base_strength + strength_step * crit_iter, 0.88)
            print(
                f"[Oracle-Synthesis] Attempt {crit_iter + 1}/{n_iters} "
                f"(strength={iter_strength:.2f}, unfilled={unfilled_frac:.2f}, guidance={guidance_scale:.1f})"
            )
            if job_dir is not None:
                (job_dir / f"prompt_{attempts:02d}.txt").write_text(prompt, encoding="utf-8")

            inpaint_img = self._refine_exact_mask_sd2(
                seed_canvas=seed_canvas,
                visible_mask=vis,
                occluded_mask=occ,
                unfilled_mask=unfilled_mask,
                prompt=prompt,
                negative_prompt=neg_prompt,
                strength=iter_strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                seed=base_seed + crit_iter * 1009,
                ref_p90_luma=float(prior_info["ref_p90_luma"]),
            )
            if job_dir is not None:
                Image.fromarray(inpaint_img).save(job_dir / f"inpainting_{attempts:02d}.png")

            blended_rgb = adaptive_seam_blend(image, inpaint_img, vis, amodal_mask)
            rgba = self.base._finalize_rgba(blended_rgb, amodal_mask, h, w)

            if not enable_critic:
                best_rgba = rgba
                best_rgb = blended_rgb
                break

            critic_crop = compose_cropped_for_critic(blended_rgb, amodal_mask, target_size=384)
            if job_dir is not None:
                Image.fromarray(critic_crop).save(job_dir / f"critic_input_{attempts:02d}.png")

            critique = self._critique_target_aware(
                critic_completed_rgb=critic_crop,
                context_highlight_rgb=context_rgb,
                object_name=parsed_vlm["object_name"],
            )
            critique_history.append(critique)
            if job_dir is not None:
                (job_dir / "critique_history.json").write_text(
                    json.dumps(critique_history, indent=2), encoding="utf-8"
                )

            print(
                f"[Critic iter {crit_iter}] score={critique['score']:.2f} "
                f"(struct={critique['structural']}, tex={critique['texture']}, ctx={critique['context']}) "
                f"-- {critique['feedback']}"
            )

            if critique["score"] >= best_score:
                best_score = float(critique["score"])
                best_rgba = rgba
                best_rgb = blended_rgb

            if critique["score"] >= critique_threshold:
                print(f"[Critic] Accepted at iter {crit_iter} (score {critique['score']:.2f} >= {critique_threshold}).")
                break

            fb_clean = critique["feedback"]
            if fb_clean and fb_clean.lower() != "good":
                prompt = _truncate_tokens_clean(
                    f"{base_prompt}, {fb_clean}, coherent texture and color continuity, plain white background",
                    sd_tok,
                    max_tokens=70,
                )
            guidance_scale = min(guidance_scale + 1.0, 9.0)

        coherence = compute_appearance_coherence_metrics(image, vis, occ, best_rgb if best_rgb is not None else image)
        t_syn = time.perf_counter() - t_syn0
        t_total = time.perf_counter() - t0
        print(
            f"[Timing] prior={t_prior:.1f}s | vlm={t_vlm:.1f}s | sd2+critic={t_syn:.1f}s | total={t_total:.1f}s "
            f"(seam_mae={coherence['seam_mae']}, color_sim={coherence['color_hist_sim']})",
            flush=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_score = best_score if critique_history else None
        return {
            "input_image": image,
            "visible_mask": vis,
            "amodal_mask": amodal_mask,
            "inpainted_rgba": best_rgba,
            "vlm_reasoning": parsed_vlm["raw"],
            "clip_verification": clip_info,
            "critique_history": critique_history,
            "final_score": final_score,
            "_trace": {
                "pad_info": pad_info,
                "synthesis_attempts": attempts,
                "critique_history": critique_history,
                "shape_diagnostics": prior_raw["diagnostics"],
                "prior_diagnostics": {
                    "unfilled_fraction": prior_info["unfilled_fraction"],
                    "shadow_suppressed_fraction": prior_info["shadow_suppressed_fraction"],
                    "color_gain": prior_info["color_gain"],
                    "color_bias": prior_info["color_bias"],
                    "ref_p90_luma": prior_info["ref_p90_luma"],
                    "seam_mae": coherence["seam_mae"],
                    "color_hist_sim": coherence["color_hist_sim"],
                },
                "backend": "SCAmodal Oracle-Mask (Exact Input Occluded Mask)",
            },
        }
