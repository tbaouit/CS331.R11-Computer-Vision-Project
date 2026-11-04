"""Appearance seed construction, color harmonization, and boundary continuation."""
from __future__ import annotations

from typing import Optional
import numpy as np

from src.vision.image_ops import (
    binary_dilate,
    binary_erode,
    gaussian_blur_rgb,
    inpaint_holes,
    nearest_foreground_fill,
)


def build_appearance_prior(
    image: np.ndarray,
    visible_mask: np.ndarray,
    amodal_mask: np.ndarray,
    p2g_rgb_full: np.ndarray,
    p2g_alpha_full: np.ndarray,
) -> dict:
    """Build a color-harmonized, edge-decontaminated RGB seed canvas (blind mode).

    - Detects unfilled studio-white dropout pixels inside `missing_mask` when
      the adjacent visible object is non-white (e.g. white microphone hole in a
      dark brown suit), while keeping genuinely white objects (e.g. white
      styrofoam container) intact.
    - Matches per-channel gain/bias of `p2g_rgb_full` to `image` over `visible_mask`.
    - Pre-fills dropout holes with Telea inpainting and decontaminates the outer
      foreground edge + a 6-px collar so SD2 never bleeds background white/gray
      inward across the object boundary.
    """
    vis = np.asarray(visible_mask, dtype=bool)
    amodal = np.asarray(amodal_mask, dtype=bool)
    missing = amodal & (~vis)

    if not missing.any():
        canvas = np.full_like(image, 255, dtype=np.uint8)
        canvas[vis] = image[vis]
        return {
            "seed_canvas": canvas,
            "dropout_mask": np.zeros_like(vis, dtype=bool),
            "dropout_fraction": 0.0,
            "color_gain": [1.0, 1.0, 1.0],
            "color_bias": [0.0, 0.0, 0.0],
            "ref_p90_luma": 255.0,
        }

    img_f = image.astype(np.float32)
    p2g_f = p2g_rgb_full.astype(np.float32)

    near_missing_vis = vis & binary_dilate(missing, 16)
    ref_mask = near_missing_vis if near_missing_vis.sum() >= 32 else vis
    ref_luma = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    vis_p90 = float(np.percentile(ref_luma[vis], 90))
    local_p90 = float(np.percentile(ref_luma[ref_mask], 90))
    ref_p90_luma = min(vis_p90, local_p90 + 15.0)

    p2g_min = p2g_f.min(axis=-1)
    p2g_chroma = p2g_f.max(axis=-1) - p2g_min
    is_studio_white = (p2g_min >= 240.0) & (p2g_chroma <= 14.0)
    low_alpha = p2g_alpha_full < 0.52

    if ref_p90_luma < 212.0:
        dropout_mask = missing & (low_alpha | is_studio_white)
    else:
        dropout_mask = missing & (p2g_alpha_full < 0.30)

    valid_p2g = (p2g_alpha_full >= 0.55) & (~(is_studio_white if ref_p90_luma < 212.0 else False))
    valid_p2g_core = binary_erode(valid_p2g & amodal, 2)
    anchor = binary_erode(vis, 1) & valid_p2g_core
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

    harmonized = np.clip(p2g_f * gain[None, None, :] + bias[None, None, :], 0.0, 255.0).astype(np.uint8)

    seed = np.full_like(image, 255, dtype=np.uint8)
    valid_missing_core = missing & valid_p2g_core & (~dropout_mask)
    seed[valid_missing_core] = harmonized[valid_missing_core]
    seed[vis] = image[vis]

    known_good = vis | valid_missing_core
    if not known_good.any():
        known_good = vis.copy()

    interior_holes = amodal & (~known_good) & (~binary_dilate(~amodal, 3))
    if interior_holes.any():
        pre_extrap = nearest_foreground_fill(seed, known_good)
        inpainted = inpaint_holes(pre_extrap, interior_holes, radius=7)
        seed[interior_holes] = inpainted[interior_holes]
        known_good = known_good | interior_holes

    collar = binary_dilate(amodal, 6)
    extrapolated = nearest_foreground_fill(seed, known_good)
    rim_and_collar = collar & (~known_good)
    seed[rim_and_collar] = extrapolated[rim_and_collar]
    seed[~collar] = 255

    dropout_fraction = float(dropout_mask.sum() / max(int(missing.sum()), 1))
    return {
        "seed_canvas": seed,
        "HarmonizedPriorRGB": harmonized,
        "dropout_mask": dropout_mask,
        "dropout_fraction": round(dropout_fraction, 4),
        "color_gain": [round(float(x), 4) for x in gain],
        "color_bias": [round(float(x), 3) for x in bias],
        "ref_p90_luma": round(ref_p90_luma, 2),
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
    nearest_vis = nearest_foreground_fill(vis_only, vis)
    telea_seed = inpaint_holes(nearest_vis, ~vis, radius=9)

    img_f = image.astype(np.float32)
    near_occ_vis = vis & binary_dilate(occ, 14)
    ref_mask = near_occ_vis if near_occ_vis.sum() >= 24 else vis
    ref_luma = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    vis_p90 = float(np.percentile(ref_luma[vis], 90))
    local_p90 = float(np.percentile(ref_luma[ref_mask], 90))
    ref_p90_luma = min(vis_p90, local_p90 + 15.0)

    if p2g_rgb_full is None or p2g_alpha_full is None:
        seed = np.full_like(image, 255, dtype=np.uint8)
        seed[vis] = image[vis]
        seed[occ] = telea_seed[occ]
        collar = binary_dilate(amodal, 6)
        extrapolated = nearest_foreground_fill(seed, amodal)
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
    valid_p2g_core = binary_erode(valid_p2g & amodal, 2)

    # 2. Affine per-channel color harmonization anchored on `vis & valid_p2g_core`.
    anchor = binary_erode(vis, 1) & valid_p2g_core
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

    # 3. Enclosed Homogeneous Light-Surface Shadow Suppressor.
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
            soft_weight = gaussian_blur_rgb(
                np.repeat((shadow_mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2), sigma=3.0
            )[:, :, 0:1] / 255.0
            blend_w = np.clip(soft_weight * 0.88, 0.0, 0.88)
            harmonized_f = harmonized_f * (1.0 - blend_w) + telea_f * blend_w

    # 4. Smooth 3-px boundary ramp from `image[vis]` (`telea_seed`) into `harmonized_f` inside `occ`.
    valid_occ_core = occ & valid_p2g_core & (~unfilled_mask)
    prior_blend_f = harmonized_f.copy()
    for step, w_telea in ((1, 0.65), (2, 0.40), (3, 0.20)):
        ring = valid_occ_core & binary_dilate(vis, step)
        if step > 1:
            ring = ring & (~binary_dilate(vis, step - 1))
        if ring.any():
            prior_blend_f[ring] = w_telea * telea_f[ring] + (1.0 - w_telea) * harmonized_f[ring]

    prior_blend = np.clip(prior_blend_f, 0.0, 255.0).astype(np.uint8)

    # 5. Assemble seed canvas + fill unfilled regions + decontaminate 6-px outer collar.
    seed = np.full_like(image, 255, dtype=np.uint8)
    seed[valid_occ_core] = prior_blend[valid_occ_core]
    seed[vis] = image[vis]

    known_good = vis | valid_occ_core
    remaining_occ = occ & (~known_good)
    if remaining_occ.any():
        base_for_holes = nearest_foreground_fill(seed, known_good)
        inpainted_holes = inpaint_holes(base_for_holes, remaining_occ, radius=7)
        seed[remaining_occ] = inpainted_holes[remaining_occ]
        known_good = known_good | remaining_occ

    collar = binary_dilate(amodal, 6)
    extrapolated = nearest_foreground_fill(seed, known_good)
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
