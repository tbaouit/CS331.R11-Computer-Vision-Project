"""Frequency-separated prior locking, adaptive boundary blending, and critic framing."""
from __future__ import annotations

import numpy as np

from src.vision.image_ops import binary_dilate, gaussian_blur_rgb, resize_array
from src.vision.roi import extract_square_roi, mask_bbox


def finalize_rgba(rgb: np.ndarray, amodal_mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Pack the completed RGB and the amodal mask (as alpha 0/255) into an HxWx4 uint8 image."""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = (amodal_mask * 255).astype(np.uint8)
    return rgba


def lock_low_frequency_to_prior(
    seed_rgb: np.ndarray,
    sd_rgb: np.ndarray,
    lock_mask: np.ndarray,
    max_low_shift: float = 10.0,
    high_gain: float = 0.60,
) -> np.ndarray:
    """Anchor low-frequency color/structure to `seed_rgb` while transferring crisp SD2 micro-detail.

    Prevents Stable Diffusion from hallucinating foreign objects (e.g. food inside an
    empty container or gray patches on dark clothing) wherever Pix2Gestalt already
    synthesized valid underlying object appearance.
    """
    mask = np.asarray(lock_mask, dtype=bool)
    if not mask.any():
        return sd_rgb.copy()
    seed_f = seed_rgb.astype(np.float32)
    sd_f = sd_rgb.astype(np.float32)
    seed_low = gaussian_blur_rgb(seed_rgb, sigma=4.0)
    sd_low = gaussian_blur_rgb(sd_rgb, sigma=4.0)
    sd_high = np.clip(sd_f - sd_low, -24.0, 24.0)
    low_delta = np.clip(sd_low - seed_low, -float(max_low_shift), float(max_low_shift)) * 0.35
    fused = np.clip(seed_f + low_delta + float(high_gain) * sd_high, 0.0, 255.0).astype(np.uint8)
    out = sd_rgb.copy()
    out[mask] = fused[mask]
    return out


def adaptive_seam_blend(
    original_image: np.ndarray,
    synthesized_image: np.ndarray,
    visible_mask: np.ndarray,
    amodal_mask: np.ndarray,
) -> np.ndarray:
    """Preserve exact visible pixels except a narrow 1-3 px feather along `missing_mask`."""
    vis = np.asarray(visible_mask, dtype=bool)
    amodal = np.asarray(amodal_mask, dtype=bool)
    missing = amodal & (~vis)

    result = np.zeros_like(original_image, dtype=np.uint8)
    if not amodal.any():
        return result
    if not missing.any():
        result[vis] = original_image[vis]
        return result

    vis_area = int(vis.sum())
    seam_px = int(np.clip(round(np.sqrt(max(vis_area, 1)) * 0.015), 1, 3))

    # Only pixels inside `vis` within `seam_px` of `missing` participate in feathering;
    # all other `vis` pixels remain 100% bit-exact to `original_image`.
    alpha = np.zeros(vis.shape, dtype=np.float32)
    alpha[vis] = 1.0

    for step in range(1, seam_px + 1):
        ring = vis & binary_dilate(missing, step)
        if step == 1:
            ring_only = ring
        else:
            ring_only = ring & (~binary_dilate(missing, step - 1))
        weight_orig = float(step) / float(seam_px + 1)
        alpha[ring_only] = weight_orig

    alpha_3ch = alpha[:, :, None]
    blended = np.clip(
        original_image.astype(np.float32) * alpha_3ch
        + synthesized_image.astype(np.float32) * (1.0 - alpha_3ch),
        0.0,
        255.0,
    ).astype(np.uint8)
    result[amodal] = blended[amodal]
    return result


def compose_cropped_for_critic(
    blended_rgb: np.ndarray,
    amodal_mask: np.ndarray,
    target_size: int = 448,
) -> np.ndarray:
    """Place completed object on a white studio canvas and zoom tightly to `amodal_mask`."""
    amodal = np.asarray(amodal_mask, dtype=bool)
    canvas = np.full_like(blended_rgb, 255, dtype=np.uint8)
    canvas[amodal] = blended_rgb[amodal]
    if not amodal.any():
        return resize_array(canvas, (target_size, target_size), mode="linear")

    ay1, ax1, ay2, ax2 = mask_bbox(amodal)
    ah, aw = ay2 - ay1 + 1, ax2 - ax1 + 1
    side = max(96, int(round(max(ah, aw) * 1.25)))
    roi = {
        "y0": ((ay1 + ay2) // 2) - side // 2,
        "x0": ((ax1 + ax2) // 2) - side // 2,
        "size": side,
        "is_full_frame": False,
    }
    patch = extract_square_roi(canvas, roi, pad_value=255)
    return resize_array(patch, (target_size, target_size), mode="linear")

