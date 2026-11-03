"""Improved training-free Open-World Amodal Appearance Completion pipeline.

Addresses four failure modes observed in the upstream SCAmodal baseline while
preserving strict zero-shot evaluation boundaries (only RGB `image` and modal
`visible_mask` enter the backend; ground-truth amodal/hidden masks are never
read here) and keeping `upstream/` + `runtime/` byte-identical:

1. Context-Aware Square ROI + Multi-Sample Pix2Gestalt Consensus + Smooth Alpha
   Matting (`alpha >= 0.48` after bilinear upscaling instead of `alpha > 0`
   nearest-neighbor, plus connected-component anchoring and outlier filtering).
2. Mask-Guided Visual Prompting for Qwen3-VL (target highlighted in green,
   hidden extent outlined in yellow, occluders dimmed + isolated visible target
   on a square white studio canvas) with structured `OBJECT / PROMPT` parsing
   and square-padded CLIP verification.
3. Hybrid Pix2Gestalt-Prior + 512x512 ROI Stable Diffusion v2 Refinement on a
   clean white studio canvas (replacing flat `[127, 127, 127]` gray background
   bleed and border outpainting ballooning, with affine color harmonization,
   white-dropout Telea pre-fill, edge color decontamination, and adaptive
   1-3 px interface seam blending).
4. Target-Aware Cropped ROI Semantic Critic with best-of-N selection across
   refinement attempts.
"""
from __future__ import annotations

from collections import deque
import inspect
import json
import os
from pathlib import Path
import re
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - local CPU data venv fallback
    cv2 = None

try:
    from scipy import ndimage as ndi  # type: ignore
except ImportError:  # pragma: no cover - local CPU data venv fallback
    ndi = None


DEFAULT_NEGATIVE_PROMPT = (
    "gray background, flat gray patch, gray fill, occluded, cut off, cropped, "
    "blurry, low quality, distorted, deformed, extra limbs, floating head, "
    "text, watermark, artifacts"
)


# ── Low-level morphology & resizing helpers (cv2/scipy + PIL/NumPy fallback) ──


def _resize_array(array: np.ndarray, size_hw: tuple[int, int], mode: str = "linear") -> np.ndarray:
    """Resize a 2D or 3D uint8/float32 array to `(height, width)`."""
    target_h, target_w = int(size_hw[0]), int(size_hw[1])
    if array.shape[:2] == (target_h, target_w):
        return array.copy()

    if cv2 is not None:
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        interp = interp_map.get(mode, cv2.INTER_LINEAR)
        return cv2.resize(array, (target_w, target_h), interpolation=interp)

    resample_map = {
        "nearest": Image.Resampling.NEAREST,
        "linear": Image.Resampling.BILINEAR,
        "cubic": Image.Resampling.BICUBIC,
        "area": Image.Resampling.BOX,
    }
    resample = resample_map.get(mode, Image.Resampling.BILINEAR)
    if array.dtype == np.uint8:
        return np.asarray(Image.fromarray(array).resize((target_w, target_h), resample=resample))

    arr_f = array.astype(np.float32)
    if arr_f.ndim == 2:
        pil_f = Image.fromarray(arr_f, mode="F").resize((target_w, target_h), resample=resample)
        return np.asarray(pil_f, dtype=np.float32)

    channels = [
        np.asarray(
            Image.fromarray(arr_f[:, :, c], mode="F").resize((target_w, target_h), resample=resample),
            dtype=np.float32,
        )
        for c in range(arr_f.shape[2])
    ]
    return np.stack(channels, axis=-1)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask_bool.any():
        return mask_bool.copy()
    k = int(radius) * 2 + 1
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask_bool.astype(np.uint8), kernel).astype(bool)
    img = Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L")
    return np.asarray(img.filter(ImageFilter.MaxFilter(k))) > 127


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask_bool.any():
        return mask_bool.copy()
    r = int(radius)
    k = r * 2 + 1
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.erode(mask_bool.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    padded = np.pad(mask_bool.astype(np.uint8) * 255, pad_width=r, mode="constant", constant_values=0)
    eroded = np.asarray(Image.fromarray(padded, mode="L").filter(ImageFilter.MinFilter(k)))
    return eroded[r:-r, r:-r] > 127


def _binary_open(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return _binary_dilate(_binary_erode(mask, radius), radius)


def _binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return _binary_erode(_binary_dilate(mask, radius), radius)


def _keep_connected_to_anchor(mask: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Keep 8-connected components of `mask` that intersect `anchor`."""
    mask_bool = np.asarray(mask, dtype=bool)
    seeds = mask_bool & np.asarray(anchor, dtype=bool)
    if not mask_bool.any() or not seeds.any():
        return np.zeros_like(mask_bool, dtype=bool)

    if cv2 is not None:
        _, labels = cv2.connectedComponents(mask_bool.astype(np.uint8), connectivity=8)
        valid_ids = np.unique(labels[seeds])
        valid_ids = valid_ids[valid_ids > 0]
        return np.isin(labels, valid_ids)

    if ndi is not None:
        labels, _ = ndi.label(mask_bool, structure=np.ones((3, 3), dtype=int))
        valid_ids = np.unique(labels[seeds])
        valid_ids = valid_ids[valid_ids > 0]
        return np.isin(labels, valid_ids)

    h, w = mask_bool.shape
    visited = seeds.copy()
    q = deque(map(tuple, np.argwhere(seeds)))
    while q:
        y, x = q.popleft()
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        sub_mask = mask_bool[y0:y1, x0:x1] & (~visited[y0:y1, x0:x1])
        if sub_mask.any():
            for dy, dx in np.argwhere(sub_mask):
                ny, nx = y0 + int(dy), x0 + int(dx)
                visited[ny, nx] = True
                q.append((ny, nx))
    return visited


def _fill_small_internal_holes(mask: np.ndarray, max_hole_pixels: int = 36) -> np.ndarray:
    """Fill tiny enclosed pinholes inside `mask` without altering real openings."""
    mask_bool = np.asarray(mask, dtype=bool)
    inv = ~mask_bool
    if not inv.any():
        return mask_bool.copy()

    h, w = mask_bool.shape
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    exterior_bg = _keep_connected_to_anchor(inv, border)
    enclosed_holes = inv & (~exterior_bg)
    if not enclosed_holes.any():
        return mask_bool.copy()

    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed_holes.astype(np.uint8), connectivity=8)
        small_ids = [idx for idx in range(1, num_labels) if stats[idx, cv2.CC_STAT_AREA] <= max_hole_pixels]
        if not small_ids:
            return mask_bool.copy()
        return mask_bool | np.isin(labels, small_ids)

    if enclosed_holes.sum() <= max_hole_pixels:
        return mask_bool | enclosed_holes
    return mask_bool.copy()


def _nearest_foreground_fill(rgb: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    """Extrapolate `rgb` from `known_mask` to `~known_mask` via nearest pixel."""
    known = np.asarray(known_mask, dtype=bool)
    out = rgb.copy()
    if not known.any() or known.all():
        return out

    if ndi is not None:
        _, indices = ndi.distance_transform_edt(~known, return_indices=True)
        out[~known] = rgb[indices[0][~known], indices[1][~known]]
        return out

    filled = known.copy()
    for _ in range(max(rgb.shape[:2])):
        if filled.all():
            break
        prev = filled.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            shifted_mask = np.zeros_like(filled)
            shifted_rgb = np.zeros_like(out)
            sy0, sy1 = max(0, -dy), filled.shape[0] - max(0, dy)
            dy0, dy1 = max(0, dy), filled.shape[0] - max(0, -dy)
            sx0, sx1 = max(0, -dx), filled.shape[1] - max(0, dx)
            dx0, dx1 = max(0, dx), filled.shape[1] - max(0, -dx)
            shifted_mask[dy0:dy1, dx0:dx1] = prev[sy0:sy1, sx0:sx1]
            shifted_rgb[dy0:dy1, dx0:dx1] = out[sy0:sy1, sx0:sx1]
            newly = shifted_mask & (~filled)
            if newly.any():
                out[newly] = shifted_rgb[newly]
                filled[newly] = True
    return out


def _inpaint_holes(rgb: np.ndarray, hole_mask: np.ndarray, radius: int = 7) -> np.ndarray:
    """Inpaint `hole_mask` smoothly from surrounding valid pixels."""
    holes = np.asarray(hole_mask, dtype=bool)
    if not holes.any():
        return rgb.copy()
    if cv2 is not None:
        return cv2.inpaint(rgb, holes.astype(np.uint8) * 255, float(radius), cv2.INPAINT_TELEA)
    filled = _nearest_foreground_fill(rgb, ~holes)
    blurred = np.asarray(Image.fromarray(filled).filter(ImageFilter.GaussianBlur(radius=2)))
    out = rgb.copy()
    out[holes] = blurred[holes]
    return out


def _gaussian_blur_rgb(rgb: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Return float32 Gaussian-blurred RGB image."""
    if cv2 is not None:
        return cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    pil_blurred = Image.fromarray(rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.asarray(pil_blurred, dtype=np.float32)


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
    seed_low = _gaussian_blur_rgb(seed_rgb, sigma=4.0)
    sd_low = _gaussian_blur_rgb(sd_rgb, sigma=4.0)
    sd_high = np.clip(sd_f - sd_low, -24.0, 24.0)
    low_delta = np.clip(sd_low - seed_low, -float(max_low_shift), float(max_low_shift)) * 0.35
    fused = np.clip(seed_f + low_delta + float(high_gain) * sd_high, 0.0, 255.0).astype(np.uint8)
    out = sd_rgb.copy()
    out[mask] = fused[mask]
    return out


# ── Step 1: Context-Aware Square ROI Cropping & Exact Projection ──────────────


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return `(y1, x1, y2, x2)` inclusive bounding box of a non-empty mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Cannot compute bounding box of an empty mask")
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def compute_square_roi(
    focus_mask: np.ndarray,
    image_shape: tuple[int, ...],
    context_scale: float = 3.8,
    min_crop: int = 256,
    full_frame_threshold: float = 0.85,
) -> dict[str, int | bool]:
    """Compute a square ROI window `(y0, x0, size)` around `focus_mask`.

    Small/medium objects receive a zoomed context window (`size >= min_crop`)
    clamped inside the image frame whenever `size <= H, W` so zero artificial
    gray padding enters Pix2Gestalt. Large objects (`size >= 0.85 * max(H, W)`)
    smoothly fall back to the full-frame centered square `S = max(H, W)`.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    s_full = max(h, w)
    y1, x1, y2, x2 = mask_bbox(focus_mask)
    vh = y2 - y1 + 1
    vw = x2 - x1 + 1
    cy = (y1 + y2) // 2
    cx = (x1 + x2) // 2

    desired = max(int(min_crop), int(round(max(vh, vw) * float(context_scale))))
    if desired >= int(round(full_frame_threshold * s_full)):
        size = s_full
        y0 = -((size - h) // 2)
        x0 = -((size - w) // 2)
        return {"y0": int(y0), "x0": int(x0), "size": int(size), "is_full_frame": True}

    size = min(desired, s_full)
    if size <= h:
        y0 = max(0, min(h - size, cy - size // 2))
    else:
        y0 = -((size - h) // 2)

    if size <= w:
        x0 = max(0, min(w - size, cx - size // 2))
    else:
        x0 = -((size - w) // 2)

    return {"y0": int(y0), "x0": int(x0), "size": int(size), "is_full_frame": False}


def extract_square_roi(
    array: np.ndarray,
    roi: dict[str, int | bool],
    pad_value: int | float | tuple[int, ...] = 0,
) -> np.ndarray:
    """Extract `(size, size[, C])` square patch for `roi`, padding out-of-bounds."""
    h, w = array.shape[:2]
    y0, x0, size = int(roi["y0"]), int(roi["x0"]), int(roi["size"])

    if array.ndim == 2:
        out = np.full((size, size), pad_value, dtype=array.dtype)
    else:
        out = np.full((size, size, array.shape[2]), pad_value, dtype=array.dtype)

    src_y1 = max(0, y0)
    src_y2 = min(h, y0 + size)
    src_x1 = max(0, x0)
    src_x2 = min(w, x0 + size)
    if src_y2 <= src_y1 or src_x2 <= src_x1:
        return out

    dst_y1 = src_y1 - y0
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = src_x1 - x0
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    out[dst_y1:dst_y2, dst_x1:dst_x2] = array[src_y1:src_y2, src_x1:src_x2]
    return out


def paste_square_roi(
    canvas: np.ndarray,
    roi_patch: np.ndarray,
    roi: dict[str, int | bool],
    resize_mode: str = "linear",
) -> np.ndarray:
    """Resize `roi_patch` to `(roi['size'], roi['size'])` and paste onto `canvas`."""
    h, w = canvas.shape[:2]
    y0, x0, size = int(roi["y0"]), int(roi["x0"]), int(roi["size"])
    scaled = _resize_array(roi_patch, (size, size), mode=resize_mode)

    out = canvas.copy()
    src_y1 = max(0, y0)
    src_y2 = min(h, y0 + size)
    src_x1 = max(0, x0)
    src_x2 = min(w, x0 + size)
    if src_y2 <= src_y1 or src_x2 <= src_x1:
        return out

    dst_y1 = src_y1 - y0
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = src_x1 - x0
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    out[src_y1:src_y2, src_x1:src_x2] = scaled[dst_y1:dst_y2, dst_x1:dst_x2]
    return out


# ── Step 2: Multi-Sample Pix2Gestalt Shape & RGB Fusion ───────────────────────


def fuse_pix2gestalt_predictions(
    samples_rgb_256: list[np.ndarray],
    alphas_256: list[np.ndarray],
    image: np.ndarray,
    visible_mask: np.ndarray,
    roi: dict[str, int | bool],
    alpha_threshold: float = 0.48,
) -> dict:
    """Project `K` Pix2Gestalt + CarveKit samples back to `(H, W)` and fuse cleanly.

    Fixes:
      - Upscale continuous alpha with bilinear interpolation before thresholding
        at `alpha_threshold` (~122/255), eliminating CarveKit `alpha > 0` soft
        matting halos and `INTER_NEAREST` stair-stepping.
      - Filter stochastic runaway outlier samples (e.g. hallucinating an entire
        human around a small held object) via visible-region consistency.
      - Recover narrow interior occluder cutouts inside `visible_mask` where
        CarveKit segmentation sees continuity (`alpha >= 0.14`) even when white
        pixels lower matting alpha.
      - Keep only components anchored to a dilated `visible_mask`.
    """
    if not samples_rgb_256 or len(samples_rgb_256) != len(alphas_256):
        raise ValueError("samples_rgb_256 and alphas_256 must be non-empty lists of equal length")

    vis = np.asarray(visible_mask, dtype=bool)
    vis_area = max(int(vis.sum()), 1)
    h, w = vis.shape
    vy1, vx1, vy2, vx2 = mask_bbox(vis)
    vh, vw = vy2 - vy1 + 1, vx2 - vx1 + 1

    rgbs_full: list[np.ndarray] = []
    alphas_full: list[np.ndarray] = []
    candidate_stats: list[dict] = []

    white_bg = np.full((h, w, 3), 255, dtype=np.uint8)
    zero_alpha = np.zeros((h, w), dtype=np.float32)

    for idx, (rgb_256, alpha_256) in enumerate(zip(samples_rgb_256, alphas_256)):
        rgb_full = paste_square_roi(white_bg, rgb_256, roi, resize_mode="cubic")
        alpha_norm = np.asarray(alpha_256, dtype=np.float32) / 255.0
        alpha_full = np.clip(paste_square_roi(zero_alpha, alpha_norm, roi, resize_mode="linear"), 0.0, 1.0)

        raw_mask = alpha_full >= float(alpha_threshold)
        vis_recall = float((raw_mask & vis).sum() / vis_area)
        expansion = float((raw_mask | vis).sum() / vis_area)
        vis_l1 = float(np.abs(rgb_full[vis].astype(np.float32) - image[vis].astype(np.float32)).mean())

        rgbs_full.append(rgb_full)
        alphas_full.append(alpha_full)
        candidate_stats.append(
            {
                "index": idx,
                "vis_recall": round(vis_recall, 4),
                "expansion": round(expansion, 4),
                "vis_l1": round(vis_l1, 3),
            }
        )

    # Reject runaway expansion or high-error outliers relative to peer samples in the batch.
    plausible = [s for s in candidate_stats if s["vis_recall"] >= 0.45] or candidate_stats
    min_exp = min(s["expansion"] for s in plausible)
    exp_ceiling = min(8.5, max(min_exp * 2.2, min_exp + 1.5))
    kept_indices = [
        s["index"]
        for s in plausible
        if s["expansion"] <= exp_ceiling
    ]
    if not kept_indices:
        kept_indices = [min(plausible, key=lambda s: s["expansion"])["index"]]

    kept_alphas = np.stack([alphas_full[i] for i in kept_indices], axis=0)
    consensus_alpha = np.median(kept_alphas, axis=0) if len(kept_indices) >= 3 else np.mean(kept_alphas, axis=0)

    exterior_fg = consensus_alpha >= float(alpha_threshold)

    # Interior concavity recovery: slots/cutouts enclosed on >= 3 of 4 cardinal
    # directions by `visible_mask` (e.g. a microphone across a suit chest or an
    # item inside a tray) where CarveKit segmentation still indicates foreground
    # continuity (`alpha >= 0.12`), while exterior edges strictly require 0.48.
    has_left = np.maximum.accumulate(vis, axis=1)
    has_right = np.maximum.accumulate(vis[:, ::-1], axis=1)[:, ::-1]
    has_up = np.maximum.accumulate(vis, axis=0)
    has_down = np.maximum.accumulate(vis[::-1, :], axis=0)[::-1]
    cardinal_enclosure = (
        has_left.astype(np.uint8)
        + has_right.astype(np.uint8)
        + has_up.astype(np.uint8)
        + has_down.astype(np.uint8)
    )
    interior_cutout_fg = (cardinal_enclosure >= 3) & (~vis) & (consensus_alpha >= 0.12)

    combined = exterior_fg | interior_cutout_fg

    morph_radius = 1 if vis_area < 5000 else 2
    cleaned = _binary_open(combined, morph_radius)
    cleaned = _binary_close(cleaned, morph_radius)

    anchor_radius = max(5, min(28, int(round(0.08 * max(vh, vw)))))
    anchor = _binary_dilate(vis, anchor_radius)
    anchored = _keep_connected_to_anchor(cleaned, anchor)

    amodal_mask = _fill_small_internal_holes(anchored | vis, max_hole_pixels=36)

    # Guard against catastrophic small-object runaway (e.g. held object absorbing
    # an adjacent person across all seeds): if visible is tiny (< 2500 px) and
    # predicted area still exploded > 10x with high visible RGB mismatch, restrict
    # growth to a local neighborhood around visible_mask.
    final_expansion = float(amodal_mask.sum() / vis_area)
    runaway_clamped = False
    if vis_area < 2500 and final_expansion > 10.0:
        best_vis_l1 = min(candidate_stats[i]["vis_l1"] for i in kept_indices)
        if best_vis_l1 > 32.0 or final_expansion > 14.0:
            local_cap = _binary_dilate(vis, max(12, int(round(0.45 * max(vh, vw)))))
            amodal_mask = (amodal_mask & local_cap) | vis
            runaway_clamped = True

    # Choose the sharpest individual RGB sample aligned with `amodal_mask` & `visible_mask`.
    def _sample_rank(i: int) -> float:
        cand_mask = (alphas_full[i] >= float(alpha_threshold)) | vis
        union = float(np.logical_or(cand_mask, amodal_mask).sum())
        shape_iou = float(np.logical_and(cand_mask, amodal_mask).sum() / max(union, 1.0))
        vis_l1_norm = candidate_stats[i]["vis_l1"] / 255.0
        return shape_iou - 0.85 * vis_l1_norm

    best_idx = max(kept_indices, key=_sample_rank)

    return {
        "amodal_mask": amodal_mask,
        "best_idx": int(best_idx),
        "p2g_rgb_full": rgbs_full[best_idx],
        "p2g_alpha_full": alphas_full[best_idx],
        "consensus_alpha": consensus_alpha,
        "diagnostics": {
            "roi": roi,
            "num_samples": len(samples_rgb_256),
            "kept_indices": kept_indices,
            "best_idx": int(best_idx),
            "candidates": candidate_stats,
            "runaway_clamped": runaway_clamped,
        },
    }


# ── Step 3: Appearance Prior Harmonization & White-Dropout Pre-Fill ───────────


def build_appearance_prior(
    image: np.ndarray,
    visible_mask: np.ndarray,
    amodal_mask: np.ndarray,
    p2g_rgb_full: np.ndarray,
    p2g_alpha_full: np.ndarray,
) -> dict:
    """Build a color-harmonized, edge-decontaminated RGB seed canvas.

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

    # Local visible reference near `missing` (captures local clothing/part brightness).
    near_missing_vis = vis & _binary_dilate(missing, 16)
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

    # Erode valid_p2g by 2px so white-background anti-aliasing never biases color stats or edges.
    valid_p2g_core = _binary_erode(valid_p2g & amodal, 2)
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

    harmonized = np.clip(p2g_f * gain[None, None, :] + bias[None, None, :], 0.0, 255.0).astype(np.uint8)

    seed = np.full_like(image, 255, dtype=np.uint8)
    valid_missing_core = missing & valid_p2g_core & (~dropout_mask)
    seed[valid_missing_core] = harmonized[valid_missing_core]
    seed[vis] = image[vis]

    known_good = vis | valid_missing_core
    if not known_good.any():
        known_good = vis.copy()

    # Fill interior dropout holes smoothly via Telea from surrounding visible + valid prior pixels.
    interior_holes = amodal & (~known_good) & (~_binary_dilate(~amodal, 3))
    if interior_holes.any():
        # Seed exterior with nearest foreground first so Telea doesn't pull [255,255,255] inward.
        pre_extrap = _nearest_foreground_fill(seed, known_good)
        inpainted = _inpaint_holes(pre_extrap, interior_holes, radius=7)
        seed[interior_holes] = inpainted[interior_holes]
        known_good = known_good | interior_holes

    # Decontaminate outer 2-3px rim of `amodal` + a 6px outward safety collar.
    collar = _binary_dilate(amodal, 6)
    extrapolated = _nearest_foreground_fill(seed, known_good)
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


# ── Step 4: Mask-Guided Visual Prompts & Structured VLM Parsing ───────────────


def _mask_outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    return _binary_dilate(m, max(1, thickness)) & (~m)


def build_vlm_visual_prompts(
    image: np.ndarray,
    visible_mask: np.ndarray,
    amodal_mask: np.ndarray,
    target_size: int = 384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create zoomed visual prompts with foreground occluders completely censored.

    Returns:
      - `context_rgb`: square crop around `amodal_mask` where `visible_mask` is
        at 100% brightness inside a bright green contour, `missing_mask` is
        completely masked out with diagonal slate stripes inside a yellow contour
        (so foreground occluders such as food inside a tray, a microphone, or a
        vehicle never leak into VLM reasoning/critique), and exterior context is
        dimmed 0.35x.
      - `isolated_rgb`: tight square crop of `visible_mask` isolated on pure white.
      - `panel_rgb`: side-by-side diagnostic image saved as `vlm_visual_prompt.jpg`.
    """
    vis = np.asarray(visible_mask, dtype=bool)
    amodal = np.asarray(amodal_mask, dtype=bool)
    missing = amodal & (~vis)

    highlighted = (image.astype(np.float32) * 0.35).astype(np.uint8)
    if missing.any():
        yy, xx = np.indices(vis.shape)
        stripe = ((yy + xx) // 8) % 2 == 0
        censor = np.where(
            stripe[:, :, None],
            np.array([112, 116, 122], dtype=np.uint8),
            np.array([138, 142, 148], dtype=np.uint8),
        )
        highlighted[missing] = censor[missing]
    highlighted[vis] = image[vis]

    yellow_ring = _mask_outline(missing, thickness=2)
    green_ring = _mask_outline(vis, thickness=2)
    highlighted[yellow_ring] = np.array([255, 215, 0], dtype=np.uint8)
    highlighted[green_ring] = np.array([0, 255, 64], dtype=np.uint8)

    ctx_roi = compute_square_roi(amodal, image.shape, context_scale=2.2, min_crop=192, full_frame_threshold=0.92)
    ctx_patch = extract_square_roi(highlighted, ctx_roi, pad_value=48)
    context_rgb = _resize_array(ctx_patch, (target_size, target_size), mode="linear")

    vy1, vx1, vy2, vx2 = mask_bbox(vis)
    vh, vw = vy2 - vy1 + 1, vx2 - vx1 + 1
    iso_side = max(64, int(round(max(vh, vw) * 1.22)))
    iso_roi = {
        "y0": ((vy1 + vy2) // 2) - iso_side // 2,
        "x0": ((vx1 + vx2) // 2) - iso_side // 2,
        "size": iso_side,
        "is_full_frame": False,
    }
    white_scene = np.full_like(image, 255, dtype=np.uint8)
    white_scene[vis] = image[vis]
    iso_patch = extract_square_roi(white_scene, iso_roi, pad_value=255)
    isolated_rgb = _resize_array(iso_patch, (target_size, target_size), mode="linear")

    sep = np.full((target_size, 8, 3), 220, dtype=np.uint8)
    panel_rgb = np.concatenate([context_rgb, sep, isolated_rgb], axis=1)
    return context_rgb, isolated_rgb, panel_rgb


def parse_structured_vlm_reasoning(raw_text: str, fallback_query: str = "") -> dict[str, str]:
    """Extract `object_name`, `occluded_parts`, and clean SD2 `sd_prompt`."""
    cleaned = re.sub(r"\*+", "", raw_text or "").strip()

    obj_match = re.search(r"OBJECT\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    occ_match = re.search(r"OCCLUDED(?:_PARTS|\s+PARTS)?\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    prm_match = re.search(r"PROMPT\s*:\s*(.+?)(?:\n[A-Z_]+\s*:|$)", cleaned, re.IGNORECASE | re.DOTALL)

    object_name = obj_match.group(1).strip(" .,\"'") if obj_match else ""
    occluded_parts = occ_match.group(1).strip() if occ_match else ""
    sd_prompt = " ".join(prm_match.group(1).split()).strip(" .,\"'") if prm_match else ""

    if not object_name:
        object_name = fallback_query.strip() or "object"

    if not sd_prompt:
        # Strip conversational meta-phrases if VLM returned free-form prose.
        prose = " ".join(cleaned.split())
        prose = re.sub(
            r"^(the target object is|this is|inpainting prompt\s*:)\s*",
            "",
            prose,
            flags=re.IGNORECASE,
        )
        sd_prompt = prose.split(".")[0].strip() or f"complete {object_name} isolated on a white background"

    return {
        "object_name": object_name,
        "occluded_parts": occluded_parts,
        "sd_prompt": sd_prompt,
        "raw": raw_text.strip() if raw_text else "",
    }


# ── Step 5: Adaptive Seam Blending & Cropped Critic Canvas ────────────────────


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
        ring = vis & _binary_dilate(missing, step)
        if step == 1:
            ring_only = ring
        else:
            ring_only = ring & (~_binary_dilate(missing, step - 1))
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
        return _resize_array(canvas, (target_size, target_size), mode="linear")

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
    return _resize_array(patch, (target_size, target_size), mode="linear")


def _truncate_tokens_clean(text: str, tokenizer, max_tokens: int = 70) -> str:
    """Truncate text to `max_tokens` without triggering HF `> 77` sequence warnings."""
    if not text:
        return ""
    try:
        ids = tokenizer(
            text,
            truncation=True,
            max_length=int(max_tokens),
            add_special_tokens=False,
        ).input_ids
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except Exception:
        words = text.split()
        return " ".join(words[:max_tokens])


# ── Main GPU Orchestrator Reusing Shared `AmodalCompleter` Weights ────────────


class ImprovedAmodalCompleter:
    """Zero-shot improved completer sharing loaded weights from `AmodalCompleter`."""

    def __init__(self, base_completer, config: Optional[dict] = None):
        self.base = base_completer
        self.cls = base_completer.__class__
        self.device = base_completer.device
        self.config = config or {}

    def _predict_shape_and_prior(self, image: np.ndarray, visible_mask: np.ndarray, job_dir: Optional[Path]) -> dict:
        import sys
        import torch

        predictor = self.cls._shape_predictor
        if predictor is None or predictor.model is None:
            raise RuntimeError("Strict SCAmodal: Pix2Gestalt model is not loaded.")

        sys.path.insert(0, os.path.join(os.getcwd(), "pix2gestalt", "pix2gestalt"))
        from inference import run_pix2gestalt

        context_scale = float(self.config.get("roi_context_scale", 3.8))
        roi = compute_square_roi(visible_mask, image.shape, context_scale=context_scale, min_crop=256)

        roi_img = extract_square_roi(image, roi, pad_value=(127, 127, 127))
        roi_vis = extract_square_roi(visible_mask.astype(np.uint8) * 255, roi, pad_value=0)

        resized_img = _resize_array(roi_img, (256, 256), mode="area")
        resized_vis = _resize_array(roi_vis, (256, 256), mode="nearest")
        # Guarantee at least one foreground pixel survives 256x256 downsampling.
        if not (resized_vis > 127).any() and roi_vis.any():
            ys, xs = np.where(roi_vis > 127)
            cy = int(round(ys.mean() * 255.0 / max(roi_vis.shape[0] - 1, 1)))
            cx = int(round(xs.mean() * 255.0 / max(roi_vis.shape[1] - 1, 1)))
            resized_vis[np.clip(cy, 0, 255), np.clip(cx, 0, 255)] = 255
        resized_vis = ((resized_vis > 127).astype(np.uint8)) * 255
        rgb_vis_256 = np.repeat(resized_vis[:, :, None], 3, axis=2)

        requested_samples = int(self.config.get("pix2gestalt_samples", 2))
        ddim_steps = int(self.config.get("pix2gestalt_ddim_steps", 100))
        n_samples = max(1, min(requested_samples, 4))
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes < int(4.5 * 1024**3):
                n_samples = 1

        p2g_kwargs = {
            "model": predictor.model,
            "device": predictor.device,
            "input_im": resized_img,
            "visible_mask": rgb_vis_256,
            "n_samples": n_samples,
        }
        try:
            if "ddim_steps" in inspect.signature(run_pix2gestalt).parameters:
                p2g_kwargs["ddim_steps"] = ddim_steps
        except Exception:
            pass

        print(
            f"[Improved-Branch B] Pix2Gestalt ROI={roi['size']}x{roi['size']} "
            f"(full_frame={roi['is_full_frame']}, n_samples={n_samples}, ddim_steps={ddim_steps})..."
        )
        try:
            with torch.inference_mode():
                samples_rgb_256 = run_pix2gestalt(**p2g_kwargs)
        except RuntimeError as err:
            if "out of memory" in str(err).lower() and n_samples > 1:
                print("[Improved-Branch B] VRAM tight for multi-sample; retrying with n_samples=1...")
                torch.cuda.empty_cache()
                p2g_kwargs["n_samples"] = 1
                with torch.inference_mode():
                    samples_rgb_256 = run_pix2gestalt(**p2g_kwargs)
            else:
                raise

        pil_samples = [Image.fromarray(s) for s in samples_rgb_256]
        rgba_mats = predictor.carvekit_interface(pil_samples)
        alphas_256 = [np.asarray(mat)[:, :, 3] for mat in rgba_mats]

        alpha_thresh = float(self.config.get("carvekit_alpha_threshold", 0.48))
        fused = fuse_pix2gestalt_predictions(
            samples_rgb_256=samples_rgb_256,
            alphas_256=alphas_256,
            image=image,
            visible_mask=visible_mask,
            roi=roi,
            alpha_threshold=alpha_thresh,
        )

        best_idx = fused["best_idx"]
        if job_dir is not None:
            Image.fromarray(samples_rgb_256[best_idx]).save(job_dir / "pix2gestalt_rgb_256.png")
            Image.fromarray(alphas_256[best_idx]).save(job_dir / "carvekit_alpha_256.png")
            Image.fromarray(fused["amodal_mask"].astype(np.uint8) * 255).save(job_dir / "shape_mask_original.png")

        return fused

    def _reason_target_aware(
        self,
        context_rgb: np.ndarray,
        isolated_rgb: np.ndarray,
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
            "You are analyzing a partially occluded target object for amodal appearance completion.\n"
            "- Image 1 shows the local scene where ONLY the target object's visible region is at full "
            "brightness inside a BRIGHT GREEN outline, and the occluded region to reconstruct is covered "
            "by a DIAGONAL GRAY STRIPED CENSOR MASK inside a YELLOW outline.\n"
            "- Image 2 shows ONLY the target object's visible pixels isolated on a pure white background.\n"
            f"{hint_line}"
            "CRITICAL RULES:\n"
            "1. Identify ONLY the BRIGHT GREEN target object shown in Image 2.\n"
            "2. Write PROMPT as a concise (12-26 words) studio photo description of the COMPLETE, "
            "UNOCCLUDED target object isolated on a plain white background, matching the exact colors, "
            "materials, patterns, and lighting visible in Image 2.\n"
            "3. If the target object is a container, takeout box, tray, bowl, plate, dish, or surface "
            "whose interior was covered by the yellow-outlined censor mask, describe it as EMPTY and clean "
            "inside (NEVER imagine food, sandwiches, or items inside it).\n"
            "4. Do NOT mention occluders, masks, stripes, green/yellow outlines, or missing parts in PROMPT.\n\n"
            "Respond in this exact 3-line format:\n"
            "OBJECT: <short noun phrase naming the target object>\n"
            "OCCLUDED_PARTS: <one short sentence naming which part of the target object is hidden>\n"
            "PROMPT: <concise description of the complete unoccluded target object on a white background>"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(context_rgb)},
                    {"type": "image", "image": Image.fromarray(isolated_rgb)},
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
        isolated_rgb: np.ndarray,
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

        crop_pil = Image.fromarray(isolated_rgb)
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

    def _refine_roi_sd2(
        self,
        seed_canvas: np.ndarray,
        visible_mask: np.ndarray,
        amodal_mask: np.ndarray,
        dropout_mask: np.ndarray,
        prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        ref_p90_luma: float,
    ) -> np.ndarray:
        import torch

        pipe = self.cls._pipe
        vis = np.asarray(visible_mask, dtype=bool)
        amodal = np.asarray(amodal_mask, dtype=bool)
        missing = amodal & (~vis)
        dropout = np.asarray(dropout_mask, dtype=bool) & missing
        if not missing.any():
            return seed_canvas.copy()

        ay1, ax1, ay2, ax2 = mask_bbox(amodal)
        ah, aw = ay2 - ay1 + 1, ax2 - ax1 + 1
        roi_side = max(128, int(round(max(ah, aw) * 1.35)))
        roi = {
            "y0": ((ay1 + ay2) // 2) - roi_side // 2,
            "x0": ((ax1 + ax2) // 2) - roi_side // 2,
            "size": roi_side,
            "is_full_frame": False,
        }

        # Dilate missing slightly into visible (2px) & collar (2px) for smooth VAE latent transition,
        # while protecting the interior of visible_mask (`erode(vis, 2)`).
        protected_vis = _binary_erode(vis, 2)
        inpaint_region = _binary_dilate(missing, 2) & _binary_dilate(amodal, 2) & (~protected_vis)
        if not inpaint_region.any():
            inpaint_region = missing.copy()

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
        out[missing] = synthesized_full[missing]

        # Where Pix2Gestalt already produced valid foreground appearance (`~dropout_region`),
        # lock low-frequency color/structure to `seed_canvas` and keep only crisp SD2 micro-detail
        # so SD2 can NEVER hallucinate food, heads, or gray patches over valid Pix2Gestalt RGB!
        dropout_region = _binary_dilate(dropout, 4) & missing
        valid_prior_region = missing & (~dropout_region)
        if valid_prior_region.any():
            out = lock_low_frequency_to_prior(
                seed_rgb=seed_canvas,
                sd_rgb=out,
                lock_mask=valid_prior_region,
                max_low_shift=10.0,
                high_gain=0.55,
            )

        # Guard against outer-rim white background bleed on non-white objects.
        if ref_p90_luma < 212.0:
            outer_rim = missing & (~_binary_erode(amodal, 4))
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
            f"You are a strict quality critic for amodal object completion of a '{object_name}'.\n"
            "- Image 1 shows the reconstructed complete target object isolated on a white background.\n"
            "- Image 2 shows the local scene where the target's visible region is inside a BRIGHT GREEN outline "
            "and the removed foreground occluder is covered by a diagonal striped censor mask inside YELLOW.\n\n"
            "Evaluate ONLY the reconstructed target object in Image 1 on a 0-10 integer scale:\n"
            "1. STRUCTURAL: whole-object shape plausibility, natural proportions, and clean silhouette.\n"
            "2. TEXTURE: color, material, and lighting continuity between visible and completed regions (penalize flat gray patches or hallucinated foreign items).\n"
            "3. CONTEXT: semantic consistency with the BRIGHT GREEN target object in Image 2 (note: if the target is a container/tray/dish, a clean EMPTY interior is 10/10 correct because the item inside was the removed occluder).\n\n"
            "Respond EXACTLY in this 4-line format:\n"
            "STRUCTURAL: <int>\n"
            "TEXTURE: <int>\n"
            "CONTEXT: <int>\n"
            "FEEDBACK: <one short actionable fix for the object appearance, or 'good' if seamless>"
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
        text_query: str = "",
        job_dir: Optional[Path] = None,
    ) -> dict:
        import torch

        t0 = time.perf_counter()
        h, w = image.shape[:2]
        vis = np.asarray(visible_mask, dtype=bool)
        pad_info = {
            "needs_outpaint": False,
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
        }
        if job_dir is not None:
            (job_dir / "outpainting.json").write_text(json.dumps(pad_info, indent=2), encoding="utf-8")

        # 1. Branch B — Context-aware ROI Pix2Gestalt + smooth consensus shape.
        t_shape0 = time.perf_counter()
        shape_out = self._predict_shape_and_prior(image, vis, job_dir)
        t_shape = time.perf_counter() - t_shape0
        amodal_mask = shape_out["amodal_mask"]
        missing_mask = amodal_mask & (~vis)

        if not missing_mask.any():
            rgba = self.base._finalize_rgba(image, amodal_mask, h, w)
            if job_dir is not None:
                (job_dir / "reasoning.txt").write_text("No missing region predicted.", encoding="utf-8")
                (job_dir / "critique_history.json").write_text("[]", encoding="utf-8")
            return {
                "input_image": image,
                "visible_mask": vis,
                "amodal_mask": amodal_mask,
                "inpainted_rgba": rgba,
                "vlm_reasoning": "No missing region predicted.",
                "critique_history": [],
                "final_score": None,
                "_trace": {
                    "pad_info": pad_info,
                    "synthesis_attempts": 0,
                    "critique_history": [],
                    "shape_diagnostics": shape_out["diagnostics"],
                    "backend": "SCAmodal Improved (ROI + Mask-Guided VLM + Hybrid Prior)",
                },
            }

        # 2. Build harmonized & decontaminated appearance prior from Pix2Gestalt + image.
        prior_info = build_appearance_prior(
            image=image,
            visible_mask=vis,
            amodal_mask=amodal_mask,
            p2g_rgb_full=shape_out["p2g_rgb_full"],
            p2g_alpha_full=shape_out["p2g_alpha_full"],
        )
        seed_canvas = prior_info["seed_canvas"]
        dropout_mask = prior_info["dropout_mask"]
        dropout_frac = float(prior_info["dropout_fraction"])
        if job_dir is not None:
            Image.fromarray(seed_canvas).save(job_dir / "prior_rgb.png")

        # 3. Branch A — Mask-guided Qwen3-VL reasoning + square-padded CLIP verify.
        t_vlm0 = time.perf_counter()
        print("[Improved-Branch A] Mask-guided semantic reasoning (Qwen3-VL)...")
        context_rgb, isolated_rgb, panel_rgb = build_vlm_visual_prompts(image, vis, amodal_mask, target_size=384)
        if job_dir is not None:
            Image.fromarray(panel_rgb).save(job_dir / "vlm_visual_prompt.jpg")

        parsed_vlm = self._reason_target_aware(context_rgb, isolated_rgb, user_query=text_query)
        print(f"[Improved VLM] OBJECT='{parsed_vlm['object_name']}' | PROMPT='{parsed_vlm['sd_prompt']}'")
        if job_dir is not None:
            (job_dir / "reasoning.txt").write_text(parsed_vlm["raw"], encoding="utf-8")

        base_prompt, clip_info = self._verify_prompt_clip(isolated_rgb, parsed_vlm, user_query=text_query)
        t_vlm = time.perf_counter() - t_vlm0
        if job_dir is not None:
            (job_dir / "prompt_selection.json").write_text(
                json.dumps({"prompt": base_prompt, "parsed_vlm": parsed_vlm, "clip_verification": clip_info}, indent=2),
                encoding="utf-8",
            )

        # 4. Hybrid ROI SD2 refinement + Target-Aware Critic loop (keeping best-scored attempt).
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
            float(self.config.get("dropout_refine_strength", 0.78))
            if dropout_frac >= 0.20
            else float(self.config.get("prior_refine_strength", 0.24))
        )
        strength_step = 0.12 if dropout_frac >= 0.20 else 0.08
        guidance_scale = 6.0 if dropout_frac < 0.20 else 7.5
        num_steps = 28
        base_seed = int(self.config.get("seed", 42))
        enable_critic = bool(self.config.get("enable_critic", True))
        critique_threshold = float(self.config.get("critique_threshold", 7.0))
        max_critic_iter = int(self.config.get("max_critic_iter", 2))
        n_iters = max_critic_iter if enable_critic else 1

        critique_history: list[dict] = []
        best_rgba: Optional[np.ndarray] = None
        best_score: float = -1.0
        attempts = 0

        for crit_iter in range(n_iters):
            attempts += 1
            iter_strength = min(base_strength + strength_step * crit_iter, 0.90)
            print(
                f"[Improved-Synthesis] Attempt {crit_iter + 1}/{n_iters} "
                f"(strength={iter_strength:.2f}, dropout={dropout_frac:.2f}, guidance={guidance_scale:.1f})"
            )
            if job_dir is not None:
                (job_dir / f"prompt_{attempts:02d}.txt").write_text(prompt, encoding="utf-8")

            inpaint_img = self._refine_roi_sd2(
                seed_canvas=seed_canvas,
                visible_mask=vis,
                amodal_mask=amodal_mask,
                dropout_mask=dropout_mask,
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

        t_syn = time.perf_counter() - t_syn0
        t_total = time.perf_counter() - t0
        print(
            f"[Timing] shape={t_shape:.1f}s | vlm={t_vlm:.1f}s | sd2+critic={t_syn:.1f}s | total={t_total:.1f}s",
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
                "shape_diagnostics": shape_out["diagnostics"],
                "prior_diagnostics": {
                    "dropout_fraction": prior_info["dropout_fraction"],
                    "color_gain": prior_info["color_gain"],
                    "color_bias": prior_info["color_bias"],
                    "ref_p90_luma": prior_info["ref_p90_luma"],
                },
                "backend": "SCAmodal Improved (ROI + Mask-Guided VLM + Hybrid Prior)",
            },
        }
