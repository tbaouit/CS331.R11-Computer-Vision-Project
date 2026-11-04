"""Multi-sample Pix2Gestalt + CarveKit shape consensus and ROI input preparation."""
from __future__ import annotations

import numpy as np

from src.vision.image_ops import (
    binary_close,
    binary_dilate,
    binary_open,
    fill_small_internal_holes,
    keep_connected_to_anchor,
    resize_array,
)
from src.vision.roi import extract_square_roi, mask_bbox, paste_square_roi


def prepare_pix2gestalt_roi_inputs(
    image: np.ndarray,
    visible_mask: np.ndarray,
    roi: dict[str, int | bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and resize `(256, 256)` RGB scene crop and 3-channel binary visible mask."""
    roi_img = extract_square_roi(image, roi, pad_value=(127, 127, 127))
    roi_vis = extract_square_roi(visible_mask.astype(np.uint8) * 255, roi, pad_value=0)

    resized_img = resize_array(roi_img, (256, 256), mode="area")
    resized_vis = resize_array(roi_vis, (256, 256), mode="nearest")
    # Guarantee at least one foreground pixel survives 256x256 downsampling.
    if not (resized_vis > 127).any() and roi_vis.any():
        ys, xs = np.where(roi_vis > 127)
        cy = int(round(ys.mean() * 255.0 / max(roi_vis.shape[0] - 1, 1)))
        cx = int(round(xs.mean() * 255.0 / max(roi_vis.shape[1] - 1, 1)))
        resized_vis[np.clip(cy, 0, 255), np.clip(cx, 0, 255)] = 255
    resized_vis = ((resized_vis > 127).astype(np.uint8)) * 255
    rgb_vis_256 = np.repeat(resized_vis[:, :, None], 3, axis=2)
    return resized_img, rgb_vis_256


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
        CarveKit segmentation sees continuity (`alpha >= 0.12`) even when white
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
    kept_indices = [s["index"] for s in plausible if s["expansion"] <= exp_ceiling]
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
    cleaned = binary_open(combined, morph_radius)
    cleaned = binary_close(cleaned, morph_radius)

    anchor_radius = max(5, min(28, int(round(0.08 * max(vh, vw)))))
    anchor = binary_dilate(vis, anchor_radius)
    anchored = keep_connected_to_anchor(cleaned, anchor)

    amodal_mask = fill_small_internal_holes(anchored | vis, max_hole_pixels=36)

    # Guard against catastrophic small-object runaway (e.g. held object absorbing
    # an adjacent person across all seeds): if visible is tiny (< 2500 px) and
    # predicted area still exploded > 10x with high visible RGB mismatch, restrict
    # growth to a local neighborhood around visible_mask.
    final_expansion = float(amodal_mask.sum() / vis_area)
    runaway_clamped = False
    if vis_area < 2500 and final_expansion > 10.0:
        best_vis_l1 = min(candidate_stats[i]["vis_l1"] for i in kept_indices)
        if best_vis_l1 > 32.0 or final_expansion > 14.0:
            local_cap = binary_dilate(vis, max(12, int(round(0.45 * max(vh, vw)))))
            amodal_mask = (amodal_mask & local_cap) | vis
            runaway_clamped = True

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

