"""Geometric IoU and reference-free appearance coherence metrics."""
from __future__ import annotations

from typing import Optional
import numpy as np

from src.vision.image_ops import binary_dilate, nearest_foreground_fill


def iou(predicted: np.ndarray, target: np.ndarray) -> Optional[float]:
    union = np.logical_or(predicted, target).sum()
    return float(np.logical_and(predicted, target).sum() / union) if union else None


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
    occ_seam_ring = occ & binary_dilate(vis, 2)
    if occ_seam_ring.any():
        vis_canvas = np.zeros_like(image, dtype=np.uint8)
        vis_canvas[vis] = image[vis]
        nearest_vis_rgb = nearest_foreground_fill(vis_canvas, vis)
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

