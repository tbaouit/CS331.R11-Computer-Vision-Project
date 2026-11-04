"""Context-aware square ROI cropping and exact canvas back-projection."""
from __future__ import annotations

import numpy as np

from src.vision.image_ops import resize_array


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
    scaled = resize_array(roi_patch, (size, size), mode=resize_mode)

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

