"""Vision, morphology, square ROI, and compositing utilities."""
from src.vision.compositing import (
    adaptive_seam_blend,
    compose_cropped_for_critic,
    finalize_rgba,
    lock_low_frequency_to_prior,
)
from src.vision.image_ops import (
    binary_close,
    binary_dilate,
    binary_erode,
    binary_open,
    fill_small_internal_holes,
    gaussian_blur_rgb,
    inpaint_holes,
    keep_connected_to_anchor,
    mask_outline,
    nearest_foreground_fill,
    resize_array,
)
from src.vision.roi import (
    compute_square_roi,
    extract_square_roi,
    mask_bbox,
    paste_square_roi,
)

__all__ = [
    "adaptive_seam_blend",
    "binary_close",
    "binary_dilate",
    "binary_erode",
    "binary_open",
    "compose_cropped_for_critic",
    "compute_square_roi",
    "extract_square_roi",
    "fill_small_internal_holes",
    "finalize_rgba",
    "gaussian_blur_rgb",
    "inpaint_holes",
    "keep_connected_to_anchor",
    "lock_low_frequency_to_prior",
    "mask_bbox",
    "mask_outline",
    "nearest_foreground_fill",
    "paste_square_roi",
    "resize_array",
]

