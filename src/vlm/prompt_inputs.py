"""Build Qwen visual inputs and truncate diffusion prompts."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.vision.image_ops import binary_erode, mask_outline, resize_array
from src.vision.roi import compute_square_roi, extract_square_roi, mask_bbox

DEFAULT_NEGATIVE_PROMPT = (
    "gray background, flat gray patch, gray fill, occluded, cut off, cropped, "
    "blurry, low quality, distorted, deformed, extra limbs, floating head, "
    "text, watermark, artifacts"
)

# Giới hạn token theo tokenizer diffusion, tránh cắt theo số ký tự.
def truncate_tokens_clean(text: str, tokenizer, max_tokens: int = 70) -> str:
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


def _render_censored_scene(image: np.ndarray, vis: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Dim background to 35%, censor `missing` with opaque slate stripes, and outline regions."""
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
    highlighted[mask_outline(missing, thickness=2)] = np.array([255, 215, 0], dtype=np.uint8)
    highlighted[mask_outline(vis, thickness=2)] = np.array([0, 255, 64], dtype=np.uint8)
    return highlighted


# Cô lập phần visible trên nền trắng làm tham chiếu màu và bề mặt.
def _render_isolated_visible_crop(image: np.ndarray, vis: np.ndarray, target_size: int) -> np.ndarray:
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
    return resize_array(iso_patch, (target_size, target_size), mode="linear")


# Tạo crop ngữ cảnh và crop visible cho proposed cùng kích thước.
def build_vlm_visual_prompts(
    image: np.ndarray,
    visible_mask: np.ndarray,
    amodal_mask: np.ndarray,
    target_size: int = 384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create zoomed visual prompts (blind mode) with foreground occluders censored."""
    vis = np.asarray(visible_mask, dtype=bool)
    amodal = np.asarray(amodal_mask, dtype=bool)
    missing = amodal & (~vis)

    highlighted = _render_censored_scene(image, vis, missing)
    ctx_roi = compute_square_roi(amodal, image.shape, context_scale=2.2, min_crop=192, full_frame_threshold=0.92)
    ctx_patch = extract_square_roi(highlighted, ctx_roi, pad_value=48)
    context_rgb = resize_array(ctx_patch, (target_size, target_size), mode="linear")

    isolated_rgb = _render_isolated_visible_crop(image, vis, target_size)

    sep = np.full((target_size, 8, 3), 220, dtype=np.uint8)
    panel_rgb = np.concatenate([context_rgb, sep, isolated_rgb], axis=1)
    return context_rgb, isolated_rgb, panel_rgb
