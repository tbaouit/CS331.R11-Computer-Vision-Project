"""Inpaint a square ROI and project its decoded RGB back onto the canvas."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.vision.image_ops import binary_dilate, binary_erode
from src.vision.roi import extract_square_roi, mask_bbox, paste_square_roi


# Cắt ROI/MASK, chạy SD1.5 trên canvas 512×512 và chiếu ảnh về kích thước ban đầu.
def run_inpaint_roi_pass(
    pipe,
    device: str,
    seed_canvas: np.ndarray,
    inpaint_region: np.ndarray,
    roi: dict[str, int | bool],
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
) -> np.ndarray:
    import torch

    roi_side = int(roi["size"])
    roi_img = extract_square_roi(seed_canvas, roi, pad_value=255)
    roi_mask = extract_square_roi(inpaint_region.astype(np.uint8) * 255, roi, pad_value=0)

    pil_img_512 = Image.fromarray(roi_img).resize((512, 512), Image.Resampling.LANCZOS)
    pil_mask_512 = Image.fromarray(roi_mask).resize((512, 512), Image.Resampling.NEAREST)

    eff_strength = float(np.clip(strength, 0.16, 1.0))
    eff_steps = max(int(num_inference_steps), int(np.ceil(10.0 / eff_strength)))

    generator = torch.Generator(device=device).manual_seed(int(seed))
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
    return paste_square_roi(seed_canvas, out_roi_np, roi, resize_mode="linear")


def refine_exact_mask(
    pipe,
    device: str,
    seed_canvas: np.ndarray,
    visible_mask: np.ndarray,
    occluded_mask: np.ndarray,
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
) -> np.ndarray:
    """Return the complete decoded ROI at its original coordinates."""
    vis = np.asarray(visible_mask, dtype=bool)
    occ = np.asarray(occluded_mask, dtype=bool) & (~vis)
    amodal = vis | occ

    ay1, ax1, ay2, ax2 = mask_bbox(amodal)
    ah, aw = ay2 - ay1 + 1, ax2 - ax1 + 1
    roi_side = max(128, int(round(max(ah, aw) * 1.28)))
    roi = {
        "y0": ((ay1 + ay2) // 2) - roi_side // 2,
        "x0": ((ax1 + ax2) // 2) - roi_side // 2,
        "size": roi_side,
        "is_full_frame": False,
    }

    protected_vis = binary_erode(vis, 2)
    inpaint_region = binary_dilate(occ, 2) & binary_dilate(amodal, 2) & (~protected_vis)
    if not inpaint_region.any():
        inpaint_region = occ.copy()

    return run_inpaint_roi_pass(
        pipe=pipe,
        device=device,
        seed_canvas=seed_canvas,
        inpaint_region=inpaint_region,
        roi=roi,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
