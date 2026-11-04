"""512x512 ROI Stable Diffusion v2 refinement stages with frequency-separated color locking."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.vision.compositing import lock_low_frequency_to_prior
from src.vision.image_ops import binary_dilate, binary_erode
from src.vision.roi import extract_square_roi, mask_bbox, paste_square_roi


def _run_sd2_roi_pass(
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


def _suppress_outer_rim_white_bleed(
    out: np.ndarray,
    seed_canvas: np.ndarray,
    missing: np.ndarray,
    amodal: np.ndarray,
    ref_p90_luma: float,
) -> np.ndarray:
    if ref_p90_luma < 212.0:
        outer_rim = missing & (~binary_erode(amodal, 4))
        out_f = out.astype(np.float32)
        white_bleed = outer_rim & (out_f.min(axis=-1) >= 232.0) & ((out_f.max(axis=-1) - out_f.min(axis=-1)) <= 16.0)
        if white_bleed.any():
            out[white_bleed] = seed_canvas[white_bleed]
    return out


def refine_roi_sd2_blind(
    pipe,
    device: str,
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
    """ROI SD2 refinement for blind mode (`roi_side = 1.35x` around predicted `amodal_mask`)."""
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

    protected_vis = binary_erode(vis, 2)
    inpaint_region = binary_dilate(missing, 2) & binary_dilate(amodal, 2) & (~protected_vis)
    if not inpaint_region.any():
        inpaint_region = missing.copy()

    synthesized_full = _run_sd2_roi_pass(
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

    out = seed_canvas.copy()
    out[missing] = synthesized_full[missing]

    dropout_region = binary_dilate(dropout, 4) & missing
    valid_prior_region = missing & (~dropout_region)
    if valid_prior_region.any():
        out = lock_low_frequency_to_prior(
            seed_rgb=seed_canvas,
            sd_rgb=out,
            lock_mask=valid_prior_region,
            max_low_shift=10.0,
            high_gain=0.55,
        )

    return _suppress_outer_rim_white_bleed(out, seed_canvas, missing, amodal, ref_p90_luma)


def refine_exact_mask_sd2_oracle(
    pipe,
    device: str,
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
    """High-density 512x512 ROI SD2 refinement inside exact `occluded_mask` with dual-zone lock."""
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

    protected_vis = binary_erode(vis, 2)
    inpaint_region = binary_dilate(occ, 2) & binary_dilate(amodal, 2) & (~protected_vis)
    if not inpaint_region.any():
        inpaint_region = occ.copy()

    synthesized_full = _run_sd2_roi_pass(
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

    out = seed_canvas.copy()
    out[occ] = synthesized_full[occ]

    unfilled_zone = binary_dilate(unfilled, 3) & occ
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

    return _suppress_outer_rim_white_bleed(out, seed_canvas, occ, amodal, ref_p90_luma)

