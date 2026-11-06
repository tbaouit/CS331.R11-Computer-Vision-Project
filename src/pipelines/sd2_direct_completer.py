"""Single-pass SD2 baseline with provided masks and a fixed, category-free prompt."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.stages.sd2_refiner import _run_sd2_roi_pass
from src.stages.vlm_guidance import DEFAULT_NEGATIVE_PROMPT
from src.vision.compositing import adaptive_seam_blend, finalize_rgba
from src.vision.image_ops import binary_dilate, binary_erode
from src.vision.roi import mask_bbox

DEFAULT_PROMPT = (
    "A photograph of a single complete object on a plain white background, "
    "natural colors, realistic texture."
)


def prepare_direct_inputs(image, visible_mask, occluded_mask):
    """Same SD2 ROI/mask geometry as oracle refinement, without any appearance prior."""
    vis = np.asarray(visible_mask, dtype=bool)
    occ = np.asarray(occluded_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("SD2 Direct requires uint8 HxWx3 RGB")
    if vis.shape != image.shape[:2] or occ.shape != vis.shape or not vis.any():
        raise ValueError("Invalid/empty visible mask or mismatched hidden mask")
    if (vis & occ).any():
        raise ValueError("Visible and hidden masks must be disjoint")
    amodal = vis | occ
    # Never expose occluder pixels to the baseline, even when strength < 1.
    canvas = np.full_like(image, 255)
    canvas[vis] = image[vis]
    canvas[occ] = 127
    y1, x1, y2, x2 = mask_bbox(amodal)
    side = max(128, int(round(max(y2 - y1 + 1, x2 - x1 + 1) * 1.28)))
    roi = dict(y0=((y1 + y2) // 2) - side // 2,
               x0=((x1 + x2) // 2) - side // 2, size=side, is_full_frame=False)
    inpaint_region = binary_dilate(occ, 2) & binary_dilate(amodal, 2) & ~binary_erode(vis, 2)
    if not inpaint_region.any():
        inpaint_region = occ.copy()
    return canvas, vis, occ, amodal, roi, inpaint_region


class SD2DirectCompleter:
    def __init__(self, models, config):
        self.models, self.config = models, config
        if config.get("text_query_policy", "empty") != "empty" or config.get("enable_critic", False):
            raise ValueError("SD2 Direct uses a fixed prompt and no critic")
        self.prompt = config.get("prompt", DEFAULT_PROMPT)
        self.strength = float(config.get("strength", 1.0))
        self.guidance = float(config.get("guidance_scale", 7.0))
        self.steps = int(config.get("num_inference_steps", 28))
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("A fixed nonempty prompt is required")
        if not 0.16 <= self.strength <= 1 or self.steps < 1 or self.guidance < 0:
            raise ValueError("Invalid SD2 baseline inference settings")

    def complete(self, image, visible_mask, occluded_mask, job_dir=None):
        if occluded_mask is None:
            raise ValueError("SD2 Direct requires an input hidden mask")
        started = time.perf_counter()
        canvas, vis, occ, amodal, roi, inpaint_region = prepare_direct_inputs(image, visible_mask, occluded_mask)
        negative = self.config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        seed = int(self.config.get("seed", 42))
        settings = dict(prompt=self.prompt, negative_prompt=negative, seed=seed,
                        strength=self.strength, guidance_scale=self.guidance,
                        num_inference_steps=self.steps, resolution=512, roi=roi,
                        category_annotation_used=False, appearance_prior=False)
        if job_dir is not None:
            job = Path(job_dir)
            Image.fromarray(canvas).save(job / "baseline_input.png")
            Image.fromarray(inpaint_region.astype(np.uint8) * 255).save(job / "inpainting_mask.png")
            (job / "prompt_01.txt").write_text(self.prompt, encoding="utf-8")
            (job / "sd2_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        model_load_seconds, inference_seconds = 0.0, 0.0
        if occ.any():
            load_started = time.perf_counter()
            pipe = self.models.pipe  # The only model this baseline activates.
            model_load_seconds = time.perf_counter() - load_started
            print(f"[SD2 Direct] One pass, ROI={roi['size']}, steps={self.steps}, strength={self.strength}", flush=True)
            inference_started = time.perf_counter()
            generated = _run_sd2_roi_pass(
                pipe=pipe, device=self.models.device, seed_canvas=canvas,
                inpaint_region=inpaint_region, roi=roi, prompt=self.prompt,
                negative_prompt=negative, strength=self.strength, guidance_scale=self.guidance,
                num_inference_steps=self.steps, seed=seed,
            )
            inference_seconds = time.perf_counter() - inference_started
            # Match oracle compositing: only copy synthesized hidden pixels into
            # the seed, then call the shared seam/alpha code. No frequency locking.
            completed = canvas.copy()
            completed[occ] = generated[occ]
        else:
            completed = canvas.copy()
        rgb = adaptive_seam_blend(image, completed, vis, amodal)
        rgba = finalize_rgba(rgb, amodal, *image.shape[:2])
        if job_dir is not None:
            Image.fromarray(completed).save(job / "inpainting_01.png")
        return dict(inpainted_rgba=rgba, amodal_mask=amodal, final_score=None, critique_history=[],
                    _trace=dict(backend="SD2 Direct (fixed prompt, input hidden mask)", pad_info={},
                                synthesis_attempts=int(occ.any()), settings=settings,
                                model_load_seconds=round(model_load_seconds, 3),
                                sd2_seconds=round(inference_seconds, 3),
                                total_seconds=round(time.perf_counter() - started, 3)))
