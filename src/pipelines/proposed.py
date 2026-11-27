"""Open-World Amodal Appearance Completion pipeline (`pipeline_mode = 'proposed'`).

Implements the 4-module Open-World Amodal pipeline (`D:/amodal/main.py`) adapted for given `visible_mask`
and Google Colab (16GB-24GB VRAM):
1. `RAM++` (Swin-L) + `CLIP ViT-B/32`: scene tagging and prompt selection from the masked visible object.
2. `GroundingDINO` (Swin-T) + `SAM ViT-H` (initialized once): instance segmentation of scene objects.
3. `InstaOrder` (`InstaOrderNet_od`, ResNet-50): pairwise occlusion order analysis.
4. `Stable Diffusion 2 Inpainting` (`stabilityai/stable-diffusion-2-inpainting`): progressive mixed-context
   diffusion on expanded canvas up to `max_iter = 3` iterations (`while run_iter and iter_id < max_iter`).
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Optional

import numpy as np
from PIL import Image

from src.eval.metrics import compute_appearance_coherence_metrics
from src.models.openworld_models import validate_openworld_proposed_config
from src.stages.openworld_completion import (
    QueryObjectState,
    alpha_blending,
    canvas_center_slice,
    handlemask,
    predict_auto_visible_mask,
    run_progressive_iteration,
    segment_with_cached_sam,
    shrink_edges_to_transparent,
)
from src.vision.compositing import finalize_rgba


def remove_duplicates(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


_SCENE_BACKGROUND_TAGS = {
    "background",
    "wall",
    "floor",
    "ceiling",
    "room",
    "bathroom",
    "bedroom",
    "kitchen",
    "living room",
    "sky",
    "grass",
    "ground",
    "field",
    "rock",
    "stone",
    "water",
    "street",
    "road",
    "building",
    "indoor",
    "outdoor",
}


class Proposed:
    """Open-World Amodal progressive completion pipeline using direct `visible_mask`."""

    def __init__(self, models, config: dict):
        validate_openworld_proposed_config(config)
        self.models = models
        self.config = config
        self.device = models.device

    def _select_prompt_with_ram_and_clip(
        self,
        img_pil: Image.Image,
        image_np: np.ndarray,
        visible_mask: np.ndarray,
        text_query: str = "",
    ) -> tuple[list[str], str, np.ndarray, str]:
        """Run RAM++ to get scene tags and CLIP ViT-B/32 on the masked visible region to choose `inpaint_prompt` and visual context."""
        classes = self.models.ram_plus.infer_tags(img_pil)
        masked_image_np = np.zeros_like(image_np, dtype=np.uint8)
        masked_image_np[visible_mask] = image_np[visible_mask]
        clip_selected = self.models.openai_clip.select_prompt(
            Image.fromarray(masked_image_np),
            classes=classes,
            text_query="",
        )
        clean_query = (text_query or "").strip()
        context_descriptor = ""
        # Keep RAM++ + CLIP tag in scene classes for occluder detection, anchor target identity to text_query,
        # and retain non-background CLIP visual descriptor to enrich SD2's semantic context.
        if clean_query and clean_query.lower() not in ("object", "main object"):
            inpaint_prompt = clean_query
            clip_clean = (clip_selected or "").strip()
            if (
                clip_clean
                and clip_clean.lower() not in _SCENE_BACKGROUND_TAGS
                and clip_clean.lower() != clean_query.lower()
                and clip_clean.lower() not in clean_query.lower()
            ):
                context_descriptor = clip_clean
            if clip_clean and clip_clean not in classes:
                classes = list(classes) + [clip_clean]
        else:
            inpaint_prompt = (clip_selected or clean_query or "object").strip()
        return classes, inpaint_prompt, masked_image_np, context_descriptor

    def _segment_scene_and_attach_query_mask(
        self,
        img_pil: Image.Image,
        image_np: np.ndarray,
        visible_mask: np.ndarray,
        classes: list[str],
        inpaint_prompt: str,
    ) -> tuple[np.ndarray, list[str], list[float], int]:
        """Run GroundingDINO + cached SAM ViT-H, attach `visible_mask` as query object, and add background contours."""
        box_thresh = float(self.config.get("gdino_box_thresh", 0.35))
        text_thresh = float(self.config.get("gdino_text_thresh", 0.35))
        _, det_masks, class_names, pred_scores = segment_with_cached_sam(
            self.models,
            img_pil,
            classes=classes,
            target_class=inpaint_prompt,
            box_thresh=box_thresh,
            text_thresh=text_thresh,
        )
        vis_u8 = visible_mask.astype(np.uint8)
        if det_masks is None or len(det_masks) == 0:
            masks = vis_u8[None, :, :]
            class_names = [inpaint_prompt]
            pred_scores = [1.0]
            query_mask_id = 0
        else:
            det_u8 = (np.asarray(det_masks) > 0).astype(np.uint8)
            query_mask_id = len(det_u8)
            masks = np.append(det_u8, [vis_u8], axis=0)
            class_names = list(class_names) + [inpaint_prompt]
            pred_scores = list(pred_scores) + [1.0]

        separated_bg = handlemask(masks)
        if separated_bg:
            bg_arr = np.asarray(separated_bg, dtype=np.uint8)
            masks = np.concatenate((masks, bg_arr), axis=0)
            for idx in range(len(separated_bg)):
                class_names.append(f"background_{idx}")
                pred_scores.append(1.0)
        return masks, class_names, pred_scores, query_mask_id

    def complete(
        self,
        image: np.ndarray,
        visible_mask: Optional[np.ndarray] = None,
        text_query: str = "",
        job_dir: Optional[Path] = None,
        occluded_mask: Optional[np.ndarray] = None,
    ) -> dict:
        t0 = time.perf_counter()
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Proposed requires RGB uint8 image")
        h, w = image.shape[:2]
        img_pil = Image.fromarray(image)
        job = Path(job_dir) if job_dir is not None else None
        if job:
            job.mkdir(parents=True, exist_ok=True)

        def save_json(name: str, data: dict | list):
            if job:
                (job / name).write_text(
                    json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
                )

        mask_source = self.config.get("visible_mask_source", "gt")
        auto_mode = (mask_source == "auto") or (visible_mask is None)
        auto_seg_info = None
        if auto_mode:
            auto_seg_info = predict_auto_visible_mask(self.models, img_pil, text_query, self.config)
            vis = np.asarray(auto_seg_info["pred_visible_mask"], dtype=bool)
            visible_pred_empty = bool(auto_seg_info["visible_pred_empty"])
            save_json(
                "auto_segmentation.json",
                {k: v for k, v in auto_seg_info.items() if k != "pred_visible_mask"},
            )
            if visible_pred_empty:
                amodal_out = np.zeros((h, w), dtype=bool)
                empty_rgb = np.full_like(image, 255)
                rgba = finalize_rgba(empty_rgb, amodal_out, h, w)
                pad = dict(needs_outpaint=False, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0)
                save_json("outpainting.json", pad)
                save_json(
                    "prompt_selection.json",
                    {
                        "prompt": f"a {(text_query or 'object').strip() or 'object'}",
                        "selected_class": (text_query or "object").strip() or "object",
                        "ram_tags": [],
                        "text_query": text_query,
                        "visible_pred_empty": True,
                    },
                )
                save_json(
                    "proposed_selection.json",
                    {
                        "selected_attempt": 0,
                        "outcome": "visible_pred_empty",
                        "visible_pred_empty": True,
                        "critic_accepted": None,
                        "synthesis_attempts": 0,
                        "inpaint_prompt": f"a {(text_query or 'object').strip() or 'object'}",
                    },
                )
                trace = {
                    "pad_info": pad,
                    "pipeline_mode": "proposed",
                    "visible_mask_source": "auto",
                    "visible_pred_empty": True,
                    "auto_segmentation": {k: v for k, v in auto_seg_info.items() if k != "pred_visible_mask"},
                    "backend": "Open-World Amodal (RAM++ + CLIP + Grounded-SAM + InstaOrder + SD2)",
                    "selected_class": (text_query or "object").strip() or "object",
                    "ram_classes": [],
                    "synthesis_attempts": 0,
                    "max_iter": int(self.config.get("max_iter", 3)),
                    "iterations": [],
                    "outcome": "visible_pred_empty",
                    "critic_accepted": None,
                    "critique_history": [],
                    "seconds": round(time.perf_counter() - t0, 3),
                    "coherence": {"seam_mae": 0.0, "color_hist_sim": 0.0},
                }
                return {
                    "inpainted_rgba": rgba,
                    "amodal_mask": amodal_out,
                    "pred_visible_mask": vis,
                    "visible_pred_empty": True,
                    "auto_segmentation": {k: v for k, v in auto_seg_info.items() if k != "pred_visible_mask"},
                    "final_score": None,
                    "critic_accepted": None,
                    "critique_history": [],
                    "_trace": trace,
                }
        else:
            vis = np.asarray(visible_mask, dtype=bool)
            if image.shape != (*vis.shape, 3) or not vis.any():
                raise ValueError("Proposed requires RGB uint8 and a nonempty matching visible mask")
            visible_pred_empty = False

        occ_gt = (np.asarray(occluded_mask, dtype=bool) & ~vis) if occluded_mask is not None else None

        # 1. RAM++ scene tagging & CLIP ViT-B/32 prompt selection on the given visible_mask
        classes, inpaint_prompt, masked_image_np, context_descriptor = self._select_prompt_with_ram_and_clip(
            img_pil, image, vis, text_query=text_query
        )
        classes = remove_duplicates(list(classes) + [inpaint_prompt])
        if job:
            Image.fromarray(masked_image_np).save(job / "masked_image.png")
            (job / "prompt.txt").write_text(inpaint_prompt, encoding="utf-8")
            (job / "reasoning.txt").write_text(
                f"RAM++ tags: {', '.join(classes)}\nSelected CLIP prompt: a {inpaint_prompt}\nContext descriptor: {context_descriptor}",
                encoding="utf-8",
            )
        save_json(
            "prompt_selection.json",
            {
                "prompt": f"a {inpaint_prompt}",
                "selected_class": inpaint_prompt,
                "context_descriptor": context_descriptor,
                "ram_tags": classes,
                "text_query": text_query,
            },
        )

        # 2. GroundingDINO + cached SAM ViT-H instance segmentation + background contour extraction
        masks, class_names, pred_scores, mask_id = self._segment_scene_and_attach_query_mask(
            img_pil, image, vis, classes, inpaint_prompt
        )

        # 3 & 4. InstaOrder occlusion ordering + SD2 Inpainting progressive loop (max_iter = 3)
        max_iter = int(self.config.get("max_iter", 3))
        query_obj = QueryObjectState(
            img=image,
            img_pil=img_pil,
            mask_id=mask_id,
            query_mask=vis.astype(np.uint8),
            query_class=inpaint_prompt,
            context_descriptor=context_descriptor,
            output_img_dir=job,
        )

        # Fixed loop boundary: runs at most `max_iter` iterations (iter_id = 0, 1, ..., max_iter - 1)
        while query_obj.run_iter and query_obj.iter_id < max_iter:
            query_obj = run_progressive_iteration(
                models=self.models,
                query_obj=query_obj,
                masks=masks,
                classes=classes,
                config=self.config,
                fallback_occluded_mask=occ_gt,
                mask_class_names=class_names,
            )

        # 5. Post-process RGBA with smooth edge alpha blending and project back to original HxW frame
        size_mult = int(self.config.get("canvas_multiplier", 6))
        row_slice, col_slice = canvas_center_slice((h, w), size_multiplier=size_mult)

        if (
            query_obj.amodal_completion is not None
            and query_obj.amodal_segmentation is not None
            and query_obj.last_crop_box is not None
            and query_obj.iter_id > 0
        ):
            crop_x_min, crop_x_max, crop_y_min, crop_y_max = query_obj.last_crop_box
            comp_crop_np = np.array(query_obj.amodal_completion.convert("RGB"))
            seg_crop_bool = query_obj.amodal_segmentation > 0

            dst_rgba = np.zeros((*comp_crop_np.shape[:2], 4), dtype=np.uint8)
            dst_rgba[:, :, :3] = comp_crop_np
            dst_rgba[:, :, 3] = seg_crop_bool.astype(np.uint8) * 255

            if job:
                Image.fromarray(dst_rgba).save(job / "amodal_completion_crop.png")
                Image.fromarray(seg_crop_bool.astype(np.uint8) * 255).save(job / "amodal_segmentation.png")

            query_obj.img_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = comp_crop_np
            query_obj.query_mask_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = seg_crop_bool.astype(np.uint8)

            completed_full_rgb = query_obj.img_canvas[row_slice, col_slice].copy()
            predicted_amodal = (query_obj.query_mask_canvas[row_slice, col_slice] > 0) | vis
        else:
            completed_full_rgb = image.copy()
            predicted_amodal = vis.copy()

        amodal_out = (vis | occ_gt) if occ_gt is not None else predicted_amodal
        missing_out = amodal_out & ~vis

        # Blend original crisp visible pixels with inpainted completion in exact HxW image coordinates
        # ONLY along the internal seam between `vis` and `missing_out` (never eroding non-occluded outer edges).
        if np.any(missing_out) and np.any(vis):
            src_rgba_full = np.zeros((h, w, 4), dtype=np.uint8)
            src_rgba_full[:, :, :3] = image
            src_rgba_full[:, :, 3] = vis.astype(np.uint8) * 255

            dst_rgba_full = np.zeros((h, w, 4), dtype=np.uint8)
            dst_rgba_full[:, :, :3] = completed_full_rgb
            dst_rgba_full[:, :, 3] = amodal_out.astype(np.uint8) * 255

            blended_full = alpha_blending(src_rgba_full, dst_rgba_full, transi_wid=5)
            completed_full_rgb = blended_full[:, :, :3].copy()
        else:
            completed_full_rgb[vis] = image[vis]

        completed_full_rgb[~amodal_out] = 255
        rgba = finalize_rgba(completed_full_rgb, amodal_out, h, w)

        pad = dict(needs_outpaint=False, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0)
        save_json("outpainting.json", pad)
        auto_meta = (
            {k: v for k, v in auto_seg_info.items() if k != "pred_visible_mask"}
            if auto_seg_info is not None
            else None
        )
        trace = {
            "pad_info": pad,
            "pipeline_mode": "proposed",
            "visible_mask_source": "auto" if auto_mode else "gt",
            "visible_pred_empty": False,
            "auto_segmentation": auto_meta,
            "backend": "Open-World Amodal (RAM++ + CLIP + Grounded-SAM + InstaOrder + SD2)",
            "selected_class": inpaint_prompt,
            "ram_classes": classes,
            "synthesis_attempts": int(query_obj.iter_id),
            "max_iter": max_iter,
            "iterations": query_obj.iteration_logs,
            "outcome": "completed",
            "critic_accepted": None,
            "critique_history": [],
            "seconds": round(time.perf_counter() - t0, 3),
            "coherence": compute_appearance_coherence_metrics(vis, missing_out, rgba[:, :, :3]),
        }
        save_json(
            "proposed_selection.json",
            {
                "selected_attempt": int(query_obj.iter_id),
                "outcome": "completed",
                "visible_pred_empty": False,
                "critic_accepted": None,
                "synthesis_attempts": int(query_obj.iter_id),
                "retries": max(0, int(query_obj.iter_id) - 1),
            },
        )
        return {
            "input_image": image,
            "visible_mask": vis,
            "pred_visible_mask": vis,
            "visible_pred_empty": False,
            "auto_segmentation": auto_meta,
            "amodal_mask": amodal_out,
            "predicted_amodal_mask": predicted_amodal,
            "inpainted_rgba": rgba,
            "vlm_reasoning": f"a {inpaint_prompt}",
            "critique_history": [],
            "final_score": None,
            "critic_accepted": None,
            "_trace": trace,
        }
