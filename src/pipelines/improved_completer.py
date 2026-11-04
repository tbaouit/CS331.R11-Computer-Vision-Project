"""Blind end-to-end Amodal Appearance Completion pipeline (`pipeline_mode = 'improved'`)."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import time
from typing import Optional

import numpy as np
from PIL import Image

from src.stages.appearance_prior import build_appearance_prior
from src.stages.sd2_refiner import refine_roi_sd2_blind
from src.stages.shape_consensus import (
    fuse_pix2gestalt_predictions,
    prepare_pix2gestalt_roi_inputs,
)
from src.stages.vlm_guidance import (
    DEFAULT_NEGATIVE_PROMPT,
    build_vlm_visual_prompts,
    run_blind_vlm_reasoning,
    run_target_aware_critique,
    truncate_tokens_clean,
    verify_prompt_with_clip,
)
from src.vision.compositing import adaptive_seam_blend, compose_cropped_for_critic
from src.vision.roi import compute_square_roi


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
        resized_img, rgb_vis_256 = prepare_pix2gestalt_roi_inputs(image, visible_mask, roi)

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
        return run_blind_vlm_reasoning(self.cls._vlm, context_rgb, isolated_rgb, user_query=user_query)

    def _verify_prompt_clip(
        self,
        isolated_rgb: np.ndarray,
        parsed_vlm: dict[str, str],
        user_query: str = "",
    ) -> tuple[str, dict]:
        return verify_prompt_with_clip(
            clip_model=self.cls._clip_model,
            clip_proc=self.cls._clip_processor,
            device=self.device,
            isolated_rgb=isolated_rgb,
            parsed_vlm=parsed_vlm,
            user_query=user_query,
        )

    def _refine_roi_sd2(self, **kwargs) -> np.ndarray:
        return refine_roi_sd2_blind(pipe=self.cls._pipe, device=self.device, **kwargs)

    def _critique_target_aware(
        self,
        critic_completed_rgb: np.ndarray,
        context_highlight_rgb: np.ndarray,
        object_name: str,
    ) -> dict:
        return run_target_aware_critique(
            vlm=self.cls._vlm,
            critic_completed_rgb=critic_completed_rgb,
            context_highlight_rgb=context_highlight_rgb,
            object_name=object_name,
            oracle_mode=False,
        )

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
        prompt = truncate_tokens_clean(
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
        neg_prompt = truncate_tokens_clean(DEFAULT_NEGATIVE_PROMPT + extra_neg, sd_tok, max_tokens=70)

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
                prompt = truncate_tokens_clean(
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
