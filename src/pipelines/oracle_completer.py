"""Dedicated Mask-Guided (Oracle-Mask) Amodal Appearance Completion pipeline (`pipeline_mode = 'oracle_mask'`)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import time
from typing import Optional

import numpy as np
from PIL import Image

from src.evaluation.metrics import compute_appearance_coherence_metrics
from src.models.pix2gestalt import load_run_pix2gestalt
from src.stages.appearance_prior import build_oracle_appearance_seed
from src.stages.sd2_refiner import refine_exact_mask_sd2_oracle
from src.stages.shape_consensus import prepare_pix2gestalt_roi_inputs
from src.stages.vlm_guidance import (
    DEFAULT_NEGATIVE_PROMPT,
    build_oracle_vlm_prompts,
    run_oracle_vlm_reasoning,
    run_target_aware_critique,
    truncate_tokens_clean,
    verify_prompt_with_clip,
)
from src.vision.compositing import adaptive_seam_blend, compose_cropped_for_critic, finalize_rgba
from src.vision.roi import compute_square_roi, paste_square_roi


class OracleMaskAmodalCompleter:
    """Dedicated Mask-Guided Amodal Appearance Completer (`pipeline_mode = 'oracle_mask'`)."""

    def __init__(self, models, config: Optional[dict] = None):
        self.models = models
        self.device = models.device
        self.config = config or {}

    def _synthesize_amodal_prior(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        amodal_mask: np.ndarray,
        job_dir: Optional[Path],
    ) -> dict:
        """Run 1 fast Pix2Gestalt pass on an `amodal_mask`-centered ROI for 3D RGB reference."""
        import torch

        predictor = self.models.shape_predictor
        if predictor is None or predictor.model is None:
            raise RuntimeError("Strict mode: Pix2Gestalt model is not loaded.")

        run_pix2gestalt = load_run_pix2gestalt()

        roi_scale = float(self.config.get("oracle_roi_scale", 1.65))
        roi = compute_square_roi(
            amodal_mask,
            image.shape,
            context_scale=roi_scale,
            min_crop=176,
            full_frame_threshold=0.88,
        )
        resized_img, rgb_vis_256 = prepare_pix2gestalt_roi_inputs(image, visible_mask, roi)

        ddim_steps = int(self.config.get("pix2gestalt_ddim_steps", 75))
        p2g_kwargs = {
            "model": predictor.model,
            "device": predictor.device,
            "input_im": resized_img,
            "visible_mask": rgb_vis_256,
            "n_samples": 1,
        }
        try:
            if "ddim_steps" in inspect.signature(run_pix2gestalt).parameters:
                p2g_kwargs["ddim_steps"] = ddim_steps
        except Exception:
            pass

        print(
            f"[Oracle-Prior] Pix2Gestalt Amodal-ROI={roi['size']}x{roi['size']} "
            f"(full_frame={roi['is_full_frame']}, n_samples=1, ddim_steps={ddim_steps})..."
        )
        with torch.inference_mode():
            samples_rgb_256 = run_pix2gestalt(**p2g_kwargs)

        rgb_256 = samples_rgb_256[0]
        rgba_mat = predictor.carvekit_interface([Image.fromarray(rgb_256)])[0]
        alpha_256 = np.asarray(rgba_mat)[:, :, 3]

        blank_rgb = np.full_like(image, 255, dtype=np.uint8)
        blank_alpha = np.zeros(image.shape[:2], dtype=np.float32)
        p2g_rgb_full = paste_square_roi(blank_rgb, rgb_256, roi, resize_mode="linear")
        p2g_alpha_full = np.clip(
            paste_square_roi(blank_alpha, alpha_256.astype(np.float32) / 255.0, roi, resize_mode="linear"),
            0.0,
            1.0,
        )

        if job_dir is not None:
            Image.fromarray(rgb_256).save(job_dir / "pix2gestalt_rgb_256.png")
            Image.fromarray(alpha_256).save(job_dir / "carvekit_alpha_256.png")
            Image.fromarray(amodal_mask.astype(np.uint8) * 255).save(job_dir / "shape_mask_original.png")

        return {
            "p2g_rgb_full": p2g_rgb_full,
            "p2g_alpha_full": p2g_alpha_full,
            "diagnostics": {
                "mask_source": "input_occluded_mask",
                "roi": roi,
                "ddim_steps": ddim_steps,
            },
        }

    def _reason_oracle_aware(
        self,
        context_rgb: np.ndarray,
        blueprint_rgb: np.ndarray,
        user_query: str = "",
    ) -> dict[str, str]:
        return run_oracle_vlm_reasoning(self.models.vlm, context_rgb, blueprint_rgb, user_query=user_query)

    def _verify_prompt_clip(
        self,
        isolated_vis_rgb: np.ndarray,
        parsed_vlm: dict[str, str],
        user_query: str = "",
    ) -> tuple[str, dict]:
        return verify_prompt_with_clip(
            clip_model=self.models.clip_model,
            clip_proc=self.models.clip_processor,
            device=self.device,
            isolated_rgb=isolated_vis_rgb,
            parsed_vlm=parsed_vlm,
            user_query=user_query,
        )

    def _refine_exact_mask_sd2(self, **kwargs) -> np.ndarray:
        return refine_exact_mask_sd2_oracle(pipe=self.models.pipe, device=self.device, **kwargs)

    def _critique_target_aware(
        self,
        critic_completed_rgb: np.ndarray,
        context_highlight_rgb: np.ndarray,
        object_name: str,
    ) -> dict:
        return run_target_aware_critique(
            vlm=self.models.vlm,
            critic_completed_rgb=critic_completed_rgb,
            context_highlight_rgb=context_highlight_rgb,
            object_name=object_name,
            oracle_mode=True,
        )

    def complete(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        occluded_mask: np.ndarray,
        text_query: str = "",
        job_dir: Optional[Path] = None,
    ) -> dict:
        import torch

        if occluded_mask is None:
            raise ValueError("OracleMaskAmodalCompleter requires an explicit `occluded_mask` input.")

        t0 = time.perf_counter()
        h, w = image.shape[:2]
        vis = np.asarray(visible_mask, dtype=bool)
        occ = np.asarray(occluded_mask, dtype=bool) & (~vis)
        if occ.shape != vis.shape:
            raise ValueError(f"occluded_mask shape {occ.shape} != visible_mask shape {vis.shape}")
        amodal_mask = vis | occ

        pad_info = {
            "needs_outpaint": False,
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
        }
        if job_dir is not None:
            (job_dir / "outpainting.json").write_text(json.dumps(pad_info, indent=2), encoding="utf-8")

        if not occ.any():
            rgba = finalize_rgba(image, amodal_mask, h, w)
            if job_dir is not None:
                (job_dir / "reasoning.txt").write_text("Empty occluded_mask input; nothing to restore.", encoding="utf-8")
                (job_dir / "critique_history.json").write_text("[]", encoding="utf-8")
            return {
                "input_image": image,
                "visible_mask": vis,
                "amodal_mask": amodal_mask,
                "inpainted_rgba": rgba,
                "vlm_reasoning": "Empty occluded_mask input; nothing to restore.",
                "critique_history": [],
                "final_score": None,
                "_trace": {
                    "pad_info": pad_info,
                    "synthesis_attempts": 0,
                    "critique_history": [],
                    "backend": "Oracle-Mask (Exact Input Occluded Mask)",
                },
            }

        # 1. Single-pass 3D whole-object appearance prior on exact amodal ROI.
        t_prior0 = time.perf_counter()
        prior_raw = self._synthesize_amodal_prior(image, vis, amodal_mask, job_dir)
        prior_info = build_oracle_appearance_seed(
            image=image,
            visible_mask=vis,
            occluded_mask=occ,
            p2g_rgb_full=prior_raw["p2g_rgb_full"],
            p2g_alpha_full=prior_raw["p2g_alpha_full"],
        )
        t_prior = time.perf_counter() - t_prior0

        seed_canvas = prior_info["seed_canvas"]
        unfilled_mask = prior_info["unfilled_mask"]
        unfilled_frac = float(prior_info["unfilled_fraction"])
        if job_dir is not None:
            Image.fromarray(seed_canvas).save(job_dir / "prior_rgb.png")

        # 2. Exact-blueprint Qwen3-VL reasoning + unbiased CLIP verification.
        t_vlm0 = time.perf_counter()
        print("[Oracle-Branch A] Exact-blueprint semantic reasoning (Qwen3-VL)...")
        context_rgb, blueprint_rgb, isolated_vis_rgb, panel_rgb = build_oracle_vlm_prompts(
            image, vis, occ, target_size=384
        )
        if job_dir is not None:
            Image.fromarray(panel_rgb).save(job_dir / "vlm_visual_prompt.jpg")

        parsed_vlm = self._reason_oracle_aware(context_rgb, blueprint_rgb, user_query=text_query)
        print(
            f"[Oracle VLM] OBJECT='{parsed_vlm['object_name']}' | "
            f"MISSING='{parsed_vlm['occluded_parts']}' | PROMPT='{parsed_vlm['sd_prompt']}'"
        )
        if job_dir is not None:
            (job_dir / "reasoning.txt").write_text(parsed_vlm["raw"], encoding="utf-8")

        base_prompt, clip_info = self._verify_prompt_clip(isolated_vis_rgb, parsed_vlm, user_query=text_query)
        t_vlm = time.perf_counter() - t_vlm0
        if job_dir is not None:
            (job_dir / "prompt_selection.json").write_text(
                json.dumps({"prompt": base_prompt, "parsed_vlm": parsed_vlm, "clip_verification": clip_info}, indent=2),
                encoding="utf-8",
            )

        # 3. Exact-mask 512x512 ROI SD2 refinement + Best-of-N Critic selection.
        t_syn0 = time.perf_counter()
        sd_tok = self.models.pipe.tokenizer
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
            float(self.config.get("unfilled_refine_strength", 0.62))
            if unfilled_frac >= 0.25
            else float(self.config.get("prior_refine_strength", 0.24))
        )
        strength_step = 0.12 if unfilled_frac >= 0.25 else 0.08
        guidance_scale = 6.0 if unfilled_frac < 0.25 else 7.0
        num_steps = 28
        base_seed = int(self.config.get("seed", 42))
        enable_critic = bool(self.config.get("enable_critic", True))
        critique_threshold = float(self.config.get("critique_threshold", 7.0))
        max_critic_iter = int(self.config.get("max_critic_iter", 2))
        n_iters = max_critic_iter if enable_critic else 1

        critique_history: list[dict] = []
        best_rgba: Optional[np.ndarray] = None
        best_rgb: Optional[np.ndarray] = None
        best_score: float = -1.0
        attempts = 0

        for crit_iter in range(n_iters):
            attempts += 1
            iter_strength = min(base_strength + strength_step * crit_iter, 0.88)
            print(
                f"[Oracle-Synthesis] Attempt {crit_iter + 1}/{n_iters} "
                f"(strength={iter_strength:.2f}, unfilled={unfilled_frac:.2f}, guidance={guidance_scale:.1f})"
            )
            if job_dir is not None:
                (job_dir / f"prompt_{attempts:02d}.txt").write_text(prompt, encoding="utf-8")

            inpaint_img = self._refine_exact_mask_sd2(
                seed_canvas=seed_canvas,
                visible_mask=vis,
                occluded_mask=occ,
                unfilled_mask=unfilled_mask,
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
            rgba = finalize_rgba(blended_rgb, amodal_mask, h, w)

            if not enable_critic:
                best_rgba = rgba
                best_rgb = blended_rgb
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
                best_rgb = blended_rgb

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

        coherence = compute_appearance_coherence_metrics(image, vis, occ, best_rgb if best_rgb is not None else image)
        t_syn = time.perf_counter() - t_syn0
        t_total = time.perf_counter() - t0
        print(
            f"[Timing] prior={t_prior:.1f}s | vlm={t_vlm:.1f}s | sd2+critic={t_syn:.1f}s | total={t_total:.1f}s "
            f"(seam_mae={coherence['seam_mae']}, color_sim={coherence['color_hist_sim']})",
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
                "shape_diagnostics": prior_raw["diagnostics"],
                "prior_diagnostics": {
                    "unfilled_fraction": prior_info["unfilled_fraction"],
                    "shadow_suppressed_fraction": prior_info["shadow_suppressed_fraction"],
                    "color_gain": prior_info["color_gain"],
                    "color_bias": prior_info["color_bias"],
                    "ref_p90_luma": prior_info["ref_p90_luma"],
                    "seam_mae": coherence["seam_mae"],
                    "color_hist_sim": coherence["color_hist_sim"],
                },
                "backend": "Oracle-Mask (Exact Input Occluded Mask)",
            },
        }

