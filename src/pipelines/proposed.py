"""Generate and critique raw decoder candidates using visual and semantic guidance."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.pix2gestalt_prior import synthesize_amodal_prior
from src.vlm.clip_verify import verify_prompt_with_clip
from src.stages.appearance_prior import build_oracle_appearance_seed
from src.stages.inpaint_refiner import refine_exact_mask
from src.models.sd15_ip_adapter import AdvancedInpaintPass, validate_proposed_config
from src.vlm.prompt_inputs import build_vlm_visual_prompts, truncate_tokens_clean, DEFAULT_NEGATIVE_PROMPT
from src.vlm.micro_attributes import run_micro_reasoning
from src.vlm.strict_critic import run_strict_critic
from src.vision.compositing import finalize_rgba
from src.vision.roi import extract_square_roi, mask_bbox
from src.eval.metrics import compute_appearance_coherence_metrics


# Cắt bốn bảng cùng ROI: cảnh gốc, vùng thấy, kết quả và bản đồ vùng sinh.
def critic_panels(image, visible, amodal, completed, size=384):
    """Four spatially registered panels; background and target cannot swap identities."""
    y1, x1, y2, x2 = mask_bbox(amodal)
    side = max(32, int(round(max(y2 - y1 + 1, x2 - x1 + 1) * 1.2)))
    roi = dict(y0=(y1 + y2) // 2 - side // 2, x0=(x1 + x2) // 2 - side // 2, size=side)
    observed = np.full_like(image, 255)
    observed[visible] = image[visible]
    result = np.full_like(image, 255)
    result[amodal] = completed[amodal]
    regions = np.zeros_like(image)
    regions[visible] = 255
    regions[amodal & ~visible] = [40, 130, 255]
    arrays = [image, observed, result, regions]
    return [np.array(Image.fromarray(extract_square_roi(a, roi, pad_value=0 if i == 3 else 255))
                     .resize((size, size), Image.Resampling.NEAREST if i == 3 else Image.Resampling.LANCZOS))
            for i, a in enumerate(arrays)]


# Ưu tiên ứng viên PASS; nếu chưa PASS thì chọn điểm hợp lệ cao nhất.
def candidate_rank(critique):
    """A valid PASS always outranks a rejection; invalid JSON is not a score of zero."""
    score = critique.get("score")
    return (bool(critique.get("accepted")), float(score) if score is not None else float("-inf"))


class Proposed:
    # Xác thực config và lưu model manager dùng chung.
    def __init__(self, models, config):
        validate_proposed_config(config)
        self.models, self.config, self.device = models, config, models.device

    # Tạo crop ngữ cảnh và crop visible riêng; suy luận trước khi tạo prior.
    def _reason(self, image, vis, query):
        # Step 1 uses only image + visible mask; no hidden-mask geometry or labels.
        context, isolated, panel = build_vlm_visual_prompts(image, vis, vis, target_size=384)
        reasoning = run_micro_reasoning(self.models.vlm, context, isolated, query)
        return reasoning, context, isolated, panel

    # Dùng hợp visible/hidden làm biên cố định để tạo prior RGB và seed tinh chỉnh.
    def _prior(self, image, vis, occ, job):
        amodal = vis | occ
        raw = synthesize_amodal_prior(self.models, self.config, image, vis, amodal, job)
        info = build_oracle_appearance_seed(image, vis, occ, raw["p2g_rgb_full"], raw["p2g_alpha_full"])
        return amodal, raw, info

    # So khớp prompt với crop visible để kiểm tra danh tính vật thể.
    def _verify_prompt(self, isolated, reasoning, query):
        return verify_prompt_with_clip(self.models.clip_model, self.models.clip_processor,
                                       self.device, isolated, reasoning, user_query=query)

    def _inpaint_pipe(self, isolated, seed):
        return AdvancedInpaintPass(self.models.pipe, self.config, Image.fromarray(isolated), self.device, seed)

    # Chạy SD1.5 trong ROI theo seed, mask hidden và tham số của lượt hiện tại.
    def _refine(self, prior, vis, amodal, isolated, prompt, negative, strength, guidance, seed):
        # Requesting .pipe offloads Qwen before activating SD1.5.
        pipe = self._inpaint_pipe(isolated, seed)
        common = dict(pipe=pipe, device=self.device, seed_canvas=prior["seed_canvas"], visible_mask=vis,
                      prompt=prompt, negative_prompt=negative, strength=strength, guidance_scale=guidance,
                      num_inference_steps=int(self.config.get("num_inference_steps", 28)), seed=seed)
        rgb = refine_exact_mask(**common, occluded_mask=amodal & ~vis)
        return rgb, getattr(pipe, "diagnostics", {})

    def _critique(self, panels, context, name):
        return run_strict_critic(self.models.vlm, panels, name, self.config["critique_threshold"])

    # Chạy đầy đủ một mẫu, lưu từng ứng viên và trả ảnh RGBA được chọn.
    def complete(self, image, visible_mask, text_query="", job_dir=None, occluded_mask=None):
        t0 = time.perf_counter()
        vis = np.asarray(visible_mask, bool)
        if image.dtype != np.uint8 or image.shape != (*vis.shape, 3) or not vis.any():
            raise ValueError("Proposed requires RGB uint8 and a nonempty matching visible mask")
        if occluded_mask is None or np.asarray(occluded_mask).shape != vis.shape:
            raise ValueError("A matching hidden mask is required")
        occ = np.asarray(occluded_mask, bool) & ~vis
        job = Path(job_dir) if job_dir is not None else None
        if job:
            job.mkdir(parents=True, exist_ok=True)
            (job / "critic_logs.jsonl").write_text("", encoding="utf-8")

        # Lưu artefact chẩn đoán khi runner có cung cấp thư mục mẫu.
        def save_json(name, data):
            if job:
                (job / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

        # 1. Tách phần nhìn thấy ra khỏi ngữ cảnh trước khi nhờ Qwen mô tả bề mặt.
        reasoning, context, isolated, panel = self._reason(image, vis, text_query)
        save_json("micro_attributes.json", reasoning)
        if job:
            (job / "reasoning.txt").write_text(reasoning["raw"], encoding="utf-8")
            Image.fromarray(panel).save(job / "vlm_visual_prompt.jpg")
            Image.fromarray(isolated).save(job / "visible_reference.png")
        # 2. Dựng seed RGB từ prior. Mask đầu ra đã biết từ annotation.
        amodal, raw_prior, prior = self._prior(image, vis, occ, job)
        missing = amodal & ~vis
        pad = dict(needs_outpaint=False, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0)
        save_json("outpainting.json", pad)
        if job:
            Image.fromarray(prior["seed_canvas"]).save(job / "prior_rgb.png")

        # Preserve CLIP verification as a guard on object identity. When Qwen is
        # accepted, retain its attributes even if CLIP prefers a shorter noun phrase.
        prompt, clip_info = self._verify_prompt(isolated, reasoning, text_query)
        if clip_info["verified"] and reasoning.get("valid_json"):
            prompt = reasoning["sd_prompt"]
        save_json("prompt_selection.json", dict(prompt=prompt, clip_verification=clip_info))
        tokenizer = self.models.pipe.tokenizer
        base_prompt = truncate_tokens_clean(prompt, tokenizer, max_tokens=60)
        negative = truncate_tokens_clean(DEFAULT_NEGATIVE_PROMPT, tokenizer, max_tokens=70)
        # 3. Prior bị thiếu nhiều thì tăng strength để SD1.5 có thêm khả năng vẽ lại.
        fraction = float(prior["unfilled_fraction"])
        high_strength = self.config.get("unfilled_refine_strength", .62)
        strength = float(high_strength if fraction >= .25 else self.config.get("prior_refine_strength", .24))
        history, best, best_critique, selected = [], None, None, None
        limit = self.config["max_critic_iter"]
        # 4. Mỗi lượt sinh rồi chấm; phản hồi gần nhất có thể sửa prompt của lượt sau.
        for attempt in range(limit):
            seed = int(self.config["seed"]) + attempt * 1009
            guidance = min(9.0, float(self.config["guidance_scale"]) + attempt * float(self.config["retry_guidance_step"]))
            repair = history[-1].get("repair_instruction", "") if history else ""
            # Keep attributes and leave room for a concise positive repair request.
            active_prompt = truncate_tokens_clean(
                (truncate_tokens_clean(base_prompt, tokenizer, 40) + ", " + repair) if repair else base_prompt,
                tokenizer, 70)
            start = time.perf_counter()
            generated, diagnostics = self._refine(prior, vis, amodal, isolated, active_prompt, negative,
                                                   strength, guidance, seed)
            rgba = finalize_rgba(generated, amodal, *image.shape[:2])
            synthesis_seconds = time.perf_counter() - start
            panels = critic_panels(image, vis, amodal, generated)
            critique = self._critique(panels, context, reasoning["object_name"])
            entry = dict(critique, attempt=attempt + 1, retries=attempt, seed=seed, guidance_scale=guidance,
                         strength=strength, prompt=active_prompt, synthesis_seconds=round(synthesis_seconds, 3),
                         attempt_seconds=round(time.perf_counter() - start, 3), refinement=diagnostics)
            history.append(entry)
            if best is None or candidate_rank(critique) > candidate_rank(best_critique):
                best, best_critique, selected = rgba, critique, attempt + 1
            if job:
                Image.fromarray(rgba).save(job / f"candidate_{attempt + 1:02d}.png")
                Image.fromarray(np.concatenate(panels, axis=1)).save(job / f"critic_input_{attempt + 1:02d}.png")
                with (job / "critic_logs.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                save_json("critique_history.json", history)
            print(f"[Proposed Critic {attempt + 1}/{limit}] {entry['final_decision']} | {entry['reason']}", flush=True)
            if critique["accepted"]:
                break

        # 5. Nếu critic từ chối mọi lượt, vẫn xuất ứng viên tốt nhất và ghi trạng thái riêng.
        accepted = best_critique["accepted"]
        outcome = "accepted" if accepted else "exhausted_retries"
        trace = dict(pad_info=pad, pipeline_mode="proposed", synthesis_attempts=len(history),
                     retries=max(0, len(history) - 1), selected_attempt=selected, outcome=outcome,
                     critic_accepted=accepted, critique_history=history, shape_diagnostics=raw_prior["diagnostics"],
                     backend="proposed",
                     seconds=round(time.perf_counter() - t0, 3),
                     coherence=compute_appearance_coherence_metrics(vis, missing, best[:, :, :3]))
        save_json("proposed_selection.json", {k: trace[k] for k in
                  ("selected_attempt", "outcome", "critic_accepted", "synthesis_attempts", "retries")})
        return dict(input_image=image, visible_mask=vis, amodal_mask=amodal, inpainted_rgba=best,
                    vlm_reasoning=reasoning["raw"], clip_verification=clip_info, critique_history=history,
                    final_score=best_critique["score"] if best_critique else None,
                    critic_accepted=accepted, _trace=trace)
