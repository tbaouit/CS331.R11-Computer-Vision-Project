"""Visual prompt rendering, Qwen3-VL reasoning/critique, and CLIP prompt verification."""
from __future__ import annotations

import re
import numpy as np
from PIL import Image

from src.vision.image_ops import binary_erode, mask_outline, resize_array
from src.vision.roi import compute_square_roi, extract_square_roi, mask_bbox

DEFAULT_NEGATIVE_PROMPT = (
    "gray background, flat gray patch, gray fill, occluded, cut off, cropped, "
    "blurry, low quality, distorted, deformed, extra limbs, floating head, "
    "text, watermark, artifacts"
)

FALLBACK_PROMPT_CANDIDATES = [
    "a person",
    "a piece of luggage or bag",
    "a container or tray",
    "a ceramic bowl or dish",
    "a glass jar or bottle",
    "a stuffed toy or teddy bear",
    "sports equipment or racket",
    "a building or lighthouse",
    "a vehicle",
    "furniture",
    "an animal",
    "an object on a white background",
]


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


def parse_structured_vlm_reasoning(raw_text: str, fallback_query: str = "") -> dict[str, str]:
    """Extract `object_name`, `occluded_parts`, and clean SD2 `sd_prompt`."""
    cleaned = re.sub(r"\*+", "", raw_text or "").strip()

    obj_match = re.search(r"OBJECT\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    occ_match = re.search(r"OCCLUDED(?:_PARTS|\s+PARTS)?\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    prm_match = re.search(r"PROMPT\s*:\s*(.+?)(?:\n[A-Z_]+\s*:|$)", cleaned, re.IGNORECASE | re.DOTALL)

    object_name = obj_match.group(1).strip(" .,\"'") if obj_match else ""
    occluded_parts = occ_match.group(1).strip() if occ_match else ""
    sd_prompt = " ".join(prm_match.group(1).split()).strip(" .,\"'") if prm_match else ""

    if not object_name:
        object_name = fallback_query.strip() or "object"

    if not sd_prompt:
        prose = " ".join(cleaned.split())
        prose = re.sub(
            r"^(the target object is|this is|inpainting prompt\s*:)\s*",
            "",
            prose,
            flags=re.IGNORECASE,
        )
        sd_prompt = prose.split(".")[0].strip() or f"complete {object_name} isolated on a white background"

    return {
        "object_name": object_name,
        "occluded_parts": occluded_parts,
        "sd_prompt": sd_prompt,
        "raw": raw_text.strip() if raw_text else "",
    }


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


def build_oracle_vlm_prompts(
    image: np.ndarray,
    visible_mask: np.ndarray,
    occluded_mask: np.ndarray,
    target_size: int = 384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact-occlusion visual prompts for Qwen3-VL + clean isolated crop for CLIP."""
    vis = np.asarray(visible_mask, dtype=bool)
    occ = np.asarray(occluded_mask, dtype=bool) & (~vis)
    amodal = vis | occ

    highlighted = _render_censored_scene(image, vis, occ)
    ctx_roi = compute_square_roi(amodal, image.shape, context_scale=2.0, min_crop=192, full_frame_threshold=0.92)
    ctx_patch = extract_square_roi(highlighted, ctx_roi, pad_value=48)
    context_rgb = resize_array(ctx_patch, (target_size, target_size), mode="linear")

    ay1, ax1, ay2, ax2 = mask_bbox(amodal)
    ah, aw = ay2 - ay1 + 1, ax2 - ax1 + 1
    bp_side = max(64, int(round(max(ah, aw) * 1.24)))
    bp_roi = {
        "y0": ((ay1 + ay2) // 2) - bp_side // 2,
        "x0": ((ax1 + ax2) // 2) - bp_side // 2,
        "size": bp_side,
        "is_full_frame": False,
    }
    blueprint_scene = np.full_like(image, 255, dtype=np.uint8)
    if occ.any():
        blueprint_scene[occ] = np.array([224, 236, 252], dtype=np.uint8)
        occ_border = occ & (~binary_erode(occ, 2))
        blueprint_scene[occ_border] = np.array([75, 135, 225], dtype=np.uint8)
    blueprint_scene[vis] = image[vis]
    bp_patch = extract_square_roi(blueprint_scene, bp_roi, pad_value=255)
    blueprint_rgb = resize_array(bp_patch, (target_size, target_size), mode="linear")

    isolated_vis_rgb = _render_isolated_visible_crop(image, vis, target_size)

    sep = np.full((target_size, 6, 3), 220, dtype=np.uint8)
    panel_rgb = np.concatenate([context_rgb, sep, blueprint_rgb, sep, isolated_vis_rgb], axis=1)
    return context_rgb, blueprint_rgb, isolated_vis_rgb, panel_rgb


def _generate_qwen_vl(vlm, image_a: np.ndarray, image_b: np.ndarray, instruction: str, max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    if vlm is None or vlm.model is None:
        raise RuntimeError("Strict mode: Qwen3-VL model is not loaded.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.fromarray(image_a)},
                {"type": "image", "image": Image.fromarray(image_b)},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm.device)

    with torch.no_grad():
        generated_ids = vlm.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None
        )

    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return vlm.processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def run_blind_vlm_reasoning(
    vlm,
    context_rgb: np.ndarray,
    isolated_rgb: np.ndarray,
    user_query: str = "",
) -> dict[str, str]:
    hint_line = f"Optional category hint: '{user_query.strip()}'.\n" if user_query and user_query.strip() else ""
    instruction = (
        "You are analyzing a partially occluded target object for amodal appearance completion.\n"
        "- Image 1 shows the local scene where ONLY the target object's visible region is at full "
        "brightness inside a BRIGHT GREEN outline, and the occluded region to reconstruct is covered "
        "by a DIAGONAL GRAY STRIPED CENSOR MASK inside a YELLOW outline.\n"
        "- Image 2 shows ONLY the target object's visible pixels isolated on a pure white background.\n"
        f"{hint_line}"
        "CRITICAL RULES:\n"
        "1. Identify ONLY the BRIGHT GREEN target object shown in Image 2.\n"
        "2. Write PROMPT as a concise (12-26 words) studio photo description of the COMPLETE, "
        "UNOCCLUDED target object isolated on a plain white background, matching the exact colors, "
        "materials, patterns, and lighting visible in Image 2.\n"
        "3. If the target object is a container, takeout box, tray, bowl, plate, dish, or surface "
        "whose interior was covered by the yellow-outlined censor mask, describe it as EMPTY and clean "
        "inside (NEVER imagine food, sandwiches, or items inside it).\n"
        "4. Do NOT mention occluders, masks, stripes, green/yellow outlines, or missing parts in PROMPT.\n\n"
        "Respond in this exact 3-line format:\n"
        "OBJECT: <short noun phrase naming the target object>\n"
        "OCCLUDED_PARTS: <one short sentence naming which part of the target object is hidden>\n"
        "PROMPT: <concise description of the complete unoccluded target object on a white background>"
    )
    raw_output = _generate_qwen_vl(vlm, context_rgb, isolated_rgb, instruction, max_new_tokens=140)
    return parse_structured_vlm_reasoning(raw_output, fallback_query=user_query)


def run_oracle_vlm_reasoning(
    vlm,
    context_rgb: np.ndarray,
    blueprint_rgb: np.ndarray,
    user_query: str = "",
) -> dict[str, str]:
    hint_line = f"Optional category hint: '{user_query.strip()}'.\n" if user_query and user_query.strip() else ""
    instruction = (
        "You are guiding exact-mask amodal appearance completion for a partially occluded object.\n"
        "- Image 1 shows the scene where ONLY the target object's visible pixels are inside a BRIGHT GREEN "
        "outline, and the exact occluded region to restore is censored by diagonal gray stripes inside YELLOW.\n"
        "- Image 2 shows the target object isolated on pure white, with the exact missing silhouette marked in "
        "LIGHT SKY-BLUE next to the visible object pixels.\n"
        f"{hint_line}"
        "CRITICAL RULES:\n"
        "1. Identify ONLY the target object shown in Image 2 (never describe any removed foreground occluder).\n"
        "2. Use the exact LIGHT SKY-BLUE silhouette in Image 2 to see which anatomical/structural part is missing.\n"
        "3. Write PROMPT as a concise (12-26 words) studio photo description of the COMPLETE, UNOCCLUDED target "
        "object on a plain white background, matching the exact colors, materials, and lighting in Image 2.\n"
        "4. If the target object is a container, takeout box, tray, bowl, plate, or dish whose interior is "
        "marked in light blue, describe it as EMPTY and clean inside (NEVER imagine food or items inside it).\n"
        "5. Do NOT mention masks, stripes, outlines, or light-blue regions in PROMPT.\n\n"
        "Respond in this exact 3-line format:\n"
        "OBJECT: <short noun phrase naming the target object>\n"
        "OCCLUDED_PARTS: <one short sentence naming the exact missing part shown in light blue>\n"
        "PROMPT: <concise description of the complete unoccluded target object on a white background>"
    )
    raw_output = _generate_qwen_vl(vlm, context_rgb, blueprint_rgb, instruction, max_new_tokens=140)
    return parse_structured_vlm_reasoning(raw_output, fallback_query=user_query)


def verify_prompt_with_clip(
    clip_model,
    clip_proc,
    device: str,
    isolated_rgb: np.ndarray,
    parsed_vlm: dict[str, str],
    user_query: str = "",
) -> tuple[str, dict]:
    """Verify VLM prompt alignment against isolated visible pixels using shared CLIP."""
    import torch

    clip_tok = clip_proc.tokenizer
    qwen_prompt = truncate_tokens_clean(parsed_vlm["sd_prompt"], clip_tok, max_tokens=65)
    obj_phrase = truncate_tokens_clean(
        f"a photo of {parsed_vlm['object_name']} isolated on a white background",
        clip_tok,
        max_tokens=40,
    )

    fallbacks = list(FALLBACK_PROMPT_CANDIDATES)
    if user_query and user_query.strip():
        fallbacks.insert(0, f"a photo of {user_query.strip()}")

    candidates = [qwen_prompt, obj_phrase] + fallbacks
    text_inputs = clip_tok(
        candidates,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    if text_inputs["input_ids"].shape[1] > 77:
        text_inputs = {k: v[:, :77] for k, v in text_inputs.items()}

    crop_pil = Image.fromarray(isolated_rgb)
    image_inputs = clip_proc.image_processor([crop_pil] * len(candidates), return_tensors="pt")
    clip_inputs = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device),
        "pixel_values": image_inputs["pixel_values"].to(device),
    }

    with torch.no_grad():
        outputs = clip_model(**clip_inputs)
        sims = (outputs.image_embeds[0:1] @ outputs.text_embeds.T).squeeze(0).detach().cpu().tolist()

    qwen_score = float(sims[0])
    obj_score = float(sims[1])
    fb_scores = sims[2:]
    best_fb_idx = max(range(len(fb_scores)), key=lambda i: fb_scores[i])
    best_fb = candidates[2 + best_fb_idx]
    best_fb_score = float(fb_scores[best_fb_idx])

    best_qwen_score = max(qwen_score, obj_score)
    best_qwen_text = qwen_prompt if qwen_score >= obj_score - 0.01 else obj_phrase

    verified = (best_qwen_score >= 0.18) and (best_qwen_score >= best_fb_score - 0.03)
    chosen = best_qwen_text if verified else best_fb
    status = "ACCEPTED" if verified else "REJECTED"
    print(
        f"[CLIP Verify] Qwen {status} "
        f"(prompt={qwen_score:.3f}, obj={obj_score:.3f}, best_fb='{best_fb}'={best_fb_score:.3f})"
    )

    return chosen, {
        "verified": bool(verified),
        "object_name": parsed_vlm["object_name"],
        "qwen_clip_score": round(qwen_score, 4),
        "object_clip_score": round(obj_score, 4),
        "best_fallback": best_fb,
        "best_fallback_score": round(best_fb_score, 4),
        "chosen": chosen,
    }


def _parse_critic_response(output_text: str) -> dict:
    cleaned = re.sub(r"\*+", "", output_text)

    def _parse_score(label: str) -> float:
        match = re.search(rf"{label}\s*:\s*([\d.]+)", cleaned, re.IGNORECASE)
        if not match:
            raise ValueError(f"Critic response missing {label}: {output_text}")
        val = float(match.group(1))
        if not 0.0 <= val <= 10.0:
            raise ValueError(f"Critic score out of range: {label}={val}")
        return val

    structural = _parse_score("STRUCTURAL")
    texture = _parse_score("TEXTURE")
    context = _parse_score("CONTEXT")
    fb_match = re.search(r"FEEDBACK\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE | re.DOTALL)
    feedback = fb_match.group(1).strip() if fb_match else ""

    return {
        "score": (structural + texture + context) / 3.0,
        "structural": structural,
        "texture": texture,
        "context": context,
        "feedback": feedback,
        "raw": output_text,
    }


def run_target_aware_critique(
    vlm,
    critic_completed_rgb: np.ndarray,
    context_highlight_rgb: np.ndarray,
    object_name: str,
    oracle_mode: bool = False,
) -> dict:
    if oracle_mode:
        instruction = (
            f"You are a strict quality critic for mask-guided amodal appearance completion of a '{object_name}'.\n"
            "- Image 1 shows the reconstructed complete target object isolated on a white background.\n"
            "- Image 2 shows the scene where the target's visible region is inside BRIGHT GREEN and the removed "
            "foreground occluder is covered by a diagonal striped censor mask inside YELLOW.\n\n"
            "Evaluate ONLY the reconstructed target object in Image 1 on a 0-10 integer scale:\n"
            "1. STRUCTURAL: internal anatomical/structural continuity across the reconstructed region.\n"
            "2. TEXTURE: seamless color, material, and lighting transition between visible and restored parts (penalize seams, gray smudges, or hallucinated items).\n"
            "3. CONTEXT: semantic fidelity to the BRIGHT GREEN target object in Image 2 (note: if the target is a container/tray/dish, a clean EMPTY interior is 10/10 correct).\n\n"
            "Respond EXACTLY in this 4-line format:\n"
            "STRUCTURAL: <int>\n"
            "TEXTURE: <int>\n"
            "CONTEXT: <int>\n"
            "FEEDBACK: <one short actionable fix for texture/color continuity, or 'good' if seamless>"
        )
    else:
        instruction = (
            f"You are a strict quality critic for amodal object completion of a '{object_name}'.\n"
            "- Image 1 shows the reconstructed complete target object isolated on a white background.\n"
            "- Image 2 shows the local scene where the target's visible region is inside a BRIGHT GREEN outline "
            "and the removed foreground occluder is covered by a diagonal striped censor mask inside YELLOW.\n\n"
            "Evaluate ONLY the reconstructed target object in Image 1 on a 0-10 integer scale:\n"
            "1. STRUCTURAL: whole-object shape plausibility, natural proportions, and clean silhouette.\n"
            "2. TEXTURE: color, material, and lighting continuity between visible and completed regions (penalize flat gray patches or hallucinated foreign items).\n"
            "3. CONTEXT: semantic consistency with the BRIGHT GREEN target object in Image 2 (note: if the target is a container/tray/dish, a clean EMPTY interior is 10/10 correct because the item inside was the removed occluder).\n\n"
            "Respond EXACTLY in this 4-line format:\n"
            "STRUCTURAL: <int>\n"
            "TEXTURE: <int>\n"
            "CONTEXT: <int>\n"
            "FEEDBACK: <one short actionable fix for the object appearance, or 'good' if seamless>"
        )

    output_text = _generate_qwen_vl(vlm, critic_completed_rgb, context_highlight_rgb, instruction, max_new_tokens=120)
    return _parse_critic_response(output_text)

