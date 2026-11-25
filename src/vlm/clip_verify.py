# Phạm vi: proposed: CLIP đối chiếu các prompt với crop vật thể nhìn thấy; đây là lựa chọn prompt,
# không phải metric CLIP cuối.
"""Verify text guidance against the observed object crop."""
from __future__ import annotations
import numpy as np
from PIL import Image

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


# So prompt Qwen, tên vật thể và prompt dự phòng trên cùng crop visible; trả prompt cùng điểm kiểm
# tra.
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
    from src.vlm.prompt_inputs import truncate_tokens_clean

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
