# Phạm vi: Proposed: giao tiếp Qwen bằng ảnh và text, kiểm tra JSON và lưu phản hồi thô nếu cần sửa định
# dạng.
"""Shared Qwen JSON generation and parsing."""
import json

# Bóc code fence nếu có, sau đó yêu cầu phản hồi phân tích được thành object JSON.
def _json_object(raw):
    text = str(raw).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0].lower() not in ("```", "```json"):
            raise ValueError("Unsupported code fence")
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


# Xác minh kiểu và độ dài của trường văn bản trước khi dùng làm prompt.
def _string(data, name, limit=500):
    value = data.get(name)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"Invalid {name}: expected a nonempty short string")
    return " ".join(value.split())


# Đóng gói ảnh/text cho Qwen, giải mã phần token của câu trả lời.
def generate_json_response(vlm, images, system, instruction):
    """Deterministic multimodal generation; JSON is subsequently schema-validated."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "system", "content": system}, {"role": "user", "content":
        [{"type": "image", "image": Image.fromarray(im)} for im in images]
        + [{"type": "text", "text": instruction}]}]
    text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm.processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(vlm.device)
    with torch.inference_mode():
        ids = vlm.model.generate(**inputs, max_new_tokens=384, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, ids)]
    return vlm.processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0].strip()


# Thử parse và yêu cầu Qwen sửa định dạng nếu cần, giữ cả phản hồi lỗi để chẩn đoán.
def validated_generation(vlm, images, system, instruction, parser, repair_attempts=1):
    """Retry JSON formatting once; preserve every raw response and validation error."""
    responses = []
    for index in range(repair_attempts + 1):
        raw = generate_json_response(vlm, images, system, instruction)
        entry = {"raw": raw, "validation_error": None}
        responses.append(entry)
        try:
            return parser(raw), responses
        except (ValueError, TypeError) as error:
            entry["validation_error"] = str(error)
            instruction = (f"{instruction}\nYour previous answer was invalid: {error}. "
                           "Reinspect the images and return one valid JSON object with ALL required fields.")
    return None, responses
