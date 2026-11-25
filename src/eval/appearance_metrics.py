# Phạm vi: Cả hai phiên bản: CLIP logit chấm ảnh hoàn chỉnh; LPIPS/SSIM/PSNR và cosine chỉ kiểm tra
# phần nhìn thấy.
"""Author protocol: full-canvas RGB masked to black outside the visible region.

These reference metrics measure visible preservation, not hidden RGB accuracy.
CLIP class relevance alone uses the entire unmasked output RGB (alpha ignored).
"""
from __future__ import annotations

import numpy as np

CLIP_METRIC_MODEL_ID = "openai/clip-vit-base-patch32"
PROTOCOL = "decoded_rgb_visible_black"
METRIC_NAMES = ("CLIP_score", "LPIPS", "Feature_Similarity", "SSIM", "PSNR")
POLICY = {
    "protocol": PROTOCOL,
    "output": "decoded ROI RGB after paste-back; given amodal alpha",
    "seam_and_color": "compare visible and hidden regions on the output RGB",
    "reference": "original input RGB; no hidden RGB ground truth",
    "CLIP_score": "full output RGB vs raw annotation class name; scaled CLIP image/text logit",
    "visible_metrics": "original canvas, black outside visible mask; no crop or manual resize",
    "feature_backbone": CLIP_METRIC_MODEL_ID,
    "lpips_backbone": "alex; uint8 RGB normalized to [-1,1]",
    "SSIM": "skimage RGB, channel_axis=2, data_range=255, default 7x7 window",
    "PSNR": "255 range; full masked-canvas MSE; exact match capped at 100 dB as in author code",
    "limitation": "identical black pixels influence scores; visible preservation is not hidden completion quality",
}
_cache: dict = {}


# Cache CLIP chấm điểm theo thiết bị; tách với CLIP chọn prompt trong proposed.
def _clip(device: str):
    key = ("clip", device)
    if key not in _cache:
        from transformers import CLIPModel, CLIPProcessor

        _cache[key] = (
            CLIPModel.from_pretrained(CLIP_METRIC_MODEL_ID).to(device).eval(),
            CLIPProcessor.from_pretrained(CLIP_METRIC_MODEL_ID, use_fast=False),
        )
    return _cache[key]


# Chuẩn hóa kiểu kết quả từ các API trích đặc trưng CLIP.
def _as_tensor(features):
    import torch

    if isinstance(features, torch.Tensor):
        return features
    # transformers >= 5 may return a model-output object instead of a raw tensor.
    pooled = getattr(features, "pooler_output", None)
    return pooled if pooled is not None else features.last_hidden_state[:, 0]


# Dùng logit ảnh–nhãn làm chỉ dấu phù hợp ngữ nghĩa của ảnh hoàn chỉnh.
def clip_score(image: np.ndarray, text: str, device: str = "cuda") -> float:
    import torch

    model, processor = _clip(device)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        return float(model(**inputs).logits_per_image.item())


# Tính cosine giữa embedding ảnh gốc và ảnh kết quả sau khi che phần không đánh giá.
def clip_feature_similarity(img_a: np.ndarray, img_b: np.ndarray, device: str = "cuda") -> float:
    import torch

    model, processor = _clip(device)
    with torch.no_grad():
        feat_a = _as_tensor(model.get_image_features(**processor(images=img_a, return_tensors="pt").to(device)))
        feat_b = _as_tensor(model.get_image_features(**processor(images=img_b, return_tensors="pt").to(device)))
    return float(torch.nn.functional.cosine_similarity(feat_a, feat_b, dim=1).item())


# Tính khoảng cách cảm nhận bằng LPIPS; ảnh đã được giới hạn vào phần visible.
def lpips_distance(img_a: np.ndarray, img_b: np.ndarray, device: str = "cuda") -> float:
    import torch

    key = ("lpips", device)
    if key not in _cache:
        import lpips

        _cache[key] = lpips.LPIPS(net="alex").to(device).eval()

    # Đổi RGB H×W×C sang batch NCHW và miền giá trị LPIPS cần.
    def to_tensor(img: np.ndarray):
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1.0

    with torch.no_grad():
        return float(_cache[key](to_tensor(img_a), to_tensor(img_b)).item())


# Đo sai khác pixel trên ảnh đã mask theo protocol.
def psnr_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    mse = float(np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))


# Đo tương đồng cấu trúc của ảnh đã mask theo protocol.
def ssim_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(img_a, img_b, channel_axis=2, data_range=255))


# Tách đường CLIP ảnh hoàn chỉnh khỏi các metric so sánh RGB trên vùng quan sát.
def compute_visible_appearance_metrics(
    image: np.ndarray,
    completed_rgb: np.ndarray,
    visible: np.ndarray,
    object_name: str,
    device: str = "cpu",
) -> dict:
    """Compare completed vs. original pixels inside the visible mask (outside = black) + CLIP text relevance."""
    if image.dtype != np.uint8 or completed_rgb.dtype != np.uint8:
        raise ValueError("Metrics require uint8 RGB; do not silently rescale images")
    if image.ndim != 3 or image.shape[2] != 3 or completed_rgb.shape != image.shape:
        raise ValueError("Original and completed RGB must share the same HxWx3 canvas")
    visible = np.asarray(visible, dtype=bool)
    if visible.shape != image.shape[:2] or not visible.any():
        raise ValueError("Visible mask is empty or has a different canvas")
    if min(image.shape[:2]) < 32:
        raise ValueError("Author LPIPS/SSIM protocol requires canvas dimensions >= 32")
    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError("A nonempty annotation class name is required for CLIP")
    original_v = np.where(visible[:, :, None], image, 0).astype(np.uint8)
    completed_v = np.where(visible[:, :, None], completed_rgb, 0).astype(np.uint8)
    return {
        "CLIP_score": clip_score(completed_rgb, object_name, device),
        "LPIPS": lpips_distance(completed_v, original_v, device),
        "Feature_Similarity": clip_feature_similarity(completed_v, original_v, device),
        "SSIM": ssim_score(completed_v, original_v),
        "PSNR": psnr_score(completed_v, original_v),
    }
