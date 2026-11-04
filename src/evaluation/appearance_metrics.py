"""Optional GPU appearance metrics on the visible region (`appearance_metrics = true` in the config).

CLIP score (image/text logit), LPIPS (AlexNet), CLIP image-feature cosine similarity, and SSIM.
Models are loaded lazily once per process so they never compete with the generation pipeline at start-up.
"""
from __future__ import annotations

import numpy as np

CLIP_METRIC_MODEL_ID = "openai/clip-vit-base-patch32"
_cache: dict = {}


def _clip(device: str):
    if "clip" not in _cache:
        from transformers import CLIPModel, CLIPProcessor

        _cache["clip"] = (
            CLIPModel.from_pretrained(CLIP_METRIC_MODEL_ID).to(device),
            CLIPProcessor.from_pretrained(CLIP_METRIC_MODEL_ID),
        )
    return _cache["clip"]


def _as_tensor(features):
    import torch

    if isinstance(features, torch.Tensor):
        return features
    # transformers >= 5 may return a model-output object instead of a raw tensor.
    pooled = getattr(features, "pooler_output", None)
    return pooled if pooled is not None else features.last_hidden_state[:, 0]


def clip_score(image: np.ndarray, text: str, device: str = "cuda") -> float:
    import torch

    model, processor = _clip(device)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        return float(model(**inputs).logits_per_image.item())


def clip_feature_similarity(img_a: np.ndarray, img_b: np.ndarray, device: str = "cuda") -> float:
    import torch

    model, processor = _clip(device)
    with torch.no_grad():
        feat_a = _as_tensor(model.get_image_features(**processor(images=img_a, return_tensors="pt").to(device)))
        feat_b = _as_tensor(model.get_image_features(**processor(images=img_b, return_tensors="pt").to(device)))
    return float(torch.nn.functional.cosine_similarity(feat_a, feat_b, dim=1).item())


def lpips_distance(img_a: np.ndarray, img_b: np.ndarray, device: str = "cuda") -> float:
    import torch

    if "lpips" not in _cache:
        import lpips

        _cache["lpips"] = lpips.LPIPS(net="alex").to(device)

    def to_tensor(img: np.ndarray):
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1.0

    with torch.no_grad():
        return float(_cache["lpips"](to_tensor(img_a), to_tensor(img_b)).item())


def ssim_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(img_a, img_b, channel_axis=2, data_range=255))


def compute_visible_appearance_metrics(
    image: np.ndarray,
    completed_rgb: np.ndarray,
    visible: np.ndarray,
    object_name: str,
) -> dict:
    """Compare completed vs. original pixels inside the visible mask (outside = black) + CLIP text relevance."""
    original_v = np.where(visible[:, :, None], image, 0).astype(np.uint8)
    completed_v = np.where(visible[:, :, None], completed_rgb, 0).astype(np.uint8)
    return {
        "CLIP_score": clip_score(completed_rgb, object_name),
        "LPIPS": lpips_distance(completed_v, original_v),
        "Feature_Similarity": clip_feature_similarity(completed_v, original_v),
        "SSIM": ssim_score(completed_v, original_v),
    }

