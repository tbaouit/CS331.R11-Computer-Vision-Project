# Phạm vi: Cả hai phiên bản (baseline và proposed): bộ metric đồng bộ theo eva_amodal.py (Open-World Amodal):
# CLIP_score, OpenCLIP_score chấm ảnh hoàn chỉnh theo nhãn; LPIPS, Feature_Similarity (CLIP),
# VGG_similarity, DINO_similarity, SSIM, PSNR chấm trên vùng quan sát.
"""Unified evaluation protocol integrating Open-World Amodal (`eva_amodal.py`) metrics.

Both `baseline` and `proposed` are evaluated with the exact same metric suite on the same COCO-A manifest:
- `CLIP_score` (HuggingFace `openai/clip-vit-base-patch32` image-text logit on full completed RGB)
- `OpenCLIP_score` (OpenCLIP `ViT-B-32` `laion2b_s34b_b79k` image-text cosine similarity on full completed RGB)
- `LPIPS` (AlexNet perceptual distance on visible region)
- `Feature_Similarity` (CLIP ViT-B/32 visual cosine similarity on visible region)
- `VGG_similarity` (VGG-16 convolutional feature cosine similarity as in `eva_amodal.py`)
- `DINO_similarity` (DINO ViT-B/16 feature cosine similarity as in `eva_amodal.py`)
- `SSIM` & `PSNR`
"""
from __future__ import annotations

import numpy as np
from PIL import Image

CLIP_METRIC_MODEL_ID = "openai/clip-vit-base-patch32"
DINO_METRIC_MODEL_ID = "facebook/dino-vitb16"
PROTOCOL = "decoded_rgb_visible_black_openworld"
METRIC_NAMES = (
    "CLIP_score",
    "OpenCLIP_score",
    "LPIPS",
    "Feature_Similarity",
    "VGG_similarity",
    "DINO_similarity",
    "SSIM",
    "PSNR",
)
POLICY = {
    "protocol": PROTOCOL,
    "output": "decoded ROI RGB after paste-back; amodal alpha",
    "seam_and_color": "compare visible and hidden regions on the output RGB",
    "reference": "original input RGB; no hidden RGB ground truth",
    "CLIP_score": "full output RGB vs raw annotation class name; scaled CLIP image/text logit",
    "OpenCLIP_score": "full output RGB vs raw annotation class name; OpenCLIP ViT-B-32 cosine similarity",
    "visible_metrics": "original canvas, black outside visible mask; no crop or manual resize",
    "feature_backbone": CLIP_METRIC_MODEL_ID,
    "vgg_backbone": "torchvision VGG16 ImageNet features cosine similarity (eva_amodal.py)",
    "dino_backbone": "DINO ViT-B/16 feature cosine similarity (eva_amodal.py)",
    "lpips_backbone": "alex; uint8 RGB normalized to [-1,1]",
    "SSIM": "skimage RGB, channel_axis=-1, data_range=255, adaptive odd win_size<=7",
    "PSNR": "255 range; full masked-canvas MSE; exact match capped at 100 dB",
    "limitation": "identical black pixels influence scores; visible preservation is paired with seam_mae and color_hist_sim",
}
_cache: dict = {}


def _clip(device: str):
    key = ("clip", device)
    if key not in _cache:
        from transformers import CLIPModel, CLIPProcessor

        _cache[key] = (
            CLIPModel.from_pretrained(CLIP_METRIC_MODEL_ID).to(device).eval(),
            CLIPProcessor.from_pretrained(CLIP_METRIC_MODEL_ID, use_fast=False),
        )
    return _cache[key]


def _openclip(device: str):
    key = ("openclip", device)
    if key not in _cache:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        model.eval()
        _cache[key] = (model, preprocess, open_clip.tokenize)
    return _cache[key]


def _dino(device: str):
    key = ("dino", device)
    if key not in _cache:
        import torch

        try:
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained(DINO_METRIC_MODEL_ID)
            model = AutoModel.from_pretrained(DINO_METRIC_MODEL_ID).to(device).eval()
            _cache[key] = ("hf", model, processor)
        except Exception:
            import torchvision.transforms as T

            model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(device).eval()
            preprocess = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            _cache[key] = ("hub", model, preprocess)
    return _cache[key]


def _vgg(device: str):
    key = ("vgg", device)
    if key not in _cache:
        from torchvision import models
        import torchvision.transforms as T

        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            vgg = models.vgg16(weights=weights).features.to(device).eval()
        except AttributeError:
            vgg = models.vgg16(pretrained=True).features.to(device).eval()
        preprocess = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        _cache[key] = (vgg, preprocess)
    return _cache[key]


def _as_tensor(features):
    import torch

    if isinstance(features, torch.Tensor):
        return features
    pooled = getattr(features, "pooler_output", None)
    return pooled if pooled is not None else features.last_hidden_state[:, 0]


def clip_score(image: np.ndarray, text: str, device: str = "cuda") -> float:
    import torch

    model, processor = _clip(device)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        return float(model(**inputs).logits_per_image.item())


def openclip_score(image: np.ndarray, text: str, device: str = "cuda") -> float:
    """Compute OpenCLIP ViT-B-32 prompt relevance cosine similarity (`compute_openclip_score` in `eva_amodal.py`)."""
    import torch
    import torch.nn.functional as F

    try:
        model, preprocess, tokenize = _openclip(device)
        image_input = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        text_input = tokenize([text]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            return float(F.cosine_similarity(image_features, text_features).item())
    except ImportError:
        # Fallback to normalized HuggingFace CLIP cosine similarity if open_clip_torch is not installed
        model, processor = _clip(device)
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_feat = _as_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            txt_feat = _as_tensor(
                model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            )
            return float(F.cosine_similarity(img_feat, txt_feat, dim=1).item())


def clip_feature_similarity(img_a: np.ndarray, img_b: np.ndarray, device: str = "cuda") -> float:
    import torch

    model, processor = _clip(device)
    with torch.no_grad():
        feat_a = _as_tensor(model.get_image_features(**processor(images=img_a, return_tensors="pt").to(device)))
        feat_b = _as_tensor(model.get_image_features(**processor(images=img_b, return_tensors="pt").to(device)))
    return float(torch.nn.functional.cosine_similarity(feat_a, feat_b, dim=1).item())


def vgg_feature_similarity(img_a: np.ndarray, img_b: np.ndarray, device: str = "cuda") -> float:
    """Compute VGG-16 feature cosine similarity (`compute_feature_similarity` in `eva_amodal.py`)."""
    import torch
    import torch.nn.functional as F

    vgg, preprocess = _vgg(device)
    t_a = preprocess(Image.fromarray(img_a)).unsqueeze(0).to(device)
    t_b = preprocess(Image.fromarray(img_b)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_a = vgg(t_a)
        feat_b = vgg(t_b)
        return float(F.cosine_similarity(feat_a, feat_b).mean().item())


def dino_similarity(img_a: np.ndarray, img_b: np.ndarray, device: str = "cuda") -> float:
    """Compute DINO ViT-B/16 feature cosine similarity (`compute_dino_similarity` in `eva_amodal.py`)."""
    import torch
    import torch.nn.functional as F

    kind, model, proc = _dino(device)
    with torch.no_grad():
        if kind == "hf":
            in_a = proc(images=Image.fromarray(img_a), return_tensors="pt").to(device)
            in_b = proc(images=Image.fromarray(img_b), return_tensors="pt").to(device)
            feat_a = _as_tensor(model(**in_a))
            feat_b = _as_tensor(model(**in_b))
        else:
            t_a = proc(Image.fromarray(img_a)).unsqueeze(0).to(device)
            t_b = proc(Image.fromarray(img_b)).unsqueeze(0).to(device)
            feat_a = model(t_a)
            feat_b = model(t_b)
        return float(F.cosine_similarity(feat_a, feat_b).mean().item())


def lpips_distance(img_a: np.ndarray, img_b: np.ndarray, device: str = "cuda") -> float:
    import torch

    key = ("lpips", device)
    if key not in _cache:
        import lpips

        _cache[key] = lpips.LPIPS(net="alex").to(device).eval()

    def to_tensor(img: np.ndarray):
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1.0

    with torch.no_grad():
        return float(_cache[key](to_tensor(img_a), to_tensor(img_b)).item())


def psnr_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    mse = float(np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))


def ssim_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    min_dim = min(img_a.shape[0], img_a.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win_size = max(3, win_size)
    return float(structural_similarity(img_a, img_b, channel_axis=2, data_range=255, win_size=win_size))


def compute_visible_appearance_metrics(
    image: np.ndarray,
    completed_rgb: np.ndarray,
    visible: np.ndarray,
    object_name: str,
    device: str = "cpu",
) -> dict:
    """Compute unified appearance metrics (`eva_amodal.py` + visible preservation protocol) on a sample."""
    if image.dtype != np.uint8 or completed_rgb.dtype != np.uint8:
        raise ValueError("Metrics require uint8 RGB; do not silently rescale images")
    if image.ndim != 3 or image.shape[2] != 3 or completed_rgb.shape != image.shape:
        raise ValueError("Original and completed RGB must share the same HxWx3 canvas")
    visible = np.asarray(visible, dtype=bool)
    if visible.shape != image.shape[:2] or not visible.any():
        raise ValueError("Visible mask is empty or has a different canvas")
    if min(image.shape[:2]) < 32:
        raise ValueError("LPIPS/SSIM protocol requires canvas dimensions >= 32")
    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError("A nonempty annotation class name is required for CLIP/OpenCLIP")
    original_v = np.where(visible[:, :, None], image, 0).astype(np.uint8)
    completed_v = np.where(visible[:, :, None], completed_rgb, 0).astype(np.uint8)
    return {
        "CLIP_score": clip_score(completed_rgb, object_name, device),
        "OpenCLIP_score": openclip_score(completed_rgb, object_name, device),
        "LPIPS": lpips_distance(completed_v, original_v, device),
        "Feature_Similarity": clip_feature_similarity(completed_v, original_v, device),
        "VGG_similarity": vgg_feature_similarity(completed_v, original_v, device),
        "DINO_similarity": dino_similarity(completed_v, original_v, device),
        "SSIM": ssim_score(completed_v, original_v),
        "PSNR": psnr_score(completed_v, original_v),
    }
