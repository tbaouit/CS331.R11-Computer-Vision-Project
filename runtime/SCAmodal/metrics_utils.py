import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# Global variables for lazy loading models (to save VRAM during initial import)
_lpips_model = None
_clip_model = None
_clip_processor = None

def calculate_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Structural Consistency: SSIM
    """
    # SSIM requires grayscale or multi-channel handling
    score, _ = ssim(img1, img2, full=True, channel_axis=2, data_range=255)
    return score

def calculate_lpips(img1, img2, device="cuda"):
    """
    Visual Consistency: LPIPS
    Requires: pip install lpips
    """
    global _lpips_model
    import lpips
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Preprocess images to [-1, 1] range and (1, C, H, W)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1
    
    with torch.no_grad():
        dist = _lpips_model(t1, t2)
    return dist.item()

def calculate_clip_score(image, text, device="cuda"):
    """
    Class Relevance: CLIP Score
    """
    from transformers import CLIPProcessor, CLIPModel
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        model_id = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(model_id).to(device)
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        
    inputs = _clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = _clip_model(**inputs)
    
    # Cosine similarity between image and text embeddings
    logits_per_image = outputs.logits_per_image # this is bridge to similarity
    return logits_per_image.item()

def calculate_feature_similarity(img1, img2, device="cuda"):
    """
    Semantic Consistency: Cosine similarity of CLIP/DINO features
    """
    from transformers import CLIPModel, CLIPProcessor
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        model_id = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(model_id).to(device)
        _clip_processor = CLIPProcessor.from_pretrained(model_id)

    inputs1 = _clip_processor(images=img1, return_tensors="pt").to(device)
    inputs2 = _clip_processor(images=img2, return_tensors="pt").to(device)

    with torch.no_grad():
        feat1 = _clip_model.get_image_features(**inputs1)
        feat2 = _clip_model.get_image_features(**inputs2)

    # transformers 5.0.0+ returns BaseModelOutputWithPooling; pull out the tensor.
    if not isinstance(feat1, torch.Tensor):
        feat1 = feat1.pooler_output if feat1.pooler_output is not None else feat1.last_hidden_state[:, 0]
    if not isinstance(feat2, torch.Tensor):
        feat2 = feat2.pooler_output if feat2.pooler_output is not None else feat2.last_hidden_state[:, 0]

    cos = nn.CosineSimilarity(dim=1)
    return cos(feat1, feat2).item()

def get_amodal_metrics(pred_mask, gt_mask, visible_mask):
    """
    Standard amodal metrics:
    - mIoU: Full shape IoU
    - oIoU (Occluded IoU): IoU specifically for the hidden regions (CVPR standard)
    """
    full_iou = calculate_iou(pred_mask, gt_mask)
    
    # Occluded region IoU
    occluded_gt = gt_mask & (~visible_mask.astype(bool))
    occluded_pred = pred_mask & (~visible_mask.astype(bool))
    
    occluded_iou = calculate_iou(occluded_pred, occluded_gt)
    
    return {
        "mIoU": full_iou,
        "oIoU": occluded_iou
    }
