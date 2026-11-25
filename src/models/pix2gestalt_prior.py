# Phạm vi: proposed: Pix2Gestalt sinh prior RGB 2D, CarveKit tách alpha tin cậy. Biên đầu ra vẫn căn
# theo mask cho sẵn.
"""Pix2Gestalt latent-diffusion loader + CarveKit matting interface (strict: fail loudly, never fall back)."""
from __future__ import annotations

from pathlib import Path
import gc
import sys

ROOT = Path(__file__).resolve().parents[2]
PIX2GESTALT_DIR = ROOT / "pix2gestalt"
DEFAULT_CKPT = ROOT / "ckpt" / "epoch=000005.ckpt"
DEFAULT_CONFIG = PIX2GESTALT_DIR / "pix2gestalt" / "configs" / "sd-finetune-pix2gestalt-c_concat-256.yaml"
VENDOR_PATHS = [
    ROOT / "vendor" / "taming-transformers",
    ROOT / "vendor" / "CLIP",
    PIX2GESTALT_DIR,
    PIX2GESTALT_DIR / "pix2gestalt",
]


# Thêm đường dẫn code Pix2Gestalt phụ thuộc trước khi import.
def ensure_vendor_paths() -> None:
    """Expose the cloned Pix2Gestalt / taming-transformers / CLIP repos on `sys.path`."""
    for path in VENDOR_PATHS:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


# Lấy hàm inference từ code vendor khi cần.
def load_run_pix2gestalt():
    """Return Pix2Gestalt's official `run_pix2gestalt(model, device, input_im, visible_mask, n_samples, ...)`."""
    ensure_vendor_paths()
    from inference import run_pix2gestalt

    return run_pix2gestalt


# Áp dụng trọng số EMA cho inference trước khi giải phóng bản lưu EMA.
def apply_inference_ema(model):
    """Keep the weights used by upstream ema_scope without GPU backup copies.

    Copy before freezing: upstream LitEma.copy_to uses requires_grad to select
    parameters. Inference never updates or restores the training weights.
    """
    if getattr(model, "use_ema", False):
        model.model_ema.copy_to(model.model)
        model.use_ema = False
        del model.model_ema
    model.requires_grad_(False)
    return model


# Gói generator và bộ tách alpha CarveKit để memory manager chuyển thiết bị cùng nhau.
class Pix2GestaltModel:
    """Holds the Pix2Gestalt LDM (`model`) and the CarveKit background-matting callable (`carvekit_interface`)."""

    # Dựng model từ YAML/checkpoint, chuẩn bị CarveKit và chế độ inference.
    def __init__(self, ckpt_path: Path = DEFAULT_CKPT, config_path: Path = DEFAULT_CONFIG, device: str = "cuda"):
        import torch

        ensure_vendor_paths()
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        from carvekit.api.high import HiInterface

        ckpt_path, config_path = Path(ckpt_path), Path(config_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Pix2Gestalt checkpoint not found: {ckpt_path} (run scripts/download_weights.py)")
        if not config_path.is_file():
            raise FileNotFoundError(f"Pix2Gestalt config not found: {config_path} (run scripts/setup_colab.py)")

        print(f"[Models] Loading Pix2Gestalt checkpoint from {ckpt_path}...", flush=True)
        self.device = device
        model = instantiate_from_config(OmegaConf.load(str(config_path)).model)
        # Lightning 1.x checkpoints pickle callback objects, so `weights_only=False` is required.
        try:
            state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False, mmap=True)
        except RuntimeError as error:
            if "mmap" not in str(error):
                raise
            state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        essential = [key for key in missing if key.startswith(("model.", "model_ema.", "first_stage_model.", "cc_projection."))]
        if essential:
            raise RuntimeError(f"Incompatible Pix2Gestalt checkpoint; missing keys: {essential[:10]}")
        del state_dict
        apply_inference_ema(model)
        gc.collect()
        self.model = model.to(device).eval()
        # Match upstream matting parameters; process samples singly on CPU so
        # matting weights and activations do not compete with diffusion on GPU.
        self.carvekit_interface = HiInterface(
            object_type="object", batch_size_seg=1, batch_size_matting=1,
            device="cpu", seg_mask_size=640, matting_mask_size=2048,
            trimap_prob_threshold=231, trimap_dilation=30,
            trimap_erosion_iters=5, fp16=False,
        )
        print("[Models] Pix2Gestalt + CarveKit ready.", flush=True)


    # Đồng bộ vị trí trọng số và thiết bị mà hàm inference sẽ dùng.
    def to(self, device):
        self.model.to(device)
        self.device = str(device)
        return self


from pathlib import Path
import inspect
from typing import Optional
import numpy as np
from PIL import Image
from src.vision.image_ops import resize_array
from src.vision.roi import compute_square_roi, extract_square_roi, paste_square_roi


# Cắt cùng ROI từ RGB và visible mask; resize mask bằng nội suy nearest để giữ biên.
def prepare_pix2gestalt_roi_inputs(
    image: np.ndarray,
    visible_mask: np.ndarray,
    roi: dict[str, int | bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and resize `(256, 256)` RGB scene crop and 3-channel binary visible mask."""
    roi_img = extract_square_roi(image, roi, pad_value=(127, 127, 127))
    roi_vis = extract_square_roi(visible_mask.astype(np.uint8) * 255, roi, pad_value=0)

    resized_img = resize_array(roi_img, (256, 256), mode="area")
    resized_vis = resize_array(roi_vis, (256, 256), mode="nearest")
    # Guarantee at least one foreground pixel survives 256x256 downsampling.
    if not (resized_vis > 127).any() and roi_vis.any():
        ys, xs = np.where(roi_vis > 127)
        cy = int(round(ys.mean() * 255.0 / max(roi_vis.shape[0] - 1, 1)))
        cx = int(round(xs.mean() * 255.0 / max(roi_vis.shape[1] - 1, 1)))
        resized_vis[np.clip(cy, 0, 255), np.clip(cx, 0, 255)] = 255
    resized_vis = ((resized_vis > 127).astype(np.uint8)) * 255
    rgb_vis_256 = np.repeat(resized_vis[:, :, None], 3, axis=2)
    return resized_img, rgb_vis_256


# Sinh RGB và alpha trong ROI 256×256, rồi dán về canvas gốc để chuẩn bị seed.
def synthesize_amodal_prior(
    models, config,
    image: np.ndarray,
    visible_mask: np.ndarray,
    amodal_mask: np.ndarray,
    job_dir: Optional[Path],
) -> dict:
    """Run 1 fast Pix2Gestalt pass on an `amodal_mask`-centered ROI for a 2D RGB prior."""
    import torch

    predictor = models.shape_predictor
    if predictor is None or predictor.model is None:
        raise RuntimeError("Strict mode: Pix2Gestalt model is not loaded.")

    run_pix2gestalt = load_run_pix2gestalt()

    roi_scale = float(config.get("oracle_roi_scale", 1.65))
    roi = compute_square_roi(
        amodal_mask,
        image.shape,
        context_scale=roi_scale,
        min_crop=176,
        full_frame_threshold=0.88,
    )
    resized_img, rgb_vis_256 = prepare_pix2gestalt_roi_inputs(image, visible_mask, roi)

    ddim_steps = int(config.get("pix2gestalt_ddim_steps", 75))
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
