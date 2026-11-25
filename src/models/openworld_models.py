"""Open-World Amodal foundation model wrappers with single-init SAM and CPU/GPU offloading.

Ports the 4 core modules from Open-World Amodal (`D:/amodal/main.py`):
1. `RAM++` (Swin-L) + `CLIP ViT-B/32` for scene tagging and prompt selection on the visible mask.
2. `GroundingDINO` (Swin-T) + `SAM ViT-H` (initialized ONCE and reused across iterations).
3. `InstaOrder` (`InstaOrderNet_od`, ResNet-50) for pairwise occlusion ordering.
4. `Stable Diffusion 2 Inpainting` (`stabilityai/stable-diffusion-2-inpainting`) with FP16 & slicing.
LISA-13B and LaMa are completely omitted.
"""
from __future__ import annotations

import gc
import math
from pathlib import Path
import re
import sys
from typing import Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SD2_INPAINTING = "stabilityai/stable-diffusion-2-inpainting"
SD2_COMMUNITY_INPAINTING = "sd2-community/stable-diffusion-2-inpainting"
ALLOWED_SD2_MODELS = (SD2_INPAINTING, SD2_COMMUNITY_INPAINTING)

OPENWORLD_VENDOR_PATHS = (
    ROOT / "vendor" / "InstaOrder",
    ROOT / "InstaOrder",
    ROOT / "vendor" / "Grounded-Segment-Anything",
    ROOT / "vendor" / "Grounded-Segment-Anything" / "GroundingDINO",
    ROOT / "Grounded-Segment-Anything",
    ROOT / "Grounded-Segment-Anything" / "GroundingDINO",
    ROOT / "vendor" / "recognize-anything",
    ROOT / "recognize-anything",
    ROOT / "vendor" / "CLIP",
)


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a relative config path against project ROOT without hardcoding absolute paths."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    # Fallback search in standard vendor/ckpt locations
    alt_candidates = [
        ROOT / "ckpt" / path.name,
        ROOT / "vendor" / path,
    ]
    for alt in alt_candidates:
        if alt.exists():
            return alt
    return candidate


def ensure_openworld_vendor_paths() -> None:
    """Expose Grounded-SAM, recognize-anything, InstaOrder, and CLIP on `sys.path`."""
    for path in OPENWORLD_VENDOR_PATHS:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _patch_transformers_legacy_compat() -> None:
    """Restore helpers removed in transformers 5.x that `RAM++` and `GroundingDINO` import from `modeling_utils`."""
    try:
        import transformers.modeling_utils as mu
        import transformers.pytorch_utils as pu
    except Exception:
        return
    for attr in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(mu, attr) and hasattr(pu, attr):
            setattr(mu, attr, getattr(pu, attr))
    if not hasattr(mu, "apply_chunking_to_forward"):
        import torch

        def _apply_chunking_to_forward(forward_fn, chunk_size: int, chunk_dim: int, *input_tensors):
            if chunk_size > 0:
                num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
                input_tensors_chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
                output_chunks = tuple(forward_fn(*chunks) for chunks in zip(*input_tensors_chunks))
                return torch.cat(output_chunks, dim=chunk_dim)
            return forward_fn(*input_tensors)

        setattr(mu, "apply_chunking_to_forward", _apply_chunking_to_forward)


def _patch_groundingdino_deform_attn() -> None:
    """Ensure GroundingDINO uses `ms_deform_attn_core_pytorch` on GPU when `_C` is uncompiled."""
    import importlib
    import types

    # 1. Patch ms_deform_attn.py on disk before/after import so it checks `_C` before calling `.apply`
    for rel in (
        "vendor/Grounded-Segment-Anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py",
        "Grounded-Segment-Anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py",
    ):
        p = ROOT / rel
        if p.is_file():
            try:
                src = p.read_text(encoding="utf-8")
                target = "if torch.cuda.is_available() and value.is_cuda:"
                replacement = 'if torch.cuda.is_available() and value.is_cuda and "_C" in globals() and _C is not None:'
                if target in src and replacement not in src:
                    p.write_text(src.replace(target, replacement), encoding="utf-8")
            except OSError:
                pass

    # 2. Import both possible module paths and inject `_C.ms_deform_attn_forward` into every loaded copy
    for mod_name in (
        "GroundingDINO.groundingdino.models.GroundingDINO.ms_deform_attn",
        "groundingdino.models.GroundingDINO.ms_deform_attn",
    ):
        try:
            importlib.import_module(mod_name)
        except Exception:
            pass

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        core_fn = getattr(mod, "ms_deform_attn_core_pytorch", None) or getattr(
            mod, "multi_scale_deform_attn_core_pytorch", None
        )
        if core_fn is None:
            continue
        if getattr(mod, "_C", None) is None:
            def _ms_deform_attn_forward(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
                _fn=core_fn,
            ):
                return _fn(value, value_spatial_shapes, sampling_locations, attention_weights)

            mod._C = types.SimpleNamespace(ms_deform_attn_forward=_ms_deform_attn_forward)


def validate_openworld_proposed_config(config: dict) -> None:
    """Validate that `configs/proposed.json` configures the Open-World Amodal SD2 pipeline."""
    if config.get("pipeline_mode") != "proposed":
        raise ValueError("Expected pipeline_mode='proposed'")
    model_id = config.get("inpainting_model_id")
    if model_id not in ALLOWED_SD2_MODELS:
        raise ValueError(
            f"Open-World Amodal proposed pipeline requires SD2 Inpainting ({ALLOWED_SD2_MODELS}), got {model_id!r}"
        )
    max_iter = config.get("max_iter", 3)
    if type(max_iter) is not int or not 1 <= max_iter <= 10:
        raise ValueError("max_iter must be an integer in [1, 10]")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    policy = config.get("text_query_policy", "dataset_object_name")
    if policy not in ("dataset_object_name", "empty"):
        raise ValueError("text_query_policy must be 'dataset_object_name' or 'empty'")
    for key in ("gdino_box_thresh", "gdino_text_thresh", "ram_threshold"):
        if key in config:
            val = config[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not 0.0 < float(val) <= 1.0:
                raise ValueError(f"{key} must be in (0, 1]")


class RAMPlusModel:
    """RAM++ (Swin-L) image tagging model wrapper with CPU/GPU offload support."""

    def __init__(
        self,
        ckpt_path: str | Path = "ckpt/ram_plus_swin_large_14m.pth",
        image_size: int = 384,
        threshold: float = 0.6,
        device: str = "cpu",
    ):
        import torch

        ensure_openworld_vendor_paths()
        _patch_transformers_legacy_compat()
        from ram import get_transform
        from ram.models import ram_plus

        resolved_ckpt = resolve_project_path(ckpt_path)
        if not resolved_ckpt.is_file():
            raise FileNotFoundError(
                f"RAM++ checkpoint not found: {resolved_ckpt} (run setup_colab.sh or scripts/download_weights.py)"
            )
        print(f"[Models] Loading RAM++ (Swin-L) from {resolved_ckpt}...", flush=True)
        self.device = device
        self.image_size = int(image_size)
        self.transform = get_transform(image_size=self.image_size)
        model = ram_plus(pretrained=str(resolved_ckpt), image_size=self.image_size, vit="swin_l")
        model.class_threshold = torch.ones(model.num_class) * float(threshold)
        self.model = model.eval().to(device)

    def to(self, device: str):
        import torch

        self.device = device
        self.model.to(device)
        if hasattr(self.model, "class_threshold") and isinstance(self.model.class_threshold, torch.Tensor):
            self.model.class_threshold = self.model.class_threshold.to(device)
        return self

    def infer_tags(self, img_pil: Image.Image) -> list[str]:
        import torch
        from ram import inference_ram

        tensor = self.transform(img_pil.convert("RGB").resize((self.image_size, self.image_size))).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            res = inference_ram(tensor, self.model)
        tags = [x.strip() for x in res[0].split("|") if x.strip()]
        if "background" not in tags:
            tags.append("background")
        return tags


class OpenAICLIPPromptSelector:
    """CLIP ViT-B/32 prompt selector matching `clip_similarity_w_query` in `D:/amodal/main.py`."""

    def __init__(self, model_id: str = "ViT-B/32", device: str = "cpu"):
        import torch

        ensure_openworld_vendor_paths()
        self.device = device
        self.model_id = model_id
        self._use_openai_clip = False
        try:
            import clip

            print(f"[Models] Loading OpenAI CLIP ({model_id})...", flush=True)
            model, preprocess = clip.load(model_id, device=device)
            self.model = model.eval()
            self.preprocess = preprocess
            self._tokenize = clip.tokenize
            self._use_openai_clip = True
        except ImportError:
            from transformers import CLIPModel, CLIPProcessor

            hf_id = "openai/clip-vit-base-patch32" if model_id == "ViT-B/32" else model_id
            print(f"[Models] Loading HuggingFace CLIP ({hf_id})...", flush=True)
            self.model = CLIPModel.from_pretrained(hf_id, torch_dtype=torch.float16).eval().to(device)
            self.processor = CLIPProcessor.from_pretrained(hf_id)

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self

    def select_prompt(self, masked_image_pil: Image.Image, classes: list[str], text_query: str = "") -> str:
        """Select the class label or query with highest CLIP similarity to the masked visible object."""
        import torch

        candidates = [c for c in classes if c and c.strip()]
        clean_query = (text_query or "").strip()
        full_candidates = list(candidates)
        if clean_query:
            full_candidates.append(clean_query)
        if not full_candidates:
            return "main object"

        prompts = [f"a photo of a {c}" for c in full_candidates]
        if self._use_openai_clip:
            image_input = self.preprocess(masked_image_pil.convert("RGB")).unsqueeze(0).to(self.device)
            text_inputs = torch.cat([self._tokenize(p) for p in prompts]).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                max_index = int(similarity[0].argmax().item())
        else:
            inputs = self.processor(
                text=prompts, images=masked_image_pil.convert("RGB"), return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits_per_image
                similarity = logits.softmax(dim=-1)
                max_index = int(similarity[0].argmax().item())
        return full_candidates[max_index]


class GroundingDINOModel:
    """GroundingDINO (Swin-T) open-set detector with CPU/GPU offload support."""

    def __init__(
        self,
        config_path: str | Path = "vendor/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ckpt_path: str | Path = "ckpt/groundingdino_swint_ogc.pth",
        device: str = "cpu",
    ):
        import torch

        ensure_openworld_vendor_paths()
        _patch_transformers_legacy_compat()
        _patch_groundingdino_deform_attn()
        try:
            import GroundingDINO.groundingdino.datasets.transforms as T
            from GroundingDINO.groundingdino.models import build_model
            from GroundingDINO.groundingdino.util.slconfig import SLConfig
            from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        except ImportError:
            import groundingdino.datasets.transforms as T
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        _patch_groundingdino_deform_attn()

        resolved_cfg = resolve_project_path(config_path)
        resolved_ckpt = resolve_project_path(ckpt_path)
        if not resolved_cfg.is_file():
            raise FileNotFoundError(f"GroundingDINO config not found: {resolved_cfg}")
        if not resolved_ckpt.is_file():
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {resolved_ckpt}")

        print(f"[Models] Loading GroundingDINO (Swin-T) from {resolved_ckpt}...", flush=True)
        self.device = device
        self._T = T
        self._get_phrases_from_posmap = get_phrases_from_posmap
        args = SLConfig.fromfile(str(resolved_cfg))
        args.device = device
        model = build_model(args)
        _patch_groundingdino_deform_attn()
        checkpoint = torch.load(str(resolved_ckpt), map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model = model.eval().to(device)

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self

    def transform_image(self, img_pil: Image.Image):
        transform = self._T.Compose(
            [
                self._T.RandomResize([800], max_size=1333),
                self._T.ToTensor(),
                self._T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img_tensor, _ = transform(img_pil.convert("RGB"), None)
        return img_tensor

    def detect(
        self,
        img_pil: Image.Image,
        caption: str,
        box_thresh: float = 0.35,
        text_thresh: float = 0.35,
        with_logits: bool = True,
    ):
        import torch

        img_tensor = self.transform_image(img_pil).to(self.device)
        clean_caption = caption.lower().strip()
        if not clean_caption.endswith("."):
            clean_caption = clean_caption + "."

        with torch.no_grad():
            _patch_groundingdino_deform_attn()
            outputs = self.model(img_tensor[None], captions=[clean_caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        filt_mask = logits.max(dim=1)[0] > float(box_thresh)
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        tokenized = self.model.tokenizer(clean_caption)
        pred_phrases: list[str] = []
        for logit in logits_filt:
            phrase = self._get_phrases_from_posmap(logit > float(text_thresh), tokenized, self.model.tokenizer)
            if with_logits:
                pred_phrases.append(f"{phrase}({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(phrase)
        return boxes_filt, pred_phrases


class SAMViTHModel:
    """SAM ViT-H wrapper initialized ONCE per pipeline lifetime and reused across all iterations."""

    def __init__(self, ckpt_path: str | Path = "ckpt/sam_vit_h_4b8939.pth", device: str = "cpu"):
        ensure_openworld_vendor_paths()
        from segment_anything import SamPredictor, build_sam

        resolved_ckpt = resolve_project_path(ckpt_path)
        if not resolved_ckpt.is_file():
            raise FileNotFoundError(
                f"SAM ViT-H checkpoint not found: {resolved_ckpt} (run setup_colab.sh or scripts/download_weights.py)"
            )
        print(f"[Models] Loading SAM ViT-H ONCE from {resolved_ckpt}...", flush=True)
        self.device = device
        self.model = build_sam(checkpoint=str(resolved_ckpt)).eval().to(device)
        self.predictor = SamPredictor(self.model)

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self

    def segment_boxes(self, img_pil: Image.Image, boxes_filt) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Run SAM mask prediction on GroundingDINO boxes using the pre-loaded `SamPredictor`."""
        import torch

        img = np.array(img_pil.convert("RGB"))
        if boxes_filt is None or int(boxes_filt.size(0)) == 0:
            return img, None

        self.predictor.set_image(img)
        w, h = img_pil.size
        boxes = boxes_filt.clone().cpu().float()
        scale = torch.tensor([w, h, w, h], dtype=torch.float32)
        for i in range(boxes.size(0)):
            boxes[i] = boxes[i] * scale
            boxes[i][:2] -= boxes[i][2:] / 2.0
            boxes[i][2:] += boxes[i][:2]
        try:
            transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, img.shape[:2]).to(self.device)
            with torch.no_grad():
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            masks_np = masks.cpu().numpy().squeeze(1)
            return img, masks_np
        except Exception as exc:
            print(f"[SAM] Warning during box segmentation: {exc}", flush=True)
            return img, None
        finally:
            self.predictor.reset_image()


def _import_instaorder_modules():
    """Import InstaOrder's `models` and `inference` without colliding with another vendor's `utils.py`."""
    ensure_openworld_vendor_paths()
    insta_dir = next((p for p in (ROOT / "vendor" / "InstaOrder", ROOT / "InstaOrder") if p.is_dir()), None)
    if insta_dir is not None:
        insta_str = str(insta_dir)
        if insta_str in sys.path:
            sys.path.remove(insta_str)
        sys.path.insert(0, insta_str)
    existing_utils = sys.modules.get("utils")
    if existing_utils is not None and not hasattr(existing_utils, "__path__"):
        sys.modules.pop("utils", None)
    import inference as insta_infer
    import models as insta_models

    return insta_models, insta_infer


class InstaOrderModel:
    """InstaOrder (`InstaOrderNet_od`, ResNet-50) occlusion ordering wrapper with CPU/GPU offload."""

    def __init__(
        self,
        ckpt_path: str | Path = "ckpt/InstaOrder_InstaOrderNet_od.pth.tar",
        device: str = "cpu",
    ):
        self.device = device
        self.model = None
        self._infer = None
        resolved_ckpt = resolve_project_path(ckpt_path)
        if not resolved_ckpt.is_file():
            print(
                f"[InstaOrder] Checkpoint {resolved_ckpt.name} not found; using geometric boundary-overlap "
                "occlusion analysis fallback.",
                flush=True,
            )
            return
        insta_models, insta_infer = _import_instaorder_modules()
        print(f"[Models] Loading InstaOrderNet_od (ResNet-50) from {resolved_ckpt}...", flush=True)
        self._infer = insta_infer
        params = {
            "algo": "InstaOrderNet_od",
            "total_iter": 60000,
            "lr_steps": [32000, 48000],
            "lr_mults": [0.1, 0.1],
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "optim": "SGD",
            "warmup_lr": [],
            "warmup_steps": [],
            "use_rgb": True,
            "backbone_arch": "resnet50_cls",
            "backbone_param": {"in_channels": 5, "num_classes": [2, 3]},
            "overlap_weight": 0.1,
            "distinct_weight": 0.9,
        }
        self.model = insta_models.__dict__["InstaOrderNet_od"](params)
        self.model.load_state(str(resolved_ckpt))
        self.model.switch_to("eval")
        self.to(device)

    def to(self, device: str):
        import torch

        self.device = device
        if self.model is None:
            return self
        if hasattr(self.model, "model") and isinstance(self.model.model, torch.nn.Module):
            self.model.model.to(device)
        for attr in dir(self.model):
            val = getattr(self.model, attr, None)
            if isinstance(val, torch.nn.Module):
                val.to(device)
        return self

    @staticmethod
    def _find_mask_sides(mask: np.ndarray, val: int = 1) -> tuple[int, int, int, int]:
        binary = (mask > 0).astype(np.uint8)
        x_arr, y_arr = np.where(binary == val)
        if len(x_arr) == 0:
            return 0, 0, 0, 0
        return int(min(x_arr)), int(max(x_arr)), int(min(y_arr)), int(max(y_arr))

    @classmethod
    def find_expand_bboxes(cls, masks: list[np.ndarray] | np.ndarray) -> np.ndarray:
        bboxes = np.zeros((len(masks), 4), dtype=np.float64)
        for i, mask in enumerate(masks):
            x_min, x_max, y_min, y_max = cls._find_mask_sides(mask)
            w = y_max - y_min
            h = x_max - x_min
            bboxes[i, 0] = y_min
            bboxes[i, 1] = x_min
            bboxes[i, 2] = w
            bboxes[i, 3] = h
        expanded = []
        for bbox in bboxes:
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            size = max([np.sqrt(bbox[2] * bbox[3] * 3.0), bbox[2] * 1.1, bbox[3] * 1.1, 1.0])
            expanded.append([int(cx - size / 2.0), int(cy - size / 2.0), int(size), int(size)])
        return np.array(expanded)

    def analyze_masks(self, img: np.ndarray, masks: list[np.ndarray] | np.ndarray, mask_id: int) -> list[np.ndarray]:
        """Determine which masks occlude `masks[mask_id]` using InstaOrderNet_od (or geometric adjacency fallback)."""
        if len(masks) <= 1:
            return []
        h, w = masks[0].shape[:2]
        modal = np.zeros((len(masks), h, w), dtype=np.uint8)
        for i, mask in enumerate(masks):
            modal[i] = (mask > 0).astype(np.uint8)

        if self.model is None or self._infer is None:
            from src.vision.image_ops import binary_dilate

            target = modal[int(mask_id)] > 0
            dilated_target = binary_dilate(target, 5)
            occluders: list[np.ndarray] = []
            for idx in range(len(modal)):
                if idx == int(mask_id):
                    continue
                cand = modal[idx] > 0
                if np.any(cand & dilated_target) and float(cand.sum()) < 0.75 * float(h * w):
                    occluders.append(modal[idx])
            return occluders

        bboxes = self.find_expand_bboxes(modal)

        pcolor_occ_order, _ = self._infer.infer_order_sup_occ_depth(
            self.model,
            img,
            modal,
            bboxes,
            pairs="all",
            method="InstaOrderNet_od",
            patch_or_image="resize",
            input_size=384,
            disp_select_method="",
        )
        pcolor_occ_order = pcolor_occ_order.transpose()
        occ_indices = pcolor_occ_order[mask_id]
        occ_mask_indices = np.where(occ_indices == 1)[0]

        occluder_masks: list[np.ndarray] = []
        for idx in occ_mask_indices:
            if int(idx) == int(mask_id):
                continue
            occluder_masks.append(modal[int(idx)])

        del modal, bboxes, pcolor_occ_order
        gc.collect()
        return occluder_masks

