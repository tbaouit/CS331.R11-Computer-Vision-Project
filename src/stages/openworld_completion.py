"""Core Open-World Amodal stages ported from `D:/amodal/main.py` and optimized for Colab.

Key improvements over original `main.py`:
- Uses pre-supplied `visible_mask` directly (no LISA-13B server/checkpoint).
- Completely omits LaMa (`big-lama`).
- Reuses a single `SAMViTHModel` instance (`models.sam`) across all iterations (no per-loop reload).
- Offloads inactive models to CPU between stages (`models.gdino` -> `models.sam` -> `models.instaorder` -> `models.pipe`).
- Fixes off-by-one iteration loop condition (`while query_obj.run_iter and query_obj.iter_id < max_iter:`).
- Safe background synthesis (`clean_background_val`) that never crashes with out-of-bounds crop indexing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Optional

import numpy as np
from PIL import Image

from src.vision.image_ops import binary_dilate, binary_erode

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - local CPU data venv fallback
    cv2 = None


def _dilate_box(mask_u8: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    if cv2 is not None:
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.dilate(mask_u8.astype(np.uint8), kernel, iterations=iterations).astype(np.uint8)
    out = mask_u8 > 0
    radius = max(1, ksize // 2)
    for _ in range(iterations):
        out = binary_dilate(out, radius)
    return out.astype(np.uint8)


def _erode_box(mask_u8: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    if cv2 is not None:
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.erode(mask_u8.astype(np.uint8), kernel, iterations=iterations).astype(np.uint8)
    out = mask_u8 > 0
    radius = max(1, ksize // 2)
    for _ in range(iterations):
        out = binary_erode(out, radius)
    return out.astype(np.uint8)


def _close_box(mask_u8: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    """Morphological closing with zero-padded borders so boundary regions do not shrink."""
    pad = max(1, (ksize // 2) * iterations + 1)
    padded = np.pad((mask_u8 > 0).astype(np.uint8), pad, mode="constant", constant_values=0)
    dilated = _dilate_box(padded, ksize, iterations=iterations)
    closed = _erode_box(dilated, ksize, iterations=iterations)
    return closed[pad:-pad, pad:-pad].astype(np.uint8)


def _fill_interior_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Fill all enclosed 0-holes inside a binary mask without altering its outer boundary."""
    binary = (mask_u8 > 0).astype(np.uint8)
    if not np.any(binary):
        return binary
    h, w = binary.shape[:2]
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    if cv2 is not None:
        inv = (1 - padded).astype(np.uint8)
        flood = inv.copy()
        ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 2)
        holes = (flood == 1)[1:-1, 1:-1]
        return np.maximum(binary, holes.astype(np.uint8))
    visited = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)])
    visited[0, 0] = True
    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h + 2 and 0 <= nx < w + 2 and not visited[ny, nx] and padded[ny, nx] == 0:
                visited[ny, nx] = True
                q.append((ny, nx))
    holes = (~visited)[1:-1, 1:-1] & (binary == 0)
    return np.maximum(binary, holes.astype(np.uint8))


def _extract_concavity_and_hole_occluders(
    query_mask: np.ndarray,
    masks: np.ndarray | list[np.ndarray],
    mask_id: int,
) -> np.ndarray:
    """Recover occluders sitting inside interior holes or deep concavities/notches of `query_mask`."""
    q_u8 = (query_mask > 0).astype(np.uint8)
    if not np.any(q_u8):
        return np.zeros_like(q_u8)
    h, w = q_u8.shape[:2]
    x_min, x_max, y_min, y_max = find_mask_sides(q_u8)
    bbox_mask = np.zeros((h, w), dtype=bool)
    bbox_mask[x_min : x_max + 1, y_min : y_max + 1] = True

    # 1. Enclosed interior holes of query_mask are guaranteed occluders on top of the object
    filled = _fill_interior_holes(q_u8)
    holes = (filled > 0) & ~(q_u8 > 0)

    # 2. Narrow notches/concavities indenting into query_mask within its tight bounding box
    closed = _close_box(q_u8, 25, iterations=2)
    concavities = (closed > 0) & ~(q_u8 > 0) & bbox_mask

    # 3. Any detected non-target foreground mask that touches query_mask and lies mostly inside its tight bbox
    dilated_q = _dilate_box(q_u8, 5, iterations=1) > 0
    in_bbox_occluders = np.zeros((h, w), dtype=bool)
    max_occ_area = 0.35 * float(h * w)
    for idx, m in enumerate(masks):
        if idx == int(mask_id):
            continue
        m_bool = m > 0
        m_area = float(m_bool.sum())
        if m_area <= 0 or m_area > max_occ_area:
            continue
        if np.any(m_bool & dilated_q):
            inside_ratio = float((m_bool & bbox_mask).sum()) / m_area
            if inside_ratio >= 0.45 or np.any(m_bool & concavities):
                in_bbox_occluders |= (m_bool & ~(q_u8 > 0))

    combined = (holes | concavities | in_bbox_occluders) & ~(q_u8 > 0)
    return combined.astype(np.uint8)


@dataclass
class QueryObjectState:
    """Mutable per-target state across progressive mixed-context diffusion iterations."""

    img: np.ndarray
    img_pil: Image.Image
    mask_id: int
    query_mask: np.ndarray
    query_class: str = "main object"
    context_descriptor: str = ""
    occluder_classes: list[str] = field(default_factory=list)
    output_img_dir: Optional[Path] = None
    run_iter: bool = True
    iter_id: int = 0
    first_sd_occ_mask: Optional[np.ndarray] = None
    skipflag: bool = False
    img_canvas: Optional[np.ndarray] = None
    query_mask_canvas: Optional[np.ndarray] = None
    occ_mask_canvas: Optional[np.ndarray] = None
    outpaint_mask_canvas: Optional[np.ndarray] = None
    init_img_canvas: Optional[np.ndarray] = None
    init_query_mask_canvas: Optional[np.ndarray] = None
    init_occ_mask_canvas: Optional[np.ndarray] = None
    init_outpaint_mask_canvas: Optional[np.ndarray] = None
    amodal_completion: Optional[Image.Image] = None
    amodal_segmentation: Optional[np.ndarray] = None
    sd_img_cut_rgba: Optional[np.ndarray] = None
    last_crop_box: Optional[tuple[int, int, int, int]] = None
    iteration_logs: list[dict] = field(default_factory=list)


def find_mask_sides(mask: np.ndarray, val: int = 1) -> tuple[int, int, int, int]:
    """Return `(x_min, x_max, y_min, y_max)` row/col bounds where `(mask > 0) == val`."""
    binary = (mask > 0).astype(np.uint8)
    x_arr, y_arr = np.where(binary == val)
    if len(x_arr) == 0:
        return 0, binary.shape[0] - 1, 0, binary.shape[1] - 1
    return int(min(x_arr)), int(max(x_arr)), int(min(y_arr)), int(max(y_arr))


def compute_iou_uint8(mask1: np.ndarray, mask2: np.ndarray) -> float:
    m1 = mask1 > 0
    m2 = mask2 > 0
    union = int(np.logical_or(m1, m2).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(m1, m2).sum() / union)


def filter_out_amodal_segmentation(
    crop_query_mask: np.ndarray,
    amodal_masks: np.ndarray | list[np.ndarray],
) -> tuple[int, np.ndarray]:
    """Select the candidate segmentation mask with highest IoU with `crop_query_mask`."""
    amodal_segmentation = (crop_query_mask > 0).astype(np.uint8)
    max_iou = 0.0
    amodal_i = 0
    for i, candidate in enumerate(amodal_masks):
        cand_u8 = (candidate > 0).astype(np.uint8)
        score = compute_iou_uint8(amodal_segmentation, cand_u8)
        if score > max_iou:
            amodal_segmentation = cand_u8
            max_iou = score
            amodal_i = i
    return amodal_i, amodal_segmentation


def parse_phrase_and_score(pred_phrase: str) -> tuple[str, float]:
    """Parse GroundingDINO phrase `'chair(0.62)'` into `('chair', 0.62)`."""
    parts = [p for p in re.split(r"\(|\)", pred_phrase) if p.strip()]
    if len(parts) >= 2:
        try:
            return parts[0].strip(), float(parts[1])
        except ValueError:
            pass
    return pred_phrase.strip(), 1.0


def _slice_single_box(boxes_filt, index: int):
    """Extract a 1-row box tensor/array `[1, 4]` at `index` across PyTorch tensors and test doubles."""
    try:
        sliced = boxes_filt[index : index + 1]
        if hasattr(sliced, "size") and int(sliced.size(0)) == 1:
            return sliced
    except Exception:
        pass
    arr = getattr(boxes_filt, "arr", None)
    if arr is not None:
        return type(boxes_filt)(arr[index : index + 1])
    return boxes_filt


def _box_cxcywh_to_xyxy_pixels(box_row, width: int, height: int) -> list[float]:
    """Convert normalized `[cx, cy, w, h]` box to clamped pixel `[x1, y1, x2, y2]`."""
    try:
        if hasattr(box_row, "cpu") and hasattr(box_row.cpu(), "tolist"):
            vals = box_row.cpu().tolist()
        elif hasattr(box_row, "arr"):
            vals = np.asarray(box_row.arr, dtype=np.float32).reshape(-1).tolist()
        else:
            vals = np.asarray(box_row, dtype=np.float32).reshape(-1).tolist()
        if isinstance(vals[0], list):
            vals = vals[0]
        cx, cy, bw, bh = [float(v) for v in vals[:4]]
    except Exception:
        return [0.0, 0.0, float(width), float(height)]
    x1 = max(0.0, min(float(width), (cx - bw / 2.0) * width))
    y1 = max(0.0, min(float(height), (cy - bh / 2.0) * height))
    x2 = max(x1, min(float(width), (cx + bw / 2.0) * width))
    y2 = max(y1, min(float(height), (cy + bh / 2.0) * height))
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def predict_auto_visible_mask(
    models,
    img_pil: Image.Image,
    object_name: str,
    config: dict,
) -> dict:
    """Predict target `visible_mask` automatically using GroundingDINO + cached SAM ViT-H."""
    w_img, h_img = img_pil.size
    query = (object_name or "object").strip() or "object"
    box_thresh = float(config.get("gdino_box_thresh", 0.35))
    text_thresh = float(config.get("gdino_text_thresh", 0.35))
    fallback_thresh = float(config.get("auto_visible_fallback_thresh", 0.15))

    gdino = models.gdino
    boxes_filt, pred_phrases = gdino.detect(
        img_pil,
        query,
        box_thresh=box_thresh,
        text_thresh=text_thresh,
    )
    used_fallback = False
    num_boxes = int(boxes_filt.size(0)) if boxes_filt is not None and hasattr(boxes_filt, "size") else 0
    if num_boxes == 0 and fallback_thresh < box_thresh:
        boxes_filt, pred_phrases = gdino.detect(
            img_pil,
            query,
            box_thresh=fallback_thresh,
            text_thresh=min(text_thresh, fallback_thresh),
        )
        used_fallback = True
        num_boxes = int(boxes_filt.size(0)) if boxes_filt is not None and hasattr(boxes_filt, "size") else 0

    if num_boxes > 0:
        scores = [parse_phrase_and_score(p)[1] for p in pred_phrases] if pred_phrases else [1.0] * num_boxes
        best_idx = int(np.argmax(scores[:num_boxes]))
        best_phrase = pred_phrases[best_idx] if best_idx < len(pred_phrases) else query
        parsed_label, best_score = parse_phrase_and_score(best_phrase)
        best_box = _slice_single_box(boxes_filt, best_idx)
        bbox_xyxy = _box_cxcywh_to_xyxy_pixels(best_box, w_img, h_img)

        _, masks_np = models.sam.segment_boxes(img_pil, best_box)
        if masks_np is not None and len(masks_np) > 0:
            pred_mask = (np.asarray(masks_np[0]) > 0).astype(bool)
        else:
            pred_mask = np.zeros((h_img, w_img), dtype=bool)
    else:
        pred_mask = np.zeros((h_img, w_img), dtype=bool)
        bbox_xyxy = None
        best_score = 0.0
        parsed_label = ""

    visible_pred_empty = not bool(pred_mask.any())
    if visible_pred_empty:
        print(f"[AutoSeg] visible_pred_empty: true (object_name={query!r})", flush=True)
    else:
        print(
            f"[AutoSeg] visible_pred_empty: false (object_name={query!r}, score={best_score:.3f}, "
            f"fallback={used_fallback}, pixels={int(pred_mask.sum())})",
            flush=True,
        )
    return {
        "pred_visible_mask": pred_mask,
        "visible_pred_empty": visible_pred_empty,
        "pred_bbox_xyxy": bbox_xyxy,
        "pred_score": float(best_score),
        "pred_phrase": parsed_label,
        "used_fallback_thresh": used_fallback,
        "query": query,
    }


def segment_with_cached_sam(
    models,
    img_pil: Image.Image,
    classes: list[str],
    target_class: Optional[str] = None,
    box_thresh: float = 0.35,
    text_thresh: float = 0.35,
) -> tuple[np.ndarray, Optional[np.ndarray], list[str], list[float]]:
    """Run GroundingDINO + cached SAM ViT-H (never rebuilding SAM inside the loop)."""
    import torch

    caption = ". ".join([c for c in classes if c]) or (target_class or "object")
    gdino = models.gdino
    boxes_filt, pred_phrases = gdino.detect(
        img_pil,
        caption,
        box_thresh=box_thresh,
        text_thresh=text_thresh,
    )

    if target_class:
        target_present = any(parse_phrase_and_score(p)[0] == target_class for p in pred_phrases)
        if not target_present:
            boxes_q, phrases_q = gdino.detect(
                img_pil,
                target_class,
                box_thresh=box_thresh,
                text_thresh=text_thresh,
            )
            if int(boxes_q.size(0)) > 0:
                if int(boxes_filt.size(0)) > 0:
                    boxes_filt = torch.cat((boxes_filt, boxes_q), dim=0)
                    pred_phrases = list(pred_phrases) + list(phrases_q)
                else:
                    boxes_filt = boxes_q
                    pred_phrases = list(phrases_q)

    sam = models.sam
    img_np, masks = sam.segment_boxes(img_pil, boxes_filt)
    class_names: list[str] = []
    pred_scores: list[float] = []
    for phrase in pred_phrases:
        cname, score = parse_phrase_and_score(phrase)
        class_names.append(cname)
        pred_scores.append(score)
    return img_np, masks, class_names, pred_scores


def handlemask(masks: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    """Identify and extract additional unsegmented foreground regions near the target object while excluding scene backgrounds."""
    if len(masks) == 0:
        return []
    h, w = masks[0].shape[:2]
    max_fg_area = 0.30 * float(h * w)
    mask_sum = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        mask_sum = np.maximum(mask_sum, (mask > 0).astype(np.uint8))

    # Restrict unsegmented candidate search to the expanded bounding box of the target query mask (masks[-1])
    # so unsegmented occluders touching the target are severed from distant walls/floors/streets.
    query_ref = (masks[-1] > 0).astype(np.uint8)
    if np.any(query_ref):
        qx1, qx2, qy1, qy2 = find_mask_sides(query_ref)
        pad_h = max(16, int(round((qx2 - qx1 + 1) * 0.25)))
        pad_w = max(16, int(round((qy2 - qy1 + 1) * 0.25)))
        bx1, bx2 = max(0, qx1 - pad_h), min(h, qx2 + 1 + pad_h)
        by1, by2 = max(0, qy1 - pad_w), min(w, qy2 + 1 + pad_w)
        roi_gate = np.zeros((h, w), dtype=np.uint8)
        roi_gate[bx1:bx2, by1:by2] = 1
    else:
        roi_gate = np.ones((h, w), dtype=np.uint8)

    new_mask = ((1 - mask_sum) * roi_gate).astype(np.uint8)
    eroded_mask = _erode_box(new_mask, 5, iterations=1)
    raw_components: list[np.ndarray] = []
    if cv2 is not None:
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area <= max_fg_area:
                m = np.zeros_like(eroded_mask, dtype=np.uint8)
                cv2.drawContours(m, [contour], -1, 1, -1)
                raw_components.append(m)
    else:
        visited = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if eroded_mask[y, x] and not visited[y, x]:
                    comp = np.zeros((h, w), dtype=np.uint8)
                    q = deque([(y, x)])
                    visited[y, x] = True
                    count = 0
                    while q:
                        cy, cx = q.popleft()
                        comp[cy, cx] = 1
                        count += 1
                        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                            if 0 <= ny < h and 0 <= nx < w and eroded_mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                q.append((ny, nx))
                    if 100 < count <= max_fg_area:
                        raw_components.append(comp)

    separated: list[np.ndarray] = []
    for comp_u8 in raw_components:
        # Skip whole-scene background walls/floors that touch 3 or more image borders
        if len(check_touch_boundary(comp_u8, gap_pixels=3)) >= 3:
            continue
        # Restore the 5px erosion so there is never an artificial 5px gap/moat around the target object
        restored = (_dilate_box(comp_u8, 5, iterations=1) > 0) & (new_mask > 0)
        if restored.any():
            separated.append(restored)
    return separated


def aggregate_occluders(query_mask: np.ndarray, all_occluder_masks: list[np.ndarray]) -> np.ndarray:
    """Aggregate foreground occluders adjacent to `query_mask` while rejecting full-scene background masks."""
    h, w = query_mask.shape[:2]
    q_bin = (query_mask > 0).astype(np.uint8)
    dilated_q = _dilate_box(q_bin, 9, iterations=1) > 0
    agg = np.zeros((h, w), dtype=np.uint8)
    for occ in all_occluder_masks:
        occ_bool = np.asarray(occ) > 0
        if not np.any(occ_bool & dilated_q):
            continue
        if float(occ_bool.sum()) > 0.65 * float(h * w):
            continue
        agg = np.maximum(agg, occ_bool.astype(np.uint8))
    agg[(q_bin + agg) > 1] = 0
    return agg


def create_canvas(input_arr: np.ndarray, size_multiplier: int = 6, canvas_val: int = 0) -> np.ndarray:
    """Place `input_arr` at the center of an expanded canvas (`size_multiplier * H`, `size_multiplier * W`)."""
    h, w = input_arr.shape[:2]
    canvas_shape = list(input_arr.shape)
    canvas_shape[0] = int(h * size_multiplier)
    canvas_shape[1] = int(w * size_multiplier)
    if canvas_val > 0:
        canvas = np.full(tuple(canvas_shape), canvas_val, dtype=input_arr.dtype)
    else:
        canvas = np.zeros(tuple(canvas_shape), dtype=input_arr.dtype)
    h_start = (canvas_shape[0] // 2) - (h // 2)
    w_start = (canvas_shape[1] // 2) - (w // 2)
    canvas[h_start : h_start + h, w_start : w_start + w] = input_arr
    return canvas


def canvas_center_slice(orig_shape: tuple[int, int], size_multiplier: int = 6) -> tuple[slice, slice]:
    """Return `(row_slice, col_slice)` mapping the original `H x W` image inside the expanded canvas."""
    h, w = orig_shape[:2]
    ch, cw = int(h * size_multiplier), int(w * size_multiplier)
    h_start = (ch // 2) - (h // 2)
    w_start = (cw // 2) - (w // 2)
    return slice(h_start, h_start + h), slice(w_start, w_start + w)


def check_touch_boundary(mask: np.ndarray, gap_pixels: int = 10) -> set[str]:
    """Check whether `mask` touches within `gap_pixels` of the image border."""
    h, w = mask.shape[:2]
    if not np.any(mask > 0):
        return set()
    x_min, x_max, y_min, y_max = find_mask_sides(mask)
    sides: set[str] = set()
    if x_max >= h - gap_pixels:
        sides.add("bottom")
    if x_min <= gap_pixels:
        sides.add("top")
    if y_max >= w - gap_pixels:
        sides.add("right")
    if y_min <= gap_pixels:
        sides.add("left")
    return sides


def find_crop_region(
    query_mask: np.ndarray,
    query_mask_canvas: np.ndarray,
    pad_pixels: int = 150,
    crop_buffer: int = 60,
) -> tuple[int, int, int, int, int]:
    """Determine square cropping region on expanded canvas with boundary-aware padding."""
    ch, cw = query_mask_canvas.shape[:2]
    crop_x_min, crop_x_max, crop_y_min, crop_y_max = find_mask_sides(query_mask_canvas)
    crop_x_min = max(0, crop_x_min - crop_buffer)
    crop_x_max = min(ch, crop_x_max + crop_buffer)
    crop_y_min = max(0, crop_y_min - crop_buffer)
    crop_y_max = min(cw, crop_y_max + crop_buffer)

    sides_touched = check_touch_boundary(query_mask)
    if "top" in sides_touched:
        crop_x_min -= pad_pixels
    if "bottom" in sides_touched:
        crop_x_max += pad_pixels
    if "left" in sides_touched:
        crop_y_min -= pad_pixels
    if "right" in sides_touched:
        crop_y_max += pad_pixels

    crop_h = crop_x_max - crop_x_min
    crop_w = crop_y_max - crop_y_min
    target_size = max(crop_h, crop_w, 64)

    if crop_w < target_size:
        diff = target_size - crop_w
        if "left" in sides_touched:
            crop_y_min -= diff
        elif "right" in sides_touched:
            crop_y_max += diff
        else:
            crop_y_min -= math.floor(diff / 2)
            crop_y_max += math.ceil(diff / 2)

    if crop_h < target_size:
        diff = target_size - crop_h
        if "top" in sides_touched:
            crop_x_min -= diff
        elif "bottom" in sides_touched:
            crop_x_max += diff
        else:
            crop_x_min -= math.floor(diff / 2)
            crop_x_max += math.ceil(diff / 2)

    crop_x_min = max(0, crop_x_min)
    crop_x_max = min(ch, crop_x_max)
    crop_y_min = max(0, crop_y_min)
    crop_y_max = min(cw, crop_y_max)
    return crop_x_min, crop_x_max, crop_y_min, crop_y_max, target_size


def crop_canvas_bundle(
    inputs: list[np.ndarray],
    query_mask: np.ndarray,
    query_mask_canvas: np.ndarray,
    pad_pixels: int = 150,
    crop_buffer: int = 60,
) -> tuple[list[np.ndarray], int, int, int, int]:
    x_min, x_max, y_min, y_max, _ = find_crop_region(
        query_mask, query_mask_canvas, pad_pixels=pad_pixels, crop_buffer=crop_buffer
    )
    cropped = [arr[x_min:x_max, y_min:y_max].astype(np.uint8) for arr in inputs]
    return cropped, x_min, x_max, y_min, y_max


def shrink_edges_to_transparent(image_rgba: np.ndarray, shrink_amount: int = 10) -> np.ndarray:
    """Shrink transparent border of an RGBA image by `shrink_amount` pixels."""
    if image_rgba.ndim != 3 or image_rgba.shape[2] != 4:
        raise ValueError("Input image must be in RGBA format.")
    alpha = image_rgba[:, :, 3]
    zero_mask = (alpha == 0).astype(np.uint8)
    dilated = _dilate_box(zero_mask, shrink_amount * 2 + 1, iterations=1)
    out = image_rgba.copy()
    out[dilated > 0, 3] = 0
    return out


def alpha_blending(src_rgba: np.ndarray, dst_rgba: np.ndarray, transi_wid: int = 5) -> np.ndarray:
    """Smoothly blend original visible RGBA (`src_rgba`) into inpainted RGBA (`dst_rgba`)."""
    if src_rgba.shape != dst_rgba.shape or src_rgba.shape[2] != 4:
        raise ValueError("Both src and dst images must be RGBA with identical dimensions.")
    src_mask = (src_rgba[:, :, 3] > 0).astype(np.uint8)
    if not np.any(src_mask):
        return dst_rgba.copy()

    src_interior = _erode_box(src_mask, transi_wid, iterations=1)
    transition = (src_mask - src_interior) > 0

    if cv2 is not None:
        dist_transform = cv2.distanceTransform((1 - src_mask).astype(np.uint8), cv2.DIST_L2, 5)
        dist_transform = np.clip(dist_transform / float(max(transi_wid, 1)), 0.0, 1.0)
    else:
        dist_transform = np.where(transition, 0.5, 1.0).astype(np.float32)

    weight_src = np.where(transition, dist_transform, 1.0)
    weight_dst = 1.0 - weight_src

    blended = dst_rgba.copy()
    blended[src_mask > 0] = src_rgba[src_mask > 0]

    if np.any(transition):
        dst_alpha_trans = dst_rgba[transition, 3]
        blend_idx = dst_alpha_trans > 0
        if np.any(blend_idx):
            w_dst = weight_dst[transition, np.newaxis][blend_idx]
            w_src = weight_src[transition, np.newaxis][blend_idx]
            blended_rgb = (
                w_dst * dst_rgba[transition, :3][blend_idx].astype(np.float32)
                + w_src * src_rgba[transition, :3][blend_idx].astype(np.float32)
            )
            trans_coords = np.where(transition)
            blended[trans_coords[0][blend_idx], trans_coords[1][blend_idx], :3] = np.clip(
                blended_rgb, 0, 255
            ).astype(np.uint8)
        no_blend_idx = ~blend_idx
        if np.any(no_blend_idx):
            trans_coords = np.where(transition)
            blended[trans_coords[0][no_blend_idx], trans_coords[1][no_blend_idx], :3] = src_rgba[
                transition, :3
            ][no_blend_idx]

    blended[:, :, 3] = np.maximum(src_rgba[:, :, 3], dst_rgba[:, :, 3])
    return blended


def check_occlusion_after_inpaint(
    models,
    amodal_completion_pil: Image.Image,
    crop_query_mask: np.ndarray,
    query_class: str,
    classes: list[str],
    query_obj: QueryObjectState,
    crop_box: tuple[int, int, int, int],
    box_thresh: float = 0.35,
    text_thresh: float = 0.35,
) -> QueryObjectState:
    """Re-segment the inpainted crop with GroundingDINO + cached SAM and check remaining occluders via InstaOrder."""
    crop_x_min, crop_x_max, crop_y_min, crop_y_max = crop_box
    comp_np, amodal_segs, _, _ = segment_with_cached_sam(
        models,
        amodal_completion_pil,
        [query_class],
        target_class=query_class,
        box_thresh=min(box_thresh, 0.25),
        text_thresh=min(text_thresh, 0.25),
    )

    allowed_region = (crop_query_mask > 0).astype(np.uint8)
    if query_obj.first_sd_occ_mask is not None and query_obj.first_sd_occ_mask.shape == allowed_region.shape:
        allowed_region = np.maximum(
            allowed_region,
            _dilate_box((query_obj.first_sd_occ_mask > 0).astype(np.uint8), 5, iterations=3),
        )

    if amodal_segs is None or len(amodal_segs) == 0:
        non_white_fg = np.any(comp_np < 248, axis=2).astype(np.uint8)
        fallback_seg = np.maximum(
            (crop_query_mask > 0).astype(np.uint8),
            np.logical_and(allowed_region > 0, non_white_fg > 0).astype(np.uint8),
        )
        fallback_seg = np.logical_and(_fill_interior_holes(_close_box(fallback_seg, 5, iterations=1)) > 0, allowed_region > 0).astype(np.uint8)
        fallback_seg = np.maximum(fallback_seg, (crop_query_mask > 0).astype(np.uint8))
        query_obj.amodal_segmentation = fallback_seg
        query_obj.img_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = comp_np
        query_obj.query_mask_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = fallback_seg
        query_obj.run_iter = False
        return query_obj

    amodal_i, amodal_seg = filter_out_amodal_segmentation(crop_query_mask, amodal_segs)
    # Clamp predicted amodal mask to the valid (visible | inpainted) region and retain non-white inpainted object pixels
    # so SAM under-segmentation never punches white Swiss-cheese holes into the completed object.
    clamped_seg = np.logical_and(amodal_seg > 0, allowed_region > 0)
    non_white_inpaint = np.logical_and(allowed_region > 0, np.any(comp_np < 246, axis=2))
    raw_union = np.maximum(
        np.logical_or(clamped_seg, non_white_inpaint).astype(np.uint8),
        (crop_query_mask > 0).astype(np.uint8),
    )
    closed_union = np.logical_and(_fill_interior_holes(_close_box(raw_union, 5, iterations=1)) > 0, allowed_region > 0)
    amodal_seg_u8 = np.maximum(closed_union.astype(np.uint8), (crop_query_mask > 0).astype(np.uint8))
    query_obj.amodal_segmentation = amodal_seg_u8

    new_occluder_masks = models.instaorder.analyze_masks(comp_np, amodal_segs, amodal_i)
    new_occluder_mask = aggregate_occluders(amodal_seg_u8, new_occluder_masks)

    query_obj.img_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = comp_np
    query_obj.query_mask_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = amodal_seg_u8
    query_obj.occ_mask_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = new_occluder_mask
    query_obj.outpaint_mask_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = np.zeros_like(amodal_seg_u8)

    sides_touched = check_touch_boundary(amodal_seg_u8)
    if "top" in sides_touched:
        query_obj.occ_mask_canvas[: crop_x_min + 5, :] = 1
    if "bottom" in sides_touched:
        query_obj.occ_mask_canvas[crop_x_max - 5 :, :] = 1
    if "left" in sides_touched:
        query_obj.occ_mask_canvas[:, : crop_y_min + 5] = 1
    if "right" in sides_touched:
        query_obj.occ_mask_canvas[:, crop_y_max - 5 :] = 1

    query_obj.run_iter = bool(new_occluder_mask.sum() > 0 or len(sides_touched) > 0)
    return query_obj


def run_progressive_iteration(
    models,
    query_obj: QueryObjectState,
    masks: np.ndarray,
    classes: list[str],
    config: dict,
    fallback_occluded_mask: Optional[np.ndarray] = None,
    mask_class_names: Optional[list[str]] = None,
) -> QueryObjectState:
    """Execute one iteration of Open-World Amodal progressive mixed-context diffusion."""
    import torch

    img = query_obj.img
    mask_id = query_obj.mask_id
    query_mask = (query_obj.query_mask > 0).astype(np.uint8)
    query_class = query_obj.query_class
    size_mult = int(config.get("canvas_multiplier", 6))
    pad_pixels = int(config.get("pad_pixels", 150))
    crop_buffer = int(config.get("crop_buffer", 60))
    sd_target_size = int(config.get("sd_target_size", 512))
    clean_bg_val = int(config.get("clean_background_val", 127))
    box_thresh = float(config.get("gdino_box_thresh", 0.35))
    text_thresh = float(config.get("gdino_text_thresh", 0.35))

    if query_obj.iter_id == 0:
        occluder_masks = models.instaorder.analyze_masks(img, masks, mask_id)
        occ_mask = aggregate_occluders(query_mask, occluder_masks)
        # Union with dataset occluded mask on iteration 0 when provided (GT mode)
        if fallback_occluded_mask is not None and np.any(fallback_occluded_mask):
            gt_occ = (np.asarray(fallback_occluded_mask) > 0).astype(np.uint8)
            gt_occ[query_mask > 0] = 0
            occ_mask = np.maximum(occ_mask, gt_occ)
        elif np.any(query_mask):
            # In Zero-GT mode, always include enclosed interior holes and deep concavities/notches of query_mask
            # so secondary/thin occluders (poles, signboards, straps, pillows, cats) are never left behind.
            concav_occ = _extract_concavity_and_hole_occluders(query_mask, masks, mask_id)
            occ_mask = np.maximum(occ_mask, concav_occ)
            if not np.any(occ_mask):
                # Fallback when InstaOrder and concavity check both return empty:
                # 1) any detected scene mask touching query_mask inside its expanded bbox; 2) expanded bbox complement
                h_img, w_img = query_mask.shape[:2]
                x_min, x_max, y_min, y_max = find_mask_sides(query_mask)
                pad_h = max(12, int(round((x_max - x_min + 1) * 0.20)))
                pad_w = max(12, int(round((y_max - y_min + 1) * 0.20)))
                bx1, bx2 = max(0, x_min - pad_h), min(h_img, x_max + 1 + pad_h)
                by1, by2 = max(0, y_min - pad_w), min(w_img, y_max + 1 + pad_w)
                bbox_region = np.zeros((h_img, w_img), dtype=bool)
                bbox_region[bx1:bx2, by1:by2] = True
                adj_cands = [
                    (m > 0) & bbox_region
                    for i, m in enumerate(masks)
                    if i != int(mask_id) and np.any((m > 0) & bbox_region)
                ]
                if adj_cands:
                    occ_mask = aggregate_occluders(query_mask, adj_cands)
                if not np.any(occ_mask):
                    occ_mask = (bbox_region & ~(query_mask > 0)).astype(np.uint8)
        # Identify specific occluder class names overlapping occ_mask for occluder-aware negative prompting
        if mask_class_names and np.any(occ_mask):
            q_low = query_class.strip().lower()
            occ_names: list[str] = []
            for idx, m in enumerate(masks):
                if idx == int(mask_id) or idx >= len(mask_class_names):
                    continue
                cname = (mask_class_names[idx] or "").strip()
                clow = cname.lower()
                if not cname or clow.startswith("background_") or clow == q_low or clow in q_low:
                    continue
                if np.any((m > 0) & (occ_mask > 0)) and cname not in occ_names:
                    occ_names.append(cname)
            query_obj.occluder_classes = occ_names
        sides_touched = check_touch_boundary(query_mask)
        if not np.any(occ_mask) and len(sides_touched) == 0:
            query_obj.run_iter = False
            query_obj.amodal_segmentation = query_mask.copy()
            return query_obj

        img_canvas = create_canvas(img, size_multiplier=size_mult, canvas_val=255)
        query_mask_canvas = create_canvas(query_mask, size_multiplier=size_mult, canvas_val=0)
        occ_mask_canvas = create_canvas(occ_mask, size_multiplier=size_mult, canvas_val=0)
        # Only mark outpaint_mask_canvas=1 when query_mask actually touches an image boundary
        outpaint_val = 1 if len(sides_touched) > 0 else 0
        outpaint_mask_canvas = create_canvas(np.zeros_like(query_mask), size_multiplier=size_mult, canvas_val=outpaint_val)

        query_obj.img_canvas = img_canvas
        query_obj.query_mask_canvas = query_mask_canvas
        query_obj.occ_mask_canvas = occ_mask_canvas
        query_obj.outpaint_mask_canvas = outpaint_mask_canvas
        query_obj.init_img_canvas = img_canvas.copy()
        query_obj.init_query_mask_canvas = query_mask_canvas.copy()
        query_obj.init_occ_mask_canvas = occ_mask_canvas.copy()
        query_obj.init_outpaint_mask_canvas = outpaint_mask_canvas.copy()

    crop_inputs = [
        query_obj.img_canvas,
        query_obj.query_mask_canvas,
        query_obj.occ_mask_canvas,
        query_obj.outpaint_mask_canvas,
    ]
    crop_outputs, crop_x_min, crop_x_max, crop_y_min, crop_y_max = crop_canvas_bundle(
        crop_inputs,
        query_mask,
        query_obj.init_query_mask_canvas,
        pad_pixels=pad_pixels,
        crop_buffer=crop_buffer,
    )
    crop_img, crop_query_mask, crop_occ_mask, crop_outpaint_mask = crop_outputs
    query_obj.last_crop_box = (crop_x_min, crop_x_max, crop_y_min, crop_y_max)

    sd_img = crop_img.copy()
    sd_modal_mask = (crop_query_mask > 0).astype(np.uint8)
    sd_occ_mask = ((crop_occ_mask.astype(np.uint16) + crop_outpaint_mask.astype(np.uint16)) > 0).astype(np.uint8)
    ctx_desc = (query_obj.context_descriptor or "").strip()
    if ctx_desc and ctx_desc.lower() not in query_class.lower():
        subject_phrase = f"{query_class} ({ctx_desc})"
    else:
        subject_phrase = query_class
    sd_prompt = (
        f"a photo of a {subject_phrase}, complete object on a plain white background, "
        "natural colors, realistic texture"
    )

    sd_modal_erode = _erode_box(sd_modal_mask, 5, iterations=1)
    if not np.any(sd_modal_erode):
        sd_modal_erode = sd_modal_mask.copy()

    if query_obj.iter_id == 0:
        cut_rgba = np.zeros((*sd_img.shape[:2], 4), dtype=np.uint8)
        cut_rgba[:, :, :3] = sd_img
        cut_rgba[:, :, 3] = sd_modal_erode * 255
        query_obj.sd_img_cut_rgba = cut_rgba
        query_obj.first_sd_occ_mask = sd_occ_mask.copy()
        if query_obj.output_img_dir is not None:
            Image.fromarray(cut_rgba).save(query_obj.output_img_dir / "sd_img_cut.png")

    if query_obj.iter_id > 0 and query_obj.first_sd_occ_mask is not None:
        if sd_occ_mask.shape == query_obj.first_sd_occ_mask.shape:
            sd_occ_mask = np.logical_and(sd_occ_mask > 0, query_obj.first_sd_occ_mask > 0).astype(np.uint8)

    if not np.any(sd_occ_mask):
        query_obj.run_iter = False
        return query_obj

    input_h, input_w = sd_img.shape[:2]
    if query_obj.iter_id == 0:
        sd_occ_mask = _dilate_box(sd_occ_mask, 5, iterations=3)
    else:
        sd_occ_mask = _dilate_box(sd_occ_mask, 3, iterations=1)
    # Place the visible/completed object on a clean white (255) canvas and seed the inpaint region with the
    # visible object's median RGB color so SD2 VAE conditioning preserves the object's color distribution.
    sd_img_syn = np.full_like(sd_img, 255, dtype=np.uint8)
    if np.any(sd_modal_mask > 0):
        vis_median_rgb = np.median(sd_img[sd_modal_mask > 0], axis=0).astype(np.uint8)
        sd_img_syn[sd_occ_mask > 0] = vis_median_rgb
    else:
        sd_img_syn[sd_occ_mask > 0] = clean_bg_val
    sd_img_syn[sd_modal_mask > 0] = sd_img[sd_modal_mask > 0]

    sd_img_pil = Image.fromarray(sd_img_syn).convert("RGB").resize((sd_target_size, sd_target_size), Image.Resampling.LANCZOS)
    sd_mask_pil = Image.fromarray(sd_occ_mask * 255).convert("L").resize((sd_target_size, sd_target_size), Image.Resampling.NEAREST)

    current_iter = query_obj.iter_id
    seed = int(config.get("seed", 42)) + current_iter * 1009
    pipe = models.pipe
    generator = torch.Generator(device=models.device).manual_seed(seed)
    base_neg_prompt = config.get(
        "negative_prompt",
        "gray background, flat gray patch, gray fill, occluded, cut off, cropped, "
        "blurry, low quality, distorted, deformed, extra limbs, floating head, "
        "text, watermark, artifacts",
    )
    if query_obj.occluder_classes:
        occ_neg = ", ".join(query_obj.occluder_classes)
        neg_prompt = f"{base_neg_prompt}, {occ_neg}"
    else:
        neg_prompt = base_neg_prompt
    call_kwargs = {
        "image": sd_img_pil,
        "mask_image": sd_mask_pil,
        "prompt": sd_prompt,
        "negative_prompt": neg_prompt,
        "generator": generator,
    }
    if "guidance_scale" in config:
        call_kwargs["guidance_scale"] = float(config["guidance_scale"])
    if "num_inference_steps" in config:
        call_kwargs["num_inference_steps"] = int(config["num_inference_steps"])

    with torch.inference_mode():
        completed_512 = pipe(**call_kwargs).images[0]

    query_obj.iter_id += 1
    amodal_completion = completed_512.resize((input_w, input_h), Image.Resampling.LANCZOS)
    query_obj.amodal_completion = amodal_completion
    if query_obj.output_img_dir is not None:
        amodal_completion.save(query_obj.output_img_dir / f"inpainting_{query_obj.iter_id:02d}.png")
        (query_obj.output_img_dir / f"prompt_{query_obj.iter_id:02d}.txt").write_text(sd_prompt, encoding="utf-8")

    query_obj = check_occlusion_after_inpaint(
        models=models,
        amodal_completion_pil=amodal_completion,
        crop_query_mask=sd_modal_mask,
        query_class=query_class,
        classes=classes,
        query_obj=query_obj,
        crop_box=(crop_x_min, crop_x_max, crop_y_min, crop_y_max),
        box_thresh=box_thresh,
        text_thresh=text_thresh,
    )
    query_obj.iteration_logs.append(
        {
            "iteration": query_obj.iter_id,
            "prompt": sd_prompt,
            "seed": seed,
            "crop_box": [crop_x_min, crop_x_max, crop_y_min, crop_y_max],
            "continue_next_iter": bool(query_obj.run_iter),
        }
    )
    return query_obj
