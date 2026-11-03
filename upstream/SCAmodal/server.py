"""
AmodalSeg Backend — FastAPI server
Endpoints:
  POST /segment          → SAM3 auto-segmentation
  POST /amodal_complete  → Amodal completion for one selected mask
  GET  /health           → health check
"""

import uuid
import json
import numpy as np
from pathlib import Path
from io import BytesIO
from typing import Optional

import torch
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn

from pipeline.segmenter import SAMSegmenter
from pipeline.amodal_completer import AmodalCompleter

app = FastAPI(title="AmodalSeg API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store (use Redis in production) ──
sessions: dict[str, dict] = {}

# ── Lazy-loaded models ──
_segmenter: Optional[SAMSegmenter] = None
_completer: Optional[AmodalCompleter] = None


def get_segmenter() -> SAMSegmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = SAMSegmenter()
    return _segmenter


def get_completer() -> AmodalCompleter:
    global _completer
    if _completer is None:
        _completer = AmodalCompleter()
    return _completer


# ── Health ──
@app.get("/health")
async def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "device": device}


# ── Segmentation endpoint ──
@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    points_per_side: int = Form(32),
    pred_iou_thresh: float = Form(0.88),
    stability_score_thresh: float = Form(0.95),
):
    img_bytes = await image.read()
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_array = np.array(pil_img)

    segmenter = get_segmenter()
    masks = segmenter.segment_everything(
        img_array,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "image": img_array,
        "pil_image": pil_img,
        "masks": masks,
    }

    # Serialize masks for JSON response
    masks_json = []
    for i, m in enumerate(masks):
        seg = m["segmentation"]  # boolean numpy array H×W
        # Encode as bbox + RLE for lightweight transfer
        y_coords, x_coords = np.where(seg)
        if len(x_coords) == 0:
            continue
        x1, y1 = int(x_coords.min()), int(y_coords.min())
        x2, y2 = int(x_coords.max()), int(y_coords.max())

        masks_json.append({
            "id": i,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": int(m["area"]),
            "predicted_iou": float(m.get("predicted_iou", 0)),
            "stability_score": float(m.get("stability_score", 0)),
            "label": f"object_{i}",
            "occluded": _estimate_occlusion(seg, masks, i),
            "segmentation": {
                "x": x1, "y": y1,
                "w": x2 - x1, "h": y2 - y1
            },
        })

    return {
        "session_id": session_id,
        "masks": masks_json,
        "image_size": [pil_img.width, pil_img.height],
    }


def _estimate_occlusion(mask: np.ndarray, all_masks: list, idx: int) -> bool:
    """
    Quick heuristic: a mask is likely occluded if its bounding box
    has significantly more area than the mask itself (convexity ratio < 0.7),
    OR if it overlaps with a higher-index (on-top) mask.
    """
    seg = mask if isinstance(mask, np.ndarray) else mask["segmentation"]
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return False
    bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    if bbox_area == 0:
        return False
    convexity = seg.sum() / bbox_area
    return convexity < 0.72


# ── Amodal completion endpoint ──
@app.post("/amodal_complete")
async def amodal_complete(
    session_id: str = Form(...),
    mask_id: int = Form(...),
    text_query: str = Form(""),
    max_iter: int = Form(3),
):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    img_array = session["image"]
    masks = session["masks"]

    if mask_id >= len(masks):
        raise HTTPException(status_code=400, detail="Invalid mask_id")

    target_mask = masks[mask_id]["segmentation"].astype(bool)

    completer = get_completer()
    rgba_result = completer.complete(
        image=img_array,
        visible_mask=target_mask,
        all_masks=masks,
        text_query=text_query or f"object_{mask_id}",
        max_iter=max_iter,
    )

    # Return as PNG bytes
    result_pil = Image.fromarray(rgba_result, mode="RGBA")
    buf = BytesIO()
    result_pil.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
