"""Read-only validation of prepared samples before expensive GPU inference."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.evaluation.runner import read_rgb, resolve_sample


def strict_mask(path, shape):
    with Image.open(path) as opened:
        array = np.array(opened)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"Mask must be single-channel on original canvas: {path}")
    if not np.isin(array, [0, 1, 255]).all():
        raise ValueError(f"Mask is not binary: {path}")
    return array > 0


def validate_record(record, base):
    if record.get("split") not in ("train", "dev", "test"):
        raise ValueError("Unknown or missing split")
    if not isinstance(record.get("name"), str) or not record["name"].strip():
        raise ValueError("Missing annotation object name")
    image = read_rgb(resolve_sample(base, record["image"]))
    if min(image.shape[:2]) < 32:
        raise ValueError("Image canvas too small for author metrics")
    visible = strict_mask(resolve_sample(base, record["visible_mask"]), image.shape[:2])
    amodal = strict_mask(resolve_sample(base, record["gt_amodal"]), image.shape[:2])
    if not visible.any() or not amodal.any() or (visible & ~amodal).any():
        raise ValueError("Empty masks or visible pixels outside amodal mask")
    hidden = amodal & ~visible
    if not hidden.any():
        raise ValueError("No hidden region to complete")
    if record.get("gt_hidden"):
        stored = strict_mask(resolve_sample(base, record["gt_hidden"]), image.shape[:2])
        if not np.array_equal(hidden, stored):
            raise ValueError("Hidden mask differs from amodal minus visible")
    ratio = float(hidden.sum() / amodal.sum())
    if record.get("hidden_ratio") is not None and not np.isclose(record["hidden_ratio"], ratio):
        raise ValueError("Recorded hidden ratio disagrees with decoded masks")
    return dict(sample_id=record["sample_id"], status="valid", hidden_ratio=ratio,
                visible_pixels=int(visible.sum()), amodal_pixels=int(amodal.sum()),
                visible_canvas_fraction=float(visible.mean()))


def validate_manifest(records, base):
    ids = [r["sample_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    rows = []
    for record in records:
        try:
            rows.append(validate_record(record, base))
        except Exception as error:
            rows.append(dict(sample_id=record["sample_id"], status="invalid", error=str(error)))
    return dict(total=len(rows), valid=sum(r["status"] == "valid" for r in rows),
                invalid=sum(r["status"] == "invalid" for r in rows), samples=rows)
