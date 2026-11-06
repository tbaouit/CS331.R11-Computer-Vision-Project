"""CPU-side run management, checksum resume verification, and sample evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from src.evaluation.metrics import iou


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def read_mask(path):
    with Image.open(path) as image:
        return np.array(image.convert("L")) > 0


def crop_original(array, pad, original_shape):
    h, w = original_shape[:2]
    pt, pb, pl, pr = (int(pad.get(k, 0)) for k in ("pad_top", "pad_bottom", "pad_left", "pad_right"))
    if array.shape[:2] != (h + pt + pb, w + pl + pr):
        raise ValueError(f"Output canvas {array.shape[:2]} disagrees with declared padding")
    return array[pt : pt + h, pl : pl + w].copy()


def preview(image, visible, amodal, rgba, path, mask_label="Amodal prediction"):
    tiles = [image, visible.astype(np.uint8) * 255, amodal.astype(np.uint8) * 255, rgba]
    labels = ["Input", "Visible input", mask_label, "Completion"]
    canvas = Image.new("RGB", (1120, 310), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        pic = Image.fromarray(tile)
        if pic.mode == "RGBA":
            bg = Image.new("RGBA", pic.size, (245, 245, 245, 255))
            bg.alpha_composite(pic)
            pic = bg.convert("RGB")
        pic.thumbnail((270, 270))
        canvas.paste(pic, (index * 280 + (280 - pic.width) // 2, 30 + (270 - pic.height) // 2))
        draw.text((index * 280 + 5, 8), label, fill="black")
    canvas.save(path)


def resolve_sample(base, value):
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Sample path escapes manifest directory")
    return path


def _uses_occluded_mask_input(config):
    return config.get("pipeline_mode") in ("oracle_mask", "gt_mask", "sd2_direct") or bool(config.get("use_occluded_mask_input"))


def run_one(record, sample_root, output_root, config, backend, profile):
    oracle_mode = _uses_occluded_mask_input(config)
    image_path = resolve_sample(sample_root, record["image"])
    visible_path = resolve_sample(sample_root, record["visible_mask"])
    sig_payload = dict(profile=profile, input=sha(image_path), visible=sha(visible_path), record=record)
    occluded_path = None
    if oracle_mode:
        if record.get("gt_hidden"):
            occluded_path = resolve_sample(sample_root, record["gt_hidden"])
            sig_payload["occluded_input"] = sha(occluded_path)
        else:
            gt_for_input = resolve_sample(sample_root, record["gt_amodal"])
            sig_payload["occluded_from_amodal"] = sha(gt_for_input)
    signature = hashlib.sha256(json.dumps(sig_payload, sort_keys=True).encode()).hexdigest()
    job = output_root / record["sample_id"] / signature[:12]
    if (job / "result.json").is_file():
        existing = json.loads((job / "result.json").read_text())
        artifacts = existing.get("artifact_sha256", {})
        if (
            existing.get("signature") == signature
            and existing.get("status") == "completed"
            and artifacts
            and all((job / name).is_file() and sha(job / name) == value for name, value in artifacts.items())
        ):
            return existing, True
    job.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    (job / "sample.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (job / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        image, visible = read_rgb(image_path), read_mask(visible_path)
        if visible.shape != image.shape[:2] or not visible.any():
            raise ValueError("Invalid/empty visible mask")
        Image.fromarray(image).save(job / "input.png")
        Image.fromarray(visible.astype(np.uint8) * 255).save(job / "visible.png")
        backend.begin_sample(job)
        if oracle_mode:
            if occluded_path is not None:
                occluded_in = read_mask(occluded_path) & (~visible)
            else:
                occluded_in = read_mask(resolve_sample(sample_root, record["gt_amodal"])) & (~visible)
            if occluded_in.shape != visible.shape:
                raise ValueError("Occluded input mask dimensions mismatch")
            Image.fromarray(occluded_in.astype(np.uint8) * 255).save(job / "occluded_input.png")
            output = backend.complete(image, visible, record.get("name", "object"), occluded_mask=occluded_in)
        else:
            # Blind mode: GT paths are deliberately not opened until after completion returns.
            output = backend.complete(image, visible, record.get("name", "object"))
        trace = output.get("_trace", {})
        full_rgba = np.asarray(output["inpainted_rgba"])
        full_mask = np.asarray(output["amodal_mask"], dtype=bool)
        if full_rgba.ndim != 3 or full_rgba.shape[2] != 4 or full_rgba.dtype != np.uint8:
            raise ValueError("Backend must return uint8 RGBA")
        if full_mask.shape != full_rgba.shape[:2]:
            raise ValueError("Output mask and RGB dimensions differ")
        pad = trace.get("pad_info", {})
        rgba = crop_original(full_rgba, pad, image.shape)
        amodal = crop_original(full_mask, pad, image.shape)
        hidden = amodal & ~visible
        Image.fromarray(full_rgba).save(job / "completion_full_canvas.png")
        Image.fromarray(rgba).save(job / "completion.png")
        Image.fromarray(full_mask.astype(np.uint8) * 255).save(job / "amodal_full_canvas.png")
        Image.fromarray(amodal.astype(np.uint8) * 255).save(job / "amodal.png")
        Image.fromarray(hidden.astype(np.uint8) * 255).save(job / "missing.png")
        gt = read_mask(resolve_sample(sample_root, record["gt_amodal"]))
        if gt.shape != visible.shape:
            raise ValueError("GT dimensions mismatch")
        gt_hidden = gt & ~visible
        metrics = dict(
            amodal_iou=iou(amodal, gt),
            hidden_iou=iou(hidden, gt_hidden),
            visible_mae=float(np.abs(rgba[:, :, :3][visible].astype(float) - image[visible]).mean()),
            visible_exact_fraction=float(np.all(rgba[:, :, :3][visible] == image[visible], axis=1).mean()),
        )
        if config.get("appearance_metrics"):
            from src.evaluation.appearance_metrics import compute_visible_appearance_metrics

            metrics.update(
                compute_visible_appearance_metrics(image, rgba[:, :, :3], visible, record.get("name", "object"),
                                                  device=config.get("appearance_metrics_device", "cpu"))
            )
        mask_lbl = "Exact Amodal (Vis+Occ)" if oracle_mode else "Amodal prediction"
        preview(image, visible, amodal, rgba, job / "preview.jpg", mask_label=mask_lbl)
        history = output.get("critique_history", [])
        result = dict(
            status="completed",
            signature=signature,
            sample_id=record["sample_id"],
            profile=profile,
            name=record.get("name"),
            seconds=round(time.perf_counter() - started, 3),
            final_score=output.get("final_score"),
            critic_accepted=(
                output.get("final_score") is not None and output["final_score"] >= config["critique_threshold"]
            ),
            critique_history=history,
            trace=trace,
            metrics=metrics,
            metric_policy="IoU on original canvas (GT-given mask has trivial IoU); optional author LPIPS/CLIP-feature/SSIM/PSNR on visible mask over black; CLIP logit on full RGB",
            config=config,
        )
        (job / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        artifacts = [
            "input.png",
            "visible.png",
            "completion.png",
            "completion_full_canvas.png",
            "amodal.png",
            "missing.png",
            "preview.jpg",
            "trace.json",
        ]
        if oracle_mode:
            artifacts.append("occluded_input.png")
        result["artifact_sha256"] = {name: sha(job / name) for name in artifacts}
        (job / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if (job / "failure.json").exists():
            (job / "failure.json").unlink()
        return result, False
    except Exception as error:
        (job / "failure.json").write_text(
            json.dumps(dict(status="failed", signature=signature, error=str(error), traceback=traceback.format_exc()), indent=2),
            encoding="utf-8",
        )
        raise

