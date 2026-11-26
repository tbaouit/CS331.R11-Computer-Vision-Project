# Phạm vi: Cả hai phiên bản: đọc một mẫu, kiểm tra khả năng resume, gọi backend, lưu ảnh RGBA,
# mask, trace và checksum.
"""CPU-side run management, checksum resume verification, and sample evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from src.eval.metrics import iou
from src.evaluation.reporting import write_json


# Băm byte của artefact để phát hiện tệp bị đổi hoặc thiếu.
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Đọc ảnh ở chế độ RGB với đúng thứ tự kênh.
def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


# Đọc PNG mask thành mảng bool cùng hệ tọa độ ảnh.
def read_mask(path):
    with Image.open(path) as image:
        return np.array(image.convert("L")) > 0


# Đưa kết quả về canvas gốc trước khi lưu và chấm.
def crop_original(array, pad, original_shape):
    h, w = original_shape[:2]
    pt, pb, pl, pr = (int(pad.get(k, 0)) for k in ("pad_top", "pad_bottom", "pad_left", "pad_right"))
    if array.shape[:2] != (h + pt + pb, w + pl + pr):
        raise ValueError(f"Output canvas {array.shape[:2]} disagrees with declared padding")
    return array[pt : pt + h, pl : pl + w].copy()


# Tạo ảnh xem nhanh đầu vào, mask và kết quả của một mẫu.
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


def save_auto_seg_debug(image, pred_visible, gt_visible, auto_info, visible_iou, path):
    """Save a 4-panel visual debug image: raw+bbox, pred_visible_mask, gt_visible_mask, and TP/FP/FN overlay."""
    box_img = Image.fromarray(image.copy())
    box_draw = ImageDraw.Draw(box_img)
    bbox = (auto_info or {}).get("pred_bbox_xyxy")
    score = (auto_info or {}).get("pred_score", 0.0)
    empty = bool((auto_info or {}).get("visible_pred_empty", not np.any(pred_visible)))
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        box_draw.rectangle([x1, y1, x2, y2], outline=(0, 230, 64), width=3)

    overlay = (image.astype(np.float32) * 0.45).astype(np.uint8)
    tp = pred_visible & gt_visible
    fp = pred_visible & (~gt_visible)
    fn = (~pred_visible) & gt_visible
    overlay[tp] = [40, 220, 80]    # Green = True Positive
    overlay[fp] = [235, 65, 65]    # Red   = False Positive
    overlay[fn] = [65, 130, 245]   # Blue  = False Negative

    tiles = [
        np.array(box_img),
        pred_visible.astype(np.uint8) * 255,
        gt_visible.astype(np.uint8) * 255,
        overlay,
    ]
    box_lbl = f"BBox (score={score:.2f})" if bbox else "BBox: EMPTY"
    labels = [
        box_lbl,
        "Pred Visible Mask" + (" [EMPTY]" if empty else ""),
        "GT Visible Mask",
        f"Overlay G:TP R:FP B:FN (IoU={visible_iou:.3f})",
    ]
    canvas = Image.new("RGB", (1120, 310), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        pic = Image.fromarray(tile)
        pic.thumbnail((270, 270))
        canvas.paste(pic, (index * 280 + (280 - pic.width) // 2, 30 + (270 - pic.height) // 2))
        draw.text((index * 280 + 5, 8), label, fill="black")
    canvas.save(path)


# Giải đường dẫn tương đối và kiểm tra tệp nằm trong thư mục của manifest.
def resolve_sample(base, value):
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Sample path escapes manifest directory")
    return path


# Xác định đường truyền mask ẩn từ config; cả hai config chính đều bật.
def _uses_occluded_mask_input(config):
    return config.get("pipeline_mode") in ("proposed", "baseline") or bool(config.get("use_occluded_mask_input"))


# Nối dữ liệu và backend cho một vật thể; trả kết quả và cờ cho biết có dùng lại artefact hay
# không.
def run_one(record, sample_root, output_root, config, backend, profile):
    oracle_mode = _uses_occluded_mask_input(config)
    image_path = resolve_sample(sample_root, record["image"])
    visible_path = resolve_sample(sample_root, record["visible_mask"])
    # Không resume nếu ảnh, mask, record hoặc cấu hình profile đã thay đổi.
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
    # Chỉ dùng lại mẫu đã hoàn thành khi toàn bộ artefact liên quan còn đúng checksum.
    if (job / "result.json").is_file():
        try:
            existing = json.loads((job / "result.json").read_text(encoding="utf-8"))
        except (ValueError, OSError):
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        artifacts = existing.get("artifact_sha256", {})
        if (
            existing.get("signature") == signature
            and existing.get("status") == "completed"
            and artifacts
            and all((job / name).is_file() and sha(job / name) == value for name, value in artifacts.items())
        ):
            return existing, True
        (job / "result.json").unlink()
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
        auto_mode = config.get("visible_mask_source", "gt") == "auto"
        visible_for_pipeline = None if auto_mode else visible
        backend.begin_sample(job)
        if oracle_mode:
            if occluded_path is not None:
                occluded_gt = read_mask(occluded_path) & (~visible)
            else:
                occluded_gt = read_mask(resolve_sample(sample_root, record["gt_amodal"])) & (~visible)
            if occluded_gt.shape != visible.shape:
                raise ValueError("Occluded input mask dimensions mismatch")
            Image.fromarray(occluded_gt.astype(np.uint8) * 255).save(job / "occluded_input.png")
            # In auto mode, do NOT pass GT visible_mask into the pipeline
            occluded_for_pipeline = (
                (occluded_gt | visible) if auto_mode else occluded_gt
            )
            output = backend.complete(
                image,
                visible_for_pipeline,
                record.get("name", "object"),
                occluded_mask=occluded_for_pipeline,
            )
        else:
            # Blind mode: GT paths are deliberately not opened until after completion returns.
            output = backend.complete(image, visible_for_pipeline, record.get("name", "object"))
        trace = output.get("_trace", {})
        if auto_mode and "pred_visible_mask" in output:
            pred_visible = np.asarray(output["pred_visible_mask"], dtype=bool)
        else:
            pred_visible = visible
        visible_pred_empty = bool(output.get("visible_pred_empty", not pred_visible.any()))
        vis_iou = iou(pred_visible, visible)
        if auto_mode:
            auto_info = dict(output.get("auto_segmentation") or trace.get("auto_segmentation") or {})
            auto_info["visible_iou"] = vis_iou
            auto_info["visible_pred_empty"] = visible_pred_empty
            trace["auto_segmentation"] = auto_info
            trace["visible_iou"] = vis_iou
            trace["visible_pred_empty"] = visible_pred_empty
            Image.fromarray(pred_visible.astype(np.uint8) * 255).save(job / "pred_visible.png")
            write_json(job / "auto_segmentation.json", auto_info)
            save_auto_seg_debug(image, pred_visible, visible, auto_info, vis_iou, job / "auto_seg_debug.jpg")
        full_rgba = np.asarray(output["inpainted_rgba"])
        full_mask = np.asarray(output["amodal_mask"], dtype=bool)
        if full_rgba.ndim != 3 or full_rgba.shape[2] != 4 or full_rgba.dtype != np.uint8:
            raise ValueError("Backend must return uint8 RGBA")
        if full_mask.shape != full_rgba.shape[:2]:
            raise ValueError("Output mask and RGB dimensions differ")
        pad = trace.get("pad_info", {})
        rgba = crop_original(full_rgba, pad, image.shape)
        amodal = crop_original(full_mask, pad, image.shape)
        hidden = amodal & ~pred_visible
        Image.fromarray(full_rgba).save(job / "completion_full_canvas.png")
        Image.fromarray(rgba).save(job / "completion.png")
        Image.fromarray(full_mask.astype(np.uint8) * 255).save(job / "amodal_full_canvas.png")
        Image.fromarray(amodal.astype(np.uint8) * 255).save(job / "amodal.png")
        Image.fromarray(hidden.astype(np.uint8) * 255).save(job / "missing.png")
        gt = read_mask(resolve_sample(sample_root, record["gt_amodal"]))
        if gt.shape != visible.shape:
            raise ValueError("GT dimensions mismatch")
        gt_hidden = gt & ~visible
        # Chấm mask và RGB visible tại kích thước ảnh gốc; GT ở đây là mask, không phải RGB vùng
        # ẩn. Appearance metrics luôn đối chiếu trên ground-truth visible_mask để công bằng với bản "gt".
        metrics = dict(
            visible_iou=vis_iou,
            amodal_iou=iou(amodal, gt),
            hidden_iou=iou(hidden, gt_hidden),
            visible_mae=float(np.abs(rgba[:, :, :3][visible].astype(float) - image[visible]).mean()),
        )
        if config.get("appearance_metrics"):
            from src.eval.appearance_metrics import compute_visible_appearance_metrics

            metrics.update(
                compute_visible_appearance_metrics(image, rgba[:, :, :3], visible, record.get("name", "object"),
                                                  device=config.get("appearance_metrics_device", "cpu"))
            )
        mask_lbl = "Exact Amodal (Vis+Occ)" if oracle_mode else "Amodal prediction"
        preview(image, pred_visible, amodal, rgba, job / "preview.jpg", mask_label=mask_lbl)
        history = output.get("critique_history", [])
        result = dict(
            status="completed",
            signature=signature,
            sample_id=record["sample_id"],
            profile=profile,
            name=record.get("name"),
            seconds=round(time.perf_counter() - started, 3),
            final_score=output.get("final_score"),
            critic_accepted=output.get("critic_accepted", (
                output.get("final_score") is not None and output["final_score"] >= config.get("critique_threshold", 7)
            )),
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
        if auto_mode:
            for extra_name in ("pred_visible.png", "auto_segmentation.json", "auto_seg_debug.jpg"):
                if (job / extra_name).is_file():
                    artifacts.append(extra_name)
        if config.get("pipeline_mode") == "proposed":
            artifacts += [
                name
                for name in ("prompt_selection.json", "proposed_selection.json")
                if (job / name).is_file()
            ]
        # Checksum cho bước resume và chấm lại; result.json được ghi sau ảnh và trace.
        result["artifact_sha256"] = {name: sha(job / name) for name in artifacts}
        write_json(job / "result.json", result)
        if (job / "failure.json").exists():
            (job / "failure.json").unlink()
        return result, False
    except Exception as error:
        (job / "failure.json").write_text(
            json.dumps(dict(status="failed", signature=signature, error=str(error), traceback=traceback.format_exc()), indent=2),
            encoding="utf-8",
        )
        raise
