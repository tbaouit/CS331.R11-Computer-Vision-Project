# Phạm vi: Cả hai phiên bản: ghi JSON/CSV và thống kê theo lớp và mức che; mẫu thiếu hoặc lỗi không
# bị tính thành điểm 0.
"""Atomic JSON and per-object macro summaries; missing scores never become zeros."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np


# Ghi vào file tạm rồi thay thế để tránh JSON dang dở khi bị ngắt.
def write_json(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(partial, path)


# Thống kê trên các điểm hợp lệ của mẫu đã hoàn thành.
def aggregate(rows):
    successful = [r for r in rows if r.get("status") == "completed"]
    names = sorted({k for r in successful for k in r.get("metrics", {})})
    metrics = {}
    for name in names:
        values = [r["metrics"].get(name) for r in successful]
        values = [float(v) for v in values if isinstance(v, (float, int)) and math.isfinite(v)]
        metrics[name] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                             std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    return dict(requested=len(rows), completed=len(successful),
                failed=sum(r.get("status") == "failed" for r in rows),
                missing=sum(r.get("status") == "missing" for r in rows), metrics=metrics)


# Xuất điểm từng mẫu, tổng hợp và phân nhóm theo lớp/mức che.
def write_reports(directory, rows, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "per_sample.json", rows)
    groups = {}
    for key in ("category", "occlusion_bin"):
        groups[key] = {str(value): aggregate([r for r in rows if r.get(key) == value])
                       for value in sorted({r.get(key, "unknown") for r in rows}, key=str)}
    report = dict(metadata=metadata or {}, overall=aggregate(rows), groups=groups)
    write_json(directory / "aggregate.json", report)
    metric_names = sorted({k for r in rows for k in r.get("metrics", {})})
    fields = ["sample_id", "category", "hidden_ratio", "occlusion_bin", "status", "error",
              "visible_canvas_fraction"] + metric_names
    partial = directory / "metrics.csv.part"
    with partial.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, row.get("metrics", {}).get(k)) for k in fields})
    os.replace(partial, directory / "metrics.csv")
    return report


# Đưa tỷ lệ che liên tục vào nhóm nhẹ, vừa hoặc nặng để đối chiếu.
def occlusion_bin(ratio):
    if ratio is None:
        return "unknown"
    return "light_<20%" if ratio < .2 else "medium_20-40%" if ratio <= .4 else "heavy_>40%"
