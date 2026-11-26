"""Recompute full-object appearance metrics (CVPR 2025 Table 3 / eva_amodal.py protocol) on saved outputs.

Compares `visible_input_white` (GT visible object on white background) against
`completed_amodal_white` (full completed amodal object—visible + inpainted occluded region—on white background)
cropped to their union bounding box, without re-running diffusion or segmentation inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.appearance_metrics import (
    _as_tensor,
    _clip,
    _dino,
    _vgg,
    lpips_distance,
    ssim_score,
)
from src.evaluation.reporting import write_json
from src.evaluation.runner import read_mask, read_rgb

PAPER_METRICS = (
    "LPIPS",
    "Feature_Similarity",
    "VGG_similarity",
    "SSIM",
    "CLIP_score",
    "DINO_similarity",
)


def composite_rgba_on_white(completion_path: Path, fallback_amodal_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load saved completion PNG and composite onto a plain white (255, 255, 255) background using its alpha mask."""
    with Image.open(completion_path) as pil_img:
        if "A" in pil_img.getbands():
            rgba = np.array(pil_img.convert("RGBA"), dtype=np.uint8)
            alpha_f = rgba[:, :, 3:4].astype(np.float32) / 255.0
            amodal_mask = rgba[:, :, 3] > 127
            rgb_f = rgba[:, :, :3].astype(np.float32)
            comp_white = np.clip(rgb_f * alpha_f + 255.0 * (1.0 - alpha_f), 0, 255).astype(np.uint8)
            return comp_white, amodal_mask
        rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    if fallback_amodal_path is not None and fallback_amodal_path.is_file():
        amodal_mask = read_mask(fallback_amodal_path)
        comp_white = np.where(amodal_mask[:, :, None], rgb, 255).astype(np.uint8)
        return comp_white, amodal_mask
    amodal_mask = np.any(rgb < 250, axis=2)
    return rgb, amodal_mask


def union_bbox_crop(
    img_a: np.ndarray,
    img_b: np.ndarray,
    union_mask: np.ndarray,
    min_side: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop both white-background images to the union bounding box of `visible_mask | amodal_mask` (min side >= 32)."""
    h, w = img_a.shape[:2]
    ys, xs = np.where(union_mask)
    if len(ys) == 0:
        y1, y2, x1, x2 = 0, h, 0, w
    else:
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1

    # Expand box symmetrically if smaller than min_side (required by LPIPS AlexNet pooling & SSIM window)
    cur_h = y2 - y1
    if cur_h < min_side:
        pad_y = min_side - cur_h
        y1 = max(0, y1 - pad_y // 2)
        y2 = min(h, y1 + min_side)
        y1 = max(0, y2 - min_side)
    cur_w = x2 - x1
    if cur_w < min_side:
        pad_x = min_side - cur_w
        x1 = max(0, x1 - pad_x // 2)
        x2 = min(w, x1 + min_side)
        x1 = max(0, x2 - min_side)

    crop_a = img_a[y1:y2, x1:x2].copy()
    crop_b = img_b[y1:y2, x1:x2].copy()
    if min(crop_a.shape[:2]) < min_side:
        pil_a = Image.fromarray(crop_a).resize((max(min_side, crop_a.shape[1]), max(min_side, crop_a.shape[0])), Image.Resampling.BILINEAR)
        pil_b = Image.fromarray(crop_b).resize((max(min_side, crop_b.shape[1]), max(min_side, crop_b.shape[0])), Image.Resampling.BILINEAR)
        crop_a = np.array(pil_a, dtype=np.uint8)
        crop_b = np.array(pil_b, dtype=np.uint8)
    return crop_a, crop_b


def prepare_paper_image_pair(job_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct `(visible_crop_white, completed_crop_white, visible_full_white, completed_full_white)` from a job dir."""
    image = read_rgb(job_dir / "input.png")
    visible_mask = read_mask(job_dir / "visible.png")
    visible_full_white = np.where(visible_mask[:, :, None], image, 255).astype(np.uint8)
    completed_full_white, amodal_mask = composite_rgba_on_white(job_dir / "completion.png", job_dir / "amodal.png")
    if completed_full_white.shape != visible_full_white.shape:
        raise ValueError(f"Canvas mismatch in {job_dir}: {completed_full_white.shape} vs {visible_full_white.shape}")
    union_mask = visible_mask | amodal_mask
    visible_crop_white, completed_crop_white = union_bbox_crop(visible_full_white, completed_full_white, union_mask)
    return visible_crop_white, completed_crop_white, visible_full_white, completed_full_white


def evaluate_batch_pairs(
    pairs: list[dict],
    device: str = "cuda",
    batch_size: int = 32,
) -> list[dict]:
    """Compute LPIPS, Feature_Similarity (CLIP), VGG_similarity, SSIM, CLIP_score, and DINO_similarity on pairs."""
    import torch
    import torch.nn.functional as F

    n = len(pairs)
    if n == 0:
        return []

    clip_model, clip_proc = _clip(device)
    dino_kind, dino_model, dino_proc = _dino(device)
    vgg_model, vgg_proc = _vgg(device)

    results: list[dict] = [{} for _ in range(n)]

    # 1. Per-pair native-resolution LPIPS & SSIM on the union-bbox white-background crops
    for i, item in enumerate(pairs):
        vis_crop = item["visible_crop"]
        comp_crop = item["completed_crop"]
        results[i]["LPIPS"] = lpips_distance(comp_crop, vis_crop, device=device)
        results[i]["SSIM"] = ssim_score(comp_crop, vis_crop)

    # 2. Batched GPU forward passes for CLIP, VGG-16, and DINO ViT-B/16
    for start in range(0, n, max(1, batch_size)):
        chunk = pairs[start : start + max(1, batch_size)]
        vis_crops = [c["visible_crop"] for c in chunk]
        comp_crops = [c["completed_crop"] for c in chunk]
        labels = [c["label"] for c in chunk]

        with torch.no_grad():
            # CLIP visual feature similarity (Feature_Similarity)
            in_vis_clip = clip_proc(images=vis_crops, return_tensors="pt").to(device)
            in_comp_clip = clip_proc(images=comp_crops, return_tensors="pt").to(device)
            feat_vis_clip = _as_tensor(clip_model.get_image_features(**in_vis_clip))
            feat_comp_clip = _as_tensor(clip_model.get_image_features(**in_comp_clip))
            clip_feat_sims = F.cosine_similarity(feat_comp_clip, feat_vis_clip, dim=1).detach().cpu().tolist()

            # CLIP image-text prompt relevance score (CLIP_score)
            txt_inputs = clip_proc(text=labels, return_tensors="pt", padding=True, truncation=True).to(device)
            txt_feat = _as_tensor(
                clip_model.get_text_features(
                    input_ids=txt_inputs["input_ids"],
                    attention_mask=txt_inputs["attention_mask"],
                )
            )
            feat_comp_norm = feat_comp_clip / feat_comp_clip.norm(dim=-1, keepdim=True)
            txt_feat_norm = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            logit_scale = clip_model.logit_scale.exp()
            clip_scores = (logit_scale * (feat_comp_norm * txt_feat_norm).sum(dim=-1)).detach().cpu().tolist()

            # VGG-16 convolutional feature cosine similarity (compute_feature_similarity in eva_amodal.py)
            t_vis_vgg = torch.stack([vgg_proc(Image.fromarray(img)) for img in vis_crops], dim=0).to(device)
            t_comp_vgg = torch.stack([vgg_proc(Image.fromarray(img)) for img in comp_crops], dim=0).to(device)
            f_vis_vgg = vgg_model(t_vis_vgg)
            f_comp_vgg = vgg_model(t_comp_vgg)
            vgg_sims_tensor = F.cosine_similarity(f_comp_vgg, f_vis_vgg, dim=1).mean(dim=(1, 2))
            vgg_sims = vgg_sims_tensor.detach().cpu().tolist()

            # DINO ViT-B/16 feature cosine similarity (compute_dino_similarity in eva_amodal.py)
            if dino_kind == "hf":
                in_vis_dino = dino_proc(images=[Image.fromarray(img) for img in vis_crops], return_tensors="pt").to(device)
                in_comp_dino = dino_proc(images=[Image.fromarray(img) for img in comp_crops], return_tensors="pt").to(device)
                f_vis_dino = _as_tensor(dino_model(**in_vis_dino))
                f_comp_dino = _as_tensor(dino_model(**in_comp_dino))
            else:
                t_vis_dino = torch.stack([dino_proc(Image.fromarray(img)) for img in vis_crops], dim=0).to(device)
                t_comp_dino = torch.stack([dino_proc(Image.fromarray(img)) for img in comp_crops], dim=0).to(device)
                f_vis_dino = dino_model(t_vis_dino)
                f_comp_dino = dino_model(t_comp_dino)
            dino_sims = F.cosine_similarity(f_vis_dino, f_comp_dino, dim=1).detach().cpu().tolist()

        for offset, (cfs, cs, vs, ds) in enumerate(zip(clip_feat_sims, clip_scores, vgg_sims, dino_sims)):
            idx = start + offset
            results[idx]["Feature_Similarity"] = float(cfs)
            results[idx]["VGG_similarity"] = float(vs)
            results[idx]["CLIP_score"] = float(cs)
            results[idx]["DINO_similarity"] = float(ds)

    return results


def summarize_rows(rows: list[dict]) -> dict:
    """Compute `{metric: {n, mean, std}}` across completed rows."""
    out = {}
    for metric in PAPER_METRICS:
        vals = [
            float(r["metrics"][metric])
            for r in rows
            if metric in r.get("metrics", {})
            and r["metrics"][metric] is not None
            and math.isfinite(float(r["metrics"][metric]))
        ]
        out[metric] = {
            "n": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        }
    return out


def collect_completed_jobs(profile_dir: Path) -> dict[str, Path]:
    """Map `sample_id -> job_dir` for all completed samples inside `profile_dir`."""
    jobs: dict[str, Path] = {}
    for res_path in sorted(profile_dir.glob("*/*/result.json"), key=lambda p: p.stat().st_mtime):
        try:
            data = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") == "completed" and data.get("sample_id"):
            jobs[str(data["sample_id"])] = res_path.parent
    return jobs


def _auto_discover_paths(
    manifest_arg: Path | None,
    runs_root_arg: Path | None,
    baseline_id: str,
    proposed_id: str,
) -> tuple[Path, Path, Path]:
    """Locate manifest and baseline/proposed profile directories on Colab Drive or local workspace."""
    search_batches: list[Path] = []
    if runs_root_arg is not None:
        search_batches.append(runs_root_arg.parent)
    for drive_root in (
        Path("/content/drive/MyDrive/CS331/results"),
        Path("/content/drive/MyDrive/amodal_project/results"),
        ROOT / "outputs",
    ):
        if drive_root.is_dir():
            for batch_dir in sorted(drive_root.glob("*/batch_*"), key=lambda p: p.stat().st_mtime, reverse=True):
                search_batches.append(batch_dir)

    def resolve_profile_dir(base_runs: Path, role: str, target_id: str) -> Path | None:
        role_dir = base_runs / role
        if (role_dir / target_id).is_dir():
            return role_dir / target_id
        marker = base_runs.parent / "profiles" / f"{role}.json"
        if marker.is_file():
            try:
                prof = json.loads(marker.read_text(encoding="utf-8")).get("profile", "")
                if prof and (role_dir / prof).is_dir():
                    return role_dir / prof
            except Exception:
                pass
        if role_dir.is_dir():
            subdirs = [d for d in role_dir.iterdir() if d.is_dir() and len(d.name) == 16]
            if subdirs:
                return max(subdirs, key=lambda p: p.stat().st_mtime)
        return None

    if manifest_arg and manifest_arg.is_file() and runs_root_arg and runs_root_arg.is_dir():
        b_dir = resolve_profile_dir(runs_root_arg, "baseline", baseline_id)
        p_dir = resolve_profile_dir(runs_root_arg, "proposed", proposed_id)
        if b_dir and p_dir:
            return manifest_arg, b_dir, p_dir

    for batch_dir in search_batches:
        m_path = manifest_arg if (manifest_arg and manifest_arg.is_file()) else (batch_dir / "manifest.json")
        r_root = runs_root_arg if (runs_root_arg and runs_root_arg.is_dir()) else (batch_dir / "runs")
        if m_path.is_file() and r_root.is_dir():
            b_dir = resolve_profile_dir(r_root, "baseline", baseline_id)
            p_dir = resolve_profile_dir(r_root, "proposed", proposed_id)
            if b_dir and p_dir:
                return m_path, b_dir, p_dir

    raise FileNotFoundError(
        "Could not auto-discover manifest and baseline/proposed profile directories. "
        "Pass --manifest and --runs-root (or --baseline-dir and --proposed-dir) explicitly."
    )


def format_summary_table(title: str, n_samples: int, base_stats: dict, prop_stats: dict) -> str:
    """Render a clean Markdown comparison table (`mean ± std`) for Baseline vs. Proposed."""
    directions = {
        "LPIPS": "↓",
        "Feature_Similarity": "↑",
        "VGG_similarity": "↑",
        "SSIM": "↑",
        "CLIP_score": "↑",
        "DINO_similarity": "↑",
    }
    labels = {
        "LPIPS": "LPIPS (AlexNet)",
        "Feature_Similarity": "Feature_Similarity (CLIP)",
        "VGG_similarity": "VGG_similarity (eva_amodal)",
        "SSIM": "SSIM",
        "CLIP_score": "CLIP_score",
        "DINO_similarity": "DINO_similarity",
    }
    lines = [
        f"\n=== {title} (N = {n_samples}) ===",
        f"{'Metric':<30} | {'Dir':^3} | {'Baseline (mean ± std)':^25} | {'Proposed (mean ± std)':^25} | {'Delta (Prop - Base)':^19}",
        "-" * 112,
    ]
    for m in PAPER_METRICS:
        b = base_stats.get(m, {})
        p = prop_stats.get(m, {})
        bm, bs = b.get("mean"), b.get("std")
        pm, ps = p.get("mean"), p.get("std")
        b_str = f"{bm:.4f} ± {(bs or 0.0):.4f}" if bm is not None else "N/A"
        p_str = f"{pm:.4f} ± {(ps or 0.0):.4f}" if pm is not None else "N/A"
        d_str = f"{(pm - bm):+.4f}" if (bm is not None and pm is not None) else "N/A"
        lines.append(f"{labels[m]:<30} | {directions[m]:^3} | {b_str:^25} | {p_str:^25} | {d_str:^19}")
    return "\n".join(lines)


def reevaluate_full_object_protocol(
    manifest_path: Path,
    baseline_dir: Path,
    proposed_dir: Path,
    output_json: Path,
    device: str = "cuda",
    batch_size: int = 32,
    min_occlusion: float = 0.15,
) -> dict:
    """Run paper-aligned full-object re-evaluation across all manifest samples without modifying existing files."""
    t0 = time.perf_counter()
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_jobs = collect_completed_jobs(baseline_dir)
    prop_jobs = collect_completed_jobs(proposed_dir)

    method_dirs = {"baseline": (baseline_dir, base_jobs), "proposed": (proposed_dir, prop_jobs)}
    per_method_rows: dict[str, list[dict]] = {"baseline": [], "proposed": []}

    for role, (p_dir, job_map) in method_dirs.items():
        print(f"[Re-Eval] Loading & cropping {len(records)} samples for {role} ({p_dir.name})...", flush=True)
        pairs: list[dict] = []
        valid_records: list[dict] = []
        for rec in records:
            sid = rec["sample_id"]
            if sid not in job_map:
                continue
            job_dir = job_map[sid]
            vis_crop, comp_crop, _, _ = prepare_paper_image_pair(job_dir)
            pairs.append(
                {
                    "sample_id": sid,
                    "visible_crop": vis_crop,
                    "completed_crop": comp_crop,
                    "label": (rec.get("name") or "object").strip() or "object",
                }
            )
            valid_records.append(rec)

        print(f"[Re-Eval] Evaluating {len(pairs)} pairs for {role} on {device} (batch_size={batch_size})...", flush=True)
        scores = evaluate_batch_pairs(pairs, device=device, batch_size=batch_size)
        for rec, metric_dict in zip(valid_records, scores):
            per_method_rows[role].append(
                {
                    "sample_id": rec["sample_id"],
                    "category": rec.get("name", ""),
                    "split": rec.get("split", ""),
                    "hidden_ratio": float(rec.get("hidden_ratio") or 0.0),
                    "metrics": metric_dict,
                }
            )

    overall_stats = {role: summarize_rows(rows) for role, rows in per_method_rows.items()}
    heavy_rows = {
        role: [r for r in rows if float(r.get("hidden_ratio") or 0.0) >= min_occlusion]
        for role, rows in per_method_rows.items()
    }
    heavy_stats = {role: summarize_rows(rows) for role, rows in heavy_rows.items()}

    report = {
        "protocol": "cvpr2025_openworld_amodal_full_object_white_bg_union_crop",
        "description": (
            "Full-object evaluation protocol (CVPR 2025 Table 3 / eva_amodal.py): compares visible_input_white "
            "against completed_amodal_white (full amodal object on white background) cropped to union bounding box."
        ),
        "manifest": str(manifest_path),
        "profiles": {
            "baseline": baseline_dir.name,
            "proposed": proposed_dir.name,
        },
        "requested_samples": len(records),
        "evaluated_samples": {
            "baseline": len(per_method_rows["baseline"]),
            "proposed": len(per_method_rows["proposed"]),
        },
        "min_occlusion_subset_threshold": min_occlusion,
        "occluded_subset_count": {
            "baseline": len(heavy_rows["baseline"]),
            "proposed": len(heavy_rows["proposed"]),
        },
        "overall": overall_stats,
        "occluded_subset_ge_15pct": heavy_stats,
        "seconds": round(time.perf_counter() - t0, 2),
        "per_sample": per_method_rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)

    csv_path = output_json.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "method", "metric", "mean", "std", "n"])
        for subset_name, stats_dict in (("all_samples", overall_stats), (f"occlusion_ge_{int(min_occlusion*100)}pct", heavy_stats)):
            for role in ("baseline", "proposed"):
                for m in PAPER_METRICS:
                    s = stats_dict[role][m]
                    writer.writerow([subset_name, role, m, s["mean"], s["std"], s["n"]])

    print(
        format_summary_table(
            "Full-Object Paper Protocol — All Samples",
            len(per_method_rows["proposed"]),
            overall_stats["baseline"],
            overall_stats["proposed"],
        ),
        flush=True,
    )
    print(
        format_summary_table(
            f"Full-Object Paper Protocol — Occlusion Ratio >= {min_occlusion * 100:.0f}%",
            len(heavy_rows["proposed"]),
            heavy_stats["baseline"],
            heavy_stats["proposed"],
        ),
        flush=True,
    )
    print(f"\n[Re-Eval] Saved JSON report to: {output_json}", flush=True)
    print(f"[Re-Eval] Saved CSV summary to: {csv_path}", flush=True)
    return report


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None, help="Path to manifest.json (auto-discovered if omitted)")
    parser.add_argument("--runs-root", type=Path, default=None, help="Path to batch runs/ directory on Drive")
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Explicit path to baseline profile folder")
    parser.add_argument("--proposed-dir", type=Path, default=None, help="Explicit path to proposed profile folder")
    parser.add_argument("--baseline-profile", type=str, default="3320c2c08535da35")
    parser.add_argument("--proposed-profile", type=str, default="f7d831fd508db25a")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/comparisons/full_820_paper_protocol_metrics.json",
        help="Non-destructive output JSON path",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-occlusion", type=float, default=0.15)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    device = args.device or _default_device()

    if args.baseline_dir and args.proposed_dir and args.manifest:
        m_path, b_dir, p_dir = args.manifest, args.baseline_dir, args.proposed_dir
    else:
        m_path, b_dir, p_dir = _auto_discover_paths(
            args.manifest,
            args.runs_root,
            args.baseline_profile,
            args.proposed_profile,
        )

    reevaluate_full_object_protocol(
        manifest_path=m_path,
        baseline_dir=b_dir,
        proposed_dir=p_dir,
        output_json=args.output,
        device=device,
        batch_size=args.batch_size,
        min_occlusion=args.min_occlusion,
    )


if __name__ == "__main__":
    main()
