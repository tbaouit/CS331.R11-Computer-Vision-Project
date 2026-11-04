"""Multi-run comparison engine (Drive backup restore + 2-way & 3-way visual/quantitative reports)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from src.evaluation.metrics import compute_appearance_coherence_metrics

DEFAULT_BACKUP_DIRS = [
    Path("/content/drive/MyDrive/CS331/results"),
    Path("/content/drive/MyDrive/amodal_project/results"),
]


def auto_restore_drive_backups(scamodal_root: Path, extra_backup_dir: Optional[Path] = None) -> list[Path]:
    """Extract the latest Google Drive backup archive for any profile missing from scamodal_root."""
    search_dirs: list[Path] = []
    if extra_backup_dir is not None:
        search_dirs.append(extra_backup_dir)
    for d in DEFAULT_BACKUP_DIRS:
        if d not in search_dirs:
            search_dirs.append(d)

    latest_zip_by_profile: dict[str, Path] = {}
    for bdir in search_dirs:
        if not bdir.is_dir():
            continue
        for zpath in sorted(bdir.glob("scamodal_*_*.zip"), key=lambda p: p.name):
            parts = zpath.stem.split("_")
            if len(parts) < 3:
                continue
            profile = parts[1]
            if len(profile) == 16 and all(c in "0123456789abcdef" for c in profile):
                prev = latest_zip_by_profile.get(profile)
                if prev is None or zpath.name >= prev.name:
                    latest_zip_by_profile[profile] = zpath

    restored: list[Path] = []
    for profile, zpath in sorted(latest_zip_by_profile.items()):
        dest = scamodal_root / profile
        existing_count = len(list(dest.glob("*/*/result.json"))) if dest.is_dir() else 0
        try:
            with zipfile.ZipFile(zpath) as bundle:
                zip_result_count = sum(1 for item in bundle.namelist() if item.endswith("/result.json"))
                if existing_count >= zip_result_count and existing_count > 0:
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                for item in bundle.infolist():
                    if not (dest / item.filename).resolve().is_relative_to(dest.resolve()):
                        raise ValueError(f"Unsafe archive path in {zpath}: {item.filename}")
                bundle.extractall(dest)
                restored.append(dest)
                print(f"[Auto-Restore] Restored profile {profile} ({zip_result_count} samples) from {zpath}")
        except Exception as exc:
            print(f"[Auto-Restore] Skipping backup {zpath}: {exc}")
    return restored


def classify_mode(data: dict) -> str:
    cfg_mode = data.get("config", {}).get("pipeline_mode", "")
    backend_name = data.get("trace", {}).get("backend", "")
    if cfg_mode in ("oracle_mask", "gt_mask") or "Oracle" in backend_name:
        return "oracle_mask"
    if cfg_mode == "improved" or "Improved" in backend_name:
        return "improved"
    return "baseline"


def load_profile_results(profile_dir: Path) -> tuple[str, str, dict[str, tuple[Path, dict]]]:
    results: dict[str, tuple[Path, dict]] = {}
    mode_kind = "baseline"
    for res_path in sorted(profile_dir.glob("*/*/result.json"), key=lambda p: p.stat().st_mtime):
        data = json.loads(res_path.read_text(encoding="utf-8"))
        if data.get("status") != "completed":
            continue
        sid = data["sample_id"]
        results[sid] = (res_path.parent, data)
        mode_kind = classify_mode(data)
    mode_label = f"{mode_kind} ({profile_dir.name})"
    return mode_kind, mode_label, results


def resolve_profile_dir(spec: Optional[str], scamodal_root: Path) -> Optional[Path]:
    if not spec:
        return None
    cand = Path(spec)
    if cand.is_dir():
        return cand
    cand2 = scamodal_root / spec
    if cand2.is_dir():
        return cand2
    raise FileNotFoundError(f"Run directory not found: {spec}")


def discover_profiles_by_mode(scamodal_root: Path, backup_dir: Optional[Path] = None) -> dict[str, list[Path]]:
    scamodal_root.mkdir(parents=True, exist_ok=True)
    auto_restore_drive_backups(scamodal_root, backup_dir)
    profiles = [p for p in scamodal_root.iterdir() if p.is_dir() and list(p.glob("*/*/result.json"))]
    if not profiles:
        raise RuntimeError(f"No completed profile directories found under {scamodal_root}.")

    by_mode: dict[str, list[Path]] = {"baseline": [], "improved": [], "oracle_mask": []}
    ordered = sorted(
        profiles,
        key=lambda d: (len(list(d.glob("*/*/result.json"))), max(f.stat().st_mtime for f in d.glob("*/*/result.json"))),
    )
    for p in ordered:
        kind, _, res = load_profile_results(p)
        if res:
            by_mode.setdefault(kind, []).append(p)
    return by_mode


def _render_rgba_tile(path: Path, bg_rgb=(245, 245, 245)) -> Image.Image:
    with Image.open(path) as im:
        if im.mode == "RGBA":
            bg = Image.new("RGBA", im.size, (*bg_rgb, 255))
            bg.alpha_composite(im)
            return bg.convert("RGB")
        return im.convert("RGB")


def _read_mask_bool(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 127


def _read_rgb_np(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode == "RGBA":
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im)
            return np.asarray(bg.convert("RGB"))
        return np.asarray(im.convert("RGB"))


def _compute_job_seam_metrics(job_dir: Path, gt_amodal_path: Path) -> dict[str, Optional[float]]:
    image = _read_rgb_np(job_dir / "input.png")
    vis = _read_mask_bool(job_dir / "visible.png")
    gt_amodal = _read_mask_bool(gt_amodal_path)
    gt_occ = gt_amodal & (~vis)
    comp_rgb = _read_rgb_np(job_dir / "completion.png")
    return compute_appearance_coherence_metrics(image, vis, gt_occ, comp_rgb)


def run_three_way_comparison(
    base_dir: Path,
    imp_dir: Path,
    ora_dir: Path,
    manifest: list[dict],
    sample_root: Path,
    output_grid: Path,
) -> None:
    _, base_label, base_map = load_profile_results(base_dir)
    _, imp_label, imp_map = load_profile_results(imp_dir)
    _, ora_label, ora_map = load_profile_results(ora_dir)

    common = [
        r for r in manifest
        if r["sample_id"] in base_map and r["sample_id"] in imp_map and r["sample_id"] in ora_map
    ]
    if not common:
        raise RuntimeError("No overlapping completed samples across Baseline, Improved, and Oracle-Mask profiles.")

    print(f"Baseline    : {base_label} ({len(base_map)} completed)")
    print(f"Improved    : {imp_label} ({len(imp_map)} completed)")
    print(f"Oracle-Mask : {ora_label} ({len(ora_map)} completed)")
    print(f"Comparing   : {len(common)} common samples (3-Way: Baseline -> Improved -> Oracle-Mask)\n")

    header = (
        f"{'Sample ID':<27} | {'Object':<14} | {'Amodal IoU (B->I->O)':<26} | "
        f"{'Hidden IoU (B->I->O)':<26} | {'SeamMAE (B->I->O)':<21} | {'Critic (B->I->O)'}"
    )
    print(header)
    print("-" * len(header))

    rows_summary = []
    tile_w, tile_h, header_h = 205, 175, 28
    cols = [
        "Input",
        "Visible",
        "Occluded (Input)",
        "Base RGB",
        "Improved RGB",
        "Oracle Mask",
        "Oracle RGB",
    ]
    grid = Image.new("RGB", (len(cols) * tile_w, len(common) * (tile_h + header_h)), "white")
    draw = ImageDraw.Draw(grid)

    for row_idx, rec in enumerate(common):
        sid = rec["sample_id"]
        cat = str(rec.get("name", "object"))[:14]
        b_job, b_res = base_map[sid]
        i_job, i_res = imp_map[sid]
        o_job, o_res = ora_map[sid]
        bm, im, om = b_res["metrics"], i_res["metrics"], o_res["metrics"]

        gt_path = sample_root / rec["gt_amodal"]
        occ_path = (
            (sample_root / rec["gt_hidden"])
            if rec.get("gt_hidden") and (sample_root / rec["gt_hidden"]).is_file()
            else (o_job / "occluded_input.png" if (o_job / "occluded_input.png").is_file() else gt_path)
        )

        b_coh = _compute_job_seam_metrics(b_job, gt_path)
        i_coh = _compute_job_seam_metrics(i_job, gt_path)
        o_coh = _compute_job_seam_metrics(o_job, gt_path)

        b_aiou, i_aiou, o_aiou = bm["amodal_iou"] or 0.0, im["amodal_iou"] or 0.0, om["amodal_iou"] or 0.0
        b_hiou = bm["hidden_iou"] if bm["hidden_iou"] is not None else 0.0
        i_hiou = im["hidden_iou"] if im["hidden_iou"] is not None else 0.0
        o_hiou = om["hidden_iou"] if om["hidden_iou"] is not None else 0.0

        b_sc, i_sc, o_sc = b_res.get("final_score"), i_res.get("final_score"), o_res.get("final_score")
        fmt_sc = lambda v: f"{v:.2f}" if v is not None else "N/A"

        print(
            f"{sid:<27} | {cat:<14} | "
            f"{b_aiou:.4f}->{i_aiou:.4f}->{o_aiou:.4f}   | "
            f"{b_hiou:.4f}->{i_hiou:.4f}->{o_hiou:.4f}   | "
            f"{b_coh['seam_mae']:5.1f}->{i_coh['seam_mae']:5.1f}->{o_coh['seam_mae']:5.1f} | "
            f"{fmt_sc(b_sc):>4}->{fmt_sc(i_sc):>4}->{fmt_sc(o_sc):>4}"
        )

        rows_summary.append(
            {
                "sample_id": sid,
                "name": rec.get("name"),
                "baseline": {**bm, **b_coh, "critic_score": b_sc},
                "improved": {**im, **i_coh, "critic_score": i_sc},
                "oracle_mask": {**om, **o_coh, "critic_score": o_sc},
            }
        )

        y_top = row_idx * (tile_h + header_h)
        tile_images = [
            _render_rgba_tile(o_job / "input.png"),
            _render_rgba_tile(o_job / "visible.png"),
            _render_rgba_tile(occ_path),
            _render_rgba_tile(b_job / "completion.png"),
            _render_rgba_tile(i_job / "completion.png"),
            _render_rgba_tile(o_job / "amodal.png"),
            _render_rgba_tile(o_job / "completion.png"),
        ]
        for col_idx, (col_title, pic) in enumerate(zip(cols, tile_images)):
            x_left = col_idx * tile_w
            pic_copy = pic.copy()
            pic_copy.thumbnail((tile_w - 8, tile_h - 6))
            px = x_left + (tile_w - pic_copy.width) // 2
            py = y_top + header_h + (tile_h - pic_copy.height) // 2
            grid.paste(pic_copy, (px, py))
            caption = f"{col_title}" if col_idx > 0 else f"{sid[-11:]} ({cat})"
            draw.text((x_left + 4, y_top + 6), caption, fill="black")

    def _mean3(key: str, side: str) -> float:
        vals = [r[side][key] for r in rows_summary if r[side].get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    ba, ia, oa = _mean3("amodal_iou", "baseline"), _mean3("amodal_iou", "improved"), _mean3("amodal_iou", "oracle_mask")
    bh, ih, oh = _mean3("hidden_iou", "baseline"), _mean3("hidden_iou", "improved"), _mean3("hidden_iou", "oracle_mask")
    bs, is_, os_ = _mean3("seam_mae", "baseline"), _mean3("seam_mae", "improved"), _mean3("seam_mae", "oracle_mask")
    bc, ic, oc = _mean3("critic_score", "baseline"), _mean3("critic_score", "improved"), _mean3("critic_score", "oracle_mask")

    print("-" * len(header))
    print(
        f"{'MEAN (' + str(len(common)) + ' samples)':<27} | {'ALL':<14} | "
        f"{ba:.4f}->{ia:.4f}->{oa:.4f}   | "
        f"{bh:.4f}->{ih:.4f}->{oh:.4f}   | "
        f"{bs:5.1f}->{is_:5.1f}->{os_:5.1f} | "
        f"{bc:4.2f}->{ic:4.2f}->{oc:4.2f}"
    )

    output_grid.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_grid, quality=92)
    summary_path = output_grid.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "mode": "three_way",
                "baseline_profile": base_dir.name,
                "improved_profile": imp_dir.name,
                "oracle_mask_profile": ora_dir.name,
                "num_samples": len(common),
                "mean_baseline": {
                    "amodal_iou": round(ba, 4),
                    "hidden_iou": round(bh, 4),
                    "seam_mae": round(bs, 3),
                    "color_hist_sim": round(_mean3("color_hist_sim", "baseline"), 4),
                    "critic_score": round(bc, 3),
                },
                "mean_improved": {
                    "amodal_iou": round(ia, 4),
                    "hidden_iou": round(ih, 4),
                    "seam_mae": round(is_, 3),
                    "color_hist_sim": round(_mean3("color_hist_sim", "improved"), 4),
                    "critic_score": round(ic, 3),
                },
                "mean_oracle_mask": {
                    "amodal_iou": round(oa, 4),
                    "hidden_iou": round(oh, 4),
                    "seam_mae": round(os_, 3),
                    "color_hist_sim": round(_mean3("color_hist_sim", "oracle_mask"), 4),
                    "critic_score": round(oc, 3),
                },
                "samples": rows_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved visual comparison grid: {output_grid}")
    print(f"Saved JSON comparison report: {summary_path}")


def run_two_way_comparison(
    base_dir: Path,
    imp_dir: Path,
    manifest: list[dict],
    sample_root: Path,
    output_grid: Path,
) -> None:
    _, base_label, base_map = load_profile_results(base_dir)
    imp_kind, imp_label, imp_map = load_profile_results(imp_dir)
    second_title = "Oracle" if imp_kind == "oracle_mask" else "Improved"

    common = [r for r in manifest if r["sample_id"] in base_map and r["sample_id"] in imp_map]
    if not common:
        raise RuntimeError("No overlapping completed samples between the two profiles.")

    print(f"Baseline : {base_label} ({len(base_map)} completed)")
    print(f"{second_title:<8} : {imp_label} ({len(imp_map)} completed)")
    print(f"Comparing: {len(common)} common samples\n")

    header = (
        f"{'Sample ID':<27} | {'Object':<14} | {'Amodal IoU (B->I)':<19} | "
        f"{'Hidden IoU (B->I)':<19} | {'VisExact (B->I)':<15} | {'Critic (B->I)'}"
    )
    print(header)
    print("-" * len(header))

    rows_summary = []
    tile_w, tile_h, header_h = 210, 175, 28
    cols = [
        "Input",
        "Visible",
        "GT Amodal",
        "Base Mask",
        "Base RGB",
        f"{second_title} Mask",
        f"{second_title} RGB",
    ]
    grid = Image.new("RGB", (len(cols) * tile_w, len(common) * (tile_h + header_h)), "white")
    draw = ImageDraw.Draw(grid)

    for row_idx, rec in enumerate(common):
        sid = rec["sample_id"]
        cat = str(rec.get("name", "object"))[:14]
        b_job, b_res = base_map[sid]
        i_job, i_res = imp_map[sid]
        bm, im = b_res["metrics"], i_res["metrics"]

        b_aiou = bm["amodal_iou"] or 0.0
        i_aiou = im["amodal_iou"] or 0.0
        b_hiou = bm["hidden_iou"] if bm["hidden_iou"] is not None else 0.0
        i_hiou = im["hidden_iou"] if im["hidden_iou"] is not None else 0.0
        b_vex, i_vex = bm["visible_exact_fraction"], im["visible_exact_fraction"]
        b_sc = b_res.get("final_score")
        i_sc = i_res.get("final_score")

        print(
            f"{sid:<27} | {cat:<14} | "
            f"{b_aiou:.4f} -> {i_aiou:.4f} | "
            f"{b_hiou:.4f} -> {i_hiou:.4f} | "
            f"{b_vex:.3f} -> {i_vex:.3f}  | "
            f"{(f'{b_sc:.2f}' if b_sc is not None else 'N/A'):>5} -> {(f'{i_sc:.2f}' if i_sc is not None else 'N/A'):>5}"
        )

        rows_summary.append(
            {
                "sample_id": sid,
                "name": rec.get("name"),
                "baseline": bm,
                "improved": im,
                "delta_amodal_iou": round(i_aiou - b_aiou, 4),
                "delta_hidden_iou": round(i_hiou - b_hiou, 4),
                "baseline_score": b_sc,
                "improved_score": i_sc,
            }
        )

        y_top = row_idx * (tile_h + header_h)
        gt_path = sample_root / rec["gt_amodal"]
        tile_images = [
            _render_rgba_tile(i_job / "input.png"),
            _render_rgba_tile(i_job / "visible.png"),
            _render_rgba_tile(gt_path),
            _render_rgba_tile(b_job / "amodal.png"),
            _render_rgba_tile(b_job / "completion.png"),
            _render_rgba_tile(i_job / "amodal.png"),
            _render_rgba_tile(i_job / "completion.png"),
        ]
        for col_idx, (col_title, pic) in enumerate(zip(cols, tile_images)):
            x_left = col_idx * tile_w
            pic_copy = pic.copy()
            pic_copy.thumbnail((tile_w - 8, tile_h - 6))
            px = x_left + (tile_w - pic_copy.width) // 2
            py = y_top + header_h + (tile_h - pic_copy.height) // 2
            grid.paste(pic_copy, (px, py))
            caption = f"{col_title}" if col_idx > 0 else f"{sid[-11:]} ({cat})"
            draw.text((x_left + 4, y_top + 6), caption, fill="black")

    def _mean(key: str, side: str) -> float:
        vals = [r[side][key] for r in rows_summary if r[side][key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    b_mean_a, i_mean_a = _mean("amodal_iou", "baseline"), _mean("amodal_iou", "improved")
    b_mean_h, i_mean_h = _mean("hidden_iou", "baseline"), _mean("hidden_iou", "improved")
    b_mean_v, i_mean_v = _mean("visible_exact_fraction", "baseline"), _mean("visible_exact_fraction", "improved")
    b_mean_mae, i_mean_mae = _mean("visible_mae", "baseline"), _mean("visible_mae", "improved")

    print("-" * len(header))
    print(
        f"{'MEAN (' + str(len(common)) + ' samples)':<27} | {'ALL':<14} | "
        f"{b_mean_a:.4f} -> {i_mean_a:.4f} ({i_mean_a - b_mean_a:+.4f}) | "
        f"{b_mean_h:.4f} -> {i_mean_h:.4f} ({i_mean_h - b_mean_h:+.4f}) | "
        f"{b_mean_v:.3f} -> {i_mean_v:.3f}  | "
        f"MAE {b_mean_mae:.3f}->{i_mean_mae:.3f}"
    )

    output_grid.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_grid, quality=92)
    summary_path = output_grid.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "baseline_profile": base_dir.name,
                "improved_profile": imp_dir.name,
                "num_samples": len(common),
                "mean_baseline": {
                    "amodal_iou": round(b_mean_a, 4),
                    "hidden_iou": round(b_mean_h, 4),
                    "visible_mae": round(b_mean_mae, 4),
                    "visible_exact_fraction": round(b_mean_v, 4),
                },
                "mean_improved": {
                    "amodal_iou": round(i_mean_a, 4),
                    "hidden_iou": round(i_mean_h, 4),
                    "visible_mae": round(i_mean_mae, 4),
                    "visible_exact_fraction": round(i_mean_v, 4),
                },
                "samples": rows_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved visual comparison grid: {output_grid}")
    print(f"Saved JSON comparison report: {summary_path}")


def compare_profiles(
    manifest_path: Path,
    scamodal_root: Path,
    output_grid: Path,
    backup_dir: Optional[Path] = None,
    baseline_spec: Optional[str] = None,
    improved_spec: Optional[str] = None,
    oracle_spec: Optional[str] = None,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_root = manifest_path.resolve().parent

    base_dir = resolve_profile_dir(baseline_spec, scamodal_root)
    imp_dir = resolve_profile_dir(improved_spec, scamodal_root)
    ora_dir = resolve_profile_dir(oracle_spec, scamodal_root)

    if base_dir is None and imp_dir is None and ora_dir is None:
        by_mode = discover_profiles_by_mode(scamodal_root, backup_dir)
        if by_mode["baseline"] and by_mode["improved"] and by_mode["oracle_mask"]:
            run_three_way_comparison(
                by_mode["baseline"][-1],
                by_mode["improved"][-1],
                by_mode["oracle_mask"][-1],
                manifest,
                sample_root,
                output_grid,
            )
            return
        ordered_candidates = (
            (by_mode["baseline"][-1:] if by_mode["baseline"] else [])
            + (by_mode["improved"][-1:] if by_mode["improved"] else [])
            + (by_mode["oracle_mask"][-1:] if by_mode["oracle_mask"] else [])
        )
        if len(ordered_candidates) >= 2:
            base_dir, imp_dir = ordered_candidates[-2], ordered_candidates[-1]
        else:
            all_profs = [x for grp in by_mode.values() for x in grp]
            if len(all_profs) >= 2:
                base_dir, imp_dir = all_profs[-2], all_profs[-1]
            else:
                print(f"[Notice] Only 1 profile ({all_profs[0].name}) found; rendering single-profile report.")
                base_dir = imp_dir = all_profs[0]
    elif ora_dir is not None and base_dir is not None and imp_dir is not None:
        run_three_way_comparison(base_dir, imp_dir, ora_dir, manifest, sample_root, output_grid)
        return
    else:
        by_mode = discover_profiles_by_mode(scamodal_root, backup_dir)
        all_profs = [x for grp in by_mode.values() for x in grp]
        if ora_dir is not None and imp_dir is None:
            imp_dir = ora_dir
        if base_dir is None:
            base_dir = by_mode["baseline"][-1] if by_mode["baseline"] else all_profs[0]
        if imp_dir is None:
            imp_dir = (
                by_mode["oracle_mask"][-1]
                if by_mode["oracle_mask"]
                else (by_mode["improved"][-1] if by_mode["improved"] else all_profs[-1])
            )

    run_two_way_comparison(base_dir, imp_dir, manifest, sample_root, output_grid)
