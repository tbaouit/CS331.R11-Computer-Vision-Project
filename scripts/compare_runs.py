"""Compare Baseline vs Improved SCAmodal runs quantitatively and visually."""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BACKUP_DIRS = [
    Path("/content/drive/MyDrive/CS331/results"),
    Path("/content/drive/MyDrive/amodal_project/results"),
]


def _auto_restore_drive_backups(scamodal_root: Path, extra_backup_dir: Path | None = None) -> list[Path]:
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


def _load_profile_results(profile_dir: Path) -> tuple[str, dict[str, tuple[Path, dict]]]:
    results: dict[str, tuple[Path, dict]] = {}
    mode_label = profile_dir.name
    for res_path in sorted(profile_dir.glob("*/*/result.json"), key=lambda p: p.stat().st_mtime):
        data = json.loads(res_path.read_text(encoding="utf-8"))
        if data.get("status") != "completed":
            continue
        sid = data["sample_id"]
        results[sid] = (res_path.parent, data)
        backend_name = data.get("trace", {}).get("backend", "")
        cfg_mode = data.get("config", {}).get("pipeline_mode", "")
        if cfg_mode == "improved" or "Improved" in backend_name:
            mode_label = f"improved ({profile_dir.name})"
        else:
            mode_label = f"baseline ({profile_dir.name})"
    return mode_label, results


def _resolve_dir(spec: str | None, scamodal_root: Path) -> Path | None:
    if not spec:
        return None
    cand = Path(spec)
    if cand.is_dir():
        return cand
    cand2 = scamodal_root / spec
    if cand2.is_dir():
        return cand2
    raise FileNotFoundError(f"Run directory not found: {spec}")


def _auto_select_profiles(scamodal_root: Path, backup_dir: Path | None = None) -> tuple[Path, Path]:
    scamodal_root.mkdir(parents=True, exist_ok=True)
    profiles = [p for p in scamodal_root.iterdir() if p.is_dir() and list(p.glob("*/*/result.json"))]
    if len(profiles) < 2:
        _auto_restore_drive_backups(scamodal_root, backup_dir)
        profiles = [p for p in scamodal_root.iterdir() if p.is_dir() and list(p.glob("*/*/result.json"))]

    if not profiles:
        raise RuntimeError(f"No completed profile directories found under {scamodal_root}.")

    if len(profiles) == 1:
        print(
            f"[Notice] Only 1 profile ({profiles[0].name}) found under {scamodal_root} "
            "(no second profile found in local outputs or Google Drive backups); rendering single-profile report."
        )
        return profiles[0], profiles[0]

    baselines, improveds = [], []
    for p in sorted(profiles, key=lambda d: (len(list(d.glob("*/*/result.json"))), max(f.stat().st_mtime for f in d.glob("*/*/result.json")))):
        label, res = _load_profile_results(p)
        if not res:
            continue
        if label.startswith("improved"):
            improveds.append(p)
        else:
            baselines.append(p)

    if baselines and improveds:
        return baselines[-1], improveds[-1]
    ordered = sorted(
        profiles,
        key=lambda d: (len(list(d.glob("*/*/result.json"))), max(f.stat().st_mtime for f in d.glob("*/*/result.json"))),
    )
    return ordered[-2], ordered[-1]


def _render_rgba_tile(path: Path, bg_rgb=(245, 245, 245)) -> Image.Image:
    with Image.open(path) as im:
        if im.mode == "RGBA":
            bg = Image.new("RGBA", im.size, (*bg_rgb, 255))
            bg.alpha_composite(im)
            return bg.convert("RGB")
        return im.convert("RGB")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/train_smoke/manifest.json")
    parser.add_argument("--scamodal-root", type=Path, default=ROOT / "outputs/scamodal")
    parser.add_argument("--backup-dir", type=Path, default=None, help="Optional Google Drive backup directory")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline profile hash or path")
    parser.add_argument("--improved", type=str, default=None, help="Improved profile hash or path")
    parser.add_argument("--output-grid", type=Path, default=ROOT / "outputs/scamodal/comparison_grid.jpg")
    args = parser.parse_args()

    base_dir = _resolve_dir(args.baseline, args.scamodal_root)
    imp_dir = _resolve_dir(args.improved, args.scamodal_root)
    if base_dir is None or imp_dir is None:
        auto_base, auto_imp = _auto_select_profiles(args.scamodal_root, args.backup_dir)
        base_dir = base_dir or auto_base
        imp_dir = imp_dir or auto_imp

    base_label, base_map = _load_profile_results(base_dir)
    imp_label, imp_map = _load_profile_results(imp_dir)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample_root = args.manifest.resolve().parent
    common = [r for r in manifest if r["sample_id"] in base_map and r["sample_id"] in imp_map]
    if not common:
        raise RuntimeError("No overlapping completed samples between the two profiles.")

    print(f"Baseline : {base_label} ({len(base_map)} completed)")
    print(f"Improved : {imp_label} ({len(imp_map)} completed)")
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
        "Improved Mask",
        "Improved RGB",
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

    args.output_grid.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output_grid, quality=92)
    summary_path = args.output_grid.with_suffix(".json")
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
    print(f"\nSaved visual comparison grid: {args.output_grid}")
    print(f"Saved JSON comparison report: {summary_path}")


if __name__ == "__main__":
    main()
