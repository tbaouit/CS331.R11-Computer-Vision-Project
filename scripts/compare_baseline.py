"""Paired SD2 Direct vs mask-given pipeline comparison on the exact same manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.comparison import auto_restore_drive_backups
from src.evaluation.reporting import aggregate, write_json
from src.evaluation.runner import read_mask, read_rgb, sha

METRICS = {
    "CLIP_score": "higher (class relevance)",
    "color_hist_sim": "higher (diagnostic only)",
    "seam_mae": "lower (diagnostic only)",
    "LPIPS": "lower (visible preservation)",
    "Feature_Similarity": "higher (visible preservation)",
    "SSIM": "higher (visible preservation)",
    "PSNR": "higher (visible preservation)",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def subset_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:12]


def load_evaluation(profile, records):
    candidates = list(profile.glob(f"evaluation/*/{subset_hash(records)}/per_sample.json"))
    if not candidates:
        raise ValueError(f"No evaluation for this exact manifest in {profile.name}; run evaluate_runs.py first")
    report = max(candidates, key=lambda p: p.stat().st_mtime)
    rows = read_json(report)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != {r["sample_id"] for r in records}:
        raise ValueError(f"Incomplete/duplicate evaluation rows: {report}")
    return report.parent, {r["sample_id"]: r for r in rows}, read_json(report.parent / "protocol.json")


def verified_jobs(profile, records, directory, rows):
    jobs = {}
    for path in sorted(profile.glob("*/*/result.json"), key=lambda p: p.stat().st_mtime):
        result = read_json(path)
        if result.get("status") == "completed":
            jobs[result["sample_id"]] = (path.parent, result)
    selected = {}
    evaluation_id = directory.parent.name
    for record in records:
        sid = record["sample_id"]
        if sid not in jobs:
            raise ValueError(f"Missing generation: {profile.name}/{sid}")
        job, result = jobs[sid]
        if read_json(job / "sample.json") != record:
            raise ValueError(f"Manifest changed since generation: {sid}")
        names = ("input.png", "visible.png", "completion.png", "amodal.png", "occluded_input.png")
        hashes = {name: sha(job / name) for name in names}
        if any(result.get("artifact_sha256", {}).get(name) != value for name, value in hashes.items()):
            raise ValueError(f"Artifact checksum mismatch: {profile.name}/{sid}")
        signature = hashlib.sha256(json.dumps(dict(evaluation_id=evaluation_id,
            generation=result["signature"], artifacts={k: v for k, v in hashes.items() if k != "occluded_input.png"},
            record=record), sort_keys=True).encode()).hexdigest()
        sidecar = read_json(job / f"appearance_{evaluation_id}.json")
        if sidecar.get("signature") != signature or sidecar.get("row") != rows[sid]:
            raise ValueError(f"Stale evaluation for {sid}; rerun evaluate_runs.py")
        selected[sid] = (job, result, hashes)
    return selected


def make_previews(output, records, baseline, main, count=20, seed=42):
    """Fixed random subset; identical object crop across the five columns."""
    chosen = random.Random(seed).sample(records, min(count, len(records)))
    files = []
    for page_start in range(0, len(chosen), 4):
        page = chosen[page_start:page_start + 4]
        sheet = Image.new("RGB", (1200, len(page) * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for row, record in enumerate(page):
            sid = record["sample_id"]
            bjob, mjob = baseline[sid][0], main[sid][0]
            mask = read_mask(bjob / "amodal.png")
            ys, xs = np.where(mask)
            side = max(32, int(round(max(np.ptp(xs) + 1, np.ptp(ys) + 1) * 1.2)))
            x0, y0 = int((xs.min() + xs.max()) // 2) - side // 2, int((ys.min() + ys.max()) // 2) - side // 2
            box = (x0, y0, x0 + side, y0 + side)
            title = f"{sid} | {record.get('name', '')} | hidden={record.get('hidden_ratio', 0):.0%}"
            draw.text((5, row * 280 + 5), title, fill="black")
            tiles = [(bjob / "input.png", "Input crop"), (bjob / "visible.png", "Visible mask"),
                     (bjob / "amodal.png", "Given amodal mask"),
                     (bjob / "completion.png", "SD2 Direct"), (mjob / "completion.png", "Main pipeline")]
            for col, (path, label) in enumerate(tiles):
                with Image.open(path) as src:
                    cropped = src.crop(box)
                    if src.mode == "RGBA":
                        bg = Image.new("RGBA", cropped.size, (245, 245, 245, 255))
                        bg.alpha_composite(cropped)
                        cropped = bg.convert("RGB")
                    else:
                        cropped = cropped.convert("RGB")
                    resample = Image.Resampling.NEAREST if col in (1, 2) else Image.Resampling.LANCZOS
                    tile = cropped.resize((230, 230), resample)
                draw.text((col * 240 + 5, row * 280 + 25), label, fill="black")
                sheet.paste(tile, (col * 240 + 5, row * 280 + 45))
        filename = f"preview_{page_start // 4 + 1:02d}.jpg"
        sheet.save(output / filename, quality=92)
        files.append(filename)
    return [r["sample_id"] for r in chosen], files


def compare_profiles(records, baseline_profile, main_profile, output, preview_count=20):
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Empty manifest or duplicate IDs")
    if baseline_profile.resolve() == main_profile.resolve():
        raise ValueError("Choose two distinct generation profiles")
    if preview_count < 0:
        raise ValueError("preview_count must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    bdir, brows, bp = load_evaluation(baseline_profile, records)
    mdir, mrows, mp = load_evaluation(main_profile, records)
    if bp != mp:
        raise ValueError("Evaluation protocols/versions differ; evaluate BOTH profiles in the same runtime first")
    coverage = {"requested": len(records), "baseline": aggregate(list(brows.values())),
                "main": aggregate(list(mrows.values()))}
    write_json(output / "coverage.json", coverage)
    if any(row.get("status") != "completed" for row in list(brows.values()) + list(mrows.values())):
        raise ValueError("Incomplete evaluation; inspect coverage.json and finish all samples before paired comparison")
    bj = verified_jobs(baseline_profile, records, bdir, brows)
    mj = verified_jobs(main_profile, records, mdir, mrows)
    paired = []
    for record in records:
        sid = record["sample_id"]
        bc, mc = bj[sid][1]["config"], mj[sid][1]["config"]
        if bc.get("pipeline_mode") != "sd2_direct" or mc.get("pipeline_mode") not in ("oracle_mask", "gt_mask"):
            raise ValueError("Expected sd2_direct baseline and oracle_mask main profiles")
        if bc.get("enable_critic") or bc.get("text_query_policy", "empty") != "empty":
            raise ValueError("Baseline must have no critic and no annotation-label prompt")
        if any(bc.get(k) != mc.get(k) for k in ("seed", "inpainting_model_id")) or mc.get("text_query_policy", "empty") != "empty":
            raise ValueError(f"Seed, SD2 checkpoint or label policy differs: {sid}")
        for name in ("input.png", "visible.png", "occluded_input.png", "amodal.png"):
            if bj[sid][2][name] != mj[sid][2][name]:
                # Different Pillow/PNG encoders may produce different bytes for
                # identical pixels. Individual artifact checksums were verified above.
                reader = read_rgb if name == "input.png" else read_mask
                if not np.array_equal(reader(bj[sid][0] / name), reader(mj[sid][0] / name)):
                    raise ValueError(f"Input RGB or masks differ between methods: {sid}/{name}")
        row = dict(sample_id=sid, category=record.get("name"), hidden_ratio=record.get("hidden_ratio"),
                   occlusion_bin=brows[sid].get("occlusion_bin"))
        for name in METRICS:
            b, m = brows[sid].get("metrics", {}).get(name), mrows[sid].get("metrics", {}).get(name)
            if name not in ("seam_mae", "color_hist_sim") and (b is None or m is None):
                raise ValueError(f"Required metric missing: {sid}/{name}")
            # Missing seam metrics are valid when no visible/hidden adjacency exists.
            if (b is None) != (m is None):
                raise ValueError(f"Metric availability differs: {sid}/{name}")
            if b is not None and (not math.isfinite(b) or not math.isfinite(m)):
                raise ValueError(f"Non-finite metric: {sid}/{name}")
            row[f"{name}_baseline"], row[f"{name}_main"] = b, m
            row[f"{name}_delta"] = m - b if b is not None else None
        row["baseline_seconds"], row["main_seconds"] = bj[sid][1].get("seconds"), mj[sid][1].get("seconds")
        paired.append(row)
    with (output / "paired_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    def summarize(selected):
        return aggregate([dict(status="completed", metrics={k: v for k, v in row.items()
                         if k.endswith(("_baseline", "_main", "_delta"))}) for row in selected])["metrics"]

    chosen, previews = make_previews(output, records, bj, mj, preview_count)
    summary = dict(baseline=baseline_profile.name, main=main_profile.name, samples=len(records),
                   subset_id=subset_hash(records), protocol=bp, delta_definition="main minus baseline",
                   metric_directions=METRICS, metrics=summarize(paired),
                   groups={key: {str(v): summarize([r for r in paired if r.get(key) == v])
                                  for v in sorted({r.get(key) for r in paired}, key=str)}
                           for key in ("category", "occlusion_bin")},
                   preview_sample_ids=chosen, preview_seed=42, previews=previews,
                   notes=["COCO-A has no hidden RGB ground truth; these scores do not measure hidden RGB accuracy.",
                          "Visible metrics are near-perfect by construction for both methods.",
                          "seam_mae/color_hist_sim are diagnostics; uniform or blurred output may score well.",
                          "Per-sample seconds include model activation/loading and I/O; runs may use different hardware.",
                          "Full-system comparison with different generation budgets, not a per-component ablation."])
    write_json(output / "comparison.json", summary)
    print(f"Paired comparison: {len(records)} objects | {baseline_profile.name} -> {main_profile.name}")
    print(f"{'Metric':22} {'SD2 Direct':>12} {'Main':>12} {'Main - Direct':>15} {'n':>5}")
    for name in METRICS:
        values = [summary["metrics"][f"{name}_{k}"] for k in ("baseline", "main", "delta")]
        formatted = [f"{v['mean']:.6g}" if v["mean"] is not None else "N/A" for v in values]
        print(f"{name:22} {formatted[0]:>12} {formatted[1]:>12} {formatted[2]:>15} {values[2]['n']:>5}")
    print("CSV, JSON and fixed-subset preview pages:", output)
    return summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--main-profile", required=True)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    if args.backup_dir:
        auto_restore_drive_backups(args.runs_root, args.backup_dir, profiles={args.baseline, args.main_profile})
    compare_profiles(read_json(args.manifest), args.runs_root / args.baseline, args.runs_root / args.main_profile,
                     args.output_dir, args.preview_count)


if __name__ == "__main__":
    main()
