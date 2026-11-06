"""Evaluate saved completions without rerunning generation (resumable sidecars)."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.appearance_metrics import POLICY, PROTOCOL, compute_visible_appearance_metrics
from src.evaluation.comparison import auto_restore_drive_backups
from src.evaluation.metrics import compute_appearance_coherence_metrics
from src.evaluation.reporting import occlusion_bin, write_json, write_reports
from src.evaluation.runner import read_mask, read_rgb, sha
from src.vision.image_ops import binary_dilate


def evaluate_profile(profile, records, device="cuda", continue_on_error=False, evaluate=None,
                     checkpoint=None, backup_every=5):
    """Read verified saved artifacts only; generation artifacts and result.json stay unchanged."""
    evaluate = evaluate or compute_visible_appearance_metrics
    source_hash = hashlib.sha256(b"".join((ROOT / path).read_bytes() for path in (
        "src/evaluation/appearance_metrics.py", "src/evaluation/metrics.py",
        "src/evaluation/reporting.py", "scripts/evaluate_runs.py"))).hexdigest()
    versions = {}
    from importlib.metadata import PackageNotFoundError, version
    for package in ("torch", "torchvision", "transformers", "lpips", "scikit-image", "Pillow", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    settings = dict(policy=POLICY, code_sha256=source_hash, device=device, versions=versions)
    evaluation_id = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    subset_id = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:12]
    destination = profile / "evaluation" / evaluation_id / subset_id
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "protocol.json", settings)
    jobs = {}
    for path in sorted(profile.glob("*/*/result.json"), key=lambda p: p.stat().st_mtime):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") == "completed":
            jobs[result["sample_id"]] = (path.parent, result)
    rows = []
    for index, record in enumerate(records, 1):
        sample_id = record["sample_id"]
        ratio = record.get("hidden_ratio")
        row = dict(sample_id=sample_id, category=record.get("name", ""), hidden_ratio=ratio,
                   occlusion_bin=occlusion_bin(ratio))
        fatal = None
        cached = False
        try:
            if sample_id not in jobs:
                row.update(status="missing", error="No completed generation for this sample in selected profile")
            else:
                job, result = jobs[sample_id]
                source_record = json.loads((job / "sample.json").read_text(encoding="utf-8"))
                if source_record != record:
                    raise ValueError("Manifest record differs from the record used for generation")
                filenames = ("input.png", "visible.png", "completion.png", "amodal.png")
                hashes = {name: sha(job / name) for name in filenames}
                expected = result.get("artifact_sha256", {})
                if any(expected.get(name) != value for name, value in hashes.items()):
                    raise ValueError("Saved generation artifacts fail checksum verification")
                signature = hashlib.sha256(json.dumps(dict(evaluation_id=evaluation_id,
                    generation=result["signature"], artifacts=hashes, record=record), sort_keys=True).encode()).hexdigest()
                sidecar = job / f"appearance_{evaluation_id}.json"
                existing = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
                if existing.get("signature") == signature and existing.get("status") == "completed":
                    row = existing["row"]
                    cached = True
                else:
                    image, output = read_rgb(job / "input.png"), read_rgb(job / "completion.png")
                    visible = read_mask(job / "visible.png")
                    amodal = read_mask(job / "amodal.png")
                    metrics = evaluate(image, output, visible, record.get("name", ""), device=device)
                    if any(v is None or not math.isfinite(v) for v in metrics.values()):
                        raise ValueError("Appearance evaluator returned missing or non-finite scores")
                    # Reference-free diagnostic on the actual reconstructed region, not hidden RGB accuracy.
                    hidden = amodal & ~visible
                    coherence = compute_appearance_coherence_metrics(image, visible, hidden, output)
                    if not hidden.any():
                        coherence = dict(seam_mae=None, color_hist_sim=None)
                    elif not (hidden & binary_dilate(visible, 2)).any():
                        coherence["seam_mae"] = None
                    metrics.update(coherence)
                    row.update(status="completed", metrics=metrics,
                               visible_canvas_fraction=float(visible.mean()),
                               visible_exact_fraction=result.get("metrics", {}).get("visible_exact_fraction"))
                    write_json(sidecar, dict(signature=signature, status="completed", row=row, settings=settings))
        except Exception as error:
            row.pop("metrics", None)
            row.update(status="failed", error=str(error))
            if not continue_on_error or "out of memory" in str(error).lower():
                fatal = error
        rows.append(row)
        report = write_reports(destination, rows, dict(**settings, generation_profile=profile.name,
                                                      requested_manifest_samples=len(records), subset_id=subset_id))
        print(f"[Metrics {index}/{len(records)}] {sample_id}: {row['status']}" +
              (" (resumed)" if cached else "") + (f" — {row['error']}" if row.get("error") else ""), flush=True)
        if checkpoint and not cached and backup_every > 0 and index % backup_every == 0:
            checkpoint()
        if fatal:
            raise RuntimeError(f"Evaluation stopped; partial results saved in {destination}") from fatal
    print("Evaluation:", destination, flush=True)
    print(json.dumps(report["overall"], indent=2), flush=True)
    return destination, rows


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/train_smoke/manifest.json")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--profile", required=True, help="Explicit profile ID or path; never automatically pick another experiment")
    parser.add_argument("--limit", type=int, help="Default: all records in manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backup-dir", type=Path, help="Restore generation backups and back up evaluation results")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--backup-every", type=int, default=5)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    profile = Path(args.profile) if Path(args.profile).is_absolute() else args.runs_root / args.profile
    if args.backup_dir:
        auto_restore_drive_backups(args.runs_root, args.backup_dir, profiles={profile.name})
    if not profile.is_dir():
        raise FileNotFoundError(f"Unknown profile: {profile}")
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit is not None:
        records = records[:args.limit]
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Empty manifest or duplicate sample IDs")
    try:
        from run_experiment import backup
        checkpoint = (lambda: backup(profile, args.backup_dir)) if args.backup_dir else None
        _, rows = evaluate_profile(profile, records, args.device, args.continue_on_error,
                                   checkpoint=checkpoint, backup_every=args.backup_every)
    finally:
        if args.backup_dir:
            backup(profile, args.backup_dir)
    if any(r["status"] != "completed" for r in rows):
        raise SystemExit("Evaluation incomplete; inspect metrics.csv/aggregate.json for failures and missing samples")


if __name__ == "__main__":
    main()
