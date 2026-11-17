# Phạm vi: Ô 5–7: chạy một phiên bản trên manifest chung, lưu kết quả và backup theo profile.
"""Run the amodal completion pipeline on prepared COCO-A targets with per-target artifacts/resume."""
import argparse, contextlib, hashlib, json, os, shutil, sys, time
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.runner import run_one
from src.evaluation.reporting import occlusion_bin, write_json, write_reports
from src.evaluation.validation import validate_manifest

# Sao nội dung terminal sang pipeline.log để đọc lại khi phiên bị ngắt.
class Tee:
    # Lưu các luồng đích (terminal và file log).
    def __init__(self, *streams): self.streams = streams
    # Gửi cùng nội dung đến mọi luồng đích.
    def write(self, text):
        for stream in self.streams: stream.write(text)
        return len(text)
    # Đẩy nội dung log đã đệm ra đĩa.
    def flush(self):
        for stream in self.streams: stream.flush()
    # Để chương trình con dùng dạng output phù hợp với file log.
    def isatty(self): return False
    # Lấy thuộc tính file còn thiếu từ luồng chính.
    def __getattr__(self, name): return getattr(self.streams[0], name)

# Nén artefact profile và chỉ thay file ZIP đích khi bản chép đã hoàn tất.
def backup(job_root, destination):
    if not destination: return
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = shutil.make_archive(str(ROOT / "outputs" / f"amodal_{job_root.name}_{stamp}"), "zip", job_root)
    try:
        target = destination / Path(archive).name
        partial = target.with_suffix(".zip.part")
        shutil.copy2(archive, partial)
        os.replace(partial, target)
        print("Backup:", target, flush=True)
    finally:
        Path(archive).unlink(missing_ok=True)

# Khóa từng vai trò vào đúng profile và danh sách mẫu trong các phiên Colab khác nhau.
def mark_profile_state(path, state, status):
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in state.items()):
            raise ValueError(f"Resume profile or manifest changed: {path}. "
                             "Use the original environment/config or set a new RUN_NAME in cell 1.")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(state, status=status))


# Preflight, tính profile, phục hồi artefact, chạy các mẫu còn thiếu và ghi báo cáo.
def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/dev50/manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v1_sd15_direct.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs/runs",
                        help="Write sample artifacts directly here; can be a mounted Drive folder")
    parser.add_argument("--profile-state", type=Path,
                        help="Track this batch across sessions; refuse an incompatible profile/manifest")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--backup-every", type=int, default=5)
    parser.add_argument("--allow-test", action="store_true", help="Explicit final-test run after configuration is frozen")
    parser.add_argument("--continue-on-error", action="store_true", help="Record sample errors and continue; CUDA OOM still stops")
    parser.add_argument("--validate-only", action="store_true", help="Validate prepared data without importing torch or loading models")
    parser.add_argument("--profile-output", type=Path, help="Write the exact successful profile ID for notebook follow-up cells")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from src.pipelines.backend import validate_experiment_config
    validate_experiment_config(config)
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit < 1: raise ValueError("--limit must be positive")
    records = records[:args.limit]
    if not records: raise ValueError("Empty manifest")
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate sample IDs")
    if any(r["split"] == "test" for r in records) and not args.allow_test:
        raise ValueError("Test split is protected; use --allow-test only after freezing configuration")
    os.chdir(ROOT)
    print(f"[Preflight] Validating {len(records)} images and visible/amodal/hidden masks...", flush=True)
    validation = validate_manifest(records, args.manifest.resolve().parent)
    preflight_path = args.manifest.parent / "preflight.json"
    write_json(preflight_path, validation)
    if validation["invalid"]:
        raise ValueError(f"{validation['invalid']} invalid samples; inspect {preflight_path} before loading models")
    print("[Preflight] All selected samples valid.", flush=True)
    if args.validate_only:
        return
    # Profile băm config, byte mã nguồn và phiên bản môi trường để sai khác tạo thư mục kết quả
    # riêng.
    files = list((ROOT / "src").rglob("*.py"))
    files += [ROOT / name for name in ["installed_gpu_versions.txt", "dependency_versions.json"] if (ROOT / name).is_file()]
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(files))).hexdigest()
    profile = hashlib.sha256(json.dumps(dict(config=config, code_hash=code_hash), sort_keys=True).encode()).hexdigest()[:16]
    state = dict(profile=profile, requested=len(records),
                 manifest_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest())
    if args.profile_state:
        mark_profile_state(args.profile_state, state, "running")
    output_root = args.runs_root / profile
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "preflight.json", validation)
    write_json(output_root / "manifest.json", records)
    if args.backup_dir:
        from src.evaluation.backups import auto_restore_drive_backups
        auto_restore_drive_backups(args.runs_root, args.backup_dir, profiles={profile})
    (output_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for name in ["installed_gpu_versions.txt", "dependency_versions.json"]:
        if (ROOT / name).is_file(): shutil.copy2(ROOT / name, output_root / name)
    results = []
    dirty_backup = False
    os.chdir(ROOT)
    with (output_root / "pipeline.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                import random, numpy as np, torch
                from src.pipelines.backend import AmodalBackend
                backend = None
                # Lazy model loading: already-completed records can resume without loading GPU models.
                class LazyBackend:
                    def begin_sample(self, job):
                        nonlocal backend
                        if backend is None:
                            t_load = time.perf_counter()
                            print("[Init] Preparing model manager (models load on first use)...", flush=True)
                            backend = AmodalBackend(config)
                            print(f"[Init] Model manager ready in {time.perf_counter() - t_load:.1f}s.", flush=True)
                        backend.begin_sample(job)
                    def complete(self, *values, **kwargs): return backend.complete(*values, **kwargs)
                lazy = LazyBackend()
                print(f"Profile: {profile} | Mode: {config['pipeline_mode']} | Targets: {len(records)}", flush=True)
                for idx, record in enumerate(records, start=1):
                    dirty_backup = True
                    failed = None
                    try:
                        random.seed(config["seed"]); np.random.seed(config["seed"])
                        torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])
                        result, skipped = run_one(record, args.manifest.resolve().parent, output_root, config, lazy, profile)
                        if config.get("pipeline_version") in (2, 3):
                            from src.evaluation.critic_logs import update_critic_log
                            update_critic_log(output_root, result)
                        results.append(dict(sample_id=record["sample_id"], status="completed", resumed=skipped, metrics=result["metrics"]))
                        print(f"[{idx}/{len(records)}] {record['sample_id']}", "resumed" if skipped else f"completed ({result['seconds']}s)", result["metrics"], flush=True)
                    except Exception as error:
                        failed = error
                        results.append(dict(sample_id=record["sample_id"], status="failed", error=str(error)))
                        print("FAILED:", record["sample_id"], str(error), flush=True)
                        if backend is not None:
                            backend.models.release_gpu()
                    results[-1].update(category=record.get("name"), hidden_ratio=record.get("hidden_ratio"),
                                       occlusion_bin=occlusion_bin(record.get("hidden_ratio")))
                    write_json(output_root / "summary.json", results)
                    write_reports(output_root / "generation_report", results, dict(profile=profile, selected=len(records)))
                    log.flush()
                    if results[-1]["status"] == "failed" or idx == len(records) or (args.backup_every > 0 and idx % args.backup_every == 0):
                        backup(output_root, args.backup_dir)
                        dirty_backup = False
                    if failed is not None and (not args.continue_on_error or "out of memory" in str(failed).lower()):
                        raise RuntimeError("Stopped after failed sample; inspect pipeline.log/failure.json before retrying")
            finally:
                log.flush()
                if dirty_backup:
                    backup(output_root, args.backup_dir)
    print("Results:", output_root, flush=True)
    if any(r["status"] == "failed" for r in results):
        raise RuntimeError("Batch finished with failed samples; inspect summary.json (failed samples are never scored as zero)")
    if args.profile_state:
        mark_profile_state(args.profile_state, state, "completed")
    if args.profile_output:
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.profile_output, dict(profile=profile, pipeline_mode=config.get("pipeline_mode"),
                                            manifest=str(args.manifest), completed=len(results)))

if __name__ == "__main__": main()
