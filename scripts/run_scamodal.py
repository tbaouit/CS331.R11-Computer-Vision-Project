"""Run full SCAmodal on prepared COCO-A targets with per-target artifacts/resume."""
import argparse, contextlib, hashlib, json, os, shutil, sys, time
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import run_one

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams: stream.write(text)
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()
    def isatty(self): return False
    def __getattr__(self, name): return getattr(self.streams[0], name)

def backup(job_root, destination):
    if not destination: return
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = shutil.make_archive(str(ROOT / "outputs" / f"scamodal_{job_root.name}_{stamp}"), "zip", job_root)
    try:
        target = destination / Path(archive).name
        partial = target.with_suffix(".zip.part")
        shutil.copy2(archive, partial)
        os.replace(partial, target)
        print("Backup:", target, flush=True)
    finally:
        Path(archive).unlink(missing_ok=True)

def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/train_smoke/manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scamodal.json")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--allow-test", action="store_true", help="Explicit final-test run after configuration is frozen")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("mask_only") or not config.get("enable_critic"):
        raise ValueError("This entry point runs full SCAmodal with critic, not mask-only/baseline")
    if not 1 <= config["max_critic_iter"] <= 10 or not 0 <= config["critique_threshold"] <= 10:
        raise ValueError("Invalid critic settings")
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit < 1: raise ValueError("--limit must be positive")
    records = records[:args.limit]
    if not records: raise ValueError("Empty manifest")
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate sample IDs")
    if any(r["split"] == "test" for r in records) and not args.allow_test:
        raise ValueError("Test split is protected; use --allow-test only after freezing configuration")
    source = json.loads((ROOT / "upstream/provenance.json").read_text(encoding="utf-8"))
    files = list((ROOT / "runtime/SCAmodal").glob("*.py")) + list((ROOT / "src").glob("*.py"))
    files += [ROOT / name for name in ["installed_gpu_versions.txt", "dependency_versions.json"] if (ROOT / name).is_file()]
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(files))).hexdigest()
    profile = hashlib.sha256(json.dumps(dict(config=config, commit=source["commit"], code_hash=code_hash), sort_keys=True).encode()).hexdigest()[:16]
    output_root = ROOT / "outputs/scamodal" / profile
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "provenance.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for name in ["installed_gpu_versions.txt", "dependency_versions.json"]:
        if (ROOT / name).is_file(): shutil.copy2(ROOT / name, output_root / name)
    results = []
    needs_final_backup = False
    os.chdir(ROOT)
    with (output_root / "pipeline.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                import random, numpy as np, torch
                from src.scamodal_backend import SCAmodalBackend
                backend = None
                # Lazy model loading: already-completed records can resume without loading GPU models.
                class LazyBackend:
                    def begin_sample(self, job):
                        nonlocal backend
                        if backend is None: backend = SCAmodalBackend(config)
                        backend.begin_sample(job)
                    def complete(self, *values): return backend.complete(*values)
                lazy = LazyBackend()
                print(f"Profile: {profile} | Mode: {config.get('pipeline_mode', 'upstream_baseline')} | Targets: {len(records)}", flush=True)
                for record in records:
                    needs_final_backup = True
                    try:
                        random.seed(config["seed"]); np.random.seed(config["seed"])
                        torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])
                        result, skipped = run_one(record, args.manifest.resolve().parent, output_root, config, lazy, profile)
                        results.append(dict(sample_id=record["sample_id"], status="completed", resumed=skipped, metrics=result["metrics"]))
                        print(record["sample_id"], "resumed" if skipped else "completed", result["metrics"], flush=True)
                    except Exception as error:
                        results.append(dict(sample_id=record["sample_id"], status="failed", error=str(error)))
                        print("FAILED:", record["sample_id"], str(error), flush=True)
                    (output_root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                    log.flush()
                    backup(output_root, args.backup_dir)
                    needs_final_backup = False
                    if results[-1]["status"] == "failed":
                        raise RuntimeError("Stopped after failed sample; inspect pipeline.log/failure.json before retrying")
            finally:
                log.flush()
                if needs_final_backup:
                    backup(output_root, args.backup_dir)
    print("Results:", output_root, flush=True)

if __name__ == "__main__": main()
