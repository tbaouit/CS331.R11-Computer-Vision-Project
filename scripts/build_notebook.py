# Phạm vi: Sinh notebook Colab từ mẫu 7 ô. Thay đổi hướng dẫn hoặc code notebook ở đây rồi sinh
# lại notebook.
"""Generate the runnable notebook with seven cells for generation, resume and evaluation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
'''# 1. Môi trường và Google Drive
import subprocess, sys, os, shutil, zipfile, json, base64, re
from pathlib import Path
# Dùng cùng tên qua các phiên để tìm lại kết quả trên Drive.
RUN_NAME = "sd15_decoded"
EVALUATION_ONLY = False  # True nếu phiên này chỉ để đánh giá ảnh đã sinh.
os.environ["MPLBACKEND"] = "Agg"
def sh(*cmd, extra_env=None):
    env = {
        **os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1",
    }
    env.update(extra_env or {})
    with subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env) as p:
        for line in p.stdout:
            # Older Pix2Gestalt/CarveKit tqdm bars may ignore TQDM_DISABLE.
            # Filter only progress-bar lines; keep status messages and errors.
            if re.search(r"[0-9]{1,3}%[|].*[|]", line):
                continue
            print(line, end="", flush=True)
        if p.wait() != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd)
sh("nvidia-smi")
from google.colab import drive
drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/CS331") if Path("/content/drive/MyDrive/CS331").is_dir() else Path("/content/drive/MyDrive/amodal_project")
(DRIVE / "results").mkdir(parents=True, exist_ok=True)
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", RUN_NAME):
    raise ValueError("RUN_NAME dùng chữ, số, dấu gạch ngang hoặc gạch dưới")
EXPERIMENT = DRIVE / "results" / RUN_NAME
EXPERIMENT.mkdir(parents=True, exist_ok=True)
print("Lưu kết quả trên Drive:", EXPERIMENT)

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
for secret_name in ("HF_TOKEN", "GITHUB_TOKEN"):
    try:
        from google.colab import userdata
        val = userdata.get(secret_name)
        if val: os.environ[secret_name] = val
    except Exception:
        pass
''',
'''# 2. Đồng bộ mã và dữ liệu
REPO_URL = "https://github.com/tbaouit/CS331.R11-Computer-Vision-Project.git"
PROJECT = Path("/content/cs331")
PYTHON = str(PROJECT / ".venv-gpu/bin/python")

git_env = {}
if os.environ.get("GITHUB_TOKEN"):
    auth = base64.b64encode(("x-access-token:" + os.environ["GITHUB_TOKEN"]).encode()).decode()
    git_env = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
               "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic " + auth}
repo_lock = EXPERIMENT / "repo_commit.json"
if EVALUATION_ONLY and not repo_lock.is_file():
    raise FileNotFoundError("Chưa có đợt chạy trên Drive. Kiểm tra RUN_NAME ở ô 1.")
if not (PROJECT / ".git").is_dir():
    if PROJECT.exists() and any(PROJECT.iterdir()):
        raise RuntimeError("Move the existing non-Git /content/cs331 directory aside before cloning")
    sh("git", "clone", "--branch", "main", REPO_URL, PROJECT, extra_env=git_env)

# Giữ cố định mã nguồn giữa các lần resume. RUN_NAME mới lấy mã mới từ main.
if repo_lock.is_file():
    commit = json.loads(repo_lock.read_text(encoding="utf-8"))["commit"]
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Mã commit trên Drive không hợp lệ")
else:
    sh("git", "-C", PROJECT, "fetch", "origin", "main", extra_env=git_env)
    commit = subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "FETCH_HEAD"], text=True).strip()
present = subprocess.run(["git", "-C", str(PROJECT), "cat-file", "-e", commit + "^{commit}"],
                         capture_output=True).returncode == 0
if not present:
    sh("git", "-C", PROJECT, "fetch", "origin", commit, extra_env=git_env)
sh("git", "-C", PROJECT, "checkout", "--detach", commit)
if not repo_lock.is_file():
    save_json(repo_lock, {"commit": commit})
os.chdir(PROJECT)
sh("git", "log", "-n", "1", "--oneline")

if EVALUATION_ONLY:
    print("Chỉ đánh giá: không cần giải nén COCO-A")
elif not (PROJECT / "dataset/cocoa/dataset.json").is_file():
    candidates = [
        DRIVE / "cocoa_dataset.zip",
        DRIVE / "data/cocoa_dataset.zip",
        Path("/content/drive/MyDrive/CS331/cocoa_dataset.zip"),
        Path("/content/drive/MyDrive/amodal_project/cocoa_dataset.zip"),
    ]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"Upload cocoa_dataset.zip vào {DRIVE}")
    local_zip = Path("/content/cocoa_dataset.zip")
    print(f"Copying {source} -> {local_zip} ...", flush=True)
    shutil.copy2(source, local_zip)
    destination = PROJECT / "dataset"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip) as bundle:
        for item in bundle.infolist():
            if not (destination / item.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
        bundle.extractall(destination)
    local_zip.unlink()
    print("Dataset extracted to", destination / "cocoa", flush=True)
else:
    print("Dataset already extracted at", PROJECT / "dataset/cocoa", "(skipped)", flush=True)

print("Free GiB:", round(shutil.disk_usage(PROJECT).free / 1024**3, 1))
''',
'''# 3. Cài thư viện, tải trọng số và kiểm thử
if EVALUATION_ONLY:
    sh(sys.executable, "scripts/setup_colab.py", "--evaluation-only")
else:
    PRELOAD_GENERATION_WEIGHTS = True  # Có thể đặt False nếu phiên này chỉ chạy Baseline.
    setup_flags = [] if PRELOAD_GENERATION_WEIGHTS else ["--skip-downloads"]
    sh(sys.executable, "scripts/setup_colab.py", *setup_flags)
    sh(PYTHON, "-m", "unittest", "discover", "-s", "tests", "-v")
''',
'''# 4. Chuẩn bị cùng tập mẫu cho mọi phiên bản
BATCH = "dev50"  # Switch to "test400" only after selecting the final settings.
FINAL_TEST = False
SMOKE_LIMIT = None  # Optional: use 3 for a first check on dev50.
sizes = {"dev50": 50, "test400": 400}
if BATCH not in sizes:
    raise ValueError("BATCH must be dev50 or test400")
if BATCH == "test400" and (not FINAL_TEST or SMOKE_LIMIT is not None):
    raise ValueError("test400 requires FINAL_TEST=True and no smoke limit")
N = sizes[BATCH] if SMOKE_LIMIT is None else SMOKE_LIMIT
if type(N) is not int or not 1 <= N <= sizes[BATCH]:
    raise ValueError("Invalid smoke limit")
extra = ["--allow-test"] if BATCH == "test400" else []
DATA_RUN = EXPERIMENT / f"{BATCH}_{N}"
RUNS_ROOT = DATA_RUN / "runs"
saved_manifest = DATA_RUN / "manifest.json"
configs = ("baseline", "proposed")

if EVALUATION_ONLY:
    manifest = saved_manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest trên Drive: {manifest}")
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if len(records) != N or any(r["split"] != BATCH.rstrip("0123456789") for r in records):
        raise ValueError("Manifest trên Drive không khớp batch đã chọn")
else:
    manifest = PROJECT / "outputs" / BATCH / "manifest.json"
    if not manifest.is_file():
        sh(PYTHON, "scripts/prepare_data.py", "--config", f"configs/data_{BATCH}.json", *extra)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if len(records) != sizes[BATCH] or any(r["split"] != BATCH.rstrip("0123456789") for r in records):
        raise ValueError("Manifest không khớp split hoặc kích thước đã chọn")
    records = records[:N]
    if N < sizes[BATCH]:
        manifest = manifest.with_name(f"first_{N}.json")
        save_json(manifest, records)
    sh(PYTHON, "scripts/run_experiment.py", "--manifest", manifest, "--limit", N,
       "--config", "configs/baseline.json", "--validate-only", *extra)
    if saved_manifest.is_file():
        if records != json.loads(saved_manifest.read_text(encoding="utf-8")):
            raise ValueError("Danh sách mẫu đã đổi. Dùng lại dataset gốc hoặc đặt RUN_NAME mới ở ô 1.")
    else:
        save_json(saved_manifest, records)

def marker(role):
    return DATA_RUN / "profiles" / f"{role}.json"

def run_version(role):
    if EVALUATION_ONLY:
        raise RuntimeError("Phiên này chỉ đánh giá; bỏ qua ô 5–6.")
    config = f"configs/{role}.json"
    sh(PYTHON, "scripts/run_experiment.py", "--manifest", manifest, "--limit", N,
       "--config", config, "--runs-root", RUNS_ROOT / role,
       "--profile-state", marker(role), *extra)

print("Batch trên Drive:", DATA_RUN)
for role in configs:
    state = json.loads(marker(role).read_text(encoding="utf-8")) if marker(role).is_file() else {}
    print(role, state.get("status", "chưa chạy"), state.get("profile", ""))
''',
'''# 5. Chạy Baseline: SD1.5 Direct
run_version("baseline")
''',
'''# 6. Chạy proposed
run_version("proposed")
''',
'''# 7. Đánh giá, so sánh và lưu báo cáo
from PIL import Image
from IPython.display import display
FID_REFERENCE = None  # For test400, set to a Drive folder of >=400 clean object crops.
import hashlib
roles = configs
profiles, pending = {}, []
saved_records = json.loads(saved_manifest.read_text(encoding="utf-8"))
manifest_sha = hashlib.sha256(json.dumps(saved_records, sort_keys=True).encode()).hexdigest()
for role in roles:
    state = json.loads(marker(role).read_text(encoding="utf-8")) if marker(role).is_file() else {}
    profile = state.get("profile", "")
    if (state.get("status") != "completed" or state.get("requested") != N
            or state.get("manifest_sha256") != manifest_sha
            or not re.fullmatch(r"[a-f0-9]{16}", profile)
            or not (RUNS_ROOT / role / profile).is_dir()):
        pending.append(role)
    else:
        profiles[role] = profile
if pending:
    raise RuntimeError("Chưa có kết quả hoàn chỉnh cho: " + ", ".join(pending)
                       + ". Chạy ô sinh ảnh tương ứng hoặc kiểm tra RUN_NAME/BATCH.")
for role, profile in profiles.items():
    # Score và sidecar cũng nằm trên Drive; chạy lại ô 7 sẽ dùng lại điểm đã chấm.
    sh(PYTHON, "scripts/evaluate_runs.py", "--profile", profile, "--manifest", saved_manifest,
       "--device", "cuda", "--runs-root", RUNS_ROOT / role)
output = PROJECT / "outputs" / "comparisons" / (
    f"{BATCH}_{N}_" + "_".join(profiles.values()))
args = [arg for role, profile in profiles.items()
        for arg in ("--" + role.replace("_", "-"), RUNS_ROOT / role / profile)]
fid_args = ["--fid-reference", FID_REFERENCE] if FID_REFERENCE else []
sh(PYTHON, "scripts/compare_versions.py", "--manifest", saved_manifest, "--output", output, *args, *fid_args)
report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
for role in roles:
    relevant = {k: v["mean"] for k, v in report["metrics"][role].items()
                if k in ("CLIP_score", "SSIM", "LPIPS", "Feature_Similarity", "seam_mae", "color_hist_sim")}
    print(role, relevant, "critic:", report["critic"][role])
if report["fid"]:
    print("FID:", report["fid"])
for name in report["previews"]:
    display(Image.open(output / name))
backup = DATA_RUN / "comparisons"
backup.mkdir(parents=True, exist_ok=True)
archive = Path(shutil.make_archive(str(output), "zip", output))
target = backup / archive.name
partial = target.with_suffix(".zip.part")
shutil.copy2(archive, partial)
os.replace(partial, target)
print("CSV, JSON and previews:", target)
'''
]


# Biên dịch từng ô để bắt lỗi cú pháp rồi xuất notebook đúng thứ tự.
def main():
    cells = []
    for index, source in enumerate(CELLS, 1):
        compile(source, f"cell_{index}", "exec")
        cells.append(dict(cell_type="code", id=f"amodal-{index:02d}", metadata={},
                          source=source.splitlines(keepends=True), outputs=[], execution_count=None))
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                    metadata=dict(accelerator="GPU", kernelspec=dict(name="python3", display_name="Python 3")))
    target = ROOT / "notebooks/Amodal_Completion_Colab.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
