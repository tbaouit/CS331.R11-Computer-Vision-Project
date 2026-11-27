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
RUN_NAME = "cocoa_openworld_amodal"
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

# Giữ cố định mã nguồn giữa các lần resume sinh ảnh. Khi EVALUATION_ONLY=True hoặc RUN_NAME mới thì lấy mã mới từ main.
if repo_lock.is_file() and not EVALUATION_ONLY:
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
else:
    import hashlib
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from scripts.mount_cocoa_zip import unpack
    archive = DRIVE / "cocoa.zip"
    dataset_root = unpack(archive, PROJECT / "dataset/named_cache")
    annotation_file = dataset_root / "cocoa_annotations.json"
    if not annotation_file.is_file():
        raise FileNotFoundError("Expected cocoa/cocoa_annotations.json inside cocoa.zip, beside dataset.json")
    data_key = dict(zip_sha256=dataset_root.name,
                    annotation_sha256=hashlib.sha256(annotation_file.read_bytes()).hexdigest())
    data_lock = EXPERIMENT / "dataset_identity.json"
    if data_lock.is_file() and json.loads(data_lock.read_text()) != data_key:
        raise ValueError("Dữ liệu đã thay đổi. Dùng RUN_NAME mới để bắt đầu đợt chạy mới.")
    save_json(data_lock, data_key)
    print("Dataset:", dataset_root, "Annotations:", annotation_file)

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
BATCH = "full"  # Chọn 10, 50 hoặc "full".
VISIBLE_MASK_SOURCE = "gt"  # Chọn "gt" (dùng toàn bộ mask ground truth có sẵn) hoặc "auto" (Zero-GT: GroundingDINO + SAM tự đoán).
BATCH = str(BATCH)
if BATCH not in ("10", "50", "full"):
    raise ValueError("BATCH phải là 10, 50 hoặc full")
if VISIBLE_MASK_SOURCE not in ("gt", "auto"):
    raise ValueError("VISIBLE_MASK_SOURCE phải là 'gt' hoặc 'auto'")
extra = ["--allow-test"]  # Inference on the named subset across all original splits.
DATA_RUN = EXPERIMENT / ("batch_" + BATCH)
RUNS_ROOT = DATA_RUN / "runs"
saved_manifest = DATA_RUN / "manifest.json"
configs = ("baseline", "proposed")

if EVALUATION_ONLY:
    manifest = saved_manifest
    records = json.loads(manifest.read_text(encoding="utf-8"))
    N = len(records)
    if not N or (BATCH != "full" and N != int(BATCH)):
        raise ValueError("Manifest trên Drive không khớp batch")
else:
    local_batch = PROJECT / "outputs/named" / dataset_root.name / data_key["annotation_sha256"] / BATCH
    manifest = local_batch / "manifest.json"
    if not manifest.exists():
        sh(PYTHON, "scripts/prepare_named_batch.py", "--dataset", dataset_root,
           "--annotations", annotation_file, "--batch", BATCH, "--output", local_batch)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    N = len(records)
    save_json(DATA_RUN / "selection_report.json", json.loads((local_batch / "selection_report.json").read_text()))
    sh(PYTHON, "scripts/run_experiment.py", "--manifest", manifest, "--limit", N,
       "--config", "configs/baseline.json", "--validate-only", *extra)
    if saved_manifest.is_file():
        if records != json.loads(saved_manifest.read_text(encoding="utf-8")):
            raise ValueError("Danh sách mẫu đã đổi. Dùng lại dataset gốc hoặc đặt RUN_NAME mới ở ô 1.")
    else:
        save_json(saved_manifest, records)

def marker(role):
    auto_marker = DATA_RUN / "profiles" / f"{role}_auto.json"
    gt_marker = DATA_RUN / "profiles" / f"{role}.json"
    if VISIBLE_MASK_SOURCE == "auto" and (not EVALUATION_ONLY or auto_marker.is_file()):
        return auto_marker
    return gt_marker

def run_version(role):
    if EVALUATION_ONLY:
        raise RuntimeError("Phiên này chỉ đánh giá; bỏ qua ô 5–6.")
    config = f"configs/{role}.json"
    sh(PYTHON, "scripts/run_experiment.py", "--manifest", manifest, "--limit", N,
       "--config", config, "--visible-mask-source", VISIBLE_MASK_SOURCE,
       "--runs-root", RUNS_ROOT / role,
       "--profile-state", marker(role), *extra)

print("Số vật thể:", N, "| Chế độ visible_mask:", VISIBLE_MASK_SOURCE, "| Batch trên Drive:", DATA_RUN)
for role in configs:
    state = json.loads(marker(role).read_text(encoding="utf-8")) if marker(role).is_file() else {}
    print(role, state.get("status", "chưa chạy"), state.get("profile", ""))
''',
'''# 5. Chạy Baseline: SD2 Direct (1 lượt, cùng tên lớp COCO-A)
run_version("baseline")
''',
'''# 6. Chạy Proposed: Open-World Amodal (RAM++, CLIP, Grounded-SAM, InstaOrder, SD2)
run_version("proposed")
''',
'''# 7. Đánh giá, so sánh và lưu báo cáo
from PIL import Image
from IPython.display import display
FID_REFERENCE = None  # For at least 400 selected objects, set to a Drive folder of >=400 clean object crops.
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
                if k in ("visible_iou", "CLIP_score", "OpenCLIP_score", "DINO_similarity", "VGG_similarity",
                         "Feature_Similarity", "SSIM", "PSNR", "LPIPS", "seam_mae", "color_hist_sim")}
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
