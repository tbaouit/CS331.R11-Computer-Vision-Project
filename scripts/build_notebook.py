import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/tbaouit/CS331.R11-Computer-Vision-Project.git"


def main():
    cells = []

    def md(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True)))

    def code(text):
        cells.append(
            dict(
                cell_type="code",
                metadata={},
                source=text.splitlines(keepends=True),
                outputs=[],
                execution_count=None,
            )
        )

    md(f"""# CS331 — Open-World Amodal Appearance Completion on COCO-A
Qwen3-VL semantic reasoning → Pix2Gestalt / CarveKit prior → Hybrid SD2 ROI refinement → Qwen3-VL target-aware critic.
- **Code**: clone / `git pull` trực tiếp từ GitHub (`{REPO_URL}`). Mỗi lần cập nhật code trên GitHub chỉ cần chạy lại **Ô 2** để pull bản mới nhất.
- **Dataset**: giữ `cocoa_dataset.zip` trên Google Drive tại `MyDrive/CS331/cocoa_dataset.zip` hoặc `MyDrive/amodal_project/cocoa_dataset.zip`.
- **Chế độ chạy**:
  - `configs/improved.json` (`pipeline_mode = "improved"`): tự dự đoán cả vùng bị che lẫn ngoại hình.
  - `configs/oracle_mask.json` (`pipeline_mode = "oracle_mask"`): **nhận thẳng mask phần bị che từ input** (`gt_hidden` / `occluded_mask`) theo thiết lập Mask-Guided Amodal Appearance Completion.
""")
    md("## 1. GPU và Google Drive\nẨn thanh tiến trình phần trăm; vẫn hiển thị trạng thái tải model, kết quả từng mẫu và lỗi. Log của pipeline vẫn được lưu trong `outputs/runs/<profile>/pipeline.log`.")
    code("""import subprocess, sys, os, shutil, zipfile, json, base64, re
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
def sh(*cmd):
    env = {
        **os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1",
    }
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
print("DRIVE:", DRIVE)
for secret_name in ("HF_TOKEN", "GITHUB_TOKEN"):
    try:
        from google.colab import userdata
        val = userdata.get(secret_name)
        if val: os.environ[secret_name] = val
    except Exception:
        pass
""")
    md("## 2. Clone / Pull code mới nhất từ GitHub & giải nén dataset từ Drive\nChạy lại ô này bất cứ khi nào bạn vừa `git push` code mới lên GitHub (dataset đã giải nén sẽ được giữ nguyên, không giải nén lại).")
    code(f"""REPO_URL = "{REPO_URL}"
BRANCH = "main"
PROJECT = Path("/content/cs331")
PYTHON = str(PROJECT / ".venv-gpu/bin/python")

git_cmd = ["git"]
if os.environ.get("GITHUB_TOKEN"):
    basic = base64.b64encode(f"x-access-token:{{os.environ['GITHUB_TOKEN']}}".encode()).decode()
    git_cmd = ["git", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {{basic}}"]

if (PROJECT / ".git").is_dir():
    print("Updating existing repository via git pull...", flush=True)
    sh(*git_cmd, "-C", PROJECT, "fetch", "origin", BRANCH)
    sh(*git_cmd, "-C", PROJECT, "reset", "--hard", f"origin/{{BRANCH}}")
elif PROJECT.exists() and any(PROJECT.iterdir()):
    print("Initializing git tracking on existing /content/cs331 directory...", flush=True)
    sh("git", "-C", PROJECT, "init", "-b", BRANCH)
    sh("git", "-C", PROJECT, "remote", "add", "origin", REPO_URL)
    sh(*git_cmd, "-C", PROJECT, "fetch", "origin", BRANCH)
    sh("git", "-C", PROJECT, "checkout", "-f", "-B", BRANCH, f"origin/{{BRANCH}}")
else:
    print("Cloning repository from GitHub...", flush=True)
    sh(*git_cmd, "clone", "--branch", BRANCH, REPO_URL, PROJECT)

os.chdir(PROJECT)
sh("git", "log", "-n", "1", "--oneline")

if not (PROJECT / "dataset/cocoa/dataset.json").is_file():
    candidates = [
        DRIVE / "cocoa_dataset.zip",
        DRIVE / "data/cocoa_dataset.zip",
        Path("/content/drive/MyDrive/CS331/cocoa_dataset.zip"),
        Path("/content/drive/MyDrive/amodal_project/cocoa_dataset.zip"),
    ]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"Upload cocoa_dataset.zip vào {{DRIVE}}")
    local_zip = Path("/content/cocoa_dataset.zip")
    print(f"Copying {{source}} -> {{local_zip}} ...", flush=True)
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
""")
    md("## 3. Setup môi trường riêng + checkpoint + unit tests\nNếu phiên Colab đang chạy đã cài xong `.venv-gpu`, chỉ cần chạy dòng unit test thứ 2.")
    code("""sh(sys.executable, "scripts/setup_colab.py")
sh(PYTHON, "-m", "unittest", "discover", "-s", "tests", "-v")
""")
    md("## 4. Chuẩn bị 10 đối tượng train")
    code("""sh(PYTHON, "scripts/prepare_data.py", "--verify-hashes")
from PIL import Image
from IPython.display import display
display(Image.open(PROJECT / "outputs/train_smoke/contact_sheet.jpg"))
print((PROJECT / "configs/oracle_mask.json").read_text())
""")
    md("## 5A. Chạy pipeline chuyên biệt nhận thẳng Mask phần bị che từ Input (`oracle_mask`)")
    code("""sh(PYTHON, "scripts/run_experiment.py", "--manifest", "outputs/train_smoke/manifest.json", "--config", "configs/oracle_mask.json", "--limit", "10", "--backup-dir", str(DRIVE / "results"))
""")
    md("## 5B. Chạy pipeline Blind Improved (tự đoán mask + khôi phục diện mạo)")
    code("""sh(PYTHON, "scripts/run_experiment.py", "--manifest", "outputs/train_smoke/manifest.json", "--config", "configs/improved.json", "--limit", "10", "--backup-dir", str(DRIVE / "results"))
""")
    md("## 6. Kiểm tra toàn bộ output của profile vừa chạy")
    code("""runs_dir = PROJECT / "outputs/runs"
profiles = sorted([p for p in runs_dir.iterdir() if p.is_dir() and list(p.glob("*/*/result.json"))],
                  key=lambda d: max(f.stat().st_mtime for f in d.glob("*/*/result.json")))
if not profiles: raise RuntimeError("No successful output; inspect pipeline.log and failure.json")
latest_profile = profiles[-1]
print("Latest profile:", latest_profile)
for res_file in sorted(latest_profile.glob("*/*/result.json")):
    res = json.loads(res_file.read_text())
    m = res["metrics"]
    print(f"=== {res['sample_id']} ({res.get('name')}) | Amodal IoU={m['amodal_iou']:.4f} | Hidden IoU={m['hidden_iou']:.4f} | Score={res.get('final_score')} ===")
    display(Image.open(res_file.parent / "preview.jpg"))
""")
    md("## 7. So sánh trực tiếp các chế độ (bảng số liệu + lưới ảnh)\nTự động khôi phục các bản backup từ Google Drive (`DRIVE / 'results'`) để xuất bảng và lưới so sánh vào `outputs/runs/comparison_grid.jpg`.")
    code("""sh(PYTHON, "scripts/compare_runs.py", "--backup-dir", str(DRIVE / "results"))
display(Image.open(PROJECT / "outputs/runs/comparison_grid.jpg"))
""")
    for i, cell in enumerate(cells):
        cell["id"] = f"amodal-{i:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{i}", "exec")
    doc = dict(
        cells=cells,
        metadata=dict(
            accelerator="GPU",
            kernelspec=dict(name="python3", display_name="Python 3"),
            language_info=dict(name="python"),
        ),
        nbformat=4,
        nbformat_minor=5,
    )
    target = ROOT / "notebooks/Amodal_Completion_Colab.ipynb"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Notebook generated and syntax checked:", target.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
