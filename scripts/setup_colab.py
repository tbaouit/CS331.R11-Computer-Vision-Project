# Phạm vi: Ô 3: tạo môi trường Colab, ghim thư viện và cài code phụ thuộc của Pix2Gestalt.
"""Isolated Linux/Colab GPU environment setup for Amodal Appearance Completion."""
import argparse, json, os, subprocess, sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]


# Thực thi lệnh cài đặt trong thư mục project và dừng nếu lệnh lỗi.
def run(command, **kwargs):
    print(">", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True, **kwargs)


# Lấy repo phụ thuộc nếu chưa có; trả commit để ghi phiên bản môi trường.
def clone(url, path):
    if not (path / ".git").exists():
        run(["git", "clone", "--depth", "1", url, path])
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


# Tạo venv GPU, cài dependency, thêm đường dẫn vendor và kiểm tra môi trường.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-downloads", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true",
                        help="Install dependencies for scoring saved images; skip generator and vendor setup")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise SystemExit("Run this setup in a Colab Linux GPU runtime, not Windows.")
    run(["nvidia-smi"])
    run([sys.executable, "-m", "pip", "install", "uv==0.6.17"])
    uv = [sys.executable, "-m", "uv"]
    python = ROOT / ".venv-gpu/bin/python"
    if not python.exists():
        run(uv + ["venv", "--python", "3.12", "--seed", str(ROOT / ".venv-gpu")])
    install = uv + ["pip", "install", "--python", str(python)]
    run(install + ["setuptools==75.8.2", "wheel==0.45.1"])
    run(install + ["torch==2.10.0", "torchvision==0.25.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
    requirements = ROOT / "requirements-colab.txt"
    if args.evaluation_only:
        names = {"numpy", "Pillow", "opencv-python", "transformers", "safetensors",
                 "scipy", "scikit-image", "lpips", "pytorch-fid", "ftfy"}
        metric_deps = [line for line in requirements.read_text(encoding="utf-8").splitlines()
                       if line.partition("==")[0] in names]
        run(install + metric_deps)
        run([python, "-c", "import torch, torchvision, transformers, lpips, skimage, pytorch_fid; "
                           "assert torch.cuda.is_available(), 'Select a Colab GPU'"])
        print("Environment ready for scoring saved images.", flush=True)
        return
    run(install + ["-r", str(requirements)])
    dependencies = {}
    dependencies["pix2gestalt"] = clone("https://github.com/cvlab-columbia/pix2gestalt.git", ROOT / "pix2gestalt")
    (ROOT / "vendor").mkdir(exist_ok=True)
    for name, url in [
        ("taming-transformers", "https://github.com/CompVis/taming-transformers.git"),
        ("CLIP", "https://github.com/openai/CLIP.git"),
    ]:
        dependencies[name] = clone(url, ROOT / "vendor" / name)
        if name == "taming-transformers":
            (ROOT / "vendor/taming-transformers/taming/__init__.py").touch()
        run(install + ["--no-deps", "--no-build-isolation", "-e", str(ROOT / "vendor" / name)])
    site_pkgs = next((ROOT / ".venv-gpu/lib").glob("python*/site-packages"))
    (site_pkgs / "amodal_vendor.pth").write_text(
        "\n".join(
            [
                str(ROOT / "vendor/taming-transformers"),
                str(ROOT / "vendor/CLIP"),
                str(ROOT / "pix2gestalt"),
                str(ROOT / "pix2gestalt/pix2gestalt"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ddpm = ROOT / "pix2gestalt/pix2gestalt/ldm/models/diffusion/ddpm.py"
    code = ddpm.read_text(encoding="utf-8")
    code = code.replace(
        "from pytorch_lightning.utilities.distributed import rank_zero_only",
        "from pytorch_lightning.utilities.rank_zero import rank_zero_only",
    )
    ddpm.write_text(code, encoding="utf-8")
    (ROOT / "dependency_versions.json").write_text(json.dumps(dependencies, indent=2), encoding="utf-8")
    frozen = subprocess.check_output(uv + ["pip", "freeze", "--python", str(python)], text=True)
    (ROOT / "installed_gpu_versions.txt").write_text(frozen, encoding="utf-8")
    run([python, ROOT / "scripts/check_environment.py", "--imports-only"])
    if not args.skip_downloads:
        run([python, ROOT / "scripts/download_weights.py"])
        run([python, ROOT / "scripts/check_environment.py", "--check-access"])
    print("Environment ready for GPU inference.", flush=True)


if __name__ == "__main__":
    main()
