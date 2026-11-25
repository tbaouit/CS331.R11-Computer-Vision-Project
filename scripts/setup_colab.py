# Phạm vi: Ô 3: tạo môi trường Colab, cài đặt Grounded-Segment-Anything, recognize-anything, InstaOrder, CLIP
# và tải checkpoint cho cả Baseline (SD2) và Proposed (Open-World Amodal SD2).
"""Isolated Linux/Colab GPU environment setup for Baseline (SD2) & Proposed (Open-World Amodal SD2)."""
import argparse, json, os, subprocess, sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    print(">", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True, **kwargs)


def clone(url, path):
    if not (path / ".git").exists():
        run(["git", "clone", "--depth", "1", url, path])
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-downloads", action="store_true")
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Install dependencies for scoring saved images; skip generator and vendor setup",
    )
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
        names = {
            "numpy",
            "Pillow",
            "opencv-python",
            "transformers",
            "safetensors",
            "scipy",
            "scikit-image",
            "scikit-learn",
            "lpips",
            "open_clip_torch",
            "pytorch-fid",
            "ftfy",
        }
        metric_deps = [
            line
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#") and line.partition("==")[0] in names
        ]
        run(install + metric_deps)
        run(
            [
                python,
                "-c",
                "import torch, torchvision, transformers, lpips, skimage, open_clip, pytorch_fid; "
                "assert torch.cuda.is_available(), 'Select a Colab GPU'",
            ]
        )
        print("Environment ready for scoring saved images.", flush=True)
        return

    run(install + ["-r", str(requirements)])
    (ROOT / "vendor").mkdir(exist_ok=True)
    dependencies = {}
    vendor_repos = [
        ("Grounded-Segment-Anything", "https://github.com/IDEA-Research/Grounded-Segment-Anything.git"),
        ("recognize-anything", "https://github.com/xinyu1205/recognize-anything.git"),
        ("InstaOrder", "https://github.com/POSTECH-CVLab/InstaOrder.git"),
        ("CLIP", "https://github.com/openai/CLIP.git"),
    ]
    for name, url in vendor_repos:
        dependencies[name] = clone(url, ROOT / "vendor" / name)

    # Install Python packages from vendor without compiling custom CUDA extensions when running on CPU-fallback
    for editable_pkg in [
        ROOT / "vendor" / "CLIP",
        ROOT / "vendor" / "recognize-anything",
    ]:
        run(install + ["--no-deps", "--no-build-isolation", "-e", str(editable_pkg)])

    gdino_dir = ROOT / "vendor" / "Grounded-Segment-Anything" / "GroundingDINO"
    if gdino_dir.is_dir():
        try:
            run(install + ["--no-deps", "--no-build-isolation", "-e", str(gdino_dir)])
        except subprocess.CalledProcessError:
            print("[Setup] Note: GroundingDINO C++ extension build skipped; using pure-PyTorch fallback.", flush=True)

    site_pkgs = next((ROOT / ".venv-gpu/lib").glob("python*/site-packages"))
    (site_pkgs / "amodal_vendor.pth").write_text(
        "\n".join(
            [
                str(ROOT / "vendor/InstaOrder"),
                str(ROOT / "vendor/Grounded-Segment-Anything"),
                str(ROOT / "vendor/Grounded-Segment-Anything/GroundingDINO"),
                str(ROOT / "vendor/recognize-anything"),
                str(ROOT / "vendor/CLIP"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ROOT / "dependency_versions.json").write_text(json.dumps(dependencies, indent=2), encoding="utf-8")
    frozen = subprocess.check_output(uv + ["pip", "freeze", "--python", str(python)], text=True)
    (ROOT / "installed_gpu_versions.txt").write_text(frozen, encoding="utf-8")
    run([python, ROOT / "scripts/check_environment.py", "--imports-only"])
    if not args.skip_downloads:
        run([python, ROOT / "scripts/download_weights.py"])
        run([python, ROOT / "scripts/check_environment.py", "--check-access"])
    print("Environment ready for GPU inference (<15GB VRAM sequential offload).", flush=True)


if __name__ == "__main__":
    main()
