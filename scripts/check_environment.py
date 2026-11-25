# Phạm vi: Ô 3: xác nhận GPU, thư viện và checkpoint có thể truy cập trước khi chạy.
"""Preflight imports, GPU availability, and optional HuggingFace model access."""
import argparse, importlib, json, os, sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imports-only", action="store_true")
    parser.add_argument("--check-access", action="store_true")
    args = parser.parse_args()
    import torch

    print("Python:", sys.version)
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose Colab GPU")
    prop = torch.cuda.get_device_properties(0)
    print("GPU:", prop.name, "VRAM GiB:", round(prop.total_memory / 1024**3, 1))
    from PIL import Image

    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    for extra in [
        "vendor/InstaOrder",
        "vendor/Grounded-Segment-Anything",
        "vendor/Grounded-Segment-Anything/GroundingDINO",
        "vendor/recognize-anything",
        "vendor/CLIP",
    ]:
        path = ROOT / extra
        if path.exists():
            sys.path.insert(0, str(path))
    for name in [
        "diffusers",
        "transformers",
        "segment_anything",
        "pycocotools",
        "lpips",
        "skimage",
        "open_clip",
        "src.models.openworld_models",
        "src.models.memory_manager",
        "src.pipelines.backend",
    ]:
        importlib.import_module(name)
        print("Import OK:", name)
    for opt_name in ["clip", "ram", "models"]:
        try:
            importlib.import_module(opt_name)
            print("Vendor import OK:", opt_name)
        except ImportError as exc:
            print(f"Vendor import notice ({opt_name}): {exc}")

    if args.check_access:
        from huggingface_hub import hf_hub_download

        for cfg_name in ("configs/baseline.json", "configs/proposed.json"):
            cfg = json.loads((ROOT / cfg_name).read_text(encoding="utf-8"))
            model_id = cfg["inpainting_model_id"]
            hf_hub_download(model_id, filename="model_index.json")
            print("Model access OK:", model_id)
    print("Preflight complete.")


if __name__ == "__main__":
    main()
