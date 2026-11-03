"""Preflight imports, source provenance, GPU and optional model access; no full model load."""
import argparse, importlib, json, os, sys
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--imports-only",action="store_true")
    parser.add_argument("--check-access",action="store_true")
    args=parser.parse_args()
    import torch
    print("Python:",sys.version)
    print("Torch:",torch.__version__,"CUDA:",torch.version.cuda)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; choose Colab GPU")
    prop=torch.cuda.get_device_properties(0)
    print("GPU:",prop.name,"VRAM GiB:",round(prop.total_memory/1024**3,1))
    if prop.total_memory<32*1024**3:
        print("NOTE: full concurrent author pipeline has not been verified on this small GPU; A100/high RAM recommended.")
    from PIL import Image
    if not hasattr(Image,"ANTIALIAS"): Image.ANTIALIAS=Image.Resampling.LANCZOS
    for extra in ["vendor/taming-transformers","vendor/CLIP","pix2gestalt","pix2gestalt/pix2gestalt"]:
        sys.path.insert(0,str(ROOT/extra))
    sys.path.insert(0,str(ROOT/"runtime/SCAmodal"))
    for name in ["diffusers","transformers","qwen_vl_utils","clip","carvekit.api.high",
                 "ldm.models.diffusion.ddpm","inference","amodal_completer","pycocotools","lpips","skimage"]:
        importlib.import_module(name)
        print("Import OK:",name)
    from transformers import Qwen3VLForConditionalGeneration
    print("Qwen3-VL class available:",Qwen3VLForConditionalGeneration.__name__)
    if not args.imports_only:
        checkpoint=ROOT/"ckpt/epoch=000005.ckpt"
        if not checkpoint.is_file() or checkpoint.stat().st_size<1024**3:
            raise RuntimeError("Missing/incomplete Pix2Gestalt checkpoint")
    if args.check_access:
        from huggingface_hub import hf_hub_download
        cfg=json.loads((ROOT/"configs/scamodal.json").read_text())
        for model,name in [("Qwen/Qwen3-VL-4B-Instruct","config.json"),
                           (cfg["inpainting_model_id"],"model_index.json")]:
            hf_hub_download(model,filename=name)
            print("Model access OK:",model)
    print("Preflight complete. This does not validate full GPU inference.")

if __name__=="__main__": main()
