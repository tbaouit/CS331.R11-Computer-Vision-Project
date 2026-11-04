"""Fetch the author's official Pix2Gestalt checkpoint without global torch.load overrides."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]

def main():
    from huggingface_hub import hf_hub_download
    folder=ROOT/"ckpt"
    folder.mkdir(exist_ok=True)
    target=folder/"epoch=000005.ckpt"
    if not target.exists():
        if shutil.disk_usage(ROOT).free < 20*1024**3:
            raise RuntimeError("Need at least 20 GiB free disk for the large checkpoint alone")
        cached=hf_hub_download("cvlab/pix2gestalt-weights",filename="epoch=000005.ckpt")
        target.symlink_to(Path(cached).resolve())
    if target.stat().st_size<1024**3:
        raise RuntimeError("Checkpoint too small; may be incomplete or a Git LFS pointer")
    print("Pix2Gestalt checkpoint:",target.stat().st_size,"bytes")
    print("Qwen, CLIP, SD2 and CarveKit weights download on first model initialization.")

if __name__=="__main__": main()
