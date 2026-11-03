"""Isolated Linux/Colab environment for the instrumented author pipeline."""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
ROOT=Path(__file__).resolve().parents[1]

def run(command,**kwargs):
    print(">", " ".join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),cwd=ROOT,check=True,**kwargs)

def clone(url,path):
    if not (path/".git").exists(): run(["git","clone","--depth","1",url,path])
    return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--skip-downloads",action="store_true")
    args=parser.parse_args()
    if sys.platform!="linux": raise SystemExit("Run this setup in a Colab Linux GPU runtime, not Windows.")
    run(["nvidia-smi"])
    run([sys.executable,"-m","pip","install","uv==0.6.17"])
    uv=[sys.executable,"-m","uv"]
    python=ROOT/".venv-gpu/bin/python"
    if not python.exists(): run(uv+["venv","--python","3.12","--seed",str(ROOT/".venv-gpu")])
    install=uv+["pip","install","--python",str(python)]
    run(install+["setuptools==75.8.2","wheel==0.45.1"])
    run(install+["torch==2.10.0","torchvision==0.25.0","--index-url","https://download.pytorch.org/whl/cu128"])
    run(install+["-r",str(ROOT/"requirements-colab.txt")])
    dependencies={}
    dependencies["pix2gestalt"]=clone("https://github.com/cvlab-columbia/pix2gestalt.git",ROOT/"pix2gestalt")
    (ROOT/"vendor").mkdir(exist_ok=True)
    for name,url in [("taming-transformers","https://github.com/CompVis/taming-transformers.git"),
                     ("CLIP","https://github.com/openai/CLIP.git")]:
        dependencies[name]=clone(url,ROOT/"vendor"/name)
        if name=="taming-transformers":
            (ROOT/"vendor/taming-transformers/taming/__init__.py").touch()
        run(install+["--no-deps","--no-build-isolation","-e",str(ROOT/"vendor"/name)])
    site_pkgs=next((ROOT/".venv-gpu/lib").glob("python*/site-packages"))
    (site_pkgs/"scamodal_vendor.pth").write_text("\n".join([
        str(ROOT/"vendor/taming-transformers"),
        str(ROOT/"vendor/CLIP"),
        str(ROOT/"pix2gestalt"),
        str(ROOT/"pix2gestalt/pix2gestalt"),
    ])+"\n",encoding="utf-8")
    # A narrow compatibility patch to the local Pix2Gestalt dependency.
    ddpm=ROOT/"pix2gestalt/pix2gestalt/ldm/models/diffusion/ddpm.py"
    code=ddpm.read_text(encoding="utf-8")
    code=code.replace("from pytorch_lightning.utilities.distributed import rank_zero_only",
                      "from pytorch_lightning.utilities.rank_zero import rank_zero_only")
    ddpm.write_text(code,encoding="utf-8")
    (ROOT/"dependency_versions.json").write_text(json.dumps(dependencies,indent=2),encoding="utf-8")
    frozen=subprocess.check_output(uv+["pip","freeze","--python",str(python)],text=True)
    (ROOT/"installed_gpu_versions.txt").write_text(frozen,encoding="utf-8")
    run([python,ROOT/"scripts/doctor_scamodal.py","--imports-only"])
    if not args.skip_downloads:
        run([python,ROOT/"scripts/download_scamodal_weights.py"])
        run([python,ROOT/"scripts/doctor_scamodal.py","--check-access"])
    print("Environment ready; full GPU inference still needs a successful smoke run.",flush=True)

if __name__=="__main__": main()
