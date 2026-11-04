"""Package code separately from the dataset; never include environments or weights."""
import argparse, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-dataset", action="store_true")
    args = parser.parse_args()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    paths = [ROOT / "README.md", ROOT / "requirements-data.txt", ROOT / "requirements-colab.txt", ROOT / ".gitignore"]
    for folder in ["configs", "src", "scripts", "tests", "docs", "notebooks"]:
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    with zipfile.ZipFile(dist / "amodal_project.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            bundle.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(dist / "amodal_project.zip") as bundle:
        assert bundle.testzip() is None
    print("Packaged code:", dist / "amodal_project.zip", flush=True)
    if args.include_dataset:
        archive = dist / "cocoa_dataset.zip"
        if not archive.exists():
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED, allowZip64=True) as bundle:
                for path in sorted((ROOT / "dataset/cocoa").rglob("*")):
                    if path.is_file():
                        bundle.write(path, path.relative_to(ROOT / "dataset").as_posix())
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None
            for split in ["train", "dev", "test"]:
                assert f"cocoa/{split}/annotations.json" in bundle.namelist()
        print("Dataset ZIP verified:", archive, archive.stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    main()
