"""Extract the named COCO-A subset into a content-addressed local cache."""
import hashlib
from pathlib import Path
import tempfile
import zipfile


def unpack(source, cache):
    if not source.is_file():
        raise FileNotFoundError(f"Upload cocoa.zip to {source.parent}")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        dest = cache / digest
        if (dest / "filename_mapping.csv").is_file():
            return dest
        cache.mkdir(parents=True, exist_ok=True)
        stream.seek(0)
        with tempfile.TemporaryDirectory(dir=cache) as tmp, zipfile.ZipFile(stream) as bundle:
            staging = Path(tmp)
            for name in bundle.namelist():
                if not (staging / name).resolve().is_relative_to(staging.resolve()):
                    raise ValueError(f"Unsafe ZIP member: {name}")
            bundle.extractall(staging)
            roots = [p.parent for p in staging.rglob("dataset.json")
                     if (p.parent / "filename_mapping.csv").is_file()]
            if len(roots) != 1:
                raise ValueError("Expected one dataset.json beside filename_mapping.csv in cocoa.zip")
            roots[0].rename(dest)
        return dest
