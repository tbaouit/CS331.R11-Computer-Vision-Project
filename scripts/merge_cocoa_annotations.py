"""Combine split annotations and retain the filename-to-region provenance."""
import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


def merge(archive):
    with zipfile.ZipFile(archive) as z:
        mappings = [n for n in z.namelist() if PurePosixPath(n).name == "filename_mapping.csv"]
        if len(mappings) != 1:
            raise ValueError("Expected one filename_mapping.csv")
        root = PurePosixPath(mappings[0]).parent
        rows = list(csv.DictReader(io.StringIO(z.read(mappings[0]).decode("utf-8-sig"))))
        by_name = {(r["split"], r["file_name"]): r for r in rows}
        images, annotations = [], []
        for split in ("train", "dev", "test"):
            data = json.loads(z.read(str(root / split / "annotations.json")))
            for img in data["images"]:
                row = by_name.pop((split, img["file_name"]))
                if int(row["image_id"]) != img["id"]:
                    raise ValueError("Image ID differs from filename mapping")
                matches = row["exact_name_candidates_annotation_id:region_index_0based"]
                images.append(dict(img, split=split, file_name=f"{split}/images/{img['file_name']}",
                                   object_hint=row["object_hint"], source_filename=row["source_filename"],
                                   exact_candidates=[list(map(int, v.split(":"))) for v in matches.split("|") if v]))
            for ann in data["annotations"]:
                annotations.append(dict(ann, source_annotation_id=ann.get("source_annotation_id", ann["id"]),
                                        id=len(annotations) + 1))
        ids = {i["id"] for i in images}
        if (by_name or len(images) != len(rows) or len(ids) != len(images)
                or {a["image_id"] for a in annotations} != ids
                or len({a["id"] for a in annotations}) != len(annotations)):
            raise ValueError("Inconsistent IDs across splits")
        return dict(images=images, annotations=annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = merge(args.archive)
    args.output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(len(data["images"]), "images,", len(data["annotations"]), "annotations")
