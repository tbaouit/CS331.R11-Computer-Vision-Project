"""Prepare nested 10, 50 or full batches using one combined annotation file."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cocoa_loader import Target, export_target


def prepare(root, labels, batch, output):
    data = json.loads(labels.read_bytes())
    images = {im["id"]: im for im in data["images"]}
    candidates = []
    for ann in data["annotations"]:
        im = images[ann["image_id"]]
        path = (root / im["file_name"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid image: {path}")
        source_ann = ann.get("source_annotation_id", ann["id"])
        exact = im["exact_candidates"]
        for idx, region in enumerate(ann["regions"]):
            if exact and [source_ann, idx] not in exact:
                continue
            if not region.get("visible_mask") or not region.get("name", "").strip():
                continue
            target = Target(im["split"], im["id"], ann["id"], idx, path,
                            im["width"], im["height"], region)
            key = (im["split"], im["source_filename"], source_ann, idx)
            candidates.append((key, target, bool(exact)))
    # Prefer a filename matching the region when an image has multiple aliases.
    unique = {}
    for key, target, exact in sorted(candidates, key=lambda v: not v[2]):
        unique.setdefault(key, (target, exact))
    pool = list(unique.values())
    random.Random(42).shuffle(pool)
    records, rejected = [], []
    limit = None if batch == "full" else int(batch)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staging = Path(tmp) / "batch"
        staging.mkdir()
        for target, exact in pool:
            try:
                visible, amodal, hidden = target.masks()
                if not visible.any() or not hidden.any():
                    raise ValueError("empty visible or hidden mask")
            except (ValueError, TypeError, KeyError, IndexError) as error:
                rejected.append(dict(sample_id=target.sample_id, reason=str(error)))
                continue
            record, _ = export_target(target, staging, visible, amodal, hidden)
            record["selection"] = "name_candidate" if exact else "unmatched_name_scene_region"
            records.append(record)
            if len(records) == limit:
                break
        if not records or (limit and len(records) != limit):
            raise ValueError(f"Only {len(records)} usable targets for {batch}")
        report = dict(batch=batch, source_files=len(images), unique_candidates=len(pool),
                      selected=len(records), by_split=Counter(r["split"] for r in records),
                      selection=Counter(r["selection"] for r in records), rejected=rejected)
        for name, value in (("manifest", records), ("selection_report", report)):
            (staging / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        staging.rename(output)
    print(f"Prepared {len(records)} objects for {batch}: {dict(report['by_split'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--batch", choices=["10", "50", "full"], required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.dataset, args.annotations, args.batch, args.output)
