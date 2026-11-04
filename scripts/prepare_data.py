"""Validate the clean dataset and export a fixed train-only smoke batch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
from src.data.cocoa import CocoaDataset


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_hash(task):
    path, expected = task
    if sha(path) != expected:
        raise ValueError(f"Image checksum mismatch: {path}")
    return expected


def audit(root, verify_hashes):
    report, ids, hashes, tasks = {}, {}, {}, []
    for split in ("train", "dev", "test"):
        ds = CocoaDataset(root, split)
        ids[split] = set(ds.images)
        filenames = [image["file_name"] for image in ds.images.values()]
        if len(set(filenames)) != len(filenames):
            raise ValueError(f"Duplicate filenames in {split}")
        missing = [name for name in filenames if not (ds.image_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{split}: missing {len(missing)} images")
        unknown = {a["image_id"] for a in ds.data["annotations"]} - ids[split]
        if unknown:
            raise ValueError(f"Unknown annotation image IDs: {unknown}")
        manifest = json.loads((root / split / "manifest.json").read_text(encoding="utf-8"))
        if len(manifest) != len(filenames) or {row["image_id"] for row in manifest} != ids[split]:
            raise ValueError(f"Manifest/annotation mismatch: {split}")
        for row in manifest:
            if row["file_name"] != ds.images[row["image_id"]]["file_name"]:
                raise ValueError(f"Manifest filename mismatch: {split}")
            hashes.setdefault(row["sha256"], []).append([split, row["image_id"]])
            tasks.append((ds.image_dir / row["file_name"], row["sha256"]))
        report[split] = dict(images=len(filenames), annotations=len(ds.data["annotations"]),
                             regions=sum(len(a["regions"]) for a in ds.data["annotations"]),
                             missing_images=0)
        print(split, report[split], flush=True)
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if ids[a] & ids[b]:
            raise ValueError(f"Image IDs overlap: {a}/{b}")
    if any(len(group) > 1 for group in hashes.values()):
        raise ValueError("Duplicate image hashes in manifests")
    if verify_hashes:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in pool.map(check_hash, tasks):
                pass
    return dict(splits=report, image_ids_disjoint=True, manifest_hashes_unique=True,
                image_bytes_verified=verify_hashes)


def overlay(image, mask, color):
    result = image.copy()
    result[mask] = (0.55 * result[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
    return result


def export_samples(ds, cfg, out):
    if ds.split != "train":
        raise ValueError("Smoke batch is train-only; keep test for final evaluation")
    records, panels, used = [], [], set()
    for target in ds.targets():
        if target.region.get("isStuff", 0):
            continue
        if cfg["one_object_per_image"] and target.image_id in used:
            continue
        if not target.region.get("visible_mask"):
            continue
        visible, amodal, hidden = target.masks()
        ratio = float(hidden.sum() / max(1, amodal.sum()))
        if (visible.sum() < cfg["min_visible_pixels"] or amodal.sum() < cfg["min_amodal_pixels"]
                or not cfg["min_hidden_ratio"] <= ratio <= cfg["max_hidden_ratio"]):
            continue
        with Image.open(target.image_path) as opened:
            image = np.array(opened.convert("RGB"))
        if image.shape[:2] != (target.height, target.width):
            raise ValueError(f"Image dimensions mismatch: {target.image_path}")
        folder = out / target.sample_id
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(folder / "image.png")
        for name, mask in (("visible", visible), ("gt_amodal", amodal), ("gt_hidden", hidden)):
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{name}.png")
        record = dict(sample_id=target.sample_id, split=ds.split, image_id=target.image_id,
                      annotation_id=target.annotation_id, region_index=target.region_index,
                      name=target.region.get("name", ""), source_filename=target.image_path.name,
                      hidden_ratio=ratio, visible_pixels=int(visible.sum()),
                      amodal_pixels=int(amodal.sum()), hidden_pixels=int(hidden.sum()),
                      image=f"{target.sample_id}/image.png", visible_mask=f"{target.sample_id}/visible.png",
                      gt_amodal=f"{target.sample_id}/gt_amodal.png", gt_hidden=f"{target.sample_id}/gt_hidden.png")
        (folder / "sample.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
        used.add(target.image_id)
        panel = Image.new("RGB", (1000, 280), "white")
        draw = ImageDraw.Draw(panel)
        tiles = [image, overlay(image, visible, (30, 220, 90)),
                 overlay(image, amodal, (30, 220, 90)), overlay(image, hidden, (255, 130, 30))]
        for col, (tile, label) in enumerate(zip(tiles, ("Input", "Visible: model input", "Amodal GT: evaluation", "Hidden GT: evaluation"))):
            thumb = Image.fromarray(tile)
            thumb.thumbnail((240, 240))
            panel.paste(thumb, (col * 250 + (250 - thumb.width) // 2, 30 + (240 - thumb.height) // 2))
            draw.text((col * 250 + 5, 7), label, fill="black")
        panel.save(folder / "preview.jpg")
        panels.append(panel)
        if len(records) == cfg["limit"]:
            break
    if len(records) != cfg["limit"]:
        raise RuntimeError(f"Only {len(records)} eligible targets; requested {cfg['limit']}")
    sheet = Image.new("RGB", (1000, 280 * len(panels)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 280))
    sheet.save(out / "contact_sheet.jpg")
    (out / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--verify-hashes", action="store_true", help="Read every JPG and verify against its manifest")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root, out = ROOT / cfg["dataset_root"], ROOT / cfg["output_dir"]
    if cfg["split"] != "train" or not 1 <= cfg["limit"] <= 100:
        raise ValueError("Use train and a limit of 1..100 for the initial smoke batch")
    report = audit(root, args.verify_hashes)
    out.mkdir(parents=True, exist_ok=True)
    records = export_samples(CocoaDataset(root, cfg["split"]), cfg, out)
    report["smoke_targets"] = len(records)
    report["smoke_unique_images"] = len({r["image_id"] for r in records})
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "selection_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} train objects. Output: {out}", flush=True)
    print("Model inputs: image.png + visible.png ONLY. gt_* are evaluation labels.")


if __name__ == "__main__":
    main()
