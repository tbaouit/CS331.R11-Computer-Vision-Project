# Phạm vi: Cả haiseline và proposed: chuyển annotation COCO-A thành danh sách vật thể cùng ảnh và ba mask.
# Không sinh RGB hoặc nạp GPU.
"""COCO-A image annotations -> individual objects. No GPU/model dependencies."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask


# Giải mã polygon hoặc RLE về mask bool H×W; dữ liệu trống thành mask rỗng.
def decode_mask(value, height, width):
    if value is None or (isinstance(value, list) and not value):
        return np.zeros((height, width), dtype=bool)
    if isinstance(value, dict):
        value = dict(value)
        value.setdefault("size", [height, width])
        if isinstance(value["counts"], str):
            value["counts"] = value["counts"].encode("ascii")
        if isinstance(value["counts"], list):
            value = coco_mask.frPyObjects(value, height, width)
        rle = value
    else:
        polygons = [value] if isinstance(value[0], (int, float)) else value
        rle = coco_mask.merge(coco_mask.frPyObjects(polygons, height, width))
    result = coco_mask.decode(rle)
    if result.ndim == 3:
        result = np.any(result, axis=2)
    if result.shape != (height, width):
        raise ValueError(f"Mask dimensions {result.shape} != {(height, width)}")
    return result.astype(bool)


# Một vật thể trong một annotation; giữ ảnh gốc và chỉ số vùng để truy lại mẫu.
@dataclass
class Target:
    split: str
    image_id: int
    annotation_id: int
    region_index: int
    image_path: Path
    width: int
    height: int
    region: dict

    # Tạo ID ổn định để nối manifest, thư mục ảnh và kết quả.
    @property
    def sample_id(self):
        return f"{self.split}_{self.image_id:012d}_a{self.annotation_id}_r{self.region_index:03d}"

    # Kiểm tra visible nằm trong amodal; hidden = amodal − visible.
    def masks(self):
        amodal = decode_mask(self.region["segmentation"], self.height, self.width)
        stored = self.region.get("visible_mask")
        if stored is None:
            if float(self.region.get("occlude_rate", 0)) != 0:
                raise ValueError(f"Occluded target has no visible mask: {self.sample_id}")
            visible = amodal.copy()
        else:
            visible = decode_mask(stored, self.height, self.width)
        if (visible & ~amodal).any():
            raise ValueError(f"Visible mask outside amodal: {self.sample_id}")
        hidden = amodal & ~visible
        if self.region.get("invisible_mask") is not None:
            annotated = decode_mask(self.region["invisible_mask"], self.height, self.width)
            if not np.array_equal(hidden, annotated):
                raise ValueError(f"Invisible mask inconsistent: {self.sample_id}")
        return visible, amodal, hidden


# Mở annotation theo split, lập chỉ mục theo image ID.
class CocoaDataset:
    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split
        settings = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))["splits"][split]
        self.image_dir = self.root / settings["images"]
        self.annotation_path = self.root / settings["annotations"]
        self.data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        images = self.data["images"]
        self.images = {image["id"]: image for image in images}
        if len(self.images) != len(images):
            raise ValueError(f"Duplicate image IDs in {split}")

    # Duyệt từng vật thể trong thứ tự cố định để seed có thể tái lập danh sách.
    def targets(self):
        for annotation in sorted(self.data["annotations"], key=lambda row: (row["image_id"], row["id"])):
            image = self.images[annotation["image_id"]]
            filename = image["file_name"]
            if Path(filename).name != filename:
                raise ValueError("Unexpected relative path inside file_name")
            for index, region in enumerate(annotation["regions"]):
                yield Target(self.split, image["id"], annotation["id"], index,
                             self.image_dir / filename, image["width"], image["height"], region)



from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import random
from PIL import Image, ImageDraw


# Đọc theo khối để tính checksum mà không nạp cả ảnh vào RAM.
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# Đối chiếu tệp ảnh với checksum đã ghi trong manifest dữ liệu.
def check_hash(task):
    path, expected = task
    if sha(path) != expected:
        raise ValueError(f"Image checksum mismatch: {path}")
    return expected


# Kiểm tra split, ảnh thiếu và trùng ID/hash; kiểm tra bytes thật khi bật verify_hashes.
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


# Tô mask để nhìn nhanh vùng được chú thích; không sửa ảnh đầu vào của model.
def overlay(image, mask, color):
    result = image.copy()
    result[mask] = (0.55 * result[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
    return result


# Lọc theo kích thước và tỷ lệ che, rồi lưu RGB/mask cùng hệ tọa độ và manifest.
def export_samples(ds, cfg, out):
    records, panels, used = [], [], set()
    rejected, invalid_examples = Counter(), []
    targets = list(ds.targets())
    if cfg.get("sampling") == "seeded_shuffle":
        random.Random(cfg.get("seed", 42)).shuffle(targets)
    elif cfg.get("sampling", "ordered") != "ordered":
        raise ValueError("sampling must be ordered or seeded_shuffle")
    for target in targets:
        if target.region.get("isStuff", target.region.get("is_stuff", 0)):
            rejected["stuff"] += 1
            continue
        if cfg["one_object_per_image"] and target.image_id in used:
            rejected["same_image"] += 1
            continue
        if not target.region.get("visible_mask"):
            rejected["missing_visible_mask"] += 1
            continue
        try:
            visible, amodal, hidden = target.masks()
        except Exception as error:
            rejected["invalid_annotation"] += 1
            if len(invalid_examples) < 20:
                invalid_examples.append(dict(sample_id=target.sample_id, error=str(error)))
            continue
        ratio = float(hidden.sum() / max(1, amodal.sum()))
        ys, xs = np.where(amodal)
        bbox_side = min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) if len(xs) else 0
        if (visible.sum() < cfg["min_visible_pixels"] or amodal.sum() < cfg["min_amodal_pixels"]
                or bbox_side < cfg.get("min_bbox_side", 0)
                or not cfg["min_hidden_ratio"] <= ratio <= cfg["max_hidden_ratio"]):
            rejected["size_or_occlusion_filter"] += 1
            continue
        if not isinstance(target.region.get("name"), str) or not target.region["name"].strip():
            rejected["missing_name"] += 1
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
        if len(panels) >= cfg.get("preview_limit", 20):
            if len(records) == cfg["limit"]:
                break
            continue
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
    (out / "selection_report.json").write_text(json.dumps(dict(
        selected=len(records), requested=cfg["limit"], rejection_counts=dict(rejected),
        invalid_examples=invalid_examples, policy=cfg,
        note="Rejection counts cover candidates visited until requested subset is filled; not a full-dataset census",
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    if len(records) != cfg["limit"]:
        raise RuntimeError(f"Only {len(records)} eligible targets; requested {cfg['limit']}")
    if panels:
        sheet = Image.new("RGB", (1000, 280 * len(panels)), "white")
        for index, panel in enumerate(panels):
            sheet.paste(panel, (0, index * 280))
        sheet.save(out / "contact_sheet.jpg")
    (out / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records
