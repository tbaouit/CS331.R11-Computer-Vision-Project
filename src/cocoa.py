"""COCO-A image annotations -> individual objects. No GPU/model dependencies."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask


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

    @property
    def sample_id(self):
        return f"{self.split}_{self.image_id:012d}_a{self.annotation_id}_r{self.region_index:03d}"

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

    def targets(self):
        for annotation in sorted(self.data["annotations"], key=lambda row: (row["image_id"], row["id"])):
            image = self.images[annotation["image_id"]]
            filename = image["file_name"]
            if Path(filename).name != filename:
                raise ValueError("Unexpected relative path inside file_name")
            for index, region in enumerate(annotation["regions"]):
                yield Target(self.split, image["id"], annotation["id"], index,
                             self.image_dir / filename, image["width"], image["height"], region)
