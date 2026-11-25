"""Prepare nested 10, 50 or full batches (1 target object per image entry, matching Open-World Amodal)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cocoa_loader import Target, export_target


def _norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _region_match_score(region: dict, hint_norm: str, is_exact: bool) -> tuple[int, int, float]:
    """Rank regions for an image entry: exact match > substring/token match > occluded largest area."""
    name_norm = _norm_label(region.get("name", ""))
    occ_rate = float(region.get("occlude_rate") or 0.0)
    is_occluded = 1 if occ_rate > 0.0 else 0
    area = float(region.get("area") or 0.0)
    if is_exact or (hint_norm and name_norm == hint_norm):
        tier = 3
    elif hint_norm and name_norm and (name_norm in hint_norm or hint_norm in name_norm):
        tier = 2
    else:
        tier = 1
    return (tier, is_occluded, area)


def prepare(root, labels, batch, output):
    data = json.loads(labels.read_bytes())
    images = {im["id"]: im for im in data["images"]}
    # Pick at most ONE best-matching occluded target object per image entry (matching author's 845-object protocol)
    selected_entries = []
    used_keys: set[tuple] = set()
    # Process entries with exact_candidates first so they claim their exact region before fallback entries
    ordered_anns = sorted(
        data["annotations"],
        key=lambda a: (0 if images[a["image_id"]].get("exact_candidates") else 1, a["id"]),
    )
    for ann in ordered_anns:
        im = images[ann["image_id"]]
        path = (root / im["file_name"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid image: {path}")
        source_ann = ann.get("source_annotation_id", ann["id"])
        exact = im.get("exact_candidates") or []
        hint_norm = _norm_label(im.get("object_hint", ""))
        ranked = []
        for idx, region in enumerate(ann["regions"]):
            if not region.get("visible_mask") or not region.get("name", "").strip():
                continue
            is_exact = [source_ann, idx] in exact
            score = _region_match_score(region, hint_norm, is_exact)
            key = (im["split"], im["source_filename"], source_ann, idx)
            ranked.append((score, key, idx, region, is_exact))
        ranked.sort(key=lambda item: item[0], reverse=True)
        for score, key, idx, region, is_exact in ranked:
            if key in used_keys:
                continue
            target = Target(im["split"], im["id"], ann["id"], idx, path,
                            im["width"], im["height"], region)
            used_keys.add(key)
            tag = "exact_match" if score[0] == 3 else ("hint_substring_match" if score[0] == 2 else "occluded_fallback")
            selected_entries.append((target, tag))
            break

    pool = selected_entries
    random.Random(42).shuffle(pool)
    records, rejected = [], []
    limit = None if batch == "full" else int(batch)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staging = Path(tmp) / "batch"
        staging.mkdir()
        for target, tag in pool:
            try:
                visible, amodal, hidden = target.masks()
                if not visible.any() or not hidden.any():
                    raise ValueError("empty visible or hidden mask")
            except (ValueError, TypeError, KeyError, IndexError) as error:
                rejected.append(dict(sample_id=target.sample_id, reason=str(error)))
                continue
            record, _ = export_target(target, staging, visible, amodal, hidden)
            record["selection"] = tag
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
