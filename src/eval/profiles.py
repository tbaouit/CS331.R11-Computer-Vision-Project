# Phạm vi: Ô 7: xác minh các kết quả được chấm từ đúng manifest và đúng ảnh/mask đã lưu trước khi
# ghép bảng.
"""Verify saved scores against job artifacts and the requested manifest."""
import hashlib
import json
from pathlib import Path

from src.evaluation.runner import sha


# Đọc JSON bằng UTF-8 để giữ tên mẫu và thông tin annotation.
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Băm toàn bộ danh sách record; thay một mẫu sẽ đổi khóa so sánh.
def subset_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:12]


# Tìm bộ điểm của đúng subset và kiểm tra mỗi mẫu xuất hiện đúng một lần.
def load_evaluation(profile, records):
    candidates = list(profile.glob(f"evaluation/*/{subset_hash(records)}/per_sample.json"))
    if not candidates:
        raise ValueError(f"No evaluation for this exact manifest in {profile.name}; run evaluate_runs.py first")
    report = max(candidates, key=lambda p: p.stat().st_mtime)
    rows = read_json(report)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != {r["sample_id"] for r in records}:
        raise ValueError(f"Incomplete/duplicate evaluation rows: {report}")
    return report.parent, {r["sample_id"]: r for r in rows}, read_json(report.parent / "protocol.json")


# Kiểm tra checksum ảnh/mask và signature của bảng điểm trước khi so sánh.
def verified_jobs(profile, records, directory, rows):
    jobs = {}
    for path in sorted(profile.glob("*/*/result.json"), key=lambda p: p.stat().st_mtime):
        result = read_json(path)
        if result.get("status") == "completed":
            jobs[result["sample_id"]] = (path.parent, result)
    selected = {}
    evaluation_id = directory.parent.name
    for record in records:
        sid = record["sample_id"]
        if sid not in jobs:
            raise ValueError(f"Missing generation: {profile.name}/{sid}")
        job, result = jobs[sid]
        if read_json(job / "sample.json") != record:
            raise ValueError(f"Manifest changed since generation: {sid}")
        names = ("input.png", "visible.png", "completion.png", "amodal.png", "occluded_input.png")
        hashes = {name: sha(job / name) for name in names}
        if any(result.get("artifact_sha256", {}).get(name) != value for name, value in hashes.items()):
            raise ValueError(f"Artifact checksum mismatch: {profile.name}/{sid}")
        signature = hashlib.sha256(json.dumps(dict(evaluation_id=evaluation_id,
            generation=result["signature"], artifacts={k: v for k, v in hashes.items() if k != "occluded_input.png"},
            record=record), sort_keys=True).encode()).hexdigest()
        sidecar = read_json(job / f"appearance_{evaluation_id}.json")
        if sidecar.get("signature") != signature or sidecar.get("row") != rows[sid]:
            raise ValueError(f"Stale evaluation for {sid}; rerun evaluate_runs.py")
        selected[sid] = (job, result, hashes)
    return selected
