# Phạm vi: Ô 7, tùy chọn: FID trên phân bố crop vật thể, cần bộ ảnh sạch bên ngoài. Không có RGB
# ground truth vùng ẩn trong COCO-A.
"""Distribution-level FID against a separate set of clean object crops."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from src.eval.visualize import object_box, render_tile
from src.evaluation.runner import read_mask


# Đưa ảnh sạch và ảnh sinh về crop cùng định dạng rồi so thống kê Inception 192 chiều.
def evaluate_fid(records, jobs, reference, device):
    if len(records) < 400:
        raise ValueError("Run FID on at least 400 completed objects")
    files = sorted(p for p in Path(reference).iterdir() if p.is_file()
                   and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if len(files) < 400:
        raise ValueError("Provide at least 400 clean object reference crops")
    # 192-dimensional Inception features keep covariance estimation viable at N=400.
    # This is a distinct protocol from standard 2048-dimensional pool3 FID.
    from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3

    dims = 192
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(device).eval()
    fingerprint = sha256()
    with TemporaryDirectory(prefix="amodal_fid_") as directory:
        root = Path(directory)
        def stats(paths):
            return calculate_activation_statistics(paths, net, batch_size=16, dims=dims,
                                                   device=device, num_workers=0)
        refs = []
        for index, source in enumerate(files):
            target = root / f"ref_{index}.png"
            tile = render_tile(source, None)
            fingerprint.update(tile.tobytes())
            tile.save(target)
            refs.append(target)
        ref_mean, ref_cov = stats(refs)
        scores = {}
        for key, selected in jobs.items():
            outputs = []
            for index, record in enumerate(records):
                job = selected[record["sample_id"]][0]
                target = root / f"generated_{index}.png"
                box = object_box(read_mask(job / "amodal.png"))
                render_tile(job / "completion.png", box).save(target)
                outputs.append(target)
            mean, cov = stats(outputs)
            scores[key] = float(calculate_frechet_distance(ref_mean, ref_cov, mean, cov))
    return dict(protocol="pytorch-fid 0.3.0 / Inception 192-d / 224px white square crops",
                reference_count=len(files), reference_sha256=fingerprint.hexdigest(),
                generated_count=len(records), scores=scores)
