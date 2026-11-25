# Phạm vi: Ô 7: tạo lưới ảnh dùng cùng crop và cùng mẫu cho baseline và proposed để so sánh trực quan.
"""Shared square crops for inspection and the reference-set evaluation."""
import random

from PIL import Image, ImageDraw

from src.vision.roi import compute_square_roi
from src.evaluation.runner import read_mask


# Lấy cùng crop vuông theo amodal mask cho tất cả cột của một mẫu.
def object_box(mask):
    roi = compute_square_roi(mask, mask.shape, context_scale=1.2, min_crop=32)
    x, y, side = roi["x0"], roi["y0"], roi["size"]
    return x, y, x + side, y + side


# Giữ tỷ lệ, đệm ngoài khung và ghép ảnh trong suốt lên nền để dễ xem.
def render_tile(path, box, size=224, mask=False):
    """Keep aspect ratio; composite transparent completions on white."""
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        if box is not None:
            x1, y1, x2, y2 = box
            padded = Image.new("RGBA", (x2 - x1, y2 - y1))
            left, top = max(0, x1), max(0, y1)
            right, bottom = min(rgba.width, x2), min(rgba.height, y2)
            if right > left and bottom > top:
                padded.paste(rgba.crop((left, top, right, bottom)), (left - x1, top - y1))
            rgba = padded
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255) if mask else (255, 255, 255, 255))
        background.alpha_composite(rgba)
        tile = background.convert("RGB")
    tile.thumbnail((size, size), Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "black" if mask else "white")
    canvas.paste(tile, ((size - tile.width) // 2, (size - tile.height) // 2))
    return canvas


# Lấy mẫu bằng seed cố định rồi chia lưới năm cột thành từng trang.
def comparison_grid(output, records, jobs, count=20, seed=42):
    chosen = random.Random(seed).sample(records, min(count, len(records)))
    files = []
    labels = ("Input Crop", "Visible Mask", "Given Mask", "Baseline", "Proposed")
    for start in range(0, len(chosen), 4):
        group = chosen[start:start+4]
        grid = Image.new("RGB", (len(labels) * 236, len(group) * 277), "white")
        draw = ImageDraw.Draw(grid)
        for row, record in enumerate(group):
            sid = record["sample_id"]
            first = jobs["baseline"][sid][0]
            box = object_box(read_mask(first / "amodal.png"))
            paths = [first / name for name in ("input.png", "visible.png", "amodal.png")]
            paths += [jobs[key][sid][0] / "completion.png" for key in ("baseline", "proposed")]
            y = row * 277
            caption = f"{sid} | {record.get('name', '')} | hidden {record.get('hidden_ratio', 0):.0%}"
            draw.text((6, y+4), caption, fill="black")
            for col, (path, label) in enumerate(zip(paths, labels)):
                x = col * 236 + 6
                draw.text((x, y+23), label, fill="black")
                grid.paste(render_tile(path, box, mask=col in (1, 2)), (x, y+43))
        name = f"preview_{start//4+1:02d}.jpg"
        grid.save(output / name, quality=92)
        files.append(name)
    return files
