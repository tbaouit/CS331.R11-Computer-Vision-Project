# 🚀 Hướng dẫn chạy AmodalSeg trên Google Colab Pro

## Yêu cầu

- Tài khoản **Google Colab Pro** (hoặc Pro+)
- GPU: **T4 (16GB)** hoặc **A100 (40GB)**

### Môi trường đã kiểm chứng

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| diffusers | 0.37.1 |
| transformers | 5.0.0 |
| GPU test | NVIDIA A100-SXM4-40GB |

---

## Bước 1: Tạo notebook & chọn GPU

1. Vào [Google Colab](https://colab.research.google.com/) → **New Notebook**
2. Chọn **Runtime → Change runtime type**
3. Chọn:
   - **Hardware accelerator**: `GPU`
   - **GPU type**: `T4` hoặc `A100` (nếu có)
4. Nhấn **Save**

---

## Bước 2: Clone repo & cài đặt

### Cell 1 — Clone project

```python
!git clone https://github.com/dpduy123/AP_ProtoSAM_Amodal.git
%cd AP_ProtoSAM_Amodal
```

### Cell 2 — Cài đặt tự động (1 lệnh duy nhất)

```python
%run colab_setup.py
```

Script này sẽ tự động:
- ✅ Kiểm tra GPU
- ✅ Cài dependencies từ `requirements.txt`
- ✅ Cài SAM2 từ GitHub
- ✅ Download SAM2.1 checkpoint (~2.4 GB)
- ✅ Clone mã nguồn Columbia University `pix2gestalt`
- ✅ Download Pix2Gestalt checkpoint (~15.5 GB qua Hugging Face CLI)
- ✅ Auto-detect GPU → chọn settings tối ưu

> ⚠️ **Lưu ý**: Phân đoạn tải 15.5 GB Pix2Gestalt sẽ mất khoảng **3-5 phút**. Vui lòng kiên nhẫn. Colab đã có sẵn `torch`, kịch bản sẽ cấu hình tự động.

---

## Bước 3: Chạy pipeline

### Option A — Chạy trực tiếp trong notebook (khuyến nghị)

```python
import torch
import numpy as np
from PIL import Image
from google.colab import files

# Upload ảnh
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
img = np.array(Image.open(image_path).convert("RGB"))
print(f"Image: {image_path} | Shape: {img.shape}")
```

```python
# ── Bước 1: SAM Segmentation ──
from segmenter import SAMSegmenter

segmenter = SAMSegmenter()
masks = segmenter.segment_everything(img, points_per_side=32)
print(f"✅ Found {len(masks)} masks")

# Giải phóng SAM để dành VRAM cho SD2
del segmenter
torch.cuda.empty_cache()
```

```python
# ── Xem masks ──
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, min(4, len(masks)), figsize=(16, 4))
if len(masks) == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    ax.imshow(img)
    ax.imshow(masks[i]["segmentation"], alpha=0.5, cmap="jet")
    ax.set_title(f"Mask {i} | Area: {masks[i]['area']}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

```python
# ── Bước 2: Amodal Completion ──
from amodal_completer import AmodalCompleter

# ⚠️ Chọn mask_id muốn complete (xem hình ở trên)
MASK_ID = 0  # <-- thay số này

target_mask = masks[MASK_ID]["segmentation"].astype(bool)

completer = AmodalCompleter()
outputs = completer.complete(
    image=img,
    visible_mask=target_mask,
    all_masks=masks,
    max_iter=3,
)
print("✅ Amodal completion done!")
```

```python
# ── Xem kết quả ──
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(outputs["input_image"])
axes[0].set_title("Original")

axes[1].imshow(outputs["visible_mask"], cmap="gray")
axes[1].set_title(f"Visible Mask (#{MASK_ID})")

axes[2].imshow(outputs["amodal_mask"], cmap="gray")
axes[2].set_title("Amodal Mask (Shape)")

rgba_result = outputs["inpainted_rgba"]
axes[3].imshow(rgba_result)
axes[3].set_title("Amodal Result (RGBA)")

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Lưu kết quả
result_pil = Image.fromarray(rgba_result, mode="RGBA")
output_name = f"amodal_result_mask{MASK_ID}.png"
result_pil.save(output_name)
print(f"✅ Saved: {output_name}")

# Download về máy
files.download(output_name)
```

---

### Option B — Dùng quick_test() (nhanh nhất)

Sau khi chạy `colab_setup.py`, dùng hàm `quick_test()` có sẵn:

```python
from google.colab import files

# Upload ảnh
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Chạy toàn bộ pipeline 1 lệnh
rgba, masks = quick_test(image_path, mask_idx=0)

# Download kết quả
files.download(image_path.rsplit(".", 1)[0] + "_amodal.png")
```

---

## Xử lý lỗi thường gặp

### ❌ CUDA Out of Memory

```python
# Giải pháp 1: Clear cache
import torch
torch.cuda.empty_cache()

# Giải pháp 2: Giảm SAM grid density
masks = segmenter.segment_everything(img, points_per_side=16)  # giảm từ 32 → 16

# Giải pháp 3: Restart runtime
# Runtime → Restart runtime → chạy lại từ đầu
```

### ❌ Module not found

```python
# Đảm bảo đang ở đúng thư mục
import os
print(os.getcwd())  # Phải là /content/AP_ProtoSAM_Amodal

# Nếu sai, chạy:
%cd /content/AP_ProtoSAM_Amodal
```

### ❌ SAM config not found (MissingConfigException)

```python
# Đảm bảo SAM2 đã cài đúng
!pip install -q git+https://github.com/facebookresearch/sam2.git
```

### ❌ SAM checkpoint not found

```python
# Download lại SAM2.1 checkpoint
!wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### ❌ Cập nhật code mới từ GitHub

```python
# KHÔNG cần clone lại, chỉ cần pull
!git pull
```

---

## Bảng VRAM tham khảo

| Bước | VRAM sử dụng | Ghi chú |
|------|:---:|---------|
| SAM2 segment | ~6 GB | Giải phóng sau khi segment xong |
| Pix2Gestalt | ~9 GB | Sinh Amodal Shape Mask (Tự động fallback về Heuristics nếu lỗi/thiếu VRAM) |
| SD2 inpainting | ~5 GB | Đã bật attention slicing. Dùng để inpaint Appearance |
| **Peak (có optimize)** | **~15 GB** | Nhờ load tuần tự: SAM → free → Pix2Gestalt → SD2 |
| **Peak (không optimize)** | **~28 GB** | Nếu giữ SAM, Pix2Gestalt và SD2 đồng thời (chỉ dùng cho A100) |

> 💡 **Tip**: Mọi quá trình đều đã được tự động hóa. Nếu Pix2Gestalt làm văng VRAM (trên T4 16GB), hệ thống sẽ **KHÔNG CRASH** mà tự động ngắt và lả lướt chuyển về thuật toán **Heuristic Predictor** nhẹ tênh (khoanh vùng Amodal Mask bằng toán học).

---

## Tối ưu cho từng loại GPU

| GPU | Khuyến nghị |
|-----|-------------|
| **T4 (16GB)** | Rất thiếu VRAM. Pipieline có thể tự động nhảy về Heuristic Predictor thay vì Pix2Gestalt. |
| **A100 (40GB)** | Khuyến nghị tối thượng! Đủ chứa 15 GB của Pix2Gestalt và SD2 cùng lúc. Chất lượng sinh chân hoàn hảo. |
| **V100 (16GB)** | Nhanh hơn T4 ~20% nhưng vẫn thiếu VRAM giống T4. |
