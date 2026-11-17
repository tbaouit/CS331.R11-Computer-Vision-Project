# CS331 — Amodal Appearance Completion trên COCO-A

Ba phiên bản nhận cùng ảnh RGB, visible mask và mask vùng khuất cho sẵn. Chúng
dùng chung Stable Diffusion 1.5 Inpainting, DDIM và seed 42. Mục tiêu là sinh
diện mạo hợp lý trong vùng khuất; hình dạng đầu ra được cố định bởi mask.

| Phiên bản | Cấu hình | Luồng |
| --- | --- | --- |
| V1 Direct | `configs/v1_sd15_direct.json` | Prompt cố định, một lượt inpainting, giữ nguyên pixel nhìn thấy |
| V2 Pipeline | `configs/v2_sd15_pipeline.json` | Qwen cơ bản → Pix2Gestalt/CarveKit → CLIP → tinh chỉnh SD1.5 → critic tổng hợp |
| V3 Proposed | `configs/v3_sd15_proposed.json` | Qwen micro-attributes, IP-Adapter Plus, latent blending và strict critic |

Mở [notebook Colab](notebooks/Amodal_Completion_Colab.ipynb). Có thể chia việc
qua nhiều phiên: chạy **ô 1–4** rồi chọn **ô 5 (V1), 6 (V2) hoặc 7 (V3)**.
Ảnh, manifest, trạng thái profile và điểm được lưu trên Drive. Giữ nguyên
`RUN_NAME` và `BATCH` để tiếp tục. Khi cả ba xong, có thể mở một phiên chỉ
đánh giá: đặt `EVALUATION_ONLY = True`, chạy **1–4 → 8**.

Ô 4 mặc định `dev50`; đặt `SMOKE_LIMIT = 3` để thử nhanh. Sau khi chốt cấu hình
trên dev, dùng `BATCH = "test400"`, `FINAL_TEST = True` và `SMOKE_LIMIT = None`.

- [Cài đặt Colab](docs/COLAB_SETUP.md)
- [Luồng xử lý và phân công module](docs/PIPELINE.md)
- [V3 và ablation](docs/PIPELINE_V3.md)
- [Protocol đánh giá](docs/EVALUATION.md)

COCO-A không cấp RGB thật của phần khuất. CLIP, SeamMAE và histogram màu là
chỉ số gián tiếp; PSNR, SSIM, LPIPS và Feature Similarity chỉ kiểm tra phần
nhìn thấy. Mã, cấu hình và tài liệu lịch sử có thể xem ở commit `9c3cfb8`.
