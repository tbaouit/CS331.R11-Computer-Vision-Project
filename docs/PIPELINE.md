# Kiến trúc Pipeline Amodal Appearance Completion

## 1. Phân tầng module (`src/`)

- `src/data/cocoa.py`: Đọc dữ liệu COCO-A và giải mã mask polygon/RLE.
- `src/models/`: Nạp một lần và chia sẻ trọng số của 4 mô hình nền tảng (`Pix2GestaltModel`, `QwenVLModel`, `StableDiffusionInpaintPipeline`, `CLIPModel`) thông qua `FoundationModels`.
- `src/vision/`: Các phép toán hình thái học nhị phân (`image_ops.py`), tính toán cửa sổ vuông ngữ cảnh (`roi.py`), khóa tần số thấp và hòa trộn biên thích nghi (`compositing.py`).
- `src/stages/`:
  - `shape_consensus.py`: Chuẩn bị đầu vào ROI cho Pix2Gestalt và hợp nhất đa mẫu kênh alpha CarveKit.
  - `appearance_prior.py`: Xây dựng bản khởi tạo màu sắc/ánh sáng vùng khuất cho chế độ Blind (`build_appearance_prior`) và Oracle-Mask (`build_oracle_appearance_seed`).
  - `vlm_guidance.py`: Xây dựng visual prompt cho Qwen3-VL, phân tích cấu trúc đầu ra, kiểm chứng bằng CLIP và chấm điểm Critic.
  - `sd2_refiner.py`: Tinh chỉnh chi tiết bằng Stable Diffusion 2 Inpainting trên cửa sổ `512x512`.
- `src/pipelines/`: `AmodalBackend`, `ImprovedAmodalCompleter`, và `OracleMaskAmodalCompleter`.
- `src/evaluation/`: Đánh giá IoU, `SeamMAE`, `ColorHistSim`, `appearance_metrics.py`, thực thi có kiểm tra SHA-256 resume (`runner.py`) và đối chiếu đa chế độ (`comparison.py`).

## 2. Chế độ nhận thẳng Mask phần bị che (`pipeline_mode = "oracle_mask"`)

Cấu hình `configs/oracle_mask.json` kích hoạt `OracleMaskAmodalCompleter`, nhận trực tiếp `occluded_mask` (`gt_hidden`) từ input cùng với `image` và `visible_mask`:
1. **Exact-Target Amodal ROI + Fast 3D RGB Prior**: Căn cửa sổ vuông trực tiếp quanh `amodal_mask = visible_mask | occluded_mask` (`oracle_roi_scale = 1.65`) và chạy 1 mẫu Pix2Gestalt (`ddim_steps = 75`) để lấy tham chiếu ánh sáng/kết cấu toàn vật thể (`Amodal IoU = 1.0`, `Hidden IoU = 1.0`).
2. **Multi-Cue Boundary + Prior Seed (`build_oracle_appearance_seed`)**: Kết hợp lan truyền biên Telea sạch vật cản từ `image[visible_mask]` với prior màu đã hiệu chỉnh tương phản; tự động khử vệt bóng tối của vật cản trên bề mặt sáng đồng nhất và lấp đầy các vùng khuất mảnh.
3. **Silhouette-Blueprint Visual Prompting (`build_oracle_vlm_prompts`)**: Cung cấp cho Qwen3-VL bản vẽ silhouette màu xanh nhạt của đúng vùng `occluded_mask` cạnh phần `visible_mask` trên nền trắng.
4. **Dual-Zone 512x512 SD2 Refinement & Best-of-N Critic**: Khóa tần số thấp chặt trên vùng có prior tốt, nới biên tổng hợp chi tiết trên vùng `unfilled_mask`, và chọn phương án có điểm Critic cao nhất.

## 3. Chế độ tự đoán Mask (`pipeline_mode = "improved"`)

Cấu hình `configs/improved.json` kích hoạt `ImprovedAmodalCompleter`:
1. **Context-Aware Square ROI + Multi-Sample Pix2Gestalt Consensus**: Lấy mẫu đa phương án trên cửa sổ vuông quanh `visible_mask`, lọc mẫu ngoại lai và nội suy tuyến tính kênh `alpha` của CarveKit trước khi ngưỡng hóa.
2. **Mask-Guided Visual Prompting cho Qwen3-VL + CLIP Verification**: Truyền đồng thời ảnh ngữ cảnh cục bộ đã làm tối vật cản và ảnh tách riêng `visible_mask` trên nền trắng.
3. **Hybrid Pix2Gestalt RGB Prior + 512x512 ROI SD2 Refinement**: Dùng RGB tổng hợp của Pix2Gestalt sau khi cân bằng màu và vá lỗ khuyết làm khởi tạo cho SD2 Inpainting kèm hòa trộn biên thích nghi `1-3 px`.
