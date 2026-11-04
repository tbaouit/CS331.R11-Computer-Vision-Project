# Pipeline và khác biệt so với repo tác giả

Nguồn: https://github.com/dpduy123/SCAmodal, commit 92e5b392cba1c767f4ddc08497f6b4450743929a.
upstream/SCAmodal giữ nguyên các source/docs của bản clone. upstream/provenance.json ghi SHA-256.
runtime/SCAmodal là bản được patch có kiểm soát, không sửa repo D:/SCAmodal.

## Luồng thực thi

1. Ảnh RGB + visible mask COCO-A (thay đầu ra SAM2 khi đánh giá).
2. Qwen3-VL-4B-Instruct mô tả phần bị che.
3. Pix2Gestalt sinh ảnh đầy đủ; CarveKit trích mask, hậu xử lý và đưa về kích thước gốc.
4. Missing = amodal AND NOT visible. Nhánh outpainting giữ mặc định của tác giả.
5. CLIP xác minh/chọn prompt từ Qwen theo _select_prompt của repo. Mặc định text_query rỗng như evaluator gốc; có thể chọn dataset_object_name trong cấu hình, khi đó phải ghi nhận thay đổi này.
6. SD2 inpaint trên canvas xám. Giữ cơ chế resize/blending của tác giả.
7. Qwen critic chấm cấu trúc/texture/context. Nếu chưa đạt ngưỡng, sửa prompt và sinh lại.
8. Xuất RGBA và các mask. Tối đa 2 lần sinh, ngưỡng critic 7 mặc định theo commit.

Không dùng RAG, không dùng mask_only và không tắt critic trong entry point này.
Không tự đưa mô tả Qwen vào Pix2Gestalt: commit gốc nhận ảnh và visible mask ở nhánh hình dạng.
Critic chỉ làm lại inpainting; mask hình dạng không được dự đoán lại trong vòng lặp gốc.
Qwen reason_occlusion của repo không highlight visible mask; hạn chế nhận diện mục tiêu trong ảnh nhiều vật thể được giữ và ghi nhận.

## Các patch có chủ đích

- Tắt heuristic dilation fallback. Nếu Pix2Gestalt không load/chạy được, mẫu thất bại rõ ràng.
- Nếu Qwen không load được, dừng thay vì đưa chuỗi lỗi vào prompt.
- Critic thiếu tiêu chí hoặc điểm ngoài 0–10: báo lỗi thay vì tự cho điểm 5.
- Lưu RGB Pix2Gestalt và alpha CarveKit để kiểm tra hình dạng.
- Tương thích Pillow ANTIALIAS và import rank_zero_only của legacy LDM.
- SSIM tùy chọn dùng data_range=255 rõ ràng cho ảnh uint8.
- Không chỉnh hình dạng, prompt, phương pháp inpainting hoặc critic loop gốc.

## Môi trường

Core bám tài liệu: Python3.12, Torch2.10/CUDA12.8, Diffusers0.37.1, Transformers5.0.
Để tương thích legacy LDM: NumPy1.26.4, OpenCV4.10, Lightning1.9.5, TorchMetrics0.11.4,
Kornia0.6.12, CarveKit4.1.0; khác các pin trong requirements gốc và đã ghi rõ ở requirements-colab.txt.
Các pin này mới là cấu hình tích hợp cần kiểm chứng trên Colab, chưa phải môi trường đã xác nhận chạy GPU.
Không dùng .venv-data trên Windows để nạp model.

## Đánh giá

Am​odal IoU và hidden IoU tính trên khung ảnh gốc; nếu outpainting mở canvas, crop đúng padding đã lưu.
IoU rỗng cả hai phía trả null, không tự tính là 0 hoặc 1.
Visible MAE và tỷ lệ pixel giữ nguyên được báo cáo; blending gốc không bảo đảm giữ mọi pixel biên.
Khi bật appearance_metrics trong configs/scamodal.json, bổ sung 4 metric theo policy ghi trong result.json:
CLIP là logit ảnh/text; LPIPS/CLIP feature similarity/SSIM so sánh RGB trong visible mask, ngoài mask là nền đen.
Đây là chỉ số nhất quán phần nhìn thấy, không chứng minh phần khuất RGB đúng.
Không mặc nhiên so sánh trực tiếp số với slide của bài báo, vì subset/preprocessing chưa được xác minh khớp.

## Trạng thái chạy và tiếp tục

Mỗi mẫu có signature phụ thuộc config, version code, input và visible mask. Kết quả hoàn thành có hash artifact.
Chạy lại cùng cấu hình/manifest sẽ bỏ qua mẫu đã hoàn thành và artifact còn nguyên.
Chỉ có result.json status=completed mới được resume; mẫu lỗi có failure.json.
completed nghĩa là đã sinh và xuất ảnh, không có nghĩa critic đã chấp nhận: xem critic_accepted.
Backup ZIP sau từng mẫu nếu truyền --backup-dir; vẫn có thể mất mẫu đang chạy nếu runtime bị xóa đột ngột.
Không chạy hai notebook ghi cùng outputs cùng lúc.


## Chế độ cải tiến (`pipeline_mode = "improved"`)

Giữ nguyên toàn bộ byte của `upstream/SCAmodal` và `runtime/SCAmodal` (chế độ gốc được giữ tại `configs/scamodal_baseline.json` với `pipeline_mode = "upstream_baseline"` để đối chiếu ablation). Cấu hình mặc định `configs/scamodal.json` kích hoạt `src/scamodal_improved.py` (`ImprovedAmodalCompleter`), khắc phục 4 lỗi hình dạng và màu sắc của bản gốc mà không dùng thêm trọng số huấn luyện và không chạm tới ground-truth mask:

1. **Context-Aware Square ROI + Multi-Sample Pix2Gestalt Consensus + Smooth Alpha Matting**:
   - Crop cửa sổ vuông có ngữ cảnh quanh `visible_mask` (`roi_context_scale = 3.8`, tối thiểu `256px`, tự động dùng toàn khung khi vật thể lớn `>= 85%` khung) giúp các vật thể nhỏ giữ độ phân giải thực thay vì bị nén xuống vài pixel trong khung `256x256`.
   - Lấy mẫu đa phương án (`pix2gestalt_samples = 3`), lọc mẫu ngoại lai nở bất thường, nội suy tuyến tính kênh `alpha` của CarveKit về kích thước gốc trước khi ngưỡng hóa (`alpha >= 0.48` ở biên ngoài thay vì `alpha > 0` gây quầng xám/tai mèo, kết hợp khôi phục khe lõm nội vùng có `alpha >= 0.12`), và giữ thành phần liên thông bám `visible_mask`.
2. **Mask-Guided Visual Prompting cho Qwen3-VL**:
   - Truyền đồng thời ảnh ngữ cảnh cục bộ (vùng `visible_mask` sáng 100% có viền xanh lá, vùng khuất dự đoán viền vàng, các vật che khuất bị làm tối) và ảnh tách riêng `visible_mask` trên nền trắng vuông, buộc Qwen xuất cấu trúc `OBJECT / OCCLUDED_PARTS / PROMPT` đúng vật thể mục tiêu (tránh mô tả nhầm vật che khuất như hot dog, xe máy, ván trượt) kết hợp CLIP kiểm chứng trên khung vuông.
3. **Hybrid Pix2Gestalt RGB Prior + 512x512 ROI SD2 Refinement trên nền trắng**:
   - Tắt outpainting làm phình biên xám vô ích; dùng chính RGB tổng hợp của Pix2Gestalt sau khi cân bằng màu (gain/bias theo `visible_mask`), khử viền trắng và vá lỗ khuyết trắng nội vùng bằng Telea làm khởi tạo cho SD2 Inpainting trên ROI `512x512` (`strength = 0.42` khi prior đầy đủ, `0.78` khi có vùng khuyết trắng), kèm hòa trộn biên thích nghi `1-3 px` giữ nguyên gần như tuyệt đối pixel vùng nhìn thấy.
4. **Target-Aware Cropped Critic & Best-of-N**:
   - Crop sát vật thể hoàn thiện trên nền trắng gửi cho Qwen Critic kèm tên vật thể mục tiêu và luôn giữ lại phương án có điểm Critic cao nhất qua các lần thử.
   - Dùng `scripts/compare_runs.py` để xuất bảng định lượng và lưới ảnh đối chiếu trực tiếp Baseline vs Improved (`outputs/scamodal/comparison_grid.jpg`).


## Chế độ chuyên biệt nhận thẳng Mask phần bị che (`pipeline_mode = "oracle_mask"`)

Cấu hình `configs/scamodal_oracle_mask.json` kích hoạt pipeline riêng `src/scamodal_oracle.py` (`OracleMaskAmodalCompleter`), nhận trực tiếp `occluded_mask` (`gt_hidden`) từ input cùng với `image` và `visible_mask` (thiết lập *Mask-Guided / Oracle-Mask Amodal Appearance Completion*):

1. **Exact-Target Amodal ROI + Fast 3D RGB Prior**: Căn cửa sổ vuông trực tiếp quanh `amodal_mask = visible_mask | occluded_mask` (`oracle_roi_scale = 1.65`, tối thiểu `176 px`) và chỉ chạy 1 mẫu Pix2Gestalt (`ddim_steps = 75`) để lấy tham chiếu ánh sáng/kết cấu toàn vật thể mà không dùng Pix2Gestalt để đoán biên mask (`Amodal IoU = 1.0`, `Hidden IoU = 1.0`).
2. **Multi-Cue Boundary + Prior Seed (`build_oracle_appearance_seed`)**: Kết hợp lan truyền biên Telea sạch vật cản từ `image[visible_mask]` với prior màu đã hiệu chỉnh tương phản; tự động khử vệt bóng tối của vật cản trên bề mặt sáng đồng nhất (như lòng hộp xốp trắng / đĩa sứ) và lấp đầy các vùng khuất mảnh mà Pix2Gestalt bỏ sót (như chân công trình hoặc cán vợt).
3. **Silhouette-Blueprint Visual Prompting (`build_oracle_vlm_prompts`)**: Cung cấp cho Qwen3-VL bản vẽ silhouette màu xanh nhạt của đúng vùng `occluded_mask` cạnh phần `visible_mask` trên nền trắng (cùng ảnh ngữ cảnh đã che sọc vật cản), giúp VLM nhận biết chính xác bộ phận giải phẫu/cấu trúc cần khôi phục.
4. **Dual-Zone 512x512 SD2 Refinement & 3-Way Comparison**: Khóa tần số thấp chặt trên vùng có prior tốt và nới biên tổng hợp chi tiết trên vùng `unfilled_mask`; `scripts/compare_runs.py` tự động xuất bảng và lưới ảnh đối chiếu 3 chiều (`Baseline -> Improved -> Oracle-Mask`) kèm chỉ số liên tục biên `SeamMAE` và tương đồng màu `ColorHistSim`.
