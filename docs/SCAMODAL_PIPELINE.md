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
