# Chạy COCO-A trên Colab

Upload `cocoa.zip` vào `MyDrive/CS331/`. Bên trong ZIP, đặt `cocoa/cocoa_annotations.json` cạnh `cocoa/dataset.json`, `filename_mapping.csv` và các thư mục split.

1. Chạy ô 1–3 với `RUN_NAME = "cocoa_embedded_sd15"` cho đợt mới. Ô 2 giải nén vào bộ nhớ local của Colab.
2. Ở ô 4, chọn `BATCH = 10`, `BATCH = 50` hoặc `BATCH = "full"` (mặc định). Ô này xuất danh sách mẫu, kiểm tra mask và in số vật thể thực tế. Hai batch nhỏ là tiền tố của cùng danh sách xáo trộn với seed 42.
3. Ô 5 sinh baseline, ô 6 sinh proposed. Có thể chạy ở hai phiên riêng. Sau khi ngắt, chạy lại ô 1–4 với cùng tên và cùng batch rồi chạy ô còn dang dở.
4. Ô 7 đánh giá cả hai phương pháp từ kết quả lưu trên Drive. Để chỉ chấm trong phiên mới, đặt `EVALUATION_ONLY = True` ở ô 1, chạy ô 1–4 rồi ô 7.

Mỗi batch lưu manifest, báo cáo chọn mẫu, tiến độ và kết quả tại `MyDrive/CS331/results/<RUN_NAME>/batch_<BATCH>/`. Tiến độ được lưu riêng cho từng batch.

## Annotation và đơn vị chạy

`cocoa_annotations.json` gồm `images` và `annotations` của cả train/dev/test. Mỗi ảnh có đường dẫn tương đối, split, tên gốc và danh sách vùng ứng viên khớp hậu tố. Mỗi annotation giữ các vùng và mask riêng. ID annotation mới là duy nhất trong toàn tập. Trường `source_annotation_id` lưu ID gốc dùng trong bảng ánh xạ.

Một mẫu suy luận là một vùng vật thể có cả pixel thấy được và bị che. Khi hậu tố khớp tên vùng, lấy các vùng ứng viên khớp. Khi không có tên khớp, duyệt các vùng có mask hợp lệ của cảnh và ghi `selection = unmatched_name_scene_region` trong manifest. Các bản sao cùng cảnh và cùng vùng chỉ tính một lần. Full bao gồm cả ba split và phù hợp cho đánh giá mô tả trên tập đã chọn. Số ảnh, số vật thể và số file có hậu tố là các đại lượng khác nhau.

Có thể tái tạo file gộp bằng `python scripts/merge_cocoa_annotations.py cocoa.zip cocoa_annotations.json`.
