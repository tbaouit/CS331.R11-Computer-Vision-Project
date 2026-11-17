# Chạy và tiếp tục trên Colab

Đặt `cocoa_dataset.zip` tại `MyDrive/CS331/`. Dùng GPU; L4 High-RAM là điểm
khởi đầu đã dùng cho thí nghiệm đầy đủ. Phiên chỉ đánh giá không cần giải nén
dataset hoặc tải checkpoint của model sinh ảnh.

1. **Ô 1:** chọn `RUN_NAME`, ví dụ `"sd15"`, và mount Drive. Giữ nguyên tên
   này qua các phiên thuộc cùng một đợt thí nghiệm. Dùng tên riêng cho mỗi đợt
   muốn chạy với code hoặc config mới.
2. **Ô 2–3:** đồng bộ repo, tạo môi trường và tải trọng số. Notebook cố định
   commit Git từ lần chạy đầu để những phiên tiếp theo dùng cùng mã nguồn.
   Nếu chỉ chạy V1 trong phiên này, có thể đặt
   `PRELOAD_GENERATION_WEIGHTS = False` ở ô 3 để bỏ tải Pix2Gestalt và các model
   của V2/V3. SD1.5 sẽ tải khi V1 chạy lần đầu.
3. **Ô 4:** chọn batch. Để chạy 400 mẫu, dùng `BATCH = "test400"`,
   `FINAL_TEST = True`, `SMOKE_LIMIT = None`. Ô này lưu manifest và hiện trạng
   thái của từng phiên bản: `chưa chạy`, `running`, `completed`.
4. Chạy **ô 5 cho V1**, **ô 6 cho V2**, **ô 7 cho V3**. Có thể chạy một ô mỗi
   phiên. Nếu Colab ngắt, mở phiên mới, chạy **1–4** với cùng lựa chọn rồi chạy
   lại ô đang dang dở. Các mẫu đã ghi đầy đủ và đúng checksum được bỏ qua.
   Tránh mở hai phiên ghi vào cùng một phiên bản cùng lúc.
5. Khi cả ba phiên bản đã xong, chạy **ô 8**. Nếu dùng phiên mới chỉ để chấm,
   đặt `EVALUATION_ONLY = True` ở ô 1, rồi chạy **1–4 → 8**. Ô 8 chấm từ ảnh đã
   lưu, không chạy sinh ảnh. Nếu ngắt giữa lúc chấm, lặp lại lộ trình này; các
   điểm đã ghi được dùng lại khi ảnh, manifest và protocol chấm còn khớp.

Để chạy ablation tắt adapter, bật `RUN_NO_IP_ABLATION` ở ô 7. Chỉ bật
`EVALUATE_ABLATION` ở ô 8 khi muốn đưa kết quả này vào báo cáo.

Kết quả nằm trong `MyDrive/CS331/results/<RUN_NAME>/<BATCH>_<N>/` (hoặc thư
mục Drive được in ở ô 1). `profiles/` lưu trạng thái mỗi phiên bản,
`runs/<profile>/` lưu ảnh, log và điểm; `comparisons/` lưu ZIP báo cáo cuối.
Notebook mới ghi trực tiếp từng mẫu vào Drive. Các ZIP từ notebook cũ
vẫn dùng được với CLI khôi phục; chúng không tự được chọn thay profile
của đợt chạy mới.

Nếu resume báo đổi profile, kiểm tra lại môi trường hoặc config. Pipeline
giữ kết quả cũ và dừng thay vì trộn hai profile. Dùng `RUN_NAME` mới nếu
muốn chạy với môi trường hoặc config khác.
