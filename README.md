# CS331 — Amodal Completion

Đồ án khôi phục hình dạng và diện mạo 2D của vật thể bị che khuất trong ảnh tĩnh, đánh giá trên COCO-A.

## Trạng thái hiện tại

Đã có dữ liệu sạch, bộ đọc CPU và runtime full SCAmodal có strict fallback, trace, resume và notebook Colab. Chưa xác nhận suy luận bằng trọng số thật trên GPU.

Xem docs/COLAB_SETUP.md để chạy. Notebook: notebooks/SCAmodal_Colab.ipynb. Code được clone/pull trực tiếp từ GitHub trên Colab; dataset giữ trên Google Drive tại `MyDrive/amodal_project/cocoa_dataset.zip`.

Pipeline: Qwen reasoning → Pix2Gestalt/CarveKit → SD2 inpainting → Qwen critic. Khác biệt so với repo gốc ghi trong docs/SCAMODAL_PIPELINE.md và upstream/provenance.json.
Repo D:/SCAmodal là nguồn tham khảo; upstream/ lưu bản gốc, runtime/ chứa bản patch được runner nạp. Không thay đổi repo nguồn D:/SCAmodal.

## Cấu trúc

- dataset/cocoa/: train 2.500 ảnh, dev 1.323 ảnh, test 1.250 ảnh; annotation JSON gốc theo từng tập.
- configs/data.json: đường dẫn và quy tắc chọn mẫu train để thử chương trình.
- src/cocoa.py: bộ đọc annotation theo ảnh và đối tượng, giải mã polygon/RLE.
- scripts/prepare_data.py: kiểm tra dataset và xuất mẫu thử.
- tests/test_cocoa.py: kiểm tra giải mã mask và quy tắc xử lý nhãn.
- outputs/train_smoke/: ảnh/mask/manifest và hình tổng hợp sau khi chạy.

## Bước 1 — chạy trong VS Code trên Windows

Mở terminal PowerShell ngay tại thư mục dự án:

```powershell
python -m venv .venv-data
.\.venv-data\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv-data\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-data\Scripts\python.exe scripts/prepare_data.py --verify-hashes
```

Không cần activate khi gọi Python bằng đường dẫn như trên. Môi trường này chỉ dành cho dữ liệu, không cài Torch hay model.
Trong VS Code có thể dùng Python: Select Interpreter và chọn .venv-data/Scripts/python.exe.

## Các kiểm tra và đầu ra

Script đối chiếu ảnh với annotation/manifest của cả ba tập, kiểm tra ID không giao nhau và checksum manifest không trùng.
Với --verify-hashes, nó đọc toàn bộ JPG để kiểm tra SHA-256 với manifest đã đóng gói.
Kiểm tra này bảo toàn bản local, không chứng minh nguồn ảnh là byte gốc từ máy chủ COCO, không phát hiện mọi ảnh gần trùng.
Không dùng nội dung/nhãn test để chọn mẫu, sửa mô hình hay tinh chỉnh tham số.

Chọn 10 đối tượng từ 10 ảnh train, theo thứ tự ID cố định; bỏ stuff, cần visible mask, diện tích tối thiểu 64 pixel và tỷ lệ khuất 5–70%.
Đây là lựa chọn cho smoke test, không phải giao thức đánh giá chính thức hoặc tập mẫu đại diện thống kê.
Có thể xem quy tắc đầy đủ trong configs/data.json. Khi chạy lại cùng cấu hình, mẫu được chọn giữ nguyên.
Nếu thay cấu hình, nên đổi output_dir để giữ lại lịch sử.

Mỗi mẫu có:

```text
image.png          ảnh RGB đầu vào
visible.png        mask phần nhìn thấy
gt_amodal.png      mask hình dạng đầy đủ — chỉ để đánh giá
gt_hidden.png      mask vùng bị che — chỉ để đánh giá
sample.json        ID ảnh, annotation, region_index, tên và tỷ lệ che
preview.jpg        ảnh so sánh các mask trên ảnh gốc
```

Chỉ đưa image.png và visible.png vào mô hình. Không đưa gt_amodal.png/gt_hidden.png vào bộ dự đoán completion.
Mask PNG xuất ra dùng 0/255; mask nội bộ là boolean. Ảnh và mask giữ nguyên kích thước.
Nhãn không bị che có thể thiếu visible_mask; bộ đọc chỉ suy ra visible=amodal khi occlude_rate=0.
Nếu vùng được ghi bị che mà thiếu visible mask, bộ đọc báo lỗi thay vì dùng amodal làm visible.

manifest.json định danh bằng image_id + annotation_id + region_index. contact_sheet.jpg giúp kiểm tra toàn bộ 10 mẫu.
audit.json lưu kết quả kiểm tra; selection_config.json lưu cấu hình lựa chọn.

## Bước tiếp theo

1. Kiểm tra overlay 10 mẫu train.
2. Mở notebooks/SCAmodal_Colab.ipynb, setup Colab GPU và chạy một mẫu bằng full SCAmodal.
3. Kiểm tra prompt, Pix2Gestalt RGB, mask, inpainting và critic; thử resume.
4. Sau smoke run GPU thành công, mở rộng limit lên 10. IoU được tính mặc định; bật appearance_metrics để có 4 metric bổ sung.
5. Dùng dev chọn tham số; chỉ đánh giá test sau khi chốt cấu hình.

Không trộn requirements-data.txt với requirements GPU của repo tác giả.

Nguồn dữ liệu/annotation: Semantic Amodal Segmentation — https://github.com/yzhuuu/amodalAPI.
