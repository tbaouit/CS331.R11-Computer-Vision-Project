# CS331 — Amodal Completion

Đồ án khôi phục hình dạng và diện mạo 2D của vật thể bị che khuất trong ảnh tĩnh, đánh giá trên COCO-A.

## Trạng thái hiện tại

Hỗ trợ đầy đủ 3 chế độ thực nghiệm trên GPU (đã kiểm chứng trên Colab A100):
- `configs/scamodal.json` (`pipeline_mode = "improved"`): tự dự đoán mask vùng khuất + khôi phục diện mạo.
- `configs/scamodal_oracle_mask.json` (`pipeline_mode = "oracle_mask"`): nhận thẳng mask vùng bị che từ input (`occluded_mask`) + khôi phục diện mạo.
- `configs/scamodal_baseline.json` (`pipeline_mode = "upstream_baseline"`): bản gốc của tác giả để đối chiếu ablation.

Xem `docs/COLAB_SETUP.md` để chạy. Notebook: `notebooks/SCAmodal_Colab.ipynb`. Code được clone/pull trực tiếp từ GitHub trên Colab; dataset giữ trên Google Drive tại `MyDrive/amodal_project/cocoa_dataset.zip` (hoặc `MyDrive/CS331/cocoa_dataset.zip`).

Chi tiết kiến trúc và pipeline xem tại `docs/SCAMODAL_PIPELINE.md` và `upstream/provenance.json`.
`upstream/` lưu bản gốc nguyên vẹn của tác giả, `runtime/` chứa bản nạp cho foundation models, và toàn bộ cải tiến/điều phối nằm độc lập trong `src/`.

## Cấu trúc thư mục

```text
├── configs/
│   ├── data.json                  # Quy tắc chọn mẫu COCO-A smoke test
│   ├── scamodal.json              # Chế độ Blind Improved (mặc định)
│   ├── scamodal_oracle_mask.json  # Chế độ nhận thẳng mask phần bị che (Oracle-Mask)
│   └── scamodal_baseline.json     # Chế độ Baseline gốc của tác giả
├── src/
│   ├── data/
│   │   └── cocoa.py               # Bộ đọc annotation COCO-A, giải mã polygon/RLE
│   ├── vision/
│   │   ├── image_ops.py           # Hình thái học mask, lọc liên thông, Telea inpaint, Gaussian blur
│   │   ├── roi.py                 # Tính toán, trích xuất và dán lại cửa sổ vuông ROI
│   │   └── compositing.py         # Khóa tần số thấp, hòa trộn biên thích nghi, crop nền trắng cho Critic
│   ├── stages/
│   │   ├── shape_consensus.py     # Multi-sample Pix2Gestalt + CarveKit smooth alpha consensus
│   │   ├── appearance_prior.py    # Khởi tạo màu/kết cấu vùng khuất (Blind & Oracle-Mask)
│   │   ├── vlm_guidance.py        # Qwen3-VL visual prompting, CLIP verification & Critic chấm điểm
│   │   └── sd2_refiner.py         # Tinh chỉnh chi tiết bằng Stable Diffusion 2 Inpainting trên ROI 512x512
│   ├── pipelines/
│   │   ├── backend.py             # Bộ điều phối chung cho cả 3 chế độ chạy
│   │   ├── baseline_tracer.py     # Trace hooks không xâm lấn cho pipeline tác giả gốc
│   │   ├── improved_completer.py  # Điều phối luồng Blind Improved (tự đoán mask + khôi phục RGB)
│   │   └── oracle_completer.py    # Điều phối luồng Oracle-Mask (nhận thẳng mask phần bị che + khôi phục RGB)
│   └── evaluation/
│       ├── metrics.py             # Amodal/Hidden IoU, SeamMAE, ColorHistSim
│       ├── runner.py              # Thực thi từng mẫu, kiểm tra resume, lưu artifact & preview
│       └── comparison.py          # Khôi phục backup từ Drive & xuất bảng/lưới so sánh 2 chiều / 3 chiều
├── scripts/
│   ├── prepare_data.py            # Kiểm tra toàn vẹn dataset và xuất mẫu thử
│   ├── run_scamodal.py            # Entry point chạy thực nghiệm và tự động backup
│   └── compare_runs.py            # Entry point đối chiếu định lượng & trực quan giữa các chế độ
├── tests/                         # Bộ unit test kiểm chứng từng module (data, vision, stages, pipelines, eval)
├── upstream/SCAmodal/             # Bản lưu nguyên gốc từ repo tác giả (bảo toàn checksum)
└── runtime/SCAmodal/              # Bản runtime có kiểm soát được nạp cho foundation models
```

## Bước 1 — chạy trong VS Code trên Windows

Mở terminal PowerShell ngay tại thư mục dự án:

```powershell
python -m venv .venv-data
.\.venv-data\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv-data\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-data\Scripts\python.exe scripts/prepare_data.py --verify-hashes
```

Không cần activate khi gọi Python bằng đường dẫn như trên. Môi trường này chỉ dành cho dữ liệu, không cài Torch hay model.
Trong VS Code có thể dùng Python: Select Interpreter và chọn `.venv-data/Scripts/python.exe`.

## Các kiểm tra và đầu ra

Script đối chiếu ảnh với annotation/manifest của cả ba tập, kiểm tra ID không giao nhau và checksum manifest không trùng.
Với `--verify-hashes`, nó đọc toàn bộ JPG để kiểm tra SHA-256 với manifest đã đóng gói.
Kiểm tra này bảo toàn bản local, không chứng minh nguồn ảnh là byte gốc từ máy chủ COCO, không phát hiện mọi ảnh gần trùng.
Không dùng nội dung/nhãn test để chọn mẫu, sửa mô hình hay tinh chỉnh tham số.

Chọn 10 đối tượng từ 10 ảnh train, theo thứ tự ID cố định; bỏ stuff, cần visible mask, diện tích tối thiểu 64 pixel và tỷ lệ khuất 5–70%.
Đây là lựa chọn cho smoke test, không phải giao thức đánh giá chính thức hoặc tập mẫu đại diện thống kê.
Có thể xem quy tắc đầy đủ trong `configs/data.json`. Khi chạy lại cùng cấu hình, mẫu được chọn giữ nguyên.
Nếu thay cấu hình, nên đổi `output_dir` để giữ lại lịch sử.

Mỗi mẫu có:

```text
image.png          ảnh RGB đầu vào
visible.png        mask phần nhìn thấy
gt_amodal.png      mask hình dạng đầy đủ — chỉ để đánh giá
gt_hidden.png      mask vùng bị che — đánh giá ở chế độ Blind, hoặc làm occluded_mask đầu vào ở chế độ Oracle-Mask
sample.json        ID ảnh, annotation, region_index, tên và tỷ lệ che
preview.jpg        ảnh so sánh các mask trên ảnh gốc
```

Ở chế độ Blind (`upstream_baseline`, `improved`), chỉ đưa `image.png` và `visible.png` vào mô hình (không đưa `gt_amodal.png`/`gt_hidden.png` vào bộ dự đoán). Ở chế độ `oracle_mask`, pipeline chuyên biệt nhận thêm `gt_hidden.png` làm `occluded_mask` đầu vào.
Mask PNG xuất ra dùng 0/255; mask nội bộ là boolean. Ảnh và mask giữ nguyên kích thước.
Nhãn không bị che có thể thiếu `visible_mask`; bộ đọc chỉ suy ra `visible=amodal` khi `occlude_rate=0`.
Nếu vùng được ghi bị che mà thiếu visible mask, bộ đọc báo lỗi thay vì dùng amodal làm visible.

`manifest.json` định danh bằng `image_id + annotation_id + region_index`. `contact_sheet.jpg` giúp kiểm tra toàn bộ 10 mẫu.
`audit.json` lưu kết quả kiểm tra; `selection_config.json` lưu cấu hình lựa chọn.

## Bước tiếp theo

1. Kiểm tra overlay 10 mẫu train.
2. Mở `notebooks/SCAmodal_Colab.ipynb`, setup Colab GPU và chạy thử nghiệm (`improved` hoặc `oracle_mask`).
3. Kiểm tra prompt, Pix2Gestalt RGB, mask, inpainting và critic; đối chiếu định lượng & trực quan qua `scripts/compare_runs.py`.
4. Dùng tập `dev` để chọn tham số; chỉ đánh giá `test` sau khi chốt cấu hình.

Không trộn `requirements-data.txt` với requirements GPU của repo tác giả.

Nguồn dữ liệu/annotation: Semantic Amodal Segmentation — https://github.com/yzhuuu/amodalAPI.
