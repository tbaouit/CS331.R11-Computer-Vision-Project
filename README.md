# CS331 — Open-World Amodal Appearance Completion

Đồ án khôi phục hình dạng và diện mạo 2D của vật thể bị che khuất trong ảnh tĩnh (Training-free Open-World Amodal Appearance Completion), đánh giá trên tập dữ liệu COCO-A.

## Chế độ thực nghiệm

Dự án hỗ trợ 2 chế độ thực thi trên GPU (Colab A100):
- **`configs/oracle_mask.json` (`pipeline_mode = "oracle_mask"`)**: Nhận thẳng mask phần bị che từ input (`gt_hidden` / `occluded_mask`) + khôi phục diện mạo vùng khuất.
- **`configs/improved.json` (`pipeline_mode = "improved"`)**: Tự dự đoán cả mask vùng bị che lẫn diện mạo từ `image.png` và `visible.png`.

Xem `docs/COLAB_SETUP.md` để chạy trên Google Colab với notebook `notebooks/Amodal_Completion_Colab.ipynb`. Chi tiết kiến trúc từng bước xem tại `docs/PIPELINE.md`.

## Metric và batch lớn

Sau dev50, đánh giá cuối dùng **test400**: notebook mới mặc định ô 9 là `BATCH="test400"`, `FINAL_TEST=True`, rồi chạy ô 10–11. Cấu hình khôi phục giữ nguyên. Báo cáo mới tách thư mục theo hash subset để không ghi đè dev50. Xem lệnh đầy đủ trong [docs/EVALUATION.md](docs/EVALUATION.md).

Đã có bước `scripts/evaluate_runs.py` chấm lại ảnh đã lưu, không chạy lại generation: CLIP logit, LPIPS AlexNet, CLIP image-feature cosine, SSIM và PSNR theo protocol visible-mask-over-black của code tác giả. Kết quả gồm CSV từng object, mean/std/n theo class và mức che; lỗi/missing được ghi riêng. Những metric so với input chỉ đo bảo toàn visible RGB, không đo độ chính xác RGB của vùng ẩn.

Notebook có ô 8 chấm profile cũ và ô 9–11 chuẩn bị/chạy/chấm batch `dev50` hoặc final `test200`, có preflight và backup/resume. Xem [docs/EVALUATION.md](docs/EVALUATION.md) trước khi chạy đánh giá cuối.

## Cấu trúc thư mục

```text
├── configs/
│   ├── data.json                  # Quy tắc chọn mẫu COCO-A smoke test
│   ├── improved.json              # Cấu hình chế độ Blind Improved (tự đoán mask + khôi phục RGB)
│   └── oracle_mask.json           # Cấu hình chế độ Oracle-Mask (nhận thẳng mask phần bị che)
├── src/
│   ├── data/
│   │   └── cocoa.py               # Bộ đọc annotation COCO-A, giải mã polygon/RLE
│   ├── models/
│   │   ├── pix2gestalt.py         # Nạp checkpoint Pix2Gestalt LDM & CarveKit matting
│   │   ├── qwen_vl.py             # Nạp Qwen3-VL-4B-Instruct (FP16)
│   │   └── registry.py            # FoundationModels: quản lý & chia sẻ trọng số dùng chung
│   ├── vision/
│   │   ├── image_ops.py           # Hình thái học mask, lọc liên thông, Telea inpaint, Gaussian blur
│   │   ├── roi.py                 # Tính toán, trích xuất và dán lại cửa sổ vuông ROI
│   │   └── compositing.py         # Khóa tần số thấp, hòa trộn biên thích nghi, đóng gói RGBA
│   ├── stages/
│   │   ├── shape_consensus.py     # Multi-sample Pix2Gestalt + CarveKit smooth alpha consensus
│   │   ├── appearance_prior.py    # Khởi tạo màu/kết cấu vùng khuất (Blind & Oracle-Mask)
│   │   ├── vlm_guidance.py        # Qwen3-VL visual prompting, CLIP verification & Critic chấm điểm
│   │   └── sd2_refiner.py         # Tinh chỉnh chi tiết bằng Stable Diffusion 2 Inpainting trên ROI 512x512
│   ├── pipelines/
│   │   ├── backend.py             # AmodalBackend: bộ điều phối chung
│   │   ├── improved_completer.py  # Điều phối luồng Blind Improved
│   │   └── oracle_completer.py    # Điều phối luồng Oracle-Mask
│   └── evaluation/
│       ├── metrics.py             # Amodal/Hidden IoU, SeamMAE, ColorHistSim
│       ├── appearance_metrics.py  # CLIP score, LPIPS, Feature Similarity, SSIM
│       ├── runner.py              # Thực thi từng mẫu, kiểm tra resume, lưu artifact & preview
│       └── comparison.py          # Khôi phục backup từ Drive & xuất bảng/lưới so sánh
├── scripts/
│   ├── prepare_data.py            # Kiểm tra toàn vẹn dataset và xuất mẫu thử
│   ├── setup_colab.py             # Cài đặt môi trường GPU độc lập trên Colab
│   ├── check_environment.py       # Kiểm tra thư viện, GPU và kết nối model
│   ├── download_weights.py        # Tải checkpoint Pix2Gestalt
│   ├── run_experiment.py          # Entry point chạy thực nghiệm và tự động backup
│   ├── compare_runs.py            # Entry point đối chiếu định lượng & trực quan giữa các chế độ
│   ├── build_notebook.py          # Sinh notebook Colab
│   └── package_project.py         # Đóng gói mã nguồn và dataset
└── tests/                         # Bộ unit test kiểm chứng toàn bộ các module
```

## Chạy kiểm thử local trên Windows

```powershell
python -m venv .venv-data
.\.venv-data\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv-data\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-data\Scripts\python.exe scripts/prepare_data.py --verify-hashes
```

## Nguồn tham khảo (Acknowledgements)

- Dữ liệu & công cụ giải mã COCO-A: [Semantic Amodal Segmentation (amodalAPI)](https://github.com/yzhuuu/amodalAPI).
- Mô hình nền tảng & ý tưởng pipeline tham khảo từ [Pix2Gestalt](https://github.com/cvlab-columbia/pix2gestalt) và [SCAmodal](https://github.com/dpduy123/SCAmodal).
