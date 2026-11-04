# Hướng dẫn chạy trên Google Colab

## 1. Mở Notebook trên Colab

- Vào Google Colab → **File → Open notebook → tab GitHub** → dán `https://github.com/tbaouit/CS331.R11-Computer-Vision-Project` → chọn `notebooks/Amodal_Completion_Colab.ipynb`.
- Đảm bảo file `cocoa_dataset.zip` đã có trên Google Drive tại `MyDrive/CS331/cocoa_dataset.zip` hoặc `MyDrive/amodal_project/cocoa_dataset.zip`.

## 2. Thứ tự các ô chạy

1. **Ô 1**: Kiểm tra GPU (khuyến nghị A100) và mount Google Drive.
2. **Ô 2**: Tự động `git clone` hoặc `git pull` code mới nhất từ nhánh `main` về `/content/cs331` và giải nén dataset nếu chưa có.
3. **Ô 3**: Chạy `scripts/setup_colab.py` (tạo `.venv-gpu`, tải checkpoint Pix2Gestalt) và chạy bộ unit test (`tests/`).
4. **Ô 4**: Chạy `scripts/prepare_data.py --verify-hashes` để trích xuất 10 mẫu smoke test.
5. **Ô 5A / 5B**:
   - **Ô 5A**: Chạy chế độ nhận thẳng mask phần bị che (`configs/oracle_mask.json`).
   - **Ô 5B**: Chạy chế độ tự đoán mask (`configs/improved.json`).
6. **Ô 6 & Ô 7**: Hiển thị kết quả từng mẫu và chạy `scripts/compare_runs.py` để xuất bảng số liệu + lưới ảnh so sánh (`outputs/runs/comparison_grid.jpg`).
