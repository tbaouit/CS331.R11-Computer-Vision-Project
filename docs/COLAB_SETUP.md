# Hướng dẫn chạy trên Google Colab

## 1. Mở Notebook trên Colab

- Vào Google Colab → **File → Open notebook → tab GitHub** → dán `https://github.com/tbaouit/CS331.R11-Computer-Vision-Project` → chọn `notebooks/Amodal_Completion_Colab.ipynb`.
- Đảm bảo file `cocoa_dataset.zip` đã có trên Google Drive tại `MyDrive/CS331/cocoa_dataset.zip` hoặc `MyDrive/amodal_project/cocoa_dataset.zip`.

## 2. Thứ tự các ô chạy

**Nếu đã có kết quả pipeline chính và chỉ cần baseline:** phiên mới chạy **1 → 2 → 3B → 12 → 13 → 14 → 15**. Phiên đã cài môi trường chạy ô 2 rồi 12–15. Ô 3B bỏ tải checkpoint Pix2Gestalt; baseline chỉ nạp SD2 khi sinh ảnh. Ô 14 dùng profile chính đã lưu trên Drive, không sinh lại 400 ảnh. Xem [BASELINE.md](BASELINE.md).

1. **Ô 1**: Kiểm tra GPU (khuyến nghị A100) và mount Google Drive.
2. **Ô 2**: Tự động `git clone` hoặc `git pull` code mới nhất từ nhánh `main` về `/content/cs331` và giải nén dataset nếu chưa có.
3. **Ô 3**: Chạy `scripts/setup_colab.py` (tạo `.venv-gpu`, tải checkpoint Pix2Gestalt) và chạy bộ unit test (`tests/`).
4. **Ô 4**: Chạy `scripts/prepare_data.py --verify-hashes` để trích xuất 10 mẫu smoke test.
5. **Ô 5A / 5B**:
   - **Ô 5A**: Chạy chế độ nhận thẳng mask phần bị che (`configs/oracle_mask.json`).
   - **Ô 5B**: Chạy chế độ tự đoán mask (`configs/improved.json`).
6. **Ô 6 & Ô 7**: Hiển thị kết quả từng mẫu và chạy `scripts/compare_runs.py` để xuất bảng số liệu + lưới ảnh so sánh (`outputs/runs/comparison_grid.jpg`).

## So sánh và kiểm thử độc lập

Ô **8** chấm metric trên output đã lưu: nhập đúng `EVAL_PROFILE`, không cần chạy lại 5A/5B. Ô **9–11** dành cho batch lớn: chọn `dev50` trước, sau khi chốt config mới chọn `test200`. CSV, mean/std/n và chính sách diễn giải xem [EVALUATION.md](EVALUATION.md). Chạy bước metric sau khi process generation kết thúc để không cạnh tranh VRAM. Ô 6 giới hạn 10 preview; ô 7 vẫn là bảng/lưới smoke 10 mẫu, không dùng để render hàng trăm mẫu.

`scripts/compare_runs.py` mặc định chỉ đọc `--runs-root`. Chỉ khi có
`--backup-dir` mới khôi phục ZIP từ thư mục backup đó. Ô 7 của notebook đã truyền
`--backup-dir` tới thư mục `results` trên Drive. Các test dùng dữ liệu giả trong
thư mục tạm, không đọc backup thật trên Drive.

## Bộ nhớ GPU

Các model được nạp khi giai đoạn đầu tiên cần dùng tới chúng. Chỉ model đang
chạy nằm trên GPU; model đã nạp được giữ trong RAM CPU và dùng lại giữa các mẫu.
CarveKit chạy trên CPU. Pix2Gestalt dùng trọng số EMA suy luận đã áp dụng một lần,
không giữ thêm bản trọng số huấn luyện để sao lưu trên GPU. Cách này giảm VRAM
nhưng tăng nhu cầu RAM hệ thống và thời gian chuyển model; nên dùng runtime có
nhiều RAM. Chưa đo mức VRAM đỉnh trên GPU Colab thực tế sau thay đổi này.

Nếu phiên trước vừa báo CUDA OOM, cập nhật code bằng ô 2, chạy lại unit tests
(chỉ dòng test của ô 3), rồi thử ô 5B với `--limit 1` trước. Không cần chạy lại
setup hoặc tải lại checkpoint nếu vẫn cùng runtime. Có thể xem `nvidia-smi` để
kiểm tra tiến trình còn chiếm GPU; không chạy đồng thời ô 5A và 5B.
