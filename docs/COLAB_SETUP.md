# Chạy full SCAmodal trên Colab (GitHub + Google Drive Dataset)

## 1. Chuẩn bị lần đầu

1. **Dataset trên Google Drive** (chỉ cần upload 1 lần duy nhất):
   Đảm bảo file `dist/cocoa_dataset.zip` đã có trên Google Drive tại:
   `MyDrive/amodal_project/cocoa_dataset.zip`
   *(Nếu cần đóng gói lại dataset từ máy local: `.\.venv-data\Scripts\python.exe scripts/package_scamodal.py --include-dataset`)*.

2. **Code trên GitHub**:
   Repo: `https://github.com/tbaouit/CS331.R11-Computer-Vision-Project.git` (nhánh `main`).
   *Lưu ý: nếu để repo ở chế độ Private, thêm `GITHUB_TOKEN` (Personal Access Token có quyền đọc repo) vào mục **Secrets** (biểu tượng chìa khóa) trên Colab.*

3. **Mở Notebook trên Colab**:
   - Cách nhanh nhất: Vào Google Colab → **File → Open notebook → tab GitHub** → dán `https://github.com/tbaouit/CS331.R11-Computer-Vision-Project` → chọn `notebooks/SCAmodal_Colab.ipynb`.
   - Hoặc upload file `notebooks/SCAmodal_Colab.ipynb` lên Colab / lưu vào Drive.

## 2. Quy trình cập nhật code mỗi lần chỉnh sửa

Khi sửa code trên máy local:

```powershell
.\.venv-data\Scripts\python.exe scripts/build_scamodal_notebook.py  # nếu có sửa cấu trúc notebook
git add -A
git commit -m "Update pipeline"
git push origin main
```

Trên phiên Colab đang chạy:
- Chỉ cần chạy lại **Ô 2** (`Clone / Pull code mới nhất từ GitHub & giải nén dataset từ Drive`).
- Ô 2 sẽ tự động chạy `git pull --ff-only origin main` để lấy code mới nhất mà **không giải nén lại dataset** và **không cần cài lại `.venv-gpu`**.

## 3. Lưu ý trên Colab

- Chọn GPU trước khi chạy, ưu tiên A100 40GB hoặc lớn hơn và High RAM (L4 22GB vẫn có thể chạy thử nghiệm nhưng cần theo dõi VRAM).
- Chạy từng ô, dừng ở lỗi đầu tiên. Không chạy Run all lần đầu.
- Notebook mount Drive, clone/pull code về `/content/cs331`, copy `cocoa_dataset.zip` về `/content` rồi giải nén và xóa bản tạm ZIP; không đọc hàng nghìn ảnh trực tiếp qua Drive.
- Setup tạo `.venv-gpu` Python 3.12 riêng, không cài đè kernel notebook.
- Setup clone phụ thuộc (`pix2gestalt`, `taming-transformers`, `CLIP`), cài core version/pin tương thích, download checkpoint Pix2Gestalt.
- Qwen/CLIP/CarveKit/SD2 tải thêm trọng số lúc load lần đầu; dành đủ đĩa và thời gian.
- Ô `prepare_data` xuất 10 mục tiêu train. Ô `run` chạy 1 mục tiêu với Qwen + Pix2Gestalt + SD2 + critic.
- Chỉ mở rộng `--limit 10` sau khi thấy artifact và log chứng minh mọi stage hoạt động.
- Khi cần 4 metric, bật `appearance_metrics=true` trong `configs/scamodal.json`; config mới tạo profile mới.

## 4. Khi runtime ngắt

- `/content` và môi trường có thể mất. `cocoa_dataset.zip` và các ZIP kết quả trên Drive (`MyDrive/amodal_project/results/`) vẫn còn.
- Tạo phiên mới, chạy lại Ô 1 → Ô 2 (tự clone code mới nhất + giải nén dataset) → Ô 3 (setup) → Ô 4 (`prepare_data`).
- Ô 5 (`RESTORE`) chọn bản backup cần dùng, khôi phục vào `outputs/scamodal/<profile>`.
- Chạy lại cùng config/manifest để bỏ qua mẫu hoàn thành. Mẫu lỗi/dở được chạy lại.

## 5. Đầu ra

`outputs/scamodal/<profile>/<sample_id>/<signature>/`:
`input.png`, `visible.png`, `reasoning.txt`, `pix2gestalt_rgb_256.png`,
`carvekit_alpha_256.png`, `shape_mask_original.png`, `prompt_selection.json`,
`prompt_01.txt`, `inpainting_01.png`, `critic_input_01.png`, `critique_history.json`,
`outpainting.json`, `completion_full_canvas.png`, `completion.png`, `amodal.png`, `missing.png`,
`preview.jpg`, `trace.json`, `result.json` (hoặc `failure.json`).

Stage không được thực thi (ví dụ không có missing region) không có artifact tương ứng.
Input GT amodal/hidden không được truyền vào backend. Chúng chỉ được đọc sau sinh ảnh để đánh giá.
`pipeline.log` và `summary.json` nằm ở thư mục profile. Backend fail được báo rõ, không chuyển sang heuristic.
