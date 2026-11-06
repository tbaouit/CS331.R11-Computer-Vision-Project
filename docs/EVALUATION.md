# Đánh giá COCO-A theo code SCAmodal

## Final test400 sau dev50

Đã chọn đánh giá cuối với 400 object từ 400 ảnh test. Notebook mới mặc định ô 9 là `BATCH="test400"`, `FINAL_TEST=True`. Giữ nguyên `configs/oracle_mask.json` và cùng bộ lọc đã dùng trên dev50. Chạy ô 9 (chuẩn bị + preflight) → 10 (generation) → 11 (metric; điền profile in ở ô 10). Nếu đang ở phiên Colab cũ, chạy ô 2 để cập nhật code, rồi lấy ô 9–11 từ notebook mới trên GitHub; nội dung các ô đang mở không tự cập nhật khi git pull.

CLI tương đương:

```bash
.venv-gpu/bin/python scripts/prepare_data.py --config configs/data_test400.json --allow-test
.venv-gpu/bin/python scripts/run_experiment.py --manifest outputs/test400/manifest.json --config configs/oracle_mask.json --limit 400 --allow-test --backup-every 5 --backup-dir /content/drive/MyDrive/CS331/results
# Điền đúng profile đã in, thay PROFILE_ID
.venv-gpu/bin/python scripts/evaluate_runs.py --manifest outputs/test400/manifest.json --profile PROFILE_ID --device cuda --backup-dir /content/drive/MyDrive/CS331/results
```

Không có thay đổi thuật toán hoặc tham số generation trong lần cập nhật test400. Báo cáo metric mới nằm ở `evaluation/<evaluation_id>/<subset_id>/` (subset_id là hash toàn bộ manifest được chọn) để dev50, test400 và các lần chạy giới hạn mẫu không ghi đè nhau. Cache từng sample vẫn tách riêng. Các báo cáo cũ ở `evaluation/<evaluation_id>/` được giữ nguyên. Khi đánh giá cuối, ghi nhận cả mẫu lỗi và không chọn lại mẫu dựa trên chất lượng output.

Bài toán chính: nhận RGB, visible mask và hidden mask được cho sẵn, rồi hoàn thiện RGB vật thể. Không huấn luyện lại model. COCO-A không có RGB ground truth ở vùng bị che tự nhiên.

## Protocol được triển khai

Tham chiếu code `dpduy123/SCAmodal`, commit `92e5b39`, `evaluate_amodal.py` và `metrics_utils.py`. Đây là tái hiện code evaluator đã kiểm tra, chưa xác nhận bảng slide được sinh bằng chính commit đó.

| Tên trong CSV | Đầu vào / triển khai | Tốt hơn |
|---|---|---|
| CLIP_score | Toàn bộ RGB output và tên lớp annotation nguyên văn; `openai/clip-vit-base-patch32`, `logits_per_image` (cosine nhân logit scale) | ↑ |
| LPIPS | Output và input đều tô đen ngoài visible mask, giữ nguyên canvas; LPIPS AlexNet, chuẩn hóa [-1,1] | ↓ |
| Feature_Similarity | Cosine giữa image embedding CLIP ViT-B/32 của hai ảnh đã mask; không dùng DINOv2 | ↑ |
| SSIM | Hai ảnh đã mask, RGB, data_range=255, cửa sổ mặc định 7x7 | ↑ |
| PSNR | MSE trên toàn canvas của hai ảnh đã mask, range 255; ảnh giống hệt trả 100 dB theo code tác giả | ↑ |

Không crop/resize thủ công trước metric; CLIP có resize/center-crop nội bộ theo processor. Processor đặt `use_fast=False` để kết quả ổn định giữa phiên bản. RGB output lấy ba kênh đầu của RGBA, không ghép alpha lên nền khác. Không coi số liệu trực tiếp tương đương bảng tác giả khi khác tập mẫu, checkpoint, preprocessing hoặc phiên bản thư viện.

Các pixel đen giống nhau ngoài mask làm SSIM/PSNR đẹp hơn, nhất là object nhỏ. `visible_canvas_fraction` trong CSV giúp nhận diện vấn đề này. Nếu pipeline sao chép chính xác visible RGB, bốn metric bảo toàn đạt LPIPS=0, Feature_Similarity≈1, SSIM=1, PSNR=100; chúng không phân biệt chất lượng vùng sinh. CLIP ảnh đầy đủ chỉ đánh giá ngữ nghĩa gián tiếp. Nên bổ sung đánh giá định tính/chấm mù của con người; điểm VLM critic dùng để chọn ảnh không phải metric đánh giá độc lập.

Các cột `seam_mae`, `color_hist_sim` là chẩn đoán không cần RGB tham chiếu cho vùng ẩn, kế thừa công thức hiện tại của project. Seam so màu vòng 2 px trong phần sinh với pixel visible gần nhất; histogram intersection dùng 16 bin/kênh. Chúng không chứng minh texture đúng thực tế. Trong protocol chấm điểm mới, trường hợp không có vùng/biên so sánh được trả null (không được chấm hoàn hảo). Nếu các phương pháp dự đoán mask khác nhau thì seam đo trên các vùng khác nhau, cần thận trọng khi so sánh. IoU trong nhánh mask-given chỉ là kiểm tra mask đầu vào.

## Chấm lại output cũ, không chạy lại generation

```bash
.venv-gpu/bin/python scripts/evaluate_runs.py \
  --manifest outputs/train_smoke/manifest.json \
  --profile 02549104b0c4d76a --device cuda \
  --backup-dir /content/drive/MyDrive/CS331/results
```

Thay profile bằng ID cần chấm. Profile phải chứa đúng các sample trong manifest. Không tự động chọn profile mới nhất vì dễ so nhầm thí nghiệm. Script khôi phục đúng backup được chỉ định, kiểm tra checksum input/visible/completion/amodal, và giữ nguyên generation `result.json`.

Kết quả trong `outputs/runs/<profile>/evaluation/<evaluation_id>/`:

- `protocol.json`: công thức, backbone, phiên bản thư viện, device và hash code đánh giá.
- `metrics.csv`, `per_sample.json`: điểm từng mẫu và trạng thái lỗi/chưa có output.
- `aggregate.json`: mean, độ lệch chuẩn mẫu và n hợp lệ; nhóm theo category và tỷ lệ che (<20%, 20–40%, >40%). Trung bình là macro theo object, mỗi object có trọng số bằng nhau.

Mỗi job có `appearance_<evaluation_id>.json`; chạy lại sẽ bỏ qua điểm hợp lệ. Thay đổi ảnh, tên lớp, code metric, device hoặc phiên bản thư viện sẽ tạo lại cache tương ứng. Các điểm lỗi/missing không được điền 0. Script thoát khác 0 khi còn thiếu/lỗi. Có thể thêm `--continue-on-error` để tiếp tục ghi nhận các lỗi thông thường; OOM luôn dừng. Báo cáo được ghi nguyên tử sau từng mẫu; backup cả profile lên Drive mỗi 5 mẫu mới (đổi bằng `--backup-every`), khi bước đánh giá kết thúc hoặc phát sinh ngoại lệ.

Metric chạy trong process riêng, sau khi process generation kết thúc, nên CLIP/LPIPS không cạnh tranh VRAM với Pix2Gestalt/Qwen/SD2. Có thể dùng `--device cpu` khi không có GPU. Giữ `appearance_metrics=false` trong config generation để chấm tập trung sau đó. Nếu cần inline, đặt true và `appearance_metrics_device=cpu`; không bật inline CUDA trên batch lớn vì model metric sẽ nằm lại trong VRAM ngoài registry generation.

## Chuẩn bị batch lớn

Không phải tạo dữ liệu mới hay che nhân tạo. Bước chuẩn bị chỉ giải mã annotation, chọn object đủ rõ, kiểm tra quan hệ mask và xuất PNG lossless trên canvas gốc.

```bash
# Phát triển / kiểm tra quy trình trước
.venv-gpu/bin/python scripts/prepare_data.py --config configs/data_dev50.json --verify-hashes
.venv-gpu/bin/python scripts/run_experiment.py --manifest outputs/dev50/manifest.json \
  --config configs/oracle_mask.json --limit 50 --backup-every 5 \
  --backup-dir /content/drive/MyDrive/CS331/results

# Chỉ thực hiện sau khi chốt config và không điều chỉnh theo kết quả test
.venv-gpu/bin/python scripts/prepare_data.py --config configs/data_test200.json --allow-test
.venv-gpu/bin/python scripts/run_experiment.py --manifest outputs/test200/manifest.json \
  --config configs/oracle_mask.json --limit 200 --allow-test --backup-every 5 \
  --backup-dir /content/drive/MyDrive/CS331/results
```

Các subset dùng seed 42, shuffle danh sách object trước khi chọn, tối đa một object/ảnh, loại stuff, thiếu mask/tên, visible <1024 px, amodal <2048 px, cạnh ngắn bbox <32 px và tỷ lệ che ngoài 10–60%. Bộ lọc này giới hạn kết luận vào các object đủ lớn, che vừa phải; không đại diện toàn bộ COCO-A. Không cân bằng lớp; báo cáo class và mức che để thấy phân bố. `selection_report.json` ghi bộ lọc và số ứng viên bị loại; không âm thầm bỏ các annotation lỗi. Contact sheet chỉ giữ tối đa 20 preview, không ghép 200 ảnh thành một ảnh khổng lồ. Đổi chính sách chọn mẫu phải dùng output_dir mới.

Trước mỗi run, preflight kiểm tra file RGB/mask đọc được, cùng kích thước, mask nhị phân, visible nằm trong amodal, hidden=amodal-visible, tỷ lệ che và sample ID duy nhất. Không sửa mask/resize ảnh để làm hợp lệ. Lỗi dừng trước khi tải model, ghi `preflight.json`. Preflight chỉ kiểm tra dữ liệu; nhánh blind vẫn không nhận GT mask khi suy luận.

Có thể kiểm tra riêng bằng `scripts/run_experiment.py --manifest outputs/dev50/manifest.json --config configs/oracle_mask.json --limit 50 --validate-only`, không cần torch/GPU.

Mỗi mẫu lưu output, checksum và trạng thái riêng; backup Drive mỗi 5 mẫu và khi lỗi. Chạy tiếp cùng code/config/dataset sẽ bỏ qua mẫu hoàn tất hợp lệ. Code generation hoặc môi trường thay đổi có thể tạo profile mới: không trộn hai profile thành một thí nghiệm. Không cần bật lại `--verify-hashes` ở mọi lần rerun nếu dataset đã audit đầy đủ và không thay đổi. Trên Colab nên copy/giải nén dataset vào `/content`, dùng Drive cho ZIP và backup; chạy tuần tự từng process GPU. Dùng 10 mẫu để kiểm tra, rồi 50 dev; khi ổn mới chạy test200 và chấm bằng `evaluate_runs.py` với manifest test200.
