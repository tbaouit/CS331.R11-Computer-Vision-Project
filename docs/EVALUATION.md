# Đánh giá baseline và proposed

Ô 7 chấm ảnh completion.png và tổng hợp kết quả theo cùng manifest, config, seed và protocol.

- CLIP đo độ phù hợp với nhãn vật thể; SeamMAE và histogram màu so phần thấy và phần khuất của cùng ảnh đầu ra để chẩn đoán tính
  liên tục. Chúng không đo RGB thật của vùng khuất.
- PSNR, SSIM, LPIPS, Feature Similarity so sánh RGB đầu ra với ảnh gốc sau khi tô đen ngoài vùng thấy. Phần nền đen chung có thể làm điểm cao khi vật thể nhỏ.
- FID là tùy chọn cho batch có ít nhất 400 vật thể: đặt `FID_REFERENCE` ở ô 7 trỏ tới một thư mục
  tối thiểu 400 crop vật thể thật không bị che, nền trắng và tỉ lệ lớp phù hợp.
  COCO-A không cấp ảnh RGB vùng khuất tương ứng nên không dùng ảnh bị che làm
  chuẩn FID. Cùng bộ tham chiếu được dùng cho mọi phương pháp; báo cáo lưu hash
  và cỡ tập tham chiếu. Dùng [pytorch-fid 0.3.0](https://github.com/mseitzer/pytorch-fid/tree/v0.3.0)
  với đặc trưng Inception 192 chiều và crop trắng 224px. Không đối chiếu trực
  tiếp với FID pool3 2048 chiều; FID đo độ gần phân bố và nhạy với cỡ mẫu.

`per_sample.csv` và `summary.csv` chứa chỉ số từng mẫu và thống kê. `comparison.json`
lưu profile, config, protocol, tỷ lệ PASS/REJECT theo lượt, tỷ lệ mẫu được
chấp nhận và số phản hồi JSON lỗi. Baseline không có critic nên tỷ lệ để trống.
`preview_*.jpg` là tối đa 20 mẫu chọn bằng seed 42, gồm Input Crop, Visible Mask,
Given Mask và đầu ra baseline, proposed.
