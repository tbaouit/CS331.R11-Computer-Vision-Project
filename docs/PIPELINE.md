# Pipeline

[Baseline](BASELINE.md) và [Proposed](PROPOSED.md) dùng cùng SD 1.5 Inpainting và DDIM. `src/pipelines/backend.py` chọn phương pháp từ `pipeline_mode`. Hai module triển khai là `src/pipelines/baseline.py` và `src/pipelines/proposed.py`.

`src/stages/inpaint_refiner.py` cắt ROI, chạy diffusion và đưa RGB về hệ tọa độ ban đầu. `src/vision/compositing.py` đóng gói RGB và alpha. Trong proposed, `src/models/sd15_ip_adapter.py` áp dụng latent blending ở từng bước DDIM trước VAE decode. Critic nhận ứng viên này cùng ảnh ngữ cảnh, phần nhìn thấy và bản đồ vùng sinh.
