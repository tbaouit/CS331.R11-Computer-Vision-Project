# Amodal Appearance Completion trên COCO-A

Hai phương pháp nhận ảnh RGB, mask nhìn thấy và mask vùng khuất. Cả hai trả RGB giải mã từ SD 1.5 sau khi chiếu ROI về tọa độ ảnh gốc. Kênh alpha là hợp của hai mask.

| Phương pháp | Config | Cách sinh |
| --- | --- | --- |
| baseline | `configs/baseline.json` | Một lượt DDIM với prompt cố định và vùng khuất tô xám |
| proposed | `configs/proposed.json` | Prior RGB, prompt Qwen, IP-Adapter Plus, latent blending và vòng phản hồi strict critic |

Mở [notebook Colab](notebooks/Amodal_Completion_Colab.ipynb), chạy ô 1–4 để chuẩn bị dev50 hoặc test400. Ô 5 sinh baseline, ô 6 sinh proposed và ô 7 chấm rồi so sánh hai phương pháp. Mỗi phiên sinh có thể tiếp tục từ Drive.

- [Cài đặt và backup](docs/COLAB_SETUP.md)
- [Pipeline](docs/PIPELINE.md)
- [Đánh giá](docs/EVALUATION.md)
