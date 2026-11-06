# SD2 Direct Inpainting baseline

Mục tiêu: so sánh hệ thống mask-given hiện tại với một cách inpainting trực tiếp, dùng cùng ảnh và mask. Cả hai nhận RGB quan sát được, visible mask và hidden mask; amodal mask = visible ∪ hidden. Tên `oracle_mask` chỉ là tên nội bộ của pipeline chính, không có nghĩa RGB hoàn hảo.

## Cấu hình đã triển khai

| Thành phần | SD2 Direct | Pipeline chính (`oracle_mask`) |
|---|---|---|
| Dữ liệu | Cùng manifest dev50/test400 | Cùng manifest |
| Mask | Mask được cung cấp | Mask được cung cấp |
| Model inpainting | `sd2-community/stable-diffusion-2-inpainting` | Cùng checkpoint |
| Khởi tạo RGB vùng ẩn | Xám 127; bỏ toàn bộ RGB vật che | Pix2Gestalt + xử lý prior |
| Prompt | Một prompt cố định, không lấy nhãn annotation | Qwen suy luận từ ảnh/mask; `text_query_policy=empty` |
| Sinh ảnh | Một pass SD2, strength 1, 28 bước, guidance 7, seed 42 | Refinement theo prior, có critic và có thể lặp |
| ROI cho SD2 | Bbox amodal × 1.28, cạnh ít nhất 128; pad vuông rồi resize 512 | Cùng quy tắc ROI SD2 |
| Ghép ảnh | Shared seam blend, giữ visible, alpha theo amodal | Cùng hàm ghép cuối |

Prompt cố định:

```text
A photograph of a single complete object on a plain white background, natural colors, realistic texture.
```

Baseline dùng negative prompt chung có sẵn của dự án. SD2 được phép sinh ở hidden mask giãn 2 pixel, trừ visible đã co 2 pixel; khi ghép chỉ lấy RGB sinh trong hidden mask. Nền canvas đầu vào trắng, visible giữ RGB gốc. Cách crop không kéo méo tỷ lệ; output ở đúng kích thước ảnh gốc, RGB ngoài alpha là đen như pipeline chính.

Không chạy bước khóa màu/tần số thấp theo prior của pipeline chính trên baseline, vì baseline không có prior. `strength=1` dùng cho việc sinh từ vùng trống; không ép bằng strength nhỏ vốn dành cho tinh chỉnh prior. Đây là so sánh **hai hệ thống hoàn chỉnh với ngân sách sinh khác nhau**, chưa tách đóng góp riêng của Qwen/Pix2Gestalt/critic.

## Chạy trên Colab

Mở lại notebook từ GitHub để thấy ô mới. Phiên mới chạy **1 → 2 → 3B → 12 → 13 → 14 → 15**. Ô 3B dùng `setup_colab.py --skip-downloads` để không tải checkpoint Pix2Gestalt; model SD2 và các model metric tải khi cần. Phiên đã cài môi trường chỉ cần cập nhật code bằng ô 2, rồi chạy 12–15.

1. Ô 12 mặc định `BASELINE_BATCH="dev50"`: kiểm tra cấu hình, dữ liệu.
2. Ô 13 sinh baseline, backup mỗi 5 mẫu và lưu chính xác profile vào `outputs/baseline_runs/dev50.json`. Đọc log `Profile: ...` để giữ ID nếu cần.
3. Ô 14 chấm baseline và **kết quả cũ** của pipeline chính; `MAIN_PROFILE="a0ab44af3e253863"`. Không gọi lại generation của pipeline chính.
4. Ô 15 xuất bảng so sánh và 20 mẫu cố định, lưu bản ZIP vào `MyDrive/CS331/results/comparisons`.
5. Khi baseline đã chốt, đổi ô 12 thành `BASELINE_BATCH="test400"`, `BASELINE_FROZEN=True`, chạy lại **12–15**. Giữ nguyên `configs/sd2_direct.json`; không điều chỉnh theo test.

Không chạy lại ô 10 để tạo pipeline chính mới: thêm source sẽ làm hash profile thay đổi. 400 kết quả cũ vẫn dùng được bằng ID đã ghi. Cùng source/config/môi trường thì ô 13 tự resume; nếu môi trường hoặc code thay đổi, profile có thể đổi. Không sửa tên thư mục profile để ép resume. Việc chọn cùng manifest sẽ được xác minh với `sample.json` và checksum ảnh/mask của mỗi phương pháp.

Muốn chỉ chấm lại sau khi mở phiên mới: chạy 1–2 (và 3B nếu cần), 12; đặt `BASELINE_PROFILE` bằng ID đã lưu rồi gọi `scripts/evaluate_runs.py` như dưới đây. Ô 14 mặc định đọc file profile do ô 13 tạo; không suy đoán profile từ thời gian sửa thư mục.

```bash
python scripts/run_experiment.py --manifest outputs/dev50/manifest.json \
  --config configs/sd2_direct.json --limit 50 --backup-every 5 \
  --backup-dir /content/drive/MyDrive/CS331/results \
  --profile-output outputs/baseline_runs/dev50.json

python scripts/evaluate_runs.py --manifest outputs/dev50/manifest.json \
  --profile BASELINE_PROFILE --device cuda --backup-dir /content/drive/MyDrive/CS331/results
python scripts/evaluate_runs.py --manifest outputs/dev50/manifest.json \
  --profile a0ab44af3e253863 --device cuda --backup-dir /content/drive/MyDrive/CS331/results

python scripts/compare_baseline.py --manifest outputs/dev50/manifest.json \
  --baseline BASELINE_PROFILE --main-profile a0ab44af3e253863 \
  --output-dir outputs/comparisons/dev50 --preview-count 20
```

Trên Colab, thay `python` bằng `/content/cs331/.venv-gpu/bin/python`. Với test400 đổi manifest/limit và thêm `--allow-test` vào lệnh sinh ảnh. Backup generation được khôi phục tự động từ đường dẫn đã chỉ định.

## Đọc kết quả

- `paired_metrics.csv`: cùng hàng là cùng đối tượng; score của hai bên và `delta = main − baseline`. Với CLIP, delta dương là tăng class relevance; với SeamMAE, delta âm là giảm chênh lệch biên. Không gộp các metric khác thang đo thành một điểm chung.
- `comparison.json`: mean/std/n theo từng metric, chênh lệch từng cặp, nhóm class/mức che và danh sách mẫu preview. `std` là độ lệch chuẩn mẫu, không phải khoảng tin cậy.
- `coverage.json`: số mẫu hoàn tất/lỗi/thiếu. So sánh dừng nếu có mẫu thiếu hoặc hai protocol khác nhau; không tự bỏ các ca thất bại để lấy trung bình đẹp hơn.
- `preview_*.jpg`: crop theo cùng bbox amodal cho ảnh gốc, visible, amodal, baseline và pipeline chính. Nhóm 20 mẫu ngẫu nhiên seed 42 là nhóm cố định, chưa phải nhóm ca khó được phân tầng riêng.
- Thời gian từng mẫu được lưu trong CSV nhưng bao gồm kích hoạt/tải model và I/O; các lượt chạy có thể khác GPU. Không dùng nó để khẳng định mức tăng tốc chính xác nếu chưa kiểm soát phần cứng và warmup.

COCO-A gốc không có RGB thật ở vùng ẩn. CLIP là mức phù hợp lớp; SeamMAE/ColorHist là chẩn đoán tính liên tục, có thể ưu ái ảnh trơn hoặc sai chất liệu. LPIPS, SSIM, PSNR và Feature Similarity hiện đo visible: do cả hai giữ visible nên chúng gần hoàn hảo là bình thường. IoU của cả hai cũng bằng 1 do dùng mask có sẵn. Cần đọc ảnh và phân tích lỗi để kết luận pipeline bổ sung giúp ở những trường hợp nào; không coi các điểm này là độ chính xác RGB bị che.
