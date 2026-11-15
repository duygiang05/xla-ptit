# FireGuard – Fusion v1

## Sự thật về pipeline cũ

`pipeline.py` gọi TV1 và TV2 độc lập trên cùng frame, tạo hai mask bằng AND, rồi đưa mask đó vào `extract_regions` của TV3. Sau đó gọi TV4. **TV3 thực sự xử lý mask sau fusion**.

## Bản patch này

- `fireguard/fusion/mask_fusion.py`: API riêng `fuse_masks(color_mask, motion_mask, mode, motion_dilate)`; tái sử dụng `fireguard.shape.morphology.combine_masks` của TV3.
- `fireguard/pipeline.py`: hỗ trợ cấu hình fusion theo từng nhánh mà không sửa TV1–TV3/TV4.
- `tests/test_fusion.py`: unit test.
- Mặc định FIRE/SMOKE đều `and`, nên hành vi pipeline cũ không đổi.
- `and_dilate` với `--motion-dilate 5` tương ứng **kernel ellipse 5x5**, không phải bán kính 5 pixel.
- `color_only` chỉ để đối chứng, không nên chọn mặc định.

## Cài đặt an toàn (PowerShell từ D:\xla-ptit)

1. Đảm bảo đã lưu/commit thay đổi khác. Sao lưu `fireguard/pipeline.py` (file này bị thay).
2. Giải nén ZIP vào thư mục gốc `D:\xla-ptit`, ghi đè `fireguard/pipeline.py`; các file mới được thêm vào.
3. `python -m pytest tests/test_fusion.py -v`
4. `python -m pytest tests/ -v`

## Chạy hai chế độ trên cùng video có thật

Thay `VIDEO.mp4` bằng video có sẵn trong `data/development`:

```powershell
python -m fireguard.pipeline --input data/development/VIDEO.mp4 --output outputs/videos/baseline_and.mp4 --csv outputs/events/baseline_and.csv --fire-fusion and --smoke-fusion and
python -m fireguard.pipeline --input data/development/VIDEO.mp4 --output outputs/videos/and_dilate5.mp4 --csv outputs/events/and_dilate5.csv --fire-fusion and_dilate --smoke-fusion and_dilate --motion-dilate 5
```

## Lưu ý đánh giá

- `event_rows` không phải số đám cháy; nó là số bbox/frame được xác nhận.
- Dilation thường tăng vùng giữ lại, không mặc nhiên cải thiện accuracy.
- Phải đánh giá các video fire, smoke và negative; tốt nhất có ground truth thời gian/bbox.
- `run_video(input, output, csv, cfg)` giữ nguyên; `app.py` vẫn có thể gọi như trước.
- Hybrid TV4 cần bật riêng trong cấu hình riêng của TV4; bản patch không thay đổi chế độ TV4.
