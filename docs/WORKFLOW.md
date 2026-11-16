# Quy trình làm việc và giao diện giữa 4 thành viên

| Thành viên | Phạm vi chính | Test liên quan |
| --- | --- | --- |
| TV1 | `fireguard/color/` | `tests/test_color.py`, `tests/test_smoke_context.py` |
| TV2 | `fireguard/motion/` | `tests/test_motion.py` |
| TV3 | `fireguard/shape/` | `tests/test_regions.py` |
| TV4 | `fireguard/fusion/`, `fireguard/temporal/`; phối hợp tích hợp | `tests/test_fusion.py`, `tests/test_temporal.py`, `tests/test_temporal_hybrid.py`, `tests/test_pipeline.py` |

`app.py`, `fireguard/pipeline.py`, `config.py`, `presets.py` là điểm tích hợp chung: thống nhất trước khi đổi API/preset/hành vi box. TV4 sở hữu **cả Fusion và Temporal**. Xem [TV1](TV1.md), [TV2](TV2.md), [TV3](TV3.md), [TV4](TV4.md).

## Hợp đồng dữ liệu

- Frame vào: `np.ndarray` BGR `uint8`, `(H,W,3)`; mask: `uint8` `(H,W)`, 0/255.
- TV1 `color_candidates(frame_bgr,cfg)` trả `{'fire':mask,'smoke':mask}`; nhánh khói bối cảnh còn dùng `context_smoke_candidates` trong pipeline.
- TV2 `MotionDetector.apply(frame)` hoặc `MOG2MotionDetector.apply(frame)` trả motion mask; flow hỗ trợ khói bối cảnh.
- TV4 Fusion `fuse_masks(color_mask,motion_mask,mode,...)` trả mask cho TV3; FIRE và SMOKE xử lý riêng.
- TV3 `extract_regions(mask,kind,cfg)` trả `Region` với `kind`, `bbox=(x,y,w,h)`, `area`, `centroid`, `score` và đặc trưng bổ sung.
- TV4 Temporal `TemporalFilter(cfg).update(regions,frame_idx)` trả vùng được xác nhận; FIRE và SMOKE có bộ lọc riêng. `frame_idx` tăng dần.
- CSV: `frame,timestamp_sec,kind,x,y,w,h,score`; `score` là điểm thuật toán, không phải xác suất cháy.

## Quy trình thay đổi

1. Kiểm tra `git status`, đọc code/preset đang dùng; không ghi đè phần sửa của thành viên khác. Một nhánh `main` vẫn có thể xung đột.
2. Sửa trong phạm vi phụ trách. Nếu phải đổi chữ ký hàm, cấu trúc `Region`, thứ tự pipeline hoặc preset, phối hợp phần tích hợp và cập nhật test/tài liệu cùng lúc.
3. Chạy test của module rồi chạy `python -m pytest -q -p no:cacheprovider --basetemp .pytest_tmp`. Khi đổi hành vi, thử app/CLI trên cả video có lửa, khói và âm tính.
4. Xem `git diff`, chỉ stage file liên quan; không force-push `main`. Chỉ `git pull --rebase origin main` khi working tree sạch.

Quy ước mã nguồn nằm trong `.editorconfig`: UTF-8, LF, thụt 4 dấu cách và mục tiêu 88 ký tự/dòng cho Python. Chú thích/docstring mới viết tiếng Việt, tập trung giải thích lý do, ràng buộc và đầu vào/đầu ra của hàm quan trọng; tránh khối tiêu đề trang trí hoặc chú thích lặp lại đúng tên lệnh.

Để kiểm tra cùng định dạng với lần chuẩn hóa này (công cụ phát triển, không bắt buộc để chạy app): `python -m pip install black==25.1.0`, rồi `python -m black --check --line-length 88 app.py fireguard tests experiments`.

## Dữ liệu và nghiệm thu

Đánh giá FIRE-only, SMOKE-only, BOTH, NEITHER; âm tính cần có mây, hơi nước, đèn, hoàng hôn. Mỗi video một hàng trong `data/manifest.csv`, nhãn `has_fire`/`has_smoke` độc lập. Dùng `data/development/` để chỉnh tham số; tạo lại `data/integration/` và `data/test/` khi có dữ liệu kiểm chứng. Không chia cùng nguồn sang các split khác nhau. Để báo precision/recall/F1 thực tế, cần nhãn box/sự kiện độc lập và quy tắc TP/FP/FN. Box ONNX chỉ là nhãn giả để đối chiếu.
