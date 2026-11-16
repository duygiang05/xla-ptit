# TV1 — Color Detection

TV1 phụ trách `fireguard/color/`: `preprocessing.py` kiểm tra/làm mịn frame; `fire_detection.py` tạo ứng viên lửa và API `color_candidates`; `smoke_detection.py` tạo mask khói cổ điển; `smoke_context.py` bổ sung ứng viên khói theo bối cảnh cho app. Không dùng mô hình AI.

`color_candidates(frame_bgr,cfg)` nhận BGR `uint8`, trả hai mask `fire`/`smoke` độc lập, cùng kích thước frame, 0/255. Pipeline dùng thêm `context_smoke_candidates` khi `smoke_engine='opencv_context'`. Mask ứng viên chưa phải kết quả cuối: TV2, Fusion, TV3 và TV4 còn lọc tiếp.

FIRE dùng ngưỡng HSV, lựa chọn luật RGB và bổ sung lõi sáng gần vùng lửa. SMOKE cổ điển dựa vào màu xám/ít bão hòa, độ sáng, độ mịn cục bộ; app chọn khói theo bối cảnh trong `smoke_context.py`. Tham số ở `fireguard/config.py`, preset app ở `fireguard/presets.py`. Nới ngưỡng khói có thể tăng bắt khói nhưng dễ lấy nhầm mây/nền sáng.

Chạy `python -m pytest tests/test_color.py tests/test_smoke_context.py -q`. Khi đổi mask, thử video lửa, khói, cả hai và âm tính; so sánh mask, box và vùng bỏ sót. Thí nghiệm TV1 ở `experiments/tv1/`. Kiểm tra tên file thật trong `data/development/` trước khi dùng `data/manifest.csv`, vì manifest chưa đồng bộ đầy đủ.
