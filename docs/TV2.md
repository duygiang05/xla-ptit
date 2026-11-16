# TV2 — Motion Detection

TV2 phụ trách `fireguard/motion/`: `frame_difference.py` đo sai khác frame, `background.py` dùng MOG2, `flow.py` tính optical flow hỗ trợ nhánh khói bối cảnh. Vùng chuyển động một mình không phải bằng chứng chắc chắn có lửa/khói.

`MotionDetector.apply(frame)` và `MOG2MotionDetector.apply(frame)` nhận frame BGR, trả mask 0/255 cùng kích thước. Pipeline dùng frame difference cho FIRE; app dùng MOG2 cho SMOKE và thêm optical flow trong nhánh bối cảnh. TV4 Fusion ghép motion với ứng viên màu theo từng loại. Video dọc có viền đen khi đưa vào khung xử lý, cần chú ý lúc đo motion/flow.

Chạy `python -m pytest tests/test_motion.py -q`. Thử camera tĩnh/rung, vật chuyển động không phải khói, khói mỏng và video âm tính. MOG2 cần thời gian thích nghi; đổi ánh sáng hoặc frame đầu dễ tạo mask giả. Script so sánh ở `experiments/tv2/`.
