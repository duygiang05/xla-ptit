# TV1 – Thí nghiệm tiền xử lý và phân tích màu

Chạy mọi lệnh từ thư mục gốc repo, trong môi trường `.venv`.

| # | Thí nghiệm | Lệnh | Dữ liệu | Nhận xét |
|---|---|---|---|---|
| 1 | Luật lửa RGB / HSV / AND / OR, có/không blur | `python experiments/tv1/compare_rgb_hsv.py` | cảnh tổng hợp | [NOTES_rgb_vs_hsv.md](results/NOTES_rgb_vs_hsv.md) |
| 2 | Luật khói HSV / RGB-xám / lọc độ mịn, quét `smoke_s_max` | `python experiments/tv1/compare_smoke.py` | cảnh tổng hợp | [NOTES_smoke.md](results/NOTES_smoke.md) |
| 3 | Quét ngưỡng lửa/khói, blur, lõi lửa trên video thật | `python experiments/tv1/sweep_dev.py` | `data/development` | [NOTES_sweep_dev.md](results/NOTES_sweep_dev.md) |
| 4 | Chỉ màu so với màu + chuyển động (dùng TV2/TV3 như hộp đen) | `python experiments/tv1/color_vs_motion.py` | chuỗi tổng hợp | [NOTES_color_vs_motion.md](results/NOTES_color_vs_motion.md) |

Công cụ:
- `inspect_videos.py <thư mục>`: thông số video, độ trượt camera, ảnh tổng quan (`results/dataset/`).
- `prepare_firesense.py [--move]`: chia bộ FIRESENSE vào development/integration/test và ghi `data/manifest.csv`.
- `visualize_frames.py [--videos ...] [--set k=v ...] [--tag t]`: frame gốc | mask lửa | mask khói | tô màu (`results/real_frames/`).

Lưu ý: đường dẫn repo có dấu tiếng Việt nên `cv2.imwrite` thất bại mà không báo lỗi. Hãy dùng
`_utils.imwrite_unicode`. `cv2.VideoCapture` thì mở được bình thường.
