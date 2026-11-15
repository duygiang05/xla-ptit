# Thực nghiệm so sánh TV4

Chạy PowerShell ở thư mục gốc của dự án (`D:\xla-ptit`):

```powershell
python -m experiments.compare_tv4 --max-frames 150
python -m experiments.compare_tv4
```

Hoặc một video cụ thể:

```powershell
python -m experiments.compare_tv4 --video data/development/ForestFire1.avi
```

Các file được tạo trong `experiments/results/tv4/`:

- `summary.csv`: thống kê mỗi video và cấu hình K/N (2/3, 3/5, 4/5).
- `<ten_video>_per_frame.csv`: số vùng FIRE/SMOKE trước và sau TV4 theo frame.

**Giải thích:** `before_*` là số vùng ứng viên sau TV3; `after_*` là số vùng còn lại sau TV4. `reduction_pct` chỉ đo phần trăm giảm **số vùng được vẽ theo frame**, không phải Precision/Recall hay false positive rate. Việc giảm vùng có thể đồng thời làm mất những vùng lửa/khói thật. `delay_from_first_candidate_sec` chỉ là độ trễ so với vùng ứng viên đầu tiên, **không phải** độ trễ phát hiện so với thời điểm cháy thực tế.

Script dùng đúng baseline TV1–TV3 từ `pipeline.py`, không thay đổi file của các thành viên. Mỗi cấu hình K/N có lịch sử độc lập. Khi TV1–TV3 cập nhật code, nên chạy lại để có kết quả mới.
