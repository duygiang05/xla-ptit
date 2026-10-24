# Quy ước 4 thành viên – một nhánh main

- TV1 chỉ sửa `fireguard/color/` và `tests/test_color.py`.
- TV2 chỉ sửa `fireguard/motion/` và `tests/test_motion.py`.
- TV3 chỉ sửa `fireguard/shape/` và `tests/test_regions.py`.
- TV4 chỉ sửa `fireguard/temporal/` và `tests/test_temporal.py`.
- Nhóm trưởng quản lý `fireguard/pipeline.py`, `fireguard/config.py`, `app.py`, `tests/test_pipeline.py`; TV4 vẫn cần được chia sẻ khối lượng công việc tích hợp khi cần.
- Mỗi người có thể lưu thí nghiệm dưới `experiments/tv1`, `tv2`, `tv3`, `tv4` (tự tạo lúc sử dụng).
- Không tự ý đổi tên API/module hoặc cấu trúc Region; nếu cần đổi phải trao đổi và cập nhật test chung.
- Mọi người `git pull --rebase origin main` trước khi làm (khi working tree sạch), `python -m pytest -q` trước khi push, và không force-push main.
- Chỉ dùng main giúp đơn giản nhưng KHÔNG loại trừ conflict hoặc lỗi tích hợp.
