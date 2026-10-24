# FIREGUARD – BTL Xử lý ảnh | 4 thành viên | 14 ngày

**Mục tiêu:** phát hiện và khoanh vùng nghi ngờ lửa và/hoặc khói trong CÙNG MỘT video. Một frame có thể chứa đồng thời vùng FIRE và SMOKE. Hệ thống là bản thử nghiệm học thuật, không thay thế cảnh báo cháy chuyên dụng.

## Cài đặt trên Windows (Python 3.10+)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

Chạy CLI trên video thật (phải tự bổ sung video):

```powershell
python -m fireguard.pipeline --input data/integration/my_video.mp4 --output outputs/videos/result.mp4 --csv outputs/events/events.csv
```

## Dự án

- `fireguard/color/` – TV1: ứng viên FIRE và SMOKE, hai mask độc lập.
- `fireguard/motion/` – TV2: phát hiện chuyển động chung.
- `fireguard/shape/` – TV3: hình thái học, vùng, đặc trưng hình dạng.
- `fireguard/temporal/` – TV4: phân tích thời gian và lọc phát hiện sai.
- `fireguard/pipeline.py`, `config.py`, `app.py` – nhóm trưởng quản lý tích hợp.
- `data/development/` – video để thử tham số; `data/integration/` – video tích hợp; `data/test/` – đánh giá cuối, không dùng để chỉnh tham số.
- `data/manifest.csv` – MỖI video MỘT hàng; `has_fire`, `has_smoke` nhận độc lập 0 hoặc 1; ví dụ cả hai bằng 1.
- `tests/` – kiểm thử code; `experiments/` – kết quả so sánh thuật toán; `outputs/` – kết quả sinh ra; `docs/` – giao tiếp module, quy trình và lịch.

**Chưa kèm video thực / ngưỡng đã tối ưu / chất lượng phát hiện được xác nhận.** Các thuật toán sẵn có chỉ là *baseline*. Không cần chia dữ liệu thành folder `fire/smoke/negative`, vì video có thể đa nhãn.

Bốn người dùng **một nhánh `main`**, mỗi người sửa chủ yếu thư mục được giao, tránh sửa chung `pipeline.py`, `config.py` nếu chưa thống nhất. Đọc `docs/WORKFLOW.md` và `docs/INTERFACES.md` trước khi code.
