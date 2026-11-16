# FireGuard — phát hiện vùng nghi lửa và khói bằng OpenCV

FireGuard là bài tập lớn xử lý video. Một frame có thể chứa cả `FIRE` và `SMOKE`. App Streamlit chỉ chạy pipeline OpenCV, không nạp mô hình AI/ONNX. Bounding box là **vùng nghi vấn theo thuật toán**, không phải xác suất hay xác nhận cháy; không dùng thay hệ thống báo cháy chuyên dụng.

## Cài đặt và chạy app

Yêu cầu Python 3.10+, FFmpeg có trong `PATH` và các gói trong `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ffmpeg -version
python -m streamlit run app.py
```

Nếu PowerShell chặn `Activate.ps1`, dùng `.venv\Scripts\python.exe -m pip install -r requirements.txt` và `.venv\Scripts\python.exe -m streamlit run app.py`. Mở URL Streamlit in ra, tải video `.mp4`, `.avi` hoặc `.mov`, bấm **Bắt đầu phân tích video**, xem kết quả và tải MP4/CSV. App cần FFmpeg để đổi MP4V thành H.264; thiếu FFmpeg thì nút phân tích bị khóa. File tải lên và đầu ra nằm trong thư mục tạm, app không tự quét `data/`.

Chạy CLI với video có sẵn (đổi `VIDEO.mp4` thành tên file thực):

```powershell
python -m fireguard.pipeline --input data/development/VIDEO.mp4 --output outputs/videos/result.mp4 --csv outputs/events/result.csv --smoke-engine opencv_context --smoke-vote-hits 5 --smoke-verify off
```

CLI có `--smoke-engine classical` mặc định; ví dụ trên chọn nhánh khói bối cảnh như app. CLI dùng `DEMO_OPTIONS` cho các tham số còn lại; xem `python -m fireguard.pipeline --help`. Video do CLI ghi chưa được chuyển H.264 như trong app.

## Cấu trúc và phân công

`video → TV1 (màu/khói theo bối cảnh) + TV2 (chuyển động) → TV4 Fusion (ghép mask) → TV3 (vùng/hình thái) → TV4 Temporal (bỏ phiếu) → box + CSV/video`.

| Thành viên | Mã nguồn | Vai trò |
| --- | --- | --- |
| TV1 | `fireguard/color/` | Mask FIRE/SMOKE từ màu, độ mịn và bối cảnh. |
| TV2 | `fireguard/motion/` | Frame difference, MOG2, optical flow. |
| TV3 | `fireguard/shape/` | Morphology, vùng, ghép vùng, tracking và kiểm tra bối cảnh. |
| TV4 | `fireguard/fusion/`, `fireguard/temporal/` | Kết hợp màu–chuyển động và lọc thời gian. |

`app.py`, `fireguard/pipeline.py`, `fireguard/config.py`, `fireguard/presets.py` là phần tích hợp chung. App chỉ có một preset khói OpenCV bối cảnh; FIRE giữ nhánh OpenCV hiện hành. Theo cấu hình app, đầu ra thường gộp tối đa một box mỗi loại trên một frame; box bao có thể lớn hơn vùng thật nếu các mảng tách rời bị gộp.

Tài liệu: [quy trình và hợp đồng dữ liệu](docs/WORKFLOW.md), [TV1](docs/TV1.md), [TV2](docs/TV2.md), [TV3](docs/TV3.md), [TV4 (Fusion + Temporal)](docs/TV4.md).

## Dữ liệu và kết quả

- `data/development/`: video thử nghiệm/chỉnh tham số. `data/integration/` và `data/test/` chỉ tạo lại khi có video tương ứng; các thư mục rỗng chỉ có `.gitkeep` đã được dọn.
- `data/manifest.csv`: danh mục video với `has_fire`, `has_smoke` độc lập. Manifest FIRESENSE lịch sử **chưa phản ánh đầy đủ các video hiện có trong development**; kiểm tra file thật trước khi chạy batch. Nhãn video không phải nhãn box.
- `data/annotations/`: mẫu nhãn sự kiện/box khi đánh giá định lượng. Video cùng nguồn phải nằm trong cùng split để tránh rò rỉ dữ liệu.
- `outputs/`: nơi CLI ghi kết quả, có thể vắng/trống; app không cần thư mục này. Không đưa video/đầu ra lớn lên Git.
- CSV có cột `frame,timestamp_sec,kind,x,y,w,h,score`. `score` không phải xác suất hiệu chuẩn; số frame phát hiện không phải precision/recall.

Video bên thứ ba có quyền sử dụng riêng: kiểm tra nguồn và điều khoản trong `data/manifest.csv` trước khi chia sẻ. License mã nguồn và license dữ liệu không đồng nhất.

## Giới hạn chất lượng và kiểm thử

Khói OpenCV **chưa tương đương ONNX** trên toàn tập. Lần đo lịch sử 25 video, ghép box SMOKE một-một với box ONNX tại IoU ≥ 0,5 cho F1 **0,528**. Đây chỉ là mức giống **nhãn giả ONNX**, không phải độ chính xác thật; một số box ONNX đã bị nghi sai. Model, công cụ so sánh và đầu ra cũ đã dọn khỏi repo, nên không thể tái lập chỉ số đó bằng repo hiện tại. Không cam kết 90% hay 100% trên video mới.

```powershell
python -m pytest -q -p no:cacheprovider --basetemp .pytest_tmp
```

Test đơn vị không thay thế thử upload→xem→tải bằng trình duyệt và đánh giá trên tập độc lập có nhãn. `models/` và `tools/` không còn là một phần của bản bàn giao hiện tại.
