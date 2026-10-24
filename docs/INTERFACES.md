# INTERFACES – Quy ước giữa bốn module

- Một video đầu vào được xử lý **đồng thời** theo hai nhánh FIRE và SMOKE; không gán một nhãn loại trừ cho toàn video.
- Frame đầu vào: `frame_bgr: np.ndarray`, shape `(H,W,3)`, dtype `uint8`, BGR.
- Mask nhị phân: `np.ndarray`, shape `(H,W)`, dtype `uint8`, giá trị 0 hoặc 255.
- TV1 `color_candidates(frame,cfg)` trả `{'fire': fire_mask, 'smoke': smoke_mask}` trong cùng frame. Hai mask có thể cùng có vùng ứng viên.
- TV2 `MotionDetector.apply(frame)` trả `motion_mask` cho frame hiện tại.
- TV3 `extract_regions(mask, kind, cfg)` trả danh sách `Region(kind,bbox,area,centroid,score)`, `kind` là `fire` hoặc `smoke`; gọi độc lập cho từng nhánh và **nối hai danh sách**, không gộp thành một nhãn.
- TV4 `TemporalFilter.update(regions,frame_idx)` nhận đồng thời cả danh sách, quyết định từng vùng theo `kind`; có thể trả cả FIRE và SMOKE trong một frame.
- `fireguard/pipeline.py` ghép hai mask màu với motion mask và chạy TV3/TV4. Phép giao cứng hiện chỉ là **baseline cần kiểm chứng**, nhất là với khói ít chuyển động.
- CSV đầu ra: `frame,timestamp_sec,kind,x,y,w,h,score`; một frame có thể có nhiều hàng, kể cả `kind=fire` và `kind=smoke`.
- Dataset: `data/manifest.csv` lưu `has_fire` và `has_smoke` độc lập; một video chỉ xuất hiện một lần theo `video_id`; giữ các đoạn cùng nguồn trong cùng split.
- Cấm tự ý đổi chữ ký hàm/kiểu dữ liệu mà không thống nhất với nhóm trưởng.
