# TV3 — Shape, Region và kiểm tra bối cảnh

TV3 phụ trách `fireguard/shape/`: `morphology.py` làm sạch mask; `regions.py` tạo `Region`; `features.py` tính đặc trưng; `tracking.py` nối vùng qua frame; `grouping.py` gộp vùng; `context_verify.py` kiểm tra/điều chỉnh vùng khói bối cảnh. TV3 xử lý mask sau Fusion; pipeline có thể thêm xác minh theo bối cảnh ở nhánh khói.

`extract_regions(mask,kind,cfg)` trả danh sách `Region(kind,bbox,area,centroid,score,...)`. `bbox=(x,y,w,h)` theo pixel. TV4 Temporal dùng `RegionTracker`; sửa cách ghép track/ID có thể đổi kết quả bỏ phiếu. Pipeline có thể `union_regions` để xuất tối đa một box mỗi loại/frame, nhưng box bao có thể chứa nền nếu gộp các mảng xa nhau.

Chạy `python -m pytest tests/test_regions.py -q`. Thử mask rỗng, vùng chạm biên, nhiều mảng tách rời, video dọc có viền đen và khói sáng trên trời. Khi sửa `context_verify.py`, đối chiếu cả khói thật với mây/nền sáng/khói giả; không đánh giá chỉ bằng số box. Script TV3 ở `experiments/tv3/`.
