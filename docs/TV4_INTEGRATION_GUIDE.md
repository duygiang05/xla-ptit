# TV4 – Hoàn thiện Temporal Analysis, kết nối TV3

## Phạm vi thay đổi

Chỉ thay 4 file `fireguard/temporal/`, thêm `tests/test_temporal_hybrid.py`,
`experiments/tv4/compare_decisions.py` và tài liệu này. **Không sửa** TV1, TV2, TV3,
`fireguard/config.py`, `fireguard/pipeline.py` hay code Streamlit.

## Bàn giao cho pipeline

```python
from fireguard.temporal import TemporalFilter
filt = TemporalFilter(cfg)  # vote: tương thích pipeline hiện tại
confirmed = filt.update(regions, frame_idx)
```

`regions`: kết quả `extract_regions()` của TV3; `confirmed`: danh sách `Region`
được xác nhận ở **frame hiện tại**. `Region.score` được giữ nguyên, không phải xác suất.

- TV3 `RegionTracker` đảm nhận ID và liên kết vùng theo IoU/khoảng cách tâm.
- TV3 `Track.shape_change()` tính `area_cv`, `perimeter_cv`, `circularity_std`,
  `area_growth`, `centroid_shift`, `bbox_iou_mean`.
- TV4 đếm K/N trên lịch sử `Track.history` bằng frame index; không đếm frame mất dấu.
- `max_missed=2` mặc định; hơn hai frame mất dấu thì vùng được coi là track mới.
- `last_decisions` lưu thống kê và lý do quyết định **để debug**, không tự xuất bởi pipeline.

## Hai chế độ và lựa chọn đúng

- `decision_mode='vote'` (mặc định): đạt K/N. Chính là baseline chốt để so sánh.
- `decision_mode='hybrid'`: phải đạt K/N VÀ bằng chứng biến dạng hình học.

Cấu hình thử nghiệm mặc định Hybrid (có thể truyền `ShapeThresholds(...)` khác):

| Nhãn | Luật biến dạng hình học | Ngưỡng mặc định THỬ NGHIỆM |
|---|---|---|
| FIRE | `area_cv` OR `perimeter_cv` OR `circularity_std` | >= 0.08 / >= 0.08 / >= 0.025 |
| SMOKE | `area_cv` OR `area_growth` OR `centroid_shift` | >= 0.08 / >= 0.10 / >= 1.5px |

**Các ngưỡng Hybrid chưa được chốt bằng nhãn thật.** Chỉ sử dụng làm phương án ablation.
Lửa ổn định, khói ít chuyển động, rung camera hoặc tách/gộp vùng có thể làm Hybrid
bỏ sót hoặc báo nhầm. Không gọi mức này là ``confidence`` hay xác suất cháy.

## Kiểm thử

```powershell
python -m pytest tests/test_temporal.py tests/test_temporal_hybrid.py -v
python -m pytest tests/ -v
```

So sánh trên video thật, giữ nguyên Color/Motion/Shape của pipeline:

```powershell
python experiments/tv4/compare_decisions.py --video data/development/ForestFire1.avi --max-frames 300
```

Đầu ra: `experiments/tv4/results/comparison_frames.csv` và
`comparison_regions.csv`. Lưu ý tổng `vote_confirmed` là số lượt **Region-frame**
(không phải số cháy hoặc false positives). Cần video âm tính + nhãn để đánh giá
Precision / Recall / F1 và độ trễ.

## Khi nào bật Hybrid trong pipeline?

Chỉ sau khi kết quả thử nghiệm trên development/integration cho thấy có lợi.
Trong `fireguard/pipeline.py` có dòng:

```python
temporal = TemporalFilter(cfg)
```

Khi đã quyết định, thay duy nhất thành:

```python
temporal = TemporalFilter(cfg, decision_mode='hybrid')
```

Không phải sửa TV1/TV2/TV3. Mặc định của gói giữ nguyên dòng cũ để an toàn.

## Giới hạn cần báo cáo

- Dùng greedy association của TV3, chưa phải multi-object tracker hoàn chỉnh.
- Track có thể đổi ID khi ngọn lửa tách/gộp hoặc di chuyển mạnh.
- Bỏ phiếu K/N cần được tối ưu cùng chất lượng mask đầu vào.
- Không cam kết phân biệt chắc chắn lửa với vật đỏ chuyển động.
