"""TV1 – Thí nghiệm 3: quét ngưỡng màu + có/không blur trên VIDEO THẬT (data/development).

Chạy từ thư mục gốc repo:
    python experiments/tv1/sweep_dev.py

Nhãn FIRESENSE chỉ ở MỨC VIDEO (has_fire / has_smoke), không có nhãn pixel. Vì vậy đo ở mức frame:
  một frame "BÁO" nếu mask có ≥ 1 thành phần liên thông ≥ cfg.min_region_area pixel
  (cùng ngưỡng diện tích TV3 dùng để giữ vùng).
  - recall   : tỉ lệ frame BÁO trong video CÓ lửa/khói        (cần cao – ưu tiên của mask màu)
  - fpr      : tỉ lệ frame BÁO trong video KHÔNG có lửa/khói  (muốn thấp)
  - cover_pos / cover_neg : % pixel bị bật trung bình ở video có / không có
Hạn chế: video "có khói" cũng có trời/tường -> mask bật ở đó vẫn bị tính là "đúng". recall ở đây
là cận trên lạc quan; fpr mới là số đáng tin. Không dùng data/test.

Mốc so sánh (BASELINE) là các giá trị mặc định TRƯỚC khi chỉnh, ghi cố định để chạy lại vẫn ra
cùng kết quả. Quy tắc chọn tự động: trong các cấu hình có recall >= recall(mốc) - 0.02, chọn fpr
nhỏ nhất (hòa thì cover_neg nhỏ nhất). Cấu hình CUỐI còn được chọn bằng mắt (giữ khói đen, giữ
>= 90% độ phủ lửa) – xem experiments/tv1/results/NOTES_sweep_dev.md.
Kết quả (experiments/tv1/results/sweep/): blur_ablation.csv, fire_sweep.csv, smoke_sweep.csv,
  fire_core_sweep.csv, final_vs_baseline.csv
"""
import csv
import itertools
import os
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fireguard.color import color_candidates  # noqa: E402
from fireguard.color.preprocessing import resize_keep_aspect  # noqa: E402
from fireguard.config import Config  # noqa: E402

OUT = ROOT / "experiments" / "tv1" / "results" / "sweep"
SAMPLES = 20  # frame rải đều mỗi video
# Mặc định TRƯỚC khi chỉnh (các trường còn lại lấy từ Config() hiện tại).
BASELINE = dict(smoke_s_max=70, smoke_texture_max=0.0, fire_core_v_min=0)
CORE_GRID = {"fire_core_v_min": (200, 220, 240), "fire_core_s_max": (60, 100), "fire_core_dilate": (7, 15)}

FIRE_GRID = {"fire_s_min": (60, 80, 100, 130), "fire_v_min": (100, 125, 150, 180),
             "fire_h_high": (25, 35), "fire_color_mode": ("hsv", "and")}
SMOKE_GRID = {"smoke_s_max": (20, 30, 40, 55, 70), "smoke_v_min": (65, 100, 130),
              "smoke_texture_max": (0.0, 8.0, 12.0)}


def load_dev_frames(cfg):
    with (ROOT / "data" / "manifest.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "development"]
    data = []
    for r in rows:
        path = os.path.relpath(ROOT / "data" / "development" / r["filename"])
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = {int((k + 0.5) * n / SAMPLES) for k in range(SAMPLES)}
        i, frames = 0, []
        while len(frames) < len(want):          # đọc tuần tự, an toàn với .avi cũ
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                frames.append(resize_keep_aspect(fr, cfg.width, cfg.height))
            i += 1
        cap.release()
        data.append({"id": r["video_id"], "fire": int(r["has_fire"]), "smoke": int(r["has_smoke"]),
                     "frames": frames})
    return data


def fires(mask, min_area):
    """Frame có báo không: thành phần liên thông lớn nhất >= min_area."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return n > 1 and stats[1:, cv2.CC_STAT_AREA].max() >= min_area


def evaluate(data, cfg, kind):
    hit = {0: [], 1: []}
    cover = {0: [], 1: []}
    speck = []
    for v in data:
        for fr in v["frames"]:
            m = color_candidates(fr, cfg)[kind]
            hit[v[kind]].append(fires(m, cfg.min_region_area))
            cover[v[kind]].append(np.count_nonzero(m) / m.size)
            n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            speck.append(int(np.sum(st[1:, cv2.CC_STAT_AREA] <= 10)))
    return {"recall": float(np.mean(hit[1])), "fpr": float(np.mean(hit[0])),
            "cover_pos": float(np.mean(cover[1])), "cover_neg": float(np.mean(cover[0])),
            "speckles_per_frame": float(np.mean(speck))}


def sweep(data, base, grid, kind):
    rows = []
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        rows.append({**params, **evaluate(data, replace(base, **params), kind)})
    default = {k: getattr(base, k) for k in keys}
    ref = next((r for r in rows if all(r[k] == default[k] for k in keys)), None)
    if ref is None:  # mặc định không nằm trong lưới -> đo riêng
        ref = {**default, **evaluate(data, base, kind)}
        rows.append(ref)
    ok = [r for r in rows if r["recall"] >= ref["recall"] - 0.02]
    best = min(ok, key=lambda r: (r["fpr"], r["cover_neg"]))
    return rows, ref, best


def write(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def show(title, r, keys):
    cfg_txt = ", ".join(f"{k}={r[k]}" for k in keys)
    print(f"  {title:9s} recall {r['recall']:.2f} | fpr {r['fpr']:.2f} | phủ có {r['cover_pos']:.1%} "
          f"| phủ không {r['cover_neg']:.1%} | {cfg_txt}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final = Config()                      # mặc định hiện tại (sau khi chỉnh)
    base = replace(final, **BASELINE)     # mốc trước khi chỉnh
    data = load_dev_frames(base)
    n_pos_f = sum(v["fire"] for v in data)
    n_pos_s = sum(v["smoke"] for v in data)
    print(f"{len(data)} video development, {sum(len(v['frames']) for v in data)} frame "
          f"(có lửa: {n_pos_f} video, có khói: {n_pos_s} video)\n")

    blur_rows = []
    for k in (1, 5, 9):
        for kind in ("fire", "smoke"):
            blur_rows.append({"kind": kind, "blur_kernel": k, **evaluate(data, replace(base, blur_kernel=k), kind)})
    write("blur_ablation.csv", blur_rows)
    print("Blur (cấu hình mặc định):")
    for r in blur_rows:
        print(f"  {r['kind']:5s} blur={r['blur_kernel']}: recall {r['recall']:.2f} fpr {r['fpr']:.2f} "
              f"đốm nhiễu/frame {r['speckles_per_frame']:.0f}")

    for kind, grid, fname in (("fire", FIRE_GRID, "fire_sweep.csv"), ("smoke", SMOKE_GRID, "smoke_sweep.csv")):
        rows, ref, best = sweep(data, base, grid, kind)
        write(fname, rows)
        print(f"\n{kind.upper()} – {len(rows)} cấu hình:")
        show("mốc", ref, list(grid))
        show("tốt nhất", best, list(grid))

    core_rows = [{**p, **evaluate(data, replace(base, **p), "fire")}
                 for p in (dict(zip(CORE_GRID, v)) for v in itertools.product(*CORE_GRID.values()))]
    write("fire_core_sweep.csv", core_rows)
    print("\nLõi lửa (luật màu lửa giữ nguyên mốc):")
    for r in core_rows:
        show("", r, list(CORE_GRID))

    final_rows = [{"kind": k, "config": name, **evaluate(data, c, k)}
                  for k in ("fire", "smoke") for name, c in (("baseline", base), ("final", final))]
    write("final_vs_baseline.csv", final_rows)
    print("\nMốc trước khi chỉnh vs cấu hình cuối (Config() hiện tại):")
    for r in final_rows:
        print(f"  {r['kind']:5s} {r['config']:8s} recall {r['recall']:.2f} | fpr {r['fpr']:.2f} | "
              f"phủ có {r['cover_pos']:.1%} | phủ không {r['cover_neg']:.1%}")
    print(f"\nĐã lưu vào {OUT}")


if __name__ == "__main__":
    main()
