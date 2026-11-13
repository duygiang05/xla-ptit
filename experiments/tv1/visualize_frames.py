"""TV1 – Minh họa mask màu trên VIDEO THẬT: frame gốc | mask lửa | mask khói | frame tô màu.

Chạy từ thư mục gốc repo:
    python experiments/tv1/visualize_frames.py                       # mọi video trong data/development
    python experiments/tv1/visualize_frames.py --videos fs_smoke_pos02 fs_fire_neg11
    python experiments/tv1/visualize_frames.py --set smoke_s_max=20 smoke_v_min=130 --tag s20

Chỉ đọc data/development (KHÔNG dùng data/test để xem/chỉnh ngưỡng).
Đầu ra CHÍNH THỨC của TV1 vẫn là mask đen trắng 0/255; ảnh tô màu ở đây chỉ để người xem.
Kết quả (experiments/tv1/results/real_frames/):
  <video_id>.jpg         : 4 ảnh cạnh nhau tại frame giữa video
  mask_coverage_dev.csv  : % pixel bị đánh dấu lửa/khói, trung bình trên ~20 frame rải đều
"""
import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fireguard.color import color_candidates  # noqa: E402
from fireguard.color.preprocessing import resize_keep_aspect  # noqa: E402
from fireguard.config import Config  # noqa: E402
from _utils import imwrite_unicode, label  # noqa: E402

OUT = ROOT / "experiments" / "tv1" / "results" / "real_frames"
FIRE_BGR, SMOKE_BGR = (0, 0, 255), (255, 150, 0)   # đỏ = lửa, xanh = khói


def overlay(frame, masks, alpha=0.5):
    """Tô màu bán trong suốt lên frame ở các pixel mask = 255 (chỉ để hiển thị)."""
    out = frame.astype(np.float32)
    for key, col in (("smoke", SMOKE_BGR), ("fire", FIRE_BGR)):
        on = masks[key] > 0
        out[on] = (1 - alpha) * out[on] + alpha * np.array(col, np.float32)
    return out.astype(np.uint8)


def read_frames(path, indices):
    """Đọc tuần tự (an toàn với .avi cũ) và giữ lại các frame có chỉ số trong `indices`."""
    cap = cv2.VideoCapture(os.path.relpath(path))
    want, got, i = set(indices), {}, 0
    while want - got.keys():
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = frame
        i += 1
    cap.release()
    return got


def load_manifest():
    with (ROOT / "data" / "manifest.csv").open(encoding="utf-8") as f:
        return {r["video_id"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", help="video_id cần xem (mặc định: cả development)")
    ap.add_argument("--samples", type=int, default=20, help="số frame lấy để tính độ phủ")
    ap.add_argument("--set", nargs="*", default=[], help="ghi đè Config, vd smoke_s_max=20")
    ap.add_argument("--tag", default="", help="hậu tố tên file kết quả, vd s20")
    args = ap.parse_args()
    cfg = Config()
    changes = {}
    for item in args.set:                      # ép kiểu theo giá trị mặc định trong Config
        k, v = item.split("=", 1)
        changes[k] = type(getattr(cfg, k))(v)
    cfg = replace(cfg, **changes)
    suffix = f"_{args.tag}" if args.tag else ""
    manifest = load_manifest()
    ids = args.videos or [v for v, r in manifest.items() if r["split"] == "development"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid in ids:
        r = manifest[vid]
        if r["split"] == "test":
            print(f"BỎ QUA {vid}: thuộc tập test")
            continue
        path = ROOT / "data" / r["split"] / r["filename"]
        cap = cv2.VideoCapture(os.path.relpath(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n <= 0:
            print(f"LỖI không đọc được {vid}")
            continue
        idx = sorted({int((k + 0.5) * n / args.samples) for k in range(args.samples)})
        frames = read_frames(path, idx)
        cover_f, cover_s = [], []
        for k in sorted(frames):
            small = resize_keep_aspect(frames[k], cfg.width, cfg.height)
            m = color_candidates(small, cfg)
            cover_f.append(np.count_nonzero(m["fire"]) / m["fire"].size)
            cover_s.append(np.count_nonzero(m["smoke"]) / m["smoke"].size)
        mid = sorted(frames)[len(frames) // 2]
        small = resize_keep_aspect(frames[mid], cfg.width, cfg.height)
        m = color_candidates(small, cfg)
        panel = np.hstack([label(small, f"{vid} frame {mid}"),
                           label(m["fire"], f"fire mask {np.mean(m['fire'] > 0):.1%}"),
                           label(m["smoke"], f"smoke mask {np.mean(m['smoke'] > 0):.1%}"),
                           label(overlay(small, m), "overlay: red=fire blue=smoke")])
        imwrite_unicode(OUT / f"{vid}{suffix}.jpg", panel)
        rows.append({"video_id": vid, "has_fire": r["has_fire"], "has_smoke": r["has_smoke"],
                     "frames_used": len(frames),
                     "fire_cover_mean": round(float(np.mean(cover_f)), 4),
                     "fire_cover_max": round(float(np.max(cover_f)), 4),
                     "smoke_cover_mean": round(float(np.mean(cover_s)), 4),
                     "smoke_cover_max": round(float(np.max(cover_s)), 4),
                     "notes": r["notes"]})
        print(f"{vid:16s} fire={r['has_fire']} smoke={r['has_smoke']} | "
              f"mask lửa TB {rows[-1]['fire_cover_mean']:.1%} | mask khói TB {rows[-1]['smoke_cover_mean']:.1%}")
    if rows:
        with (OUT / f"mask_coverage_dev{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nĐã lưu vào {OUT}")


if __name__ == "__main__":
    main()
