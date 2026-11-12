"""TV1 – Kiểm tra nhanh một thư mục video: thông số + ảnh tổng quan (contact sheet).

Chạy từ thư mục gốc repo:
    python experiments/tv1/inspect_videos.py data/raw/firesense
    python experiments/tv1/inspect_videos.py data/development --frames 6

Kết quả (experiments/tv1/results/dataset/):
  <tên thư mục>_overview.csv : thông số, số frame đọc được, độ trượt camera (px), ghi chú
  <tên thư mục>_sheet_XX.jpg : mỗi hàng là 1 video, gồm N frame rải đều + tên file
Dùng để: phát hiện video hỏng, xem nhanh nội dung (lửa? khói? camera rung?) trước khi chia
development/integration/test và điền data/manifest.csv.

Lưu ý Windows: đường dẫn repo có dấu ("Học tập"). Script mở video bằng đường dẫn TƯƠNG ĐỐI
so với thư mục hiện tại; nếu mở bằng đường dẫn tuyệt đối thất bại thì ghi chú vào cột note.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import imwrite_unicode  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".mpg", ".mpeg", ".m4v", ".3gp"}
OUT = ROOT / "experiments" / "tv1" / "results" / "dataset"
THUMB_W, THUMB_H, ROWS_PER_SHEET = 240, 135, 12


def open_video(path: Path):
    """Thử đường dẫn tương đối trước (an toàn với thư mục có dấu), rồi tới tuyệt đối."""
    rel = os.path.relpath(path)
    cap = cv2.VideoCapture(rel)
    if cap.isOpened():
        abs_ok = cv2.VideoCapture(str(path.resolve()))
        note = "" if abs_ok.isOpened() else "abs-path-fails"
        abs_ok.release()
        return cap, note
    cap.release()
    return None, "cannot-open"


def camera_shift(prev_gray, gray):
    """Độ trượt toàn khung (pixel) giữa 2 frame, ước lượng bằng tương quan pha (phaseCorrelate).

    Camera cố định -> ~0 px (vật chuyển động nhỏ không làm trượt cả khung);
    camera cầm tay/quét -> vài px trở lên. Dùng để loại video không hợp với frame difference.
    """
    (dx, dy), _ = cv2.phaseCorrelate(prev_gray, gray)
    return float(np.hypot(dx, dy))


def inspect(path: Path, n_frames: int):
    row = {"path": os.path.relpath(path).replace("\\", "/"), "width": 0, "height": 0, "fps": 0.0,
           "frames_meta": 0, "frames_read": 0, "duration_sec": 0.0,
           "cam_shift_med": 0.0, "cam_shift_p90": 0.0, "ok": False, "note": ""}
    cap, row["note"] = open_video(path)
    if cap is None:
        return row, []
    row["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    row["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    row["fps"] = round(cap.get(cv2.CAP_PROP_FPS) or 0.0, 2)
    row["frames_meta"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(row["frames_meta"], 1)
    targets = {int((i + 0.5) * total / n_frames): i for i in range(n_frames)}
    gap = max(1, round((row["fps"] or 25) / 2))          # so sánh 2 frame cách nhau ~0.5 s
    thumbs = [np.zeros((THUMB_H, THUMB_W, 3), np.uint8)] * n_frames
    shifts, anchor, idx = [], None, 0
    # Đọc TUẦN TỰ (không tua) – tua bằng CAP_PROP_POS_FRAMES hay hỏng với một số codec .avi cũ
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in targets:
            thumbs[targets[idx]] = cv2.resize(frame, (THUMB_W, THUMB_H))
        if idx % gap == 0:
            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if anchor is not None:
                shifts.append(camera_shift(anchor, gray) * row["width"] / 320)  # đổi về pixel gốc
            anchor = gray
        idx += 1
    cap.release()
    row["frames_read"] = idx
    if row["fps"] > 0:
        row["duration_sec"] = round(idx / row["fps"], 1)
    if shifts:
        row["cam_shift_med"] = round(float(np.median(shifts)), 2)
        row["cam_shift_p90"] = round(float(np.percentile(shifts, 90)), 2)
    row["ok"] = idx > 0
    notes = [row["note"]] if row["note"] else []
    if idx == 0:
        notes.append("no-frames")
    elif row["frames_meta"] and idx < 0.9 * row["frames_meta"]:
        notes.append(f"read-only-{idx}-of-{row['frames_meta']}")
    if row["cam_shift_p90"] > 3:
        notes.append("camera-moving?")
    row["note"] = " ".join(notes)
    return row, thumbs


def sheet_row(name, thumbs):
    strip = np.hstack(thumbs)
    bar = np.zeros((22, strip.shape[1], 3), np.uint8)
    cv2.putText(bar, name[-70:], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([bar, strip])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--frames", type=int, default=4, help="số frame lấy mẫu mỗi video")
    args = ap.parse_args()
    folder = Path(args.folder)
    videos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        print(f"Không tìm thấy video trong {folder}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    tag = folder.name
    rows, strips = [], []
    for p in videos:
        row, thumbs = inspect(p, args.frames)
        rows.append(row)
        if thumbs:
            strips.append(sheet_row(row["path"], thumbs))
        print(f"{'OK ' if row['ok'] else 'ERR'} {row['width']}x{row['height']} {row['fps']:>5}fps "
              f"{row['duration_sec']:>6}s  shift med/p90 {row['cam_shift_med']:>5}/{row['cam_shift_p90']:<6} "
              f"{Path(row['path']).name}  {row['note']}")

    with (OUT / f"{tag}_overview.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for k in range(0, len(strips), ROWS_PER_SHEET):
        imwrite_unicode(OUT / f"{tag}_sheet_{k // ROWS_PER_SHEET + 1:02d}.jpg",
                        np.vstack(strips[k:k + ROWS_PER_SHEET]))
    print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} video đọc được. Kết quả: {OUT}")


if __name__ == "__main__":
    main()
