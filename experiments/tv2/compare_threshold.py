import csv
import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = "data/development/vd_khoi.mp4"
OUTPUT_DIR = Path("experiments/tv2/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [8, 12, 18]
TARGET_FRAME = 149

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(VIDEO_PATH)

previous = None
frame_count = 0
ratios = {t: [] for t in THRESHOLDS}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (640, 360))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if previous is None:
        diff = np.zeros_like(gray)
    else:
        diff = cv2.absdiff(gray, previous)

    previous = gray.copy()

    for t in THRESHOLDS:
        _, mask = cv2.threshold(
            diff, t, 255, cv2.THRESH_BINARY
        )

        ratios[t].append(
            np.count_nonzero(mask) / mask.size
        )

        if frame_count == TARGET_FRAME:
            path = OUTPUT_DIR / (
                f"vd_khoi_T{t}_frame_{frame_count:04d}_diff.png"
            )
            cv2.imwrite(str(path), mask)

    if frame_count == TARGET_FRAME:
        cv2.imwrite(
            str(OUTPUT_DIR / "vd_khoi_frame_0149_original.png"),
            frame
        )
        print("Da luu ket qua frame 149")

cap.release()

print("\n===== SO SANH THRESHOLD =====")
for t in THRESHOLDS:
    print(
        f"T={t}: "
        f"{np.mean(ratios[t]):.4%} pixel trang TB"
    )


# ==============================
# XUAT THONG KE THRESHOLD RA CSV
# ==============================

csv_path = OUTPUT_DIR / "threshold_statistics.csv"

if frame_count > 0:
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Video",
            "Threshold",
            "Total Frames",
            "White Pixel Ratio (%)"
        ])

        for t in THRESHOLDS:
            writer.writerow([
                Path(VIDEO_PATH).name,
                t,
                frame_count,
                round(
                    float(np.mean(ratios[t])) * 100,
                    4
                )
            ])

    print(f"Da luu thong ke: {csv_path}")


print("Hoan thanh!")
