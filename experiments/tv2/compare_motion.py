
import cv2
import time
import numpy as np
from pathlib import Path
import csv

from fireguard.config import Config
from fireguard.motion.frame_difference import MotionDetector
from fireguard.motion.background import MOG2MotionDetector


# ==============================
# 1. CAU HINH
# ==============================

VIDEO_PATH = "data/development/vd_khoi.mp4"

OUTPUT_DIR = Path("experiments/tv2/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MOTION_THRESHOLD = 18
MOG2_HISTORY = 200
MOG2_VAR_THRESHOLD = 16

FRAME_WIDTH = 640
FRAME_HEIGHT = 360


# ==============================
# 2. MO VIDEO
# ==============================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Khong the mo video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

print("===== TV2 - MOTION DETECTION =====")
print(f"Video: {VIDEO_PATH}")
print(f"FPS: {fps:.2f}")
print(f"Thu muc luu anh: {OUTPUT_DIR.resolve()}")


# ==============================
# 3. KHOI TAO THUAT TOAN
# ==============================

frame_diff = MotionDetector(
    Config(motion_threshold=MOTION_THRESHOLD)
)

mog2 = MOG2MotionDetector(
    history=MOG2_HISTORY,
    var_threshold=MOG2_VAR_THRESHOLD
)

times_diff = []
times_mog2 = []

ratios_diff = []
ratios_mog2 = []

frame_count = 0


# ==============================
# 4. XU LY VIDEO
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    # ----- FRAME DIFFERENCE -----

    start = time.perf_counter()

    mask_diff = frame_diff.apply(frame)

    elapsed_diff = (
        time.perf_counter() - start
    ) * 1000

    times_diff.append(elapsed_diff)

    # ----- MOG2 -----

    start = time.perf_counter()

    mask_mog2 = mog2.apply(frame)

    elapsed_mog2 = (
        time.perf_counter() - start
    ) * 1000

    times_mog2.append(elapsed_mog2)

    # ----- THONG KE PIXEL -----

    ratio_diff = (
        np.count_nonzero(mask_diff)
        / mask_diff.size
    )

    ratio_mog2 = (
        np.count_nonzero(mask_mog2)
        / mask_mog2.size
    )

    ratios_diff.append(ratio_diff)
    ratios_mog2.append(ratio_mog2)

    # ----- TAO ANH HIEN THI -----

    display_diff = mask_diff.copy()
    display_mog2 = mask_mog2.copy()
    display_original = frame.copy()

    cv2.putText(
        display_original,
        f"Frame: {frame_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_diff,
        f"Frame Diff: {ratio_diff:.1%}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        255,
        2
    )

    cv2.putText(
        display_mog2,
        f"MOG2: {ratio_mog2:.1%}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        255,
        2
    )

    # ----- HIEN THI -----

    cv2.imshow(
        "Original Video",
        display_original
    )

    cv2.imshow(
        f"Frame Difference - T={MOTION_THRESHOLD}",
        display_diff
    )

    cv2.imshow(
        "MOG2 - History=200",
        display_mog2
    )

    key = cv2.waitKey(30) & 0xFF

    # ----- NHAN Q DE THOAT -----

    if key == ord("q"):
        break

    # ----- NHAN S DE LUU ANH -----

    if key == ord("s"):

        filename = (
            f"{Path(VIDEO_PATH).stem}"
            f"_T{MOTION_THRESHOLD}"
            f"_frame_{frame_count:04d}"
        )

        original_path = (
            OUTPUT_DIR / f"{filename}_original.png"
        )

        diff_path = (
            OUTPUT_DIR / f"{filename}_diff.png"
        )

        mog2_path = (
            OUTPUT_DIR / f"{filename}_mog2.png"
        )

        saved_original = cv2.imwrite(
            str(original_path),
            display_original
        )

        saved_diff = cv2.imwrite(
            str(diff_path),
            mask_diff
        )

        saved_mog2 = cv2.imwrite(
            str(mog2_path),
            mask_mog2
        )

        if all([
            saved_original,
            saved_diff,
            saved_mog2
        ]):
            print(
                f"[SAVED] Frame {frame_count}"
                f" -> {OUTPUT_DIR}"
            )
        else:
            print(
                f"[ERROR] Khong luu duoc "
                f"day du anh frame {frame_count}"
            )


# ==============================
# 5. GIAI PHONG TAI NGUYEN
# ==============================

cap.release()
cv2.destroyAllWindows()


# ==============================
# 6. KET QUA THONG KE
# ==============================

print("\n===== KET QUA THUC NGHIEM =====")

print(f"Tong so frame: {frame_count}")

if frame_count > 0:

    print("\n--- FRAME DIFFERENCE ---")

    print(
        f"Threshold: {MOTION_THRESHOLD}"
    )

    print(
        f"Thoi gian TB: "
        f"{np.mean(times_diff):.3f} ms/frame"
    )

    print(
        f"Ty le pixel trang TB: "
        f"{np.mean(ratios_diff):.2%}"
    )

    print("\n--- MOG2 ---")

    print(
        f"History: {MOG2_HISTORY}"
    )

    print(
        f"VarThreshold: {MOG2_VAR_THRESHOLD}"
    )

    print(
        f"Thoi gian TB: "
        f"{np.mean(times_mog2):.3f} ms/frame"
    )

    print(
        f"Ty le pixel trang TB: "
        f"{np.mean(ratios_mog2):.2%}"
    )



# ==============================
# 7. XUAT THONG KE RA CSV
# ==============================

if frame_count > 0:
    csv_path = OUTPUT_DIR / "motion_statistics.csv"

    # Kiem tra file da ton tai hay chua
    write_header = (
        not csv_path.exists()
        or csv_path.stat().st_size == 0
    )

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        # Ghi tieu de neu file moi
        if write_header:
            writer.writerow([
                "Video",
                "FPS",
                "Total Frames",
                "Algorithm",
                "Parameters",
                "Time (ms/frame)",
                "White Pixel Ratio (%)"
            ])

        # Frame Difference
        writer.writerow([
            Path(VIDEO_PATH).name,
            round(fps, 2),
            frame_count,
            "Frame Difference",
            f"threshold={MOTION_THRESHOLD}",
            round(float(np.mean(times_diff)), 3),
            round(float(np.mean(ratios_diff)) * 100, 4)
        ])

        # MOG2
        writer.writerow([
            Path(VIDEO_PATH).name,
            round(fps, 2),
            frame_count,
            "MOG2",
            f"history={MOG2_HISTORY};"
            f"varThreshold={MOG2_VAR_THRESHOLD}",
            round(float(np.mean(times_mog2)), 3),
            round(float(np.mean(ratios_mog2)) * 100, 4)
        ])

    print(f"\nDa luu thong ke tai: {csv_path}")

print("\nHoan thanh!")
