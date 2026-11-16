"""So sánh trực quan sai khác frame với MOG2 trên một video TV2.

Nhấn Q để dừng, S để lưu ảnh ba khung đang hiển thị.
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from fireguard.config import Config
from fireguard.motion import MOG2MotionDetector, MotionDetector


VIDEO_PATH = Path("data/development/demo1_30fps.mp4")
OUTPUT_DIR = Path("experiments/tv2/results")
MOTION_THRESHOLD = 18
MOG2_HISTORY = 200
MOG2_VAR_THRESHOLD = 16
FRAME_SIZE = (640, 360)


def label_frame(frame, title: str, color) -> np.ndarray:
    """Thêm nhãn so sánh lên bản sao của frame/mask."""
    result = frame.copy()
    cv2.putText(result, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return result


def save_statistics(
    video_path: Path,
    fps: float,
    frames: int,
    times_diff: list[float],
    times_mog2: list[float],
    ratios_diff: list[float],
    ratios_mog2: list[float],
) -> Path:
    """Ghi một dòng thống kê cho mỗi thuật toán vào CSV."""
    csv_path = OUTPUT_DIR / "motion_statistics.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        if write_header:
            writer.writerow(
                [
                    "Video",
                    "FPS",
                    "Total Frames",
                    "Algorithm",
                    "Parameters",
                    "Time (ms/frame)",
                    "White Pixel Ratio (%)",
                ]
            )
        for name, parameters, times, ratios in (
            (
                "Frame Difference",
                f"threshold={MOTION_THRESHOLD}",
                times_diff,
                ratios_diff,
            ),
            (
                "MOG2",
                f"history={MOG2_HISTORY};varThreshold={MOG2_VAR_THRESHOLD}",
                times_mog2,
                ratios_mog2,
            ),
        ):
            writer.writerow(
                [
                    video_path.name,
                    round(fps, 2),
                    frames,
                    name,
                    parameters,
                    round(float(np.mean(times)), 3),
                    round(float(np.mean(ratios)) * 100, 4),
                ]
            )
    return csv_path


def main() -> None:
    """Đọc video, hiển thị hai mask và lưu thống kê khi kết thúc."""
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(f"Không thể mở video: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_diff = MotionDetector(Config(motion_threshold=MOTION_THRESHOLD))
    mog2 = MOG2MotionDetector(history=MOG2_HISTORY, var_threshold=MOG2_VAR_THRESHOLD)
    times_diff, times_mog2 = [], []
    ratios_diff, ratios_mog2 = [], []
    frame_count = 0
    print(f"TV2 — video: {VIDEO_PATH}, FPS: {fps:.2f}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            frame = cv2.resize(frame, FRAME_SIZE)

            start = time.perf_counter()
            mask_diff = frame_diff.apply(frame)
            times_diff.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            mask_mog2 = mog2.apply(frame)
            times_mog2.append((time.perf_counter() - start) * 1000)

            ratio_diff = np.count_nonzero(mask_diff) / mask_diff.size
            ratio_mog2 = np.count_nonzero(mask_mog2) / mask_mog2.size
            ratios_diff.append(ratio_diff)
            ratios_mog2.append(ratio_mog2)

            original = label_frame(frame, f"Frame: {frame_count}", (0, 255, 0))
            diff_view = label_frame(mask_diff, f"Frame Diff: {ratio_diff:.1%}", 255)
            mog2_view = label_frame(mask_mog2, f"MOG2: {ratio_mog2:.1%}", 255)
            cv2.imshow("Original Video", original)
            cv2.imshow(f"Frame Difference - T={MOTION_THRESHOLD}", diff_view)
            cv2.imshow("MOG2 - History=200", mog2_view)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                stem = f"{VIDEO_PATH.stem}_T{MOTION_THRESHOLD}_frame_{frame_count:04d}"
                images = (
                    ("original", original),
                    ("diff", mask_diff),
                    ("mog2", mask_mog2),
                )
                saved = [
                    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_{kind}.png"), image)
                    for kind, image in images
                ]
                print(f"{'Đã lưu' if all(saved) else 'Lỗi lưu'} frame {frame_count}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Tổng số frame: {frame_count}")
    if frame_count == 0:
        return
    print(
        f"Sai khác frame: {np.mean(times_diff):.3f} ms, {np.mean(ratios_diff):.2%} pixel"
    )
    print(f"MOG2: {np.mean(times_mog2):.3f} ms, {np.mean(ratios_mog2):.2%} pixel")
    csv_path = save_statistics(
        VIDEO_PATH, fps, frame_count, times_diff, times_mog2, ratios_diff, ratios_mog2
    )
    print(f"Đã lưu thống kê: {csv_path}")


if __name__ == "__main__":
    main()
