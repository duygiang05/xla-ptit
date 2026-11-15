"""So sánh trực quan regions trước TV4 và sau Temporal Voting.

Chạy từ thư mục gốc FireGuard:
    python -m experiments.visualize_tv4 --video data/development/barbeq.avi --k 3 --n 5 --max-frames 150

Chỉ sử dụng pipeline baseline để lấy cùng regions cho cả hai nửa màn hình.
Kết quả là video MP4 side-by-side và bảng thống kê; không phải đánh giá độ chính xác.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from fireguard.color import color_candidates
from fireguard.config import Config
from fireguard.motion import MotionDetector
from fireguard.shape import extract_regions
from fireguard.temporal import TemporalFilter


def draw_regions(frame, regions):
    result = frame.copy()
    for region in regions:
        x, y, w, h = region.bbox
        color = (0, 0, 255) if region.kind == 'fire' else (255, 200, 0)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result, region.kind.upper(), (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
    return result


def make_panel(frame, regions, label, counts):
    annotated = draw_regions(frame, regions)
    bar = np.zeros((52, annotated.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(bar, f"FIRE: {counts['fire']}  SMOKE: {counts['smoke']}",
                (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)
    return np.vstack((bar, annotated))


def count_kinds(regions):
    return {label: sum(r.kind == label for r in regions) for label in ('fire', 'smoke')}


def run(video_path: Path, output_path: Path, k: int, n: int, max_frames: int):
    if not video_path.is_file():
        raise FileNotFoundError(f'Không tìm thấy video: {video_path}')
    cfg = replace(Config(), vote_hits=k, vote_window=n)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'OpenCV không mở được video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_size = (cfg.width * 2, cfg.height + 52)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, output_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Không tạo được MP4 tại: {output_path}')

    motion = MotionDetector(cfg)
    temporal = TemporalFilter(cfg)
    frame_idx = 0
    total_before = 0
    total_after = 0
    try:
        while max_frames == 0 or frame_idx < max_frames:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.resize(frame, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
            masks = color_candidates(frame, cfg)
            movement = motion.apply(frame)
            fire_mask = cv2.bitwise_and(masks['fire'], movement)
            smoke_mask = cv2.bitwise_and(masks['smoke'], movement)
            regions = (extract_regions(fire_mask, 'fire', cfg)
                       + extract_regions(smoke_mask, 'smoke', cfg))
            accepted = temporal.update(regions, frame_idx)
            total_before += len(regions)
            total_after += len(accepted)
            left = make_panel(frame, regions, 'TRUOC TV4 - Ung vien', count_kinds(regions))
            right = make_panel(frame, accepted, f'SAU TV4 - K={k}, N={n}',
                               count_kinds(accepted))
            writer.write(np.hstack((left, right)))
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    if frame_idx == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError('Không đọc được frame nào từ video')
    percent = 100.0 * (total_before - total_after) / total_before if total_before else 0.0
    print(f'Video: {video_path.name}; frames: {frame_idx}')
    print(f'Truoc TV4: {total_before} luot vung; Sau TV4: {total_after} luot vung')
    print(f'Ty le giam so LUOT VUNG: {percent:.2f}% (KHONG phai ty le bao sai)')
    print(f'Video so sanh: {output_path.resolve()}')


def main():
    parser = argparse.ArgumentParser(description='Xuất video so sánh trước/sau TV4')
    parser.add_argument('--video', type=Path, required=True, help='Đường dẫn AVI hoặc MP4')
    parser.add_argument('--k', type=int, default=3, help='Số lần xuất hiện tối thiểu')
    parser.add_argument('--n', type=int, default=5, help='Độ dài cửa sổ frame')
    parser.add_argument('--max-frames', type=int, default=150,
                        help='Số frame thử nhanh; 0 để chạy hết video')
    parser.add_argument('--output', type=Path, default=None,
                        help='Đường dẫn video đầu ra, mặc định trong outputs/videos/')
    args = parser.parse_args()
    if not 1 <= args.k <= args.n:
        parser.error('Cần 1 <= K <= N')
    if args.max_frames < 0:
        parser.error('--max-frames phải >= 0')
    name = f'{args.video.stem}_before_after_K{args.k}_N{args.n}.mp4'
    output = args.output or (Path('outputs/videos') / name)
    run(args.video, output, args.k, args.n, args.max_frames)


if __name__ == '__main__':
    main()
