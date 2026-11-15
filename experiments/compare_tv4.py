"""Thực nghiệm TV4: vùng trước và sau Temporal Voting trên cùng video.

Chạy từ thư mục gốc FireGuard:
    python -m experiments.compare_tv4
    python -m experiments.compare_tv4 --video data/development/ForestFire1.avi

Không sửa TV1/TV2/TV3 hoặc pipeline của nhóm. Đây là so sánh số lượng vùng
ứng viên, KHÔNG phải Precision/Recall/false-positive rate khi chưa có nhãn thật.
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import cv2

from fireguard.color import color_candidates
from fireguard.config import Config
from fireguard.motion import MotionDetector
from fireguard.shape import extract_regions
from fireguard.temporal import TemporalFilter


CONFIGS = [(2, 3), (3, 5), (4, 5)]  # (K, N), cấu hình THỬ NGHIỆM
EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv'}


def save_csv(path: Path, columns: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze(video_path: Path, output_dir: Path, max_frames: int = 0):
    cfg = Config()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'OpenCV không mở được: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0

    motion = MotionDetector(cfg)
    filters = {(k, n): TemporalFilter(replace(cfg, vote_hits=k, vote_window=n))
               for k, n in CONFIGS}
    frame_rows = []
    summary = {(k, n): {'before_fire': 0, 'before_smoke': 0,
                        'after_fire': 0, 'after_smoke': 0,
                        'frames_with_any_before': 0, 'frames_with_any_after': 0,
                        'first_candidate_frame': None, 'first_confirmed_frame': None}
               for k, n in CONFIGS}
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames > 0 and frame_idx >= max_frames):
                break
            frame = cv2.resize(frame, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
            color_masks = color_candidates(frame, cfg)
            motion_mask = motion.apply(frame)
            # Phải giống baseline pipeline.py để so sánh công bằng.
            fire_mask = cv2.bitwise_and(color_masks['fire'], motion_mask)
            smoke_mask = cv2.bitwise_and(color_masks['smoke'], motion_mask)
            regions = (extract_regions(fire_mask, 'fire', cfg)
                       + extract_regions(smoke_mask, 'smoke', cfg))
            before_fire = sum(r.kind == 'fire' for r in regions)
            before_smoke = sum(r.kind == 'smoke' for r in regions)
            for (k, n), temporal_filter in filters.items():
                confirmed = temporal_filter.update(regions, frame_idx)
                after_fire = sum(r.kind == 'fire' for r in confirmed)
                after_smoke = sum(r.kind == 'smoke' for r in confirmed)
                row = {'video': video_path.name, 'frame': frame_idx,
                       'time_sec': round(frame_idx / fps, 3),
                       'K': k, 'N': n,
                       'before_fire': before_fire, 'before_smoke': before_smoke,
                       'after_fire': after_fire, 'after_smoke': after_smoke}
                frame_rows.append(row)
                total = summary[(k, n)]
                for label in ('fire', 'smoke'):
                    total['before_' + label] += row['before_' + label]
                    total['after_' + label] += row['after_' + label]
                if before_fire + before_smoke:
                    total['frames_with_any_before'] += 1
                    if total['first_candidate_frame'] is None:
                        total['first_candidate_frame'] = frame_idx
                if after_fire + after_smoke:
                    total['frames_with_any_after'] += 1
                    if total['first_confirmed_frame'] is None:
                        total['first_confirmed_frame'] = frame_idx
            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        raise ValueError(f'Video không có frame đọc được: {video_path}')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / f'{video_path.stem}_per_frame.csv',
             ['video', 'frame', 'time_sec', 'K', 'N', 'before_fire', 'before_smoke',
              'after_fire', 'after_smoke'], frame_rows)
    result_rows = []
    for (k, n), values in summary.items():
        before = values['before_fire'] + values['before_smoke']
        after = values['after_fire'] + values['after_smoke']
        first_candidate = values['first_candidate_frame']
        first_confirmed = values['first_confirmed_frame']
        result_rows.append({
            'video': video_path.name, 'K': k, 'N': n, 'frames': frame_idx,
            'before_fire': values['before_fire'], 'before_smoke': values['before_smoke'],
            'after_fire': values['after_fire'], 'after_smoke': values['after_smoke'],
            'total_before': before, 'total_after': after,
            'reduction_pct': round(100 * (before-after)/before, 2) if before else 0,
            'frames_with_any_before': values['frames_with_any_before'],
            'frames_with_any_after': values['frames_with_any_after'],
            'first_candidate_sec': round(first_candidate/fps, 3) if first_candidate is not None else '',
            'first_confirmed_sec': round(first_confirmed/fps, 3) if first_confirmed is not None else '',
            'delay_from_first_candidate_sec': (
                round((first_confirmed-first_candidate)/fps, 3)
                if first_candidate is not None and first_confirmed is not None else ''),
        })
    return result_rows


def main():
    parser = argparse.ArgumentParser(description='So sánh trước/sau TV4 trên video thật')
    parser.add_argument('--video', type=Path, help='Một video cụ thể (nếu bỏ qua sẽ chạy cả folder)')
    parser.add_argument('--folder', type=Path, default=Path('data/development'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/tv4'))
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Giới hạn số frame để thử nhanh, 0 là cả video')
    args = parser.parse_args()
    if args.video:
        videos = [args.video]
    else:
        videos = sorted(p for p in args.folder.iterdir()
                        if p.is_file() and p.suffix.lower() in EXTENSIONS) if args.folder.exists() else []
    if not videos:
        print('Chưa thấy video. Đặt AVI/MP4 trong data/development hoặc truyền --video.', file=sys.stderr)
        return 1

    all_results = []
    for video in videos:
        try:
            print(f'Đang phân tích: {video.name} ...', flush=True)
            rows = analyze(video, args.output, args.max_frames)
            all_results.extend(rows)
            for row in rows:
                print(f"  K/N={row['K']}/{row['N']} | trước={row['total_before']} "
                      f"sau={row['total_after']} | giảm {row['reduction_pct']}%")
        except (ValueError, cv2.error) as exc:
            print(f'Bỏ qua {video.name}: {exc}', file=sys.stderr)

    if not all_results:
        return 1
    fields = list(all_results[0])
    save_csv(args.output / 'summary.csv', fields, all_results)
    print(f'\nĐã lưu CSV tại: {args.output.resolve()}')
    print('CHÚ Ý: giảm số vùng không đồng nghĩa giảm phát hiện sai; cần nhãn thật để đánh giá.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
