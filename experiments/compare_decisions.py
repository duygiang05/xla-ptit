"""So sánh 2 luật quyết định TV4 trên CÙNG vùng ứng viên của pipeline hiện tại.

Chạy từ gốc dự án:
python experiments/tv4/compare_decisions.py --video data/development/ForestFire1.avi --max-frames 300
Tạo file CSV theo từng frame và từng vùng, không ghi đè kết quả TV3/TV1.
Chỉ đo số vùng, KHÔNG suy ra precision/recall nếu chưa có nhãn.
"""
import argparse
import csv
from pathlib import Path
import sys

import cv2
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from fireguard.color import color_candidates
from fireguard.config import Config
from fireguard.motion import MotionDetector
from fireguard.shape import extract_regions
from fireguard.temporal import TemporalFilter


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',required=True)
    parser.add_argument('--max-frames',type=int,default=300)
    parser.add_argument('--output-dir',default='experiments/tv4/results')
    args=parser.parse_args()
    if args.max_frames<=0:parser.error('--max-frames phải > 0')
    outdir=ROOT/args.output_dir;outdir.mkdir(parents=True,exist_ok=True)
    video=Path(args.video)
    if not video.is_absolute():video=ROOT/video
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise SystemExit(f'Không mở được video: {video}')
    cfg=Config();motion=MotionDetector(cfg)
    filters={'vote':TemporalFilter(cfg,decision_mode='vote'),
             'hybrid':TemporalFilter(cfg,decision_mode='hybrid')}
    frame_rows=[];region_rows=[]
    try:
        for idx in range(args.max_frames):
            ok,frame=cap.read()
            if not ok:break
            frame=cv2.resize(frame,(cfg.width,cfg.height),interpolation=cv2.INTER_AREA)
            masks=color_candidates(frame,cfg)
            mov=motion.apply(frame)
            regions=[]
            for kind in ('fire','smoke'):
                regions += extract_regions(cv2.bitwise_and(masks[kind],mov),kind,cfg)
            counts={}
            for mode,filter_ in filters.items():
                confirmed=filter_.update(regions,idx)
                counts[mode]=len(confirmed)
                for d in filter_.last_decisions:
                    region_rows.append({'frame':idx,'mode':mode,**d})
            frame_rows.append({'frame':idx,'candidate_regions':len(regions),
                               'vote_confirmed':counts['vote'],
                               'hybrid_confirmed':counts['hybrid']})
    finally:cap.release()
    if not frame_rows:raise SystemExit('Video không có frame đọc được')
    def save(path,rows):
        with open(path,'w',newline='',encoding='utf-8-sig') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]),extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    save(outdir/'comparison_frames.csv',frame_rows)
    if region_rows:save(outdir/'comparison_regions.csv',region_rows)
    print(f'Video: {video.name} | Frames: {len(frame_rows)}')
    for key in ('candidate_regions','vote_confirmed','hybrid_confirmed'):
        print(f'{key}: {sum(r[key] for r in frame_rows)} lượt vùng/frame')
    print(f'Đã lưu: {outdir}')
    print('LƯU Ý: số vùng giảm không phải số false positive giảm. Cần nhãn để đánh giá.')


if __name__=='__main__':main()
