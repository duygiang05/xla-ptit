"""Integration baseline. The smoke branch is heuristic and needs testing."""
import argparse
import csv
from pathlib import Path
import cv2
from .config import Config
from .color import color_candidates
from .motion import MotionDetector
from .shape import extract_regions
from .temporal import TemporalFilter


def run_video(input_path: str, output_path: str, csv_path: str | None = None, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot read video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_path=Path(output_path);out_path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(out_path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(cfg.width,cfg.height))
    if not writer.isOpened():
        cap.release();raise ValueError(f"Cannot write output video: {output_path}")
    motion=MotionDetector(cfg)
    temporal=TemporalFilter(cfg)
    frame_no=0
    events=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame=cv2.resize(frame,(cfg.width,cfg.height),interpolation=cv2.INTER_AREA)
            colors=color_candidates(frame,cfg)
            movement=motion.apply(frame)
            # Candidate rules are hypotheses, not tuned results.
            fire_mask=cv2.bitwise_and(colors['fire'],movement)
            smoke_mask=cv2.bitwise_and(colors['smoke'],movement)
            regions=extract_regions(fire_mask,'fire',cfg)+extract_regions(smoke_mask,'smoke',cfg)
            confirmed=temporal.update(regions,frame_no)
            annotated=frame.copy()
            timestamp=frame_no/fps
            for obj in confirmed:
                x,y,w,h=obj.bbox
                col=(0,0,255) if obj.kind=='fire' else (255,200,0)
                cv2.rectangle(annotated,(x,y),(x+w,y+h),col,2)
                cv2.putText(annotated,obj.kind.upper(),(x,max(y-7,15)),cv2.FONT_HERSHEY_SIMPLEX,.55,col,2)
                events.append([frame_no,round(timestamp,3),obj.kind,x,y,w,h,obj.score])
            writer.write(annotated)
            frame_no+=1
    finally:
        cap.release();writer.release()
    if csv_path:
        p=Path(csv_path);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w',newline='',encoding='utf-8') as f:
            c=csv.writer(f);c.writerow(['frame','timestamp_sec','kind','x','y','w','h','score']);c.writerows(events)
    return {'frames':frame_no,'event_rows':len(events),'output_video':str(out_path)}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--output',default='outputs/videos/out.mp4')
    parser.add_argument('--csv',default='outputs/events/events.csv')
    args=parser.parse_args()
    print(run_video(args.input,args.output,args.csv))
