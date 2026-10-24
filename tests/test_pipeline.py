from pathlib import Path
import numpy as np
import cv2
from fireguard.pipeline import run_video
from fireguard.config import Config
from fireguard.color import color_candidates
from fireguard.shape import extract_regions


def test_pipeline_generates_video(tmp_path: Path):
    p=tmp_path/'in.avi'
    w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,90))
    assert w.isOpened()
    for i in range(8):
        f=np.zeros((90,160,3),dtype=np.uint8)
        cv2.rectangle(f,(10+i*5,10),(50+i*5,70),(0,100,255),-1)
        w.write(f)
    w.release()
    out=tmp_path/'out.mp4'; csv_path=tmp_path/'out.csv'
    result=run_video(str(p),str(out),str(csv_path),Config(width=160,height=90,min_region_area=20))
    assert result['frames']==8
    assert out.exists() and out.stat().st_size>0 and csv_path.exists()


def test_same_frame_can_hold_fire_and_smoke_candidates():
    # Kiểm thử giao tiếp module đa nhãn; KHÔNG khẳng định thuật toán phát hiện ảnh thực chính xác.
    cfg=Config(width=160,height=90,min_region_area=15)
    frame=np.zeros((90,160,3),dtype=np.uint8)
    cv2.rectangle(frame,(10,10),(55,70),(0,100,255),-1) # vùng màu lửa
    cv2.rectangle(frame,(100,10),(145,70),(130,130,130),-1) # vùng màu khói
    candidates=color_candidates(frame,cfg)
    assert set(candidates)=={'fire','smoke'}
    assert candidates['fire'].shape==candidates['smoke'].shape==frame.shape[:2]
    assert np.count_nonzero(candidates['fire'])>0
    assert np.count_nonzero(candidates['smoke'])>0
    regions=(extract_regions(candidates['fire'],'fire',cfg)+
             extract_regions(candidates['smoke'],'smoke',cfg))
    assert {'fire','smoke'} <= {r.kind for r in regions}
