import cv2
import numpy as np
from fireguard.config import Config
from fireguard.shape import extract_regions, bbox_iou

def test_regions_and_iou():
    mask=np.zeros((90,90),dtype=np.uint8)
    cv2.rectangle(mask,(10,10),(35,35),255,-1)
    regions=extract_regions(mask,'fire',Config())
    assert len(regions)==1
    assert regions[0].kind=='fire'
    assert bbox_iou((0,0,10,10),(0,0,10,10))==1
