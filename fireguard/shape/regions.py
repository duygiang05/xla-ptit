from dataclasses import dataclass
import cv2
from fireguard.config import Config
from .morphology import clean_mask

@dataclass(frozen=True)
class Region:
    kind: str
    bbox: tuple[int,int,int,int]
    area: float
    centroid: tuple[float,float]
    score: float = 1.0

def extract_regions(mask, kind: str, cfg: Config) -> list[Region]:
    mask = clean_mask(mask, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < cfg.min_region_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        m = cv2.moments(contour)
        center = (m['m10']/m['m00'], m['m01']/m['m00']) if m['m00'] else (x+w/2, y+h/2)
        items.append(Region(kind, (x,y,w,h), float(area), center))
    return items
