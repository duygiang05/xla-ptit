from fireguard.shape.regions import Region
from fireguard.shape.tracking import bbox_iou

def same_candidate(current: Region, previous: Region, cfg) -> bool:
    if current.kind != previous.kind:
        return False
    dx = previous.centroid[0] - current.centroid[0]
    dy = previous.centroid[1] - current.centroid[1]
    return (dx*dx+dy*dy)**0.5 <= cfg.max_center_distance or bbox_iou(previous.bbox, current.bbox) >= cfg.min_iou
