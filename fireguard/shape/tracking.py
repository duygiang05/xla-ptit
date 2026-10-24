def bbox_iou(a: tuple[int,int,int,int], b: tuple[int,int,int,int]) -> float:
    """Helper for basic association; NOT a full multi-object tracker."""
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    x1=max(ax,bx); y1=max(ay,by); x2=min(ax+aw,bx+bw); y2=min(ay+ah,by+bh)
    intersection=max(0,x2-x1)*max(0,y2-y1)
    union=aw*ah+bw*bh-intersection
    return intersection/float(union) if union>0 else 0.0
