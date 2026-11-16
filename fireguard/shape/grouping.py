"""Biểu diễn các vùng đã xác nhận bằng một hộp bao cho mỗi loại."""

from .regions import Region


def union_regions(regions: list[Region], kind: str) -> list[Region]:
    """Gộp mọi vùng cùng loại trong frame thành một hộp bao."""
    selected = [r for r in regions if r.kind == kind]
    if not selected:
        return []
    x = min(r.bbox[0] for r in selected)
    y = min(r.bbox[1] for r in selected)
    right = max(r.bbox[0] + r.bbox[2] for r in selected)
    bottom = max(r.bbox[1] + r.bbox[3] for r in selected)
    return [
        Region(
            kind,
            (x, y, right - x, bottom - y),
            sum(r.area for r in selected),
            ((x + right) / 2, (y + bottom) / 2),
            max(r.score for r in selected),
        )
    ]
