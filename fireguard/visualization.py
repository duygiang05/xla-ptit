"""Render all pipeline stages on the same video timeline."""

import cv2
import numpy as np


STAGE_LAYOUT = (
    ("fire_color", "FIRE - TV1 MAU", 0, 0),
    ("fire_motion", "FIRE - TV2 CHUYEN DONG", 0, 1),
    ("fire_fusion", "FIRE - FUSION", 0, 2),
    ("fire_regions", "FIRE - TV3 VUNG", 1, 0),
    ("fire_temporal", "FIRE - TEMPORAL", 1, 1),
    ("smoke_color", "SMOKE - TV1 MAU", 2, 0),
    ("smoke_motion", "SMOKE - TV2 CHUYEN DONG", 2, 1),
    ("smoke_fusion", "SMOKE - FUSION", 2, 2),
    ("smoke_regions", "SMOKE - TV3 VUNG", 3, 0),
    ("smoke_temporal", "SMOKE - TEMPORAL", 3, 1),
)


def dashboard_size() -> tuple[int, int]:
    """Size of the standalone three-column stage video."""
    return 720, 648


def compose_pipeline_frame(
    stages: dict[str, np.ndarray], frame_no: int, fps: float
) -> np.ndarray:
    """Compose only the pipeline stages for the given source frame."""
    width, height = dashboard_size()
    board = np.full((height, width, 3), 16, dtype=np.uint8)
    tile_width = width // 3
    row_height = height // 4
    image_height = 135
    for key, label, row, column in STAGE_LAYOUT:
        image = stages[key]
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_AREA
        x = column * tile_width
        y = row * row_height
        board[y + 27 : y + 27 + image_height, x : x + tile_width] = cv2.resize(
            image, (tile_width, image_height), interpolation=interpolation
        )
        cv2.putText(
            board, label, (x + 8, y + 19), cv2.FONT_HERSHEY_SIMPLEX,
            0.39, (235, 235, 235), 1, cv2.LINE_AA,
        )
        cv2.rectangle(
            board, (x, y), (x + tile_width - 1, y + row_height - 1),
            (65, 65, 65), 1,
        )

    cv2.putText(
        board, f"{frame_no / fps:.2f}s | FRAME {frame_no + 1}",
        (2 * tile_width + 10, row_height + 85),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        board, "FIRE: DO | SMOKE: XANH", (2 * tile_width + 10, height - 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 190, 190), 1, cv2.LINE_AA,
    )
    return board
