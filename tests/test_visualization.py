import numpy as np

from fireguard.visualization import STAGE_LAYOUT, compose_pipeline_frame


def test_dashboard_places_each_stage_from_the_same_frame():
    stages = {}
    for index, (key, _, _, _) in enumerate(STAGE_LAYOUT):
        stages[key] = np.full((40, 80, 3), (index + 1) * 18, np.uint8)
    stages["fire_color"] = np.full((40, 80), 255, np.uint8)
    originals = {key: image.copy() for key, image in stages.items()}

    board = compose_pipeline_frame(stages, frame_no=7, fps=10)

    assert board.shape == (648, 720, 3)
    assert "input" not in stages and "output" not in stages
    for key, _, row, column in STAGE_LAYOUT:
        x = column * 240 + 120
        y = row * 162 + 27 + 67
        expected = originals[key][20, 40]
        if originals[key].ndim == 2:
            expected = np.repeat(expected, 3)
        assert np.array_equal(board[y, x], expected)
        assert np.array_equal(stages[key], originals[key])
