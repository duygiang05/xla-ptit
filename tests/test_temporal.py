import pytest
from fireguard.config import Config
from fireguard.shape import Region
from fireguard.temporal import TemporalFilter
from fireguard.temporal.features import calculate_temporal_features


def region(kind='fire', x=5, area=400):
    return Region(kind, (x, 5, 20, 20), area, (x + 10, 15))


def test_temporal_voting_needs_history():
    filt = TemporalFilter(Config(vote_window=5, vote_hits=3))
    assert filt.update([region()], 0) == []
    assert filt.update([region()], 1) == []
    assert len(filt.update([region()], 2)) == 1


def test_missing_frame_does_not_count_as_hit():
    filt = TemporalFilter(Config(vote_window=3, vote_hits=3))
    assert not filt.update([region()], 0)
    assert not filt.update([], 1)
    assert not filt.update([region()], 2)
    assert not filt.update([region()], 3)  # frame 0 đã ra ngoài cửa sổ
    assert len(filt.update([region()], 4)) == 1


def test_fire_and_smoke_independent():
    filt = TemporalFilter(Config(vote_window=4, vote_hits=2))
    for i in range(2):
        result = filt.update([region('fire', 5), region('smoke', 100)], i)
    assert {r.kind for r in result} == {'fire', 'smoke'}


def test_two_distant_fire_regions_not_merged():
    filt = TemporalFilter(Config(vote_window=4, vote_hits=2))
    assert not filt.update([region('fire', 0), region('fire', 150)], 0)
    out = filt.update([region('fire', 0)], 1)
    assert len(out) == 1
    assert out[0].bbox[0] == 0


def test_feature_area_center_aspect():
    a = Region('fire', (10, 10, 20, 20), 100, (20, 20))
    b = Region('fire', (12, 11, 30, 20), 120, (23, 24))
    f = calculate_temporal_features(a, b)
    assert f['area_change'] == pytest.approx(.2)
    assert f['centroid_displacement'] == pytest.approx(5)
    assert f['aspect_ratio_change'] == pytest.approx(.5)


def test_non_increasing_frame_is_rejected():
    filt = TemporalFilter(Config())
    filt.update([], 0)
    with pytest.raises(ValueError):
        filt.update([], 0)
