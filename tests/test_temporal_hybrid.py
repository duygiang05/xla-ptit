"""Test thêm cho TV4 + TV3 RegionTracker (không sửa test của thành viên khác)."""

import pytest
from fireguard.config import Config
from fireguard.shape.regions import Region
from fireguard.temporal import TemporalFilter, ShapeThresholds
from fireguard.temporal.decision import passes_shape_evidence


def region(kind="fire", x=10, area=400, perimeter=100, circularity=0.5, y=10):
    return Region(
        kind, (x, y, 20, 20), area, (x + 10, y + 10), 1.0, perimeter, circularity
    )


def test_uses_tv3_tracker_not_local_duplicate():
    from fireguard.shape.tracking import RegionTracker

    f = TemporalFilter(Config())
    assert isinstance(f.tracker, RegionTracker)


def test_vote_three_of_five_nonconsecutive_hits():
    f = TemporalFilter(Config(vote_window=5, vote_hits=3))
    assert f.update([region()], 0) == []
    assert f.update([], 1) == []
    assert f.update([region()], 2) == []
    assert len(f.update([region()], 3)) == 1
    assert f.last_decisions[0]["hits"] == 3


def test_vote_drops_old_hits_outside_time_window():
    f = TemporalFilter(Config(vote_window=3, vote_hits=3))
    f.update([region()], 0)
    f.update([], 1)
    f.update([region()], 2)
    assert f.update([region()], 3) == []
    assert f.last_decisions[0]["hits"] == 2
    assert len(f.update([region()], 4)) == 1


def test_track_id_stable_and_two_regions_separated():
    f = TemporalFilter(Config(vote_window=3, vote_hits=2))
    f.update([region(x=0), region(x=200)], 0)
    ids = [v["track_id"] for v in f.last_decisions]
    f.update([region(x=202), region(x=2)], 1)
    assert {v["track_id"] for v in f.last_decisions} == set(ids)
    assert len(f.last_decisions) == 2
    assert len(f.update([region(x=4), region(x=204)], 2)) == 2


def test_fire_smoke_never_share_track():
    f = TemporalFilter(Config(vote_window=3, vote_hits=2))
    f.update([region("fire"), region("smoke")], 0)
    assert len({d["track_id"] for d in f.last_decisions}) == 2
    out = f.update([region("fire"), region("smoke")], 1)
    assert {v.kind for v in out} == {"fire", "smoke"}


def test_lost_track_no_confirmation_after_timeout():
    f = TemporalFilter(Config(vote_window=7, vote_hits=2), max_missed=1)
    f.update([region()], 0)
    f.update([], 1)
    f.update([], 2)
    assert f.update([region()], 3) == []
    assert f.last_decisions[0]["hits"] == 1


def test_skipped_frame_indices_expire_tracks():
    f = TemporalFilter(Config(vote_window=7, vote_hits=2), max_missed=2)
    f.update([region()], 0)
    assert f.update([region()], 6) == []
    assert f.last_decisions[0]["hits"] == 1


def test_non_increasing_index_rejected():
    f = TemporalFilter(Config())
    f.update([], 5)
    with pytest.raises(ValueError):
        f.update([], 5)
    with pytest.raises(ValueError):
        f.update([], 4)


def test_invalid_settings():
    with pytest.raises(ValueError):
        TemporalFilter(Config(vote_hits=6, vote_window=5))
    with pytest.raises(ValueError):
        TemporalFilter(Config(), decision_mode="foo")
    with pytest.raises(ValueError):
        TemporalFilter(Config(), max_missed=-1)


def test_hybrid_rejects_static_fire_but_vote_keeps():
    cfg = Config(vote_window=5, vote_hits=3)
    vote = TemporalFilter(cfg, decision_mode="vote")
    hybrid = TemporalFilter(cfg, decision_mode="hybrid")
    for i in range(3):
        v = vote.update([region()], i)
        h = hybrid.update([region()], i)
    assert len(v) == 1
    assert h == []
    assert hybrid.last_decisions[0]["vote_pass"]
    assert not hybrid.last_decisions[0]["shape_pass"]


def test_hybrid_accepts_variable_flame_shape():
    f = TemporalFilter(Config(vote_window=5, vote_hits=3), decision_mode="hybrid")
    for i, area in enumerate((400, 550, 320)):
        out = f.update(
            [region(area=area, perimeter=100 + i * 15, circularity=0.4 + i * 0.05)], i
        )
    assert len(out) == 1
    assert f.last_decisions[0]["area_cv"] > 0.08


def test_hybrid_accepts_growing_smoke():
    f = TemporalFilter(Config(vote_window=5, vote_hits=3), decision_mode="hybrid")
    for i, area in enumerate((300, 360, 430)):
        out = f.update([region("smoke", area=area)], i)
    assert len(out) == 1
    assert f.last_decisions[0]["area_growth"] > 0.1


def test_thresholds_tunable_without_shared_config_change():
    cfg = Config(vote_window=3, vote_hits=2)
    f = TemporalFilter(
        cfg, decision_mode="hybrid", shape_thresholds=ShapeThresholds(fire_area_cv=0.02)
    )
    f.update([region(area=400)], 0)
    assert len(f.update([region(area=450)], 1)) == 1


def test_unknown_kind_not_shape_accepted():
    assert not passes_shape_evidence("unknown", {"area_cv": 2.0})


def test_shape_history_excludes_expired_frames():
    f = TemporalFilter(
        Config(vote_window=3, vote_hits=2), decision_mode="hybrid", max_missed=2
    )
    f.update([region(area=800)], 0)
    f.update([], 1)
    f.update([region(area=200)], 2)
    f.update([], 3)
    f.update([region(area=200)], 4)
    # Frame 0 đã ra khỏi [2,4], nên area_cv của chỉ [2,4] phải bằng 0.
    assert f.last_decisions[0]["hits"] == 2
    assert f.last_decisions[0]["area_cv"] == pytest.approx(0)
    assert f.update([region(area=200)], 5) == []  # nhánh Hybrid vẫn yêu cầu biến dạng
