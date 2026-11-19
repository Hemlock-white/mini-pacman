"""Observation layout (obs_layout is done, so this test always runs)."""
import pytest

from minipacman.dodge_env import obs_layout


def _check_contiguous(slices):
    stops = 0
    for s in slices.values():
        assert s.start == stops, "each part must start right where the previous one ends"
        stops = s.stop
    return stops


def test_default_dims(cfg):
    lay = obs_layout(cfg)
    assert _check_contiguous(lay["actor"]) == 589
    assert _check_contiguous(lay["critic"]) == 632
    assert lay["actor"]["depth"] == slice(0, 576)
    assert all(lay["critic"][k] == v for k, v in lay["actor"].items()), "the start of the critic = the actor"


def test_tracker_fallback_dims(cfg):
    cfg.obs.actor_ball_input = "tracker"
    lay = obs_layout(cfg)
    assert "depth" not in lay["actor"] and lay["actor"]["tracked_balls"] == slice(0, 40)
    assert _check_contiguous(lay["actor"]) == 53
    assert _check_contiguous(lay["critic"]) == 96


def test_unknown_input_rejected(cfg):
    cfg.obs.actor_ball_input = "pixels"
    with pytest.raises(ValueError):
        obs_layout(cfg)
