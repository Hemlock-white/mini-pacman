"""Selected-seed comparison video planning and layout."""
import numpy as np
import pytest

from minipacman.evaluate import Condition
from minipacman import video


def _condition(name, reward="task", filter_mode=None, source="truth"):
    return Condition(name, "scripted", reward, filter_mode, "every_step", source)


def test_comparison_groups_match_experiments():
    exp1 = [
        _condition("exp1_task_nofilter"),
        _condition("exp1_task_filter", filter_mode="lag_aware"),
        _condition("exp1_task_cbf_nofilter", reward="task_cbf"),
        _condition("exp1_task_cbf_filter", reward="task_cbf", filter_mode="lag_aware"),
    ]
    groups = video.comparison_groups("exp1", exp1)
    assert [group.name for group in groups] == ["exp1_task", "exp1_task_cbf"]
    assert [len(group.conditions) for group in groups] == [2, 2]

    exp3 = [_condition(f"exp3_{name}") for name in ("no_filter", "truth", "noisy_truth", "depth_tracker")]
    groups = video.comparison_groups("exp3", exp3)
    assert len(groups) == 1 and groups[0].name == "exp3_sources"
    assert len(groups[0].conditions) == 4

    with pytest.raises(ValueError, match="not defined"):
        video.comparison_groups("summary", exp3)


def test_tile_panels_uses_horizontal_and_two_by_two_layouts():
    panel = np.zeros((10, 20, 3), dtype=np.uint8)
    assert video.tile_panels([panel, panel]).shape == (10, 40, 3)
    assert video.tile_panels([panel, panel, panel]).shape == (10, 60, 3)
    assert video.tile_panels([panel, panel, panel, panel]).shape == (20, 40, 3)
    with pytest.raises(ValueError, match="at least one"):
        video.tile_panels([])
    with pytest.raises(ValueError, match="at most four"):
        video.tile_panels([panel] * 5)


def test_record_experiment_videos_dispatches_selected_seeds(tmp_path, monkeypatch, cfg):
    conditions = [
        _condition("exp2_naive", reward="scripted", filter_mode="naive"),
        _condition("exp2_lag_aware", reward="scripted", filter_mode="lag_aware"),
    ]
    calls = []

    def fake_record(group, seed, cfg_, path, device, fps):
        calls.append((tuple(condition.name for condition in group), seed, path, device, fps))
        return [
            {"condition": condition.name, "success": True, "fell": False, "n_hit": 0, "n_thrown": 1}
            for condition in group
        ]

    monkeypatch.setattr(video, "record_comparison", fake_record)
    paths = video.record_experiment_videos(
        "exp2", conditions, [10041, 10090], cfg, tmp_path, device="cpu", fps=20,
    )

    assert paths == [
        tmp_path / "exp2_filters_seed10041.mp4",
        tmp_path / "exp2_filters_seed10090.mp4",
    ]
    assert [call[1] for call in calls] == [10041, 10090]
    assert all(call[0] == ("exp2_naive", "exp2_lag_aware") for call in calls)
    assert all(call[3:] == ("cpu", 20) for call in calls)


def test_condition_labels_are_experiment_specific():
    assert video._condition_label(_condition("exp1_task_filter", filter_mode="lag_aware")) == "task | filter on"
    assert video._condition_label(
        _condition("exp2_lag_aware", reward="scripted", filter_mode="lag_aware")
    ) == "scripted | lag-aware"
    assert video._condition_label(
        _condition("exp3_depth_tracker", reward="task_cbf", filter_mode="lag_aware", source="depth_tracker")
    ) == "depth tracker | lag-aware"
