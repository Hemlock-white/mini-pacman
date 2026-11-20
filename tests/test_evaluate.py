"""Tests for scripted evaluation and resumable condition runs."""

import numpy as np
import pytest

from conftest import need_tita
from minipacman import report
from minipacman.evaluate import Condition, ScriptedPolicy


def _episode(seed, condition="test"):
    return {
        "condition": condition,
        "seed": seed,
        "success": True,
        "fell": False,
        "t_goal": 1.0,
        "n_thrown": 0,
        "n_hit": 0,
        "balls": [],
        "intervention_mean": None,
        "filter_active_frac": None,
        "episode_seconds": 2.0,
        "device": "cpu",
    }


def test_scripted_policy_drives_then_stops(cfg):
    from minipacman.dodge_env import obs_layout

    policy = ScriptedPolicy(cfg)
    layout = obs_layout(cfg)["actor"]
    actor = np.zeros(max(part.stop for part in layout.values()), dtype=np.float32)
    observation = {"actor": actor}
    actor[layout["goal"]] = [2.0, 0.0]
    drive = policy.act(observation, None)
    actor[layout["goal"]] = [0.1, 0.0]
    stop = policy.act(observation, None)
    assert drive[0] > stop[0]
    assert drive[1] == stop[1] == 0.0
    assert drive[2] == stop[2] == -1.0


@need_tita
def test_run_scripted_episode(cfg):
    from minipacman.evaluate import run_episode

    cfg.task.goal_distance = 0.5
    cfg.sim.episode_seconds = 2.0
    cfg.balls.first_throw_range = (10.0, 10.0)
    condition = Condition("smoke", "scripted", "scripted", None)
    result = run_episode(condition, 11, cfg, ScriptedPolicy(cfg), "cpu")
    assert result["condition"] == "smoke" and result["seed"] == 11
    assert result["success"] and not result["fell"]
    assert result["t_goal"] is not None
    assert result["episode_seconds"] == pytest.approx(2.0)
    assert result["device"] == "cpu"


def test_run_condition_resume(tmp_path, monkeypatch, cfg):
    import minipacman.evaluate as evaluate

    condition = Condition("resume", "scripted", "scripted", None)
    path = tmp_path / "episodes.jsonl"
    report.save_jsonl([_episode(1, condition.name)], path)
    calls = []

    def fake_run_episode(condition_, seed, cfg_, policy, device):
        calls.append(seed)
        return _episode(seed, condition_.name)

    monkeypatch.setattr(evaluate, "run_episode", fake_run_episode)
    episodes = evaluate.run_condition(condition, [1, 2], cfg, path, "cpu", resume=True)
    assert calls == [2]
    assert [episode["seed"] for episode in episodes] == [1, 2]
    assert len(report.load_jsonl(path)) == 2

    calls.clear()
    episodes = evaluate.run_condition(condition, [1, 2], cfg, path, "cpu", resume=False)
    assert calls == [1, 2]
    assert len(episodes) == 2 and len(report.load_jsonl(path)) == 2
