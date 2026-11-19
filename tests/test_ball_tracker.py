"""追蹤器與球資訊來源：追得上彈跳的球、延遲正確。"""
import numpy as np
import pytest

from minipacman import scene
from minipacman.ball_sources import NoisyTruthBallSource, make_ball_source
from minipacman.balls import BallLauncher, plan_throw
from minipacman.depth_camera import DepthCamera

from conftest import RobotFreezer, step


def _track_one_throw(m, d, cfg, mode, seed):
    ids = scene.scene_ids(m, cfg)
    launcher = BallLauncher(m, d, cfg, ids)
    freezer = RobotFreezer(m, d, ids)
    cam = DepthCamera(m, cfg, ids)
    source = make_ball_source("depth_tracker", m, cfg, launcher, cam)
    rng = np.random.default_rng(seed)
    source.reset(rng)
    base = d.xpos[ids.base_body].copy()
    # 從正前方丟一顆會先彈跳的球，瞄準機器人旁邊 0.8 m（不會打到，才看得到完整軌跡）
    plan = plan_throw(rng, cfg.balls, base[0:2] + np.array([0.0, 0.8]), 0.0, np.zeros(2),
                      0.1, 0.136, 0.7, mode=mode)
    launcher.throw_now(plan, 0)
    pos_err, vel_err, ids_seen = [], [], set()
    cam_pos, _ = cam.pose(d)
    for _ in range(int(plan.flight_time / cfg.sim.policy_dt)):
        step(m, d, cfg.sim.policy_every, freezer, launcher)
        est = source.update(d, rng)
        p, v = launcher.ball_state(0)
        if 0.8 < np.linalg.norm(p - cam_pos) < 4.5 and est:
            e = min(est, key=lambda b: np.linalg.norm(b.position - p))
            ids_seen.add(e.id)
            pos_err.append(np.linalg.norm(e.position - p))
            vel_err.append(np.linalg.norm(e.velocity - v))
    return np.array(pos_err), np.array(vel_err), ids_seen


@pytest.mark.parametrize("mode", ["direct", "bounce"])
@pytest.mark.parametrize("seed", [0, 1])
def test_tracker_follows_ball(sim, cfg, mode, seed):
    pos_err, vel_err, ids_seen = _track_one_throw(*sim, cfg, mode, seed)
    assert len(pos_err) > 10, "大部分時間應該追得到"
    assert np.median(pos_err) < 0.06, f"位置誤差中位數 {np.median(pos_err):.3f} m"
    assert np.median(vel_err[3:]) < 0.5, f"速度誤差中位數 {np.median(vel_err[3:]):.3f} m/s"
    assert len(ids_seen) <= 2, "同一顆球不應該一直被當成新的球"


def test_tracker_geometric_mode(sim, cfg):
    cfg.tracker.mode = "geometric"
    pos_err, vel_err, _ = _track_one_throw(*sim, cfg, "bounce", 3)
    assert len(pos_err) > 10
    assert np.median(pos_err) < 0.10 and np.median(vel_err[3:]) < 0.8


def test_noisy_truth_latency(sim, cfg):
    m, d = sim
    ids = scene.scene_ids(m, cfg)
    cfg.noisy_truth.pos_sigma = cfg.noisy_truth.vel_sigma = cfg.noisy_truth.dropout = 0.0
    launcher = BallLauncher(m, d, cfg, ids)
    freezer = RobotFreezer(m, d, ids)
    src = NoisyTruthBallSource(launcher, cfg)
    rng = np.random.default_rng(0)
    plan = plan_throw(rng, cfg.balls, np.array([20.0, 20.0]), 0.0, np.zeros(2), 0.1, 0.136, 0.7, mode="direct")
    launcher.throw_now(plan, 0)
    history = []
    for k in range(10):
        step(m, d, cfg.sim.policy_every, freezer)
        history.append((d.time, launcher.ball_state(0)[0]))
        out = src.update(d, rng)
        if k >= src.delay:
            t_old, p_old = history[k - src.delay]
            assert len(out) == 1 and np.isclose(out[0].stamp, t_old) and np.allclose(out[0].position, p_old)
