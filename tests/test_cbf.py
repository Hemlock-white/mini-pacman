"""cbf.py 的測試。【實作後把 pytestmark 刪掉】"""
import numpy as np
import pytest

from minipacman.ball_tracker import BallEstimate

pytestmark = pytest.mark.skip(reason="TODO：實作 cbf.py 後刪掉這行")


def _ball(p, v, r=0.1):
    return BallEstimate(id=0, position=np.array(p, float), velocity=np.array(v, float), radius=r,
                        restitution=0.7, stamp=0.0, source="truth")


def test_link_barrier_values(cfg):
    """球心距 link 中心 1 m、兩個半徑 0.1 + 0.2、margin 0.05 → h = 0.65；球以 2 m/s 正面靠近 → hdot = -2。"""
    from minipacman.cbf import link_barriers
    h, hdot = link_barriers(np.zeros((1, 3)), np.array([0.2]), np.zeros((1, 3)),
                            _ball([1.0, 0, 0], [-2.0, 0, 0]), cfg.cbf)
    assert np.isclose(h[0], 0.65) and np.isclose(hdot[0], -2.0)


def test_reward_term_is_nonpositive_and_clipped(cfg):
    """CBF 項 <= 0，最小是 -constraint_clip * reward_weight。"""
    pytest.fail("TODO")


def test_filter_passes_through_without_balls(cfg):
    from minipacman.cbf import RobotPlanarState, SafetyFilter
    f = SafetyFilter(cfg, "lag_aware")
    f.reset()
    u = np.array([0.5, 0.0, 0.0])
    u_safe, info = f.filter(u, RobotPlanarState(np.zeros(2), 0.0, 0.5, 0.0, 0.0), [], 0.0)
    assert np.allclose(u_safe, u) and not info["active"]


def test_filter_reacts_to_incoming_ball(cfg):
    """球從正前方 3 m、高 0.4 m 以 5 m/s 飛來：過濾器應該改指令（煞車、轉彎或蹲下），而且 h_pred 變大。"""
    pytest.fail("TODO")


def test_lag_aware_more_conservative(cfg):
    """同一個情境，lag_aware 預測的機器人位移比 naive 慢（速度延遲），所以會更早介入。"""
    pytest.fail("TODO")
