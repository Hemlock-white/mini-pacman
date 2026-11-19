"""Tests for cbf.py. [Delete pytestmark after implementing it]"""
import numpy as np
import pytest

from minipacman.ball_tracker import BallEstimate

pytestmark = pytest.mark.skip(reason="TODO: delete this line after implementing cbf.py")


def _ball(p, v, r=0.1):
    return BallEstimate(id=0, position=np.array(p, float), velocity=np.array(v, float), radius=r,
                        restitution=0.7, stamp=0.0, source="truth")


def test_link_barrier_values(cfg):
    """Ball centre 1 m from the link centre, radii 0.1 + 0.2, margin 0.05 → h = 0.65; ball approaching head-on at
    2 m/s → hdot = -2."""
    from minipacman.cbf import link_barriers
    h, hdot = link_barriers(np.zeros((1, 3)), np.array([0.2]), np.zeros((1, 3)),
                            _ball([1.0, 0, 0], [-2.0, 0, 0]), cfg.cbf)
    assert np.isclose(h[0], 0.65) and np.isclose(hdot[0], -2.0)


def test_reward_term_is_nonpositive_and_clipped(cfg):
    """The CBF term is <= 0, and no lower than -constraint_clip * reward_weight."""
    pytest.fail("TODO")


def test_filter_passes_through_without_balls(cfg):
    from minipacman.cbf import RobotPlanarState, SafetyFilter
    f = SafetyFilter(cfg, "lag_aware")
    f.reset()
    u = np.array([0.5, 0.0, 0.0])
    u_safe, info = f.filter(u, RobotPlanarState(np.zeros(2), 0.0, 0.5, 0.0, 0.0), [], 0.0)
    assert np.allclose(u_safe, u) and not info["active"]


def test_filter_reacts_to_incoming_ball(cfg):
    """Ball coming from 3 m straight ahead, 0.4 m high, at 5 m/s: the filter should change the command (brake, turn
    or duck), and h_pred should go up."""
    pytest.fail("TODO")


def test_lag_aware_more_conservative(cfg):
    """Same situation: lag_aware predicts the robot moving more slowly (speed lag) than naive, so it steps in earlier."""
    pytest.fail("TODO")
