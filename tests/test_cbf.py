"""Tests for the CBF reward and predictive safety filter."""

import numpy as np

from minipacman.ball_tracker import BallEstimate


def _ball(position, velocity, radius=0.1):
    return BallEstimate(
        id=0,
        position=np.array(position, float),
        velocity=np.array(velocity, float),
        radius=radius,
        restitution=0.7,
        stamp=0.0,
        source="truth",
    )


def test_link_barrier_values(cfg):
    from minipacman.cbf import link_barriers

    h, hdot = link_barriers(
        np.zeros((1, 3)), np.array([0.2]), np.zeros((1, 3)),
        _ball([1.0, 0, 0], [-2.0, 0, 0]), cfg.cbf,
    )
    assert np.isclose(h[0], 0.65)
    assert np.isclose(hdot[0], -2.0)


def test_reward_term_is_nonpositive_and_clipped(sim, cfg):
    from minipacman.cbf import cbf_reward_term, robot_link_spheres
    from minipacman.scene import scene_ids

    model, data = sim
    ids = scene_ids(model, cfg)
    centers, _, _ = robot_link_spheres(model, data, ids.robot_collision_geoms)
    incoming = _ball(centers[0] + [1.0, 0.0, 0.0], [-100.0, 0.0, 0.0])
    reward = cbf_reward_term(model, data, ids.robot_collision_geoms, [incoming], cfg.cbf)
    lower_bound = -cfg.cbf.constraint_clip * cfg.cbf.reward_weight
    assert lower_bound <= reward <= 0.0
    assert np.isclose(reward, lower_bound)
    assert cbf_reward_term(model, data, ids.robot_collision_geoms, [], cfg.cbf) == 0.0


def test_filter_passes_through_without_balls(cfg):
    from minipacman.cbf import RobotPlanarState, SafetyFilter

    safety_filter = SafetyFilter(cfg, "lag_aware")
    safety_filter.reset()
    command = np.array([0.5, 0.0, 0.0])
    safe, info = safety_filter.filter(
        command, RobotPlanarState(np.zeros(2), 0.0, 0.5, 0.0, 0.0), [], 0.0,
    )
    assert np.allclose(safe, command)
    assert not info["active"]
    assert info["intervention"] == 0.0
    assert np.isinf(info["h_pred"])


def test_filter_reacts_to_incoming_ball(cfg):
    from minipacman.cbf import RobotPlanarState, SafetyFilter

    safety_filter = SafetyFilter(cfg, "lag_aware")
    robot = RobotPlanarState(np.zeros(2), 0.0, 0.5, 0.0, 0.0)
    command = np.array([0.5, 0.0, 0.0])
    incoming = _ball([3.0, 0.0, 0.4], [-5.0, 0.0, 0.0])
    nominal_clearance = safety_filter.predicted_clearance(command, robot, [incoming], 0.0)

    safe, info = safety_filter.filter(command, robot, [incoming], 0.0)

    assert info["active"]
    assert not np.allclose(safe, command)
    assert info["h_pred"] > nominal_clearance


def test_lag_aware_more_conservative(cfg):
    from minipacman.cbf import RobotPlanarState, SafetyFilter

    robot = RobotPlanarState(np.zeros(2), 0.0, 0.0, 0.0, 0.0)
    command = np.array([0.8, 0.0, 0.0])
    incoming = _ball([1.5, 2.0, 0.4], [-3.5, -3.5, 0.0])
    naive = SafetyFilter(cfg, "naive")
    aware = SafetyFilter(cfg, "lag_aware")

    naive_safe, naive_info = naive.filter(command, robot, [incoming], 0.0)
    aware_nominal = aware.predicted_clearance(command, robot, [incoming], 0.0)
    aware_safe, aware_info = aware.filter(command, robot, [incoming], 0.0)

    assert not naive_info["active"]
    assert np.allclose(naive_safe, command)
    assert aware_info["active"]
    assert not np.allclose(aware_safe, command)
    assert aware_nominal < cfg.cbf.filter_margin < naive_info["h_pred"]
    assert aware_info["h_pred"] > aware_nominal
