"""The command envelope, crouch coupling, reference leash, and the filter's robot model (no MuJoCo needed).

Background: with random high-level commands the round-1 controller fell in 26/28 episodes (stress test 2026-11-23).
"""
import numpy as np
import pytest

from minipacman.cbf import RobotPlanarState, SafetyFilter
from minipacman.lqr import ReferenceGenerator, clip_to_envelope, envelope_limits, max_crouch_for


def test_reverse_limit_shrinks_with_crouch(cfg):
    c = cfg.command
    assert clip_to_envelope(np.array([-5.0, 0.0, 0.0]), 0.0, c)[0] == pytest.approx(c.v_min)
    assert clip_to_envelope(np.array([-5.0, 0.0, 0.0]), 1.0, c)[0] == pytest.approx(c.v_min_crouch)


def test_max_crouch_for_matches_the_envelope(cfg):
    c = cfg.command
    for v, w in [(0.0, 0.0), (0.5, 0.0), (0.8, 0.0), (-0.3, 0.0), (0.2, 1.5), (0.1, 0.5)]:
        s = max_crouch_for(v, w, c)
        lo, hi, wmax = envelope_limits(s, c)
        assert lo - 1e-9 <= v <= hi + 1e-9 and abs(w) <= wmax + 1e-9
        if s < 1.0:                       # and one step deeper would leave the envelope
            lo, hi, wmax = envelope_limits(min(1.0, s + 0.02), c)
            assert not (lo <= v <= hi and abs(w) <= wmax)
    assert max_crouch_for(0.0, 0.0, c) == 1.0
    assert max_crouch_for(c.v_max_stand, 0.0, c) == 0.0


def test_reference_crouches_only_as_fast_as_it_slows_down(cfg):
    """Crouch command at full speed: s may only grow while the reference speed stays inside the envelope."""
    c, dt = cfg.command, cfg.sim.timestep
    gen = ReferenceGenerator(cfg)
    gen.reset(np.zeros(2), 0.0, v=c.v_max_stand)
    for _ in range(1000):
        ref = gen.update(np.array([c.v_max_stand, 0.0, c.sdot_max]), dt)
        lo, hi, wmax = envelope_limits(ref.s, c)
        assert lo - 1e-9 <= ref.v <= hi + 1e-9
    assert ref.s == pytest.approx(1.0) and ref.v == pytest.approx(c.v_max_crouch)


def test_leash_pulls_back_a_runaway_reference_only(cfg):
    gen = ReferenceGenerator(cfg)
    gen.reset(np.zeros(2), 0.0, v=0.5)
    gen._reference.p = np.array([0.5, 0.0])             # reference 0.5 m ahead of a robot at the origin
    gen.leash(np.zeros(2), 0.1)
    assert gen._reference.p[0] == pytest.approx(0.1)
    gen._reference.p = np.array([-0.5, 0.0])            # robot ahead of the reference: left alone
    gen.leash(np.zeros(2), 0.1)
    assert gen._reference.p[0] == pytest.approx(-0.5)
    gen.reset(np.zeros(2), 0.0, v=-0.3)                 # reversing: "ahead" means behind the robot
    gen._reference.p = np.array([-0.5, 0.0])
    gen.leash(np.zeros(2), 0.1)
    assert gen._reference.p[0] == pytest.approx(-0.1)


def test_lag_aware_model_is_the_reference_generator(cfg):
    """The filter's vectorised lag-aware rollout must reproduce ReferenceGenerator (with the extra lag off)."""
    cfg.cbf.lag_tau_aware = 0.0
    safety = SafetyFilter(cfg, "lag_aware")
    robot = RobotPlanarState(np.array([1.0, -0.5]), 0.3, 0.6, 0.2, 0.1)
    rng = np.random.default_rng(0)
    commands = np.column_stack([rng.uniform(-0.5, 0.8, 20), rng.uniform(-2, 2, 20), rng.uniform(-3.4, 3.4, 20)])
    steps = 40
    xy, yaw, speed, crouch = safety._predict_robot(commands, robot, steps)
    dt = cfg.sim.policy_dt
    for i, u in enumerate(commands):
        gen = ReferenceGenerator(cfg)
        gen.reset(robot.xy, robot.yaw, v=robot.v, w=robot.w, s=robot.s)
        for k in range(1, steps):
            ref = gen.update(u, dt)
            assert ref.s == pytest.approx(crouch[i, k], abs=1e-9)
            assert ref.v == pytest.approx(speed[i, k], abs=1e-9)
            assert ref.psi == pytest.approx(yaw[i, k], abs=1e-9)
            assert np.allclose(ref.p, xy[i, k], atol=1e-9)


def test_filter_does_not_chatter_between_extremes(cfg):
    """With a ball coming, consecutive outputs should stay put rather than flip between extreme commands."""
    from minipacman.ball_tracker import BallEstimate

    safety = SafetyFilter(cfg, "lag_aware")
    robot = RobotPlanarState(np.zeros(2), 0.0, 0.5, 0.0, 0.0)
    ball = BallEstimate(id=0, position=np.array([3.0, 0.0, 0.4]), velocity=np.array([-4.5, 0.0, 0.0]), radius=0.1,
                        restitution=0.7, stamp=0.0, source="truth")
    outputs = []
    for k in range(5):
        ball_k = BallEstimate(id=0, position=ball.position + ball.velocity * 0.02 * k, velocity=ball.velocity,
                              radius=0.1, restitution=0.7, stamp=0.02 * k, source="truth")
        u, info = safety.filter(np.array([0.5, 0.0, -cfg.command.sdot_max]), robot, [ball_k], 0.02 * k)
        outputs.append(u)
    jumps = [np.abs(a - b).max() for a, b in zip(outputs, outputs[1:])]
    assert max(jumps) <= cfg.command.sdot_max + 1e-9      # never a full swing from -sdot_max to +sdot_max
