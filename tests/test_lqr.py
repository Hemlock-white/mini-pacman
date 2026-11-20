"""Tests for the TITA LQR balance controller."""

import mujoco
import numpy as np
import pytest

from conftest import need_tita


def _build_controller(cfg):
    from minipacman.env import TitaEnv
    from minipacman.lqr import LQRBalancer
    from minipacman.scene import build_scene, scene_ids

    env = TitaEnv(str(build_scene(cfg=cfg)), cfg=cfg)
    ids = scene_ids(env.m, cfg)
    controller = LQRBalancer(env, cfg, ids.robot_nv)
    controller.design_all()
    return env, controller


def _simulate(env, controller, cfg, seconds, command, push=0.0):
    from minipacman.lqr import ReferenceGenerator

    env.reset(*controller.leg_poses["stand"])
    initial = env.planar_state()
    references = ReferenceGenerator(cfg)
    references.reset(initial.xy, initial.yaw)
    times, speeds, turn_rates, pitches = [], [], [], []

    for _ in range(round(seconds / cfg.sim.timestep)):
        time = float(env.d.time)
        env.d.xfrc_applied[env.base, 0] = push if 1.0 <= time < 1.1 else 0.0
        reference = references.update(np.asarray(command, dtype=float), cfg.sim.timestep)
        env.set_leg_targets(controller.leg_targets(reference.s))
        env.set_wheel_torques(controller.wheel_torques(env.d.qpos, env.d.qvel, reference))
        env.step()
        state = env.planar_state(reference.s)
        times.append(float(env.d.time))
        speeds.append(state.v)
        turn_rates.append(state.w)
        pitches.append(env.pitch())
    return {
        "times": np.asarray(times),
        "speeds": np.asarray(speeds),
        "turn_rates": np.asarray(turn_rates),
        "pitches": np.asarray(pitches),
    }


def test_clip_to_envelope(cfg):
    from minipacman.lqr import clip_to_envelope

    command = cfg.command
    u = clip_to_envelope(np.array([5.0, 5.0, 5.0]), 0.0, command)
    assert np.allclose(u, [command.v_max_stand, command.w_max_stand, command.sdot_max])
    u = clip_to_envelope(np.array([5.0, -5.0, 5.0]), 1.0, command)
    assert np.allclose(u, [command.v_max_crouch, -command.w_max_crouch, 0.0])
    u = clip_to_envelope(np.array([0.0, 0.0, -5.0]), 0.0, command)
    assert u[2] == 0.0


def test_reference_rate_limit(cfg):
    from minipacman.lqr import ReferenceGenerator

    generator = ReferenceGenerator(cfg)
    generator.reset(np.zeros(2), 0.0)
    reference = generator.update(np.array([0.8, 2.0, 0.0]), cfg.sim.timestep)
    assert reference.v <= cfg.command.accel_limit * cfg.sim.timestep + 1e-12
    assert reference.w <= cfg.command.yaw_accel_limit * cfg.sim.timestep + 1e-12


@need_tita
def test_gains_are_finite_and_abad_zero(cfg):
    env, controller = _build_controller(cfg)
    n = controller.robot_nv
    abad_dofs = []
    for name in (cfg.robot.legs[0], cfg.robot.legs[3]):
        joint = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_JOINT, name)
        abad_dofs.append(int(env.m.jnt_dofadr[joint]))

    for gain in controller.gains.values():
        assert gain.shape == (2, 2 * n)
        assert np.all(np.isfinite(gain))
        for dof in abad_dofs:
            np.testing.assert_array_equal(gain[:, [dof, n + dof]], 0.0)


@need_tita
@pytest.mark.parametrize("push", [30.0, 60.0, 100.0])
def test_survives_push(cfg, push):
    env, controller = _build_controller(cfg)
    result = _simulate(env, controller, cfg, 3.0, [0.0, 0.0, 0.0], push=push)
    assert np.max(np.abs(result["pitches"])) < 1.0


@need_tita
def test_drive_speed(cfg):
    env, controller = _build_controller(cfg)
    result = _simulate(env, controller, cfg, 3.0, [0.5, 0.0, 0.0])
    tail_speed = float(np.mean(result["speeds"][-100:]))
    assert tail_speed == pytest.approx(0.5, abs=0.08)
    reached = np.flatnonzero(result["speeds"] >= 0.45)
    assert reached.size, "the robot never reached 90% of the requested speed"
    assert 0.2 < result["times"][reached[0]] < 0.8


@need_tita
def test_turning(cfg):
    env, controller = _build_controller(cfg)
    result = _simulate(env, controller, cfg, 2.0, [0.5, 1.5, 0.0])
    tail_turn_rate = float(np.mean(result["turn_rates"][-100:]))
    assert np.max(np.abs(result["pitches"])) < 1.0
    assert abs(tail_turn_rate - 1.5) < 0.2
