"""Tests for the TITA MuJoCo wrapper."""

import mujoco
import numpy as np
import pytest

from conftest import HAS_TITA


@pytest.fixture
def tita_env(cfg):
    if not HAS_TITA:
        pytest.skip("download the TITA model first")
    from minipacman.env import TitaEnv
    from minipacman.scene import build_scene

    return TitaEnv(str(build_scene(cfg=cfg)), cfg=cfg)


def test_actuator_interface_validates_and_clips(tita_env, cfg):
    leg_targets = np.linspace(-0.3, 0.3, 6)
    tita_env.set_leg_targets(leg_targets)
    np.testing.assert_allclose(tita_env.d.ctrl[tita_env.pact], leg_targets)

    tita_env.set_wheel_torques([100.0, -100.0])
    limit = cfg.robot.wheel_torque_limit
    np.testing.assert_allclose(tita_env.d.ctrl[tita_env.wact], [limit, -limit])

    with pytest.raises(ValueError, match="shape"):
        tita_env.set_leg_targets(np.zeros(5))
    with pytest.raises(ValueError, match="finite"):
        tita_env.set_wheel_torques([np.nan, 0.0])


def test_robot_reset_preserves_ball_state(tita_env, cfg):
    from minipacman.scene import scene_ids

    ids = scene_ids(tita_env.m, cfg)
    for index, (qadr, dadr) in enumerate(zip(ids.ball_qadr, ids.ball_dadr)):
        tita_env.d.qpos[qadr:qadr + 7] = [10 + index, -2, 1, 1, 0, 0, 0]
        tita_env.d.qvel[dadr:dadr + 6] = np.arange(6) + index
    ball_qpos = tita_env.d.qpos[ids.robot_nq:].copy()
    ball_qvel = tita_env.d.qvel[ids.robot_nv:].copy()

    tita_env.reset(0.5, -1.6)

    np.testing.assert_array_equal(tita_env.d.qpos[ids.robot_nq:], ball_qpos)
    np.testing.assert_array_equal(tita_env.d.qvel[ids.robot_nv:], ball_qvel)


def test_planar_state_and_step(tita_env):
    tita_env.reset(0.5, -1.6)
    state = tita_env.planar_state(s=0.25)

    np.testing.assert_allclose(state.xy, tita_env.d.xpos[tita_env.base, :2])
    assert state.yaw == pytest.approx(0.0, abs=1e-12)
    assert state.v == pytest.approx(0.0, abs=1e-12)
    assert state.w == pytest.approx(0.0, abs=1e-12)
    assert state.s == pytest.approx(0.25)
    assert np.isfinite(tita_env.pitch())

    yaw = np.pi / 2
    qadr, dadr = tita_env.root_qadr, tita_env.root_dadr
    tita_env.d.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    tita_env.d.qvel[dadr:dadr + 6] = [0, 0.4, 0, 0, 0, 0.7]
    mujoco.mj_forward(tita_env.m, tita_env.d)
    moving = tita_env.planar_state()
    assert moving.yaw == pytest.approx(yaw)
    assert moving.v == pytest.approx(0.4, abs=1e-4)
    assert moving.w == pytest.approx(0.7)

    start = tita_env.d.time
    tita_env.step(3)
    assert tita_env.d.time == pytest.approx(start + 3 * tita_env.m.opt.timestep)
    with pytest.raises(ValueError, match="non-negative integer"):
        tita_env.step(-1)


def test_model_uses_configured_simulation_options(tita_env, cfg):
    assert tita_env.robot_nq < tita_env.m.nq
    assert tita_env.robot_nv < tita_env.m.nv
    assert tita_env.m.opt.timestep == pytest.approx(cfg.sim.timestep)
    assert tita_env.m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert tita_env.m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
