"""Tests for simulated sensors and robot state estimation."""

import mujoco
import numpy as np

from conftest import need_tita


def _build(cfg, seed=0):
    from minipacman.env import TitaEnv
    from minipacman.lqr import LQRBalancer, ReferenceGenerator
    from minipacman.scene import build_scene, scene_ids
    from minipacman.sensors import Encoders, Imu
    from minipacman.state_estimator import StateEstimator

    env = TitaEnv(str(build_scene(cfg=cfg)), cfg=cfg)
    ids = scene_ids(env.m, cfg)
    controller = LQRBalancer(env, cfg, ids.robot_nv)
    controller.design_all()
    rng = np.random.default_rng(seed)
    imu = Imu(env.m, cfg)
    encoders = Encoders(env.m, cfg)
    imu.reset(rng)
    encoders.reset(rng)
    estimator = StateEstimator(env.m, cfg, ids.robot_nq, ids.robot_nv)
    estimator.reset(env.d.qpos)
    references = ReferenceGenerator(cfg)
    initial = env.planar_state()
    references.reset(initial.xy, initial.yaw)
    return env, ids, controller, references, imu, encoders, estimator, rng


def _pitch_from_qpos(qpos, root_qadr=0):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, qpos[root_qadr + 3:root_qadr + 7])
    rotation = rotation.reshape(3, 3)
    return float(np.arctan2(rotation[0, 2], rotation[2, 2]))


@need_tita
def test_imu_at_rest(cfg):
    env, _, controller, references, imu, _, _, rng = _build(cfg, seed=1)
    readings = []
    steps = round(2.0 / cfg.sim.timestep)
    for step in range(steps):
        reference = references.update(np.zeros(3), cfg.sim.timestep)
        env.set_leg_targets(controller.leg_targets(reference.s))
        env.set_wheel_torques(controller.wheel_torques(env.d.qpos, env.d.qvel, reference))
        env.step()
        if step >= steps - 100:
            readings.append(imu.read(env.d, rng))
    gyro = np.mean([reading.gyro for reading in readings], axis=0)
    accel = np.mean([reading.accel for reading in readings], axis=0)
    assert np.linalg.norm(gyro) < 0.05
    np.testing.assert_allclose(accel, [0.0, 0.0, 9.81], atol=0.2)


@need_tita
def test_encoder_quantization(cfg):
    env, _, _, _, _, encoders, _, rng = _build(cfg, seed=2)
    reading = encoders.read(env.d, rng)
    quantum = 2 * np.pi / cfg.encoder.counts_per_rev
    np.testing.assert_allclose(reading.pos / quantum, np.round(reading.pos / quantum), atol=1e-12)
    assert reading.pos.shape == (8,)
    assert reading.vel.shape == (8,)


@need_tita
def test_estimator_tracks_truth(cfg):
    env, _, controller, references, imu, encoders, estimator, rng = _build(cfg, seed=3)
    q_estimate = None
    for _ in range(round(3.0 / cfg.sim.timestep)):
        reference = references.update(np.array([0.5, 0.0, 0.0]), cfg.sim.timestep)
        env.set_leg_targets(controller.leg_targets(reference.s))
        env.set_wheel_torques(controller.wheel_torques(env.d.qpos, env.d.qvel, reference))
        env.step()
        q_estimate, _ = estimator.update(
            imu.read(env.d, rng), encoders.read(env.d, rng), cfg.sim.timestep,
        )

    truth = env.planar_state()
    estimate = estimator.planar_state()
    assert abs(estimate.v - truth.v) < 0.05
    assert np.linalg.norm(estimate.xy - truth.xy) < 0.1
    pitch_error = abs(_pitch_from_qpos(q_estimate, estimator.root_qadr) - env.pitch())
    assert pitch_error < np.deg2rad(2.0)


@need_tita
def test_lqr_with_estimator(cfg):
    env, ids, controller, references, imu, encoders, estimator, rng = _build(cfg, seed=4)
    q_estimate = env.d.qpos[:ids.robot_nq].copy()
    v_estimate = env.d.qvel[:ids.robot_nv].copy()
    peak_pitch = 0.0
    for _ in range(round(3.0 / cfg.sim.timestep)):
        env.d.xfrc_applied[env.base, 0] = 60.0 if 1.0 <= env.d.time < 1.1 else 0.0
        reference = references.update(np.zeros(3), cfg.sim.timestep)
        env.set_leg_targets(controller.leg_targets(reference.s))
        env.set_wheel_torques(controller.wheel_torques(q_estimate, v_estimate, reference))
        env.step()
        q_estimate, v_estimate = estimator.update(
            imu.read(env.d, rng), encoders.read(env.d, rng), cfg.sim.timestep,
        )
        peak_pitch = max(peak_pitch, abs(env.pitch()))
    assert peak_pitch < cfg.task.fall_pitch
    assert np.all(np.isfinite(q_estimate))
    assert np.all(np.isfinite(v_estimate))
