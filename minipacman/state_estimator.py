"""State estimation: estimate the state the LQR needs from the IMU + encoders. [YOURS, the hardest of the four]

Why we need it
--------------
The LQR takes the full qpos / qvel, and a real robot can't measure some of it:
    * base position (x, y, z)        → no sensor; integrate how far the wheels turned (odometry)
    * base linear velocity           → from wheel speeds + body pitch rate
    * base orientation, angular vel. → IMU
    * 8 joint angles and velocities  → encoders
With config.EstimatorConfig.use_true_state = True the simulator's true values are used directly. **Use that for the
whole pipeline until the estimator works.**

Suggested approach (simple, good enough)
----------------------------------------
1. Orientation: use the IMU's trunk_quat directly (it has noise; treat it as the IMU's built-in orientation fusion).
   If you want more: fuse integrated gyro with the accelerometer's gravity direction using a complementary filter
   (complementary_alpha).
2. Angular velocity: the gyro reading (IMU frame ≈ body frame).
3. Forward speed: average wheel speed × wheel radius. But wheel speed is measured "relative to the body", and the
   wheels also turn when the body pitches, so add the body pitch rate back:
       v_forward ≈ R * (mean(wheel_vel) * sign + pitch_rate)
   (sign depends on the wheel axis direction; use the sign of coef in env.roll. Compare against the true value and
   you'll know right away if it's right.)
4. Position: integrate v_forward along the current heading; z from the leg kinematics (or fixed at standing height).
5. Check: in tests/test_sensors_estimator.py, run the robot on the true state for a while and compare estimate vs
   truth. Then set use_true_state to False and make sure the LQR still stands (push test 30/60/100 N).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .config import Config
from .sensors import EncoderReading, ImuReading


class StateEstimator:
    def __init__(self, m: mujoco.MjModel, cfg: Config, robot_nq: int, robot_nv: int):
        if not 0 < robot_nq <= m.nq or not 0 < robot_nv <= m.nv:
            raise ValueError("robot_nq and robot_nv must describe a non-empty prefix of the model state")
        self.m = m
        self.cfg = cfg
        self.robot_nq = int(robot_nq)
        self.robot_nv = int(robot_nv)

        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg.robot.base_body)
        if base < 0:
            raise ValueError(f"body {cfg.robot.base_body!r} is missing from the model")
        root_joint = int(m.body_jntadr[base])
        self.root_qadr = int(m.jnt_qposadr[root_joint])
        self.root_dadr = int(m.jnt_dofadr[root_joint])

        self.joint_names = tuple(cfg.robot.legs) + tuple(cfg.robot.wheels)
        self.joint_ids = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ], dtype=int)
        if np.any(self.joint_ids < 0):
            raise ValueError("one or more configured robot joints are missing from the model")
        self.joint_qadr = m.jnt_qposadr[self.joint_ids].astype(int)
        self.joint_dadr = m.jnt_dofadr[self.joint_ids].astype(int)
        self.wheel_indices = np.arange(len(cfg.robot.legs), len(self.joint_names))
        self.hip_indices = np.array([
            index for index, name in enumerate(self.joint_names)
            if name in (cfg.robot.legs[1], cfg.robot.legs[4])
        ])
        self.initialized = False

    def reset(self, qpos0: np.ndarray) -> None:
        """Start of an episode: initialise from the true pose after reset (at power-on the robot knows it's at the origin)."""
        initial = np.asarray(qpos0, dtype=float)
        if initial.ndim != 1 or len(initial) < self.robot_nq:
            raise ValueError(f"qpos0 must contain at least {self.robot_nq} values")
        self.qpos = initial[:self.robot_nq].copy()
        self.qvel = np.zeros(self.robot_nv)
        root = self.root_qadr
        self.quat = self.qpos[root + 3:root + 7].copy()
        norm = np.linalg.norm(self.quat)
        if norm < 1e-12:
            raise ValueError("the initial root quaternion is invalid")
        self.quat /= norm
        self.position = self.qpos[root:root + 3].copy()

        data = mujoco.MjData(self.m)
        data.qpos[:self.robot_nq] = self.qpos
        mujoco.mj_forward(self.m, data)
        coefficients = data.xaxis[self.joint_ids[self.wheel_indices], 1] / self.cfg.robot.wheel_radius
        if np.any(np.abs(coefficients) < 1e-9):
            raise RuntimeError("could not determine wheel rolling directions")
        self.wheel_signs = np.sign(coefficients)
        self.forward_speed = 0.0
        self.turn_rate = 0.0
        self.crouch = self._crouch_from_encoders(self.qpos[self.joint_qadr])
        self.initialized = True

    def update(self, imu: ImuReading, enc: EncoderReading, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the estimated (qpos[:robot_nq], qvel[:robot_nv]) in MuJoCo's format, so the LQR can use them directly."""
        if not self.initialized:
            raise RuntimeError("call reset() before update()")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        gyro = np.asarray(imu.gyro, dtype=float)
        accel = np.asarray(imu.accel, dtype=float)
        measured_quat = np.asarray(imu.quat, dtype=float)
        joint_pos = np.asarray(enc.pos, dtype=float)
        joint_vel = np.asarray(enc.vel, dtype=float)
        if gyro.shape != (3,) or accel.shape != (3,) or measured_quat.shape != (4,):
            raise ValueError("IMU gyro/acceleration/quaternion have invalid shapes")
        if joint_pos.shape != (len(self.joint_names),) or joint_vel.shape != joint_pos.shape:
            raise ValueError("encoder position/velocity have invalid shapes")
        if not all(np.all(np.isfinite(value)) for value in
                   (gyro, accel, measured_quat, joint_pos, joint_vel)):
            raise ValueError("sensor readings must be finite")

        measured_norm = np.linalg.norm(measured_quat)
        if measured_norm < 1e-12:
            raise ValueError("the measured IMU quaternion is invalid")
        measured_quat = measured_quat / measured_norm
        angular_step = np.linalg.norm(gyro) * dt
        if angular_step > 1e-12:
            axis = gyro / np.linalg.norm(gyro)
            delta_quat = np.r_[np.cos(angular_step / 2), axis * np.sin(angular_step / 2)]
        else:
            delta_quat = np.array([1.0, 0.0, 0.0, 0.0])
        predicted_quat = np.empty(4)
        mujoco.mju_mulQuat(predicted_quat, self.quat, delta_quat)
        if predicted_quat @ measured_quat < 0.0:
            measured_quat = -measured_quat
        alpha = self.cfg.estimator.complementary_alpha
        self.quat = alpha * predicted_quat + (1.0 - alpha) * measured_quat
        self.quat /= np.linalg.norm(self.quat)

        pitch_rate = gyro[1]
        wheel_speed = np.mean(joint_vel[self.wheel_indices] * self.wheel_signs)
        wheel_forward_speed = self.cfg.robot.wheel_radius * (wheel_speed + pitch_rate)
        speed_margin = self.cfg.command.accel_limit * self.cfg.command.speed_lag_tau
        wheel_forward_speed = float(np.clip(
            wheel_forward_speed,
            self.cfg.command.v_min - speed_margin,
            self.cfg.command.v_max_stand + speed_margin,
        ))
        self.turn_rate = float(gyro[2])
        yaw = self._yaw(self.quat)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.quat)
        world_accel = rotation.reshape(3, 3) @ accel + self.m.opt.gravity
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        inertial_speed = self.forward_speed + float(world_accel[:2] @ heading) * dt
        alpha = self.cfg.estimator.complementary_alpha
        self.forward_speed = float(alpha * inertial_speed + (1.0 - alpha) * wheel_forward_speed)
        self.position[:2] += self.forward_speed * dt * heading
        self.crouch = self._crouch_from_encoders(joint_pos)

        root_q, root_v = self.root_qadr, self.root_dadr
        self.qpos[root_q:root_q + 3] = self.position
        self.qpos[root_q + 3:root_q + 7] = self.quat
        self.qpos[self.joint_qadr] = joint_pos
        self.qvel[:] = 0.0
        self.qvel[root_v:root_v + 3] = self.forward_speed * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        self.qvel[root_v + 3:root_v + 6] = gyro
        self.qvel[self.joint_dadr] = joint_vel
        return self.qpos.copy(), self.qvel.copy()

    def planar_state(self):
        """Return a cbf.RobotPlanarState (xy, yaw, forward speed, turn rate, crouch level) for the CBF and the policy."""
        if not self.initialized:
            raise RuntimeError("call reset() before planar_state()")
        from .cbf import RobotPlanarState

        return RobotPlanarState(
            xy=self.position[:2].copy(),
            yaw=self._yaw(self.quat),
            v=self.forward_speed,
            w=self.turn_rate,
            s=self.crouch,
        )

    def _crouch_from_encoders(self, joint_position: np.ndarray) -> float:
        hip = float(np.mean(np.abs(joint_position[self.hip_indices])))
        span = self.cfg.robot.crouch_hip - self.cfg.robot.stand_hip
        return float(np.clip((hip - self.cfg.robot.stand_hip) / span, 0.0, 1.0))

    @staticmethod
    def _yaw(quat: np.ndarray) -> float:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        rotation = rotation.reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))
