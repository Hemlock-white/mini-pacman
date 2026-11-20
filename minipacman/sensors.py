"""IMU and joint encoders: read the MuJoCo sensors and add noise. [YOURS]

TITA's robot.xml already has these sensors (no XML changes needed):

=================  ==========================================  ====================
name               content                                     size in sensordata
=================  ==========================================  ====================
trunk_gyro         angular velocity, IMU frame (rad/s)         3
trunk_accel        accelerometer (specific force = accel - g)  3
trunk_quat         orientation quaternion (w, x, y, z)         4
*_leg_N_pos        8 joint angles (rad)                        1 each
*_leg_N_vel        8 joint velocities (rad/s)                  1 each
=================  ==========================================  ====================

How to read a sensor
--------------------
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "trunk_gyro")
    adr, dim = m.sensor_adr[sid], m.sensor_dim[sid]
    value = d.sensordata[adr:adr + dim]
(Look up every sensor's adr and dim once in __init__, then each step is just a slice.)

Noise model (parameters in config.ImuConfig / EncoderConfig)
------------------------------------------------------------
* Gyro: true value + a bias fixed per episode (drawn in reset) + white noise every step.
* Accelerometer: same.
* Orientation: multiply the true quaternion by a small random rotation (angle ~ N(0, quat_noise_deg)), standing in for
  the error of the IMU's built-in fusion.
* Encoder angle: quantised to integer multiples of 2π / counts_per_rev.
* Encoder velocity: true value + white noise.

Note that robot.xml's jointactuatorfrc sensors have noise="0.01", but MuJoCo does **not** add noise by itself
(checked: noise only lives in m.sensor_noise for the user's reference; sensordata is clean), so add it here.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import Config


@dataclass
class ImuReading:
    gyro: np.ndarray       # (3,) rad/s, IMU frame
    accel: np.ndarray      # (3,) m/s^2, IMU frame (about (0, 0, +9.81) at rest)
    quat: np.ndarray       # (4,) w, x, y, z


@dataclass
class EncoderReading:
    pos: np.ndarray        # (8,) order: cfg.robot.legs + cfg.robot.wheels
    vel: np.ndarray        # (8,)


class Imu:
    def __init__(self, m: mujoco.MjModel, cfg: Config):
        """Look up where trunk_gyro, trunk_accel and trunk_quat are in sensordata."""
        self.cfg = cfg
        self.ic = cfg.imu
        self._gyro = self._sensor_slice(m, self.ic.gyro_sensor, 3)
        self._accel = self._sensor_slice(m, self.ic.accel_sensor, 3)
        self._quat = self._sensor_slice(m, self.ic.quat_sensor, 4)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

    @staticmethod
    def _sensor_slice(m: mujoco.MjModel, name: str, expected_dim: int) -> slice:
        sensor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor < 0:
            raise ValueError(f"sensor {name!r} is missing from the model")
        dimension = int(m.sensor_dim[sensor])
        if dimension != expected_dim:
            raise ValueError(f"sensor {name!r} has dimension {dimension}, expected {expected_dim}")
        address = int(m.sensor_adr[sensor])
        return slice(address, address + dimension)

    def reset(self, rng: np.random.Generator) -> None:
        """Start of an episode: draw the gyro and accelerometer biases."""
        self.gyro_bias = rng.normal(0.0, self.ic.gyro_bias_std, 3)
        self.accel_bias = rng.normal(0.0, self.ic.accel_bias_std, 3)

    def read(self, d: mujoco.MjData, rng: np.random.Generator) -> ImuReading:
        gyro = (d.sensordata[self._gyro] + self.gyro_bias
                + rng.normal(0.0, self.ic.gyro_noise, 3))
        accel = (d.sensordata[self._accel] + self.accel_bias
                 + rng.normal(0.0, self.ic.accel_noise, 3))

        true_quat = d.sensordata[self._quat]
        axis = rng.normal(size=3)
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
        angle = rng.normal(0.0, np.deg2rad(self.ic.quat_noise_deg))
        noise_quat = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, true_quat, noise_quat)
        quat /= np.linalg.norm(quat)
        return ImuReading(gyro=gyro, accel=accel, quat=quat)


class Encoders:
    def __init__(self, m: mujoco.MjModel, cfg: Config):
        """Look up the *_pos and *_vel sensors of the 8 joints (order: legs + wheels).

        Sensor name = joint name without "joint_", plus "_pos" / "_vel", e.g. joint_left_leg_1 → left_leg_1_pos.
        """
        self.cfg = cfg
        self.ec = cfg.encoder
        joints = tuple(cfg.robot.legs) + tuple(cfg.robot.wheels)
        self._position = [Imu._sensor_slice(m, name.removeprefix("joint_") + "_pos", 1)
                          for name in joints]
        self._velocity = [Imu._sensor_slice(m, name.removeprefix("joint_") + "_vel", 1)
                          for name in joints]
        self.quantum = 2 * np.pi / self.ec.counts_per_rev

    def reset(self, rng: np.random.Generator) -> None:
        del rng

    def read(self, d: mujoco.MjData, rng: np.random.Generator) -> EncoderReading:
        position = np.array([d.sensordata[address][0] for address in self._position])
        position = np.round(position / self.quantum) * self.quantum
        velocity = np.array([d.sensordata[address][0] for address in self._velocity])
        velocity += rng.normal(0.0, self.ec.vel_noise, len(velocity))
        return EncoderReading(pos=position, vel=velocity)
