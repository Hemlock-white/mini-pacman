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
        raise NotImplementedError("TODO: Imu.__init__")

    def reset(self, rng: np.random.Generator) -> None:
        """Start of an episode: draw the gyro and accelerometer biases."""
        raise NotImplementedError

    def read(self, d: mujoco.MjData, rng: np.random.Generator) -> ImuReading:
        raise NotImplementedError


class Encoders:
    def __init__(self, m: mujoco.MjModel, cfg: Config):
        """Look up the *_pos and *_vel sensors of the 8 joints (order: legs + wheels).

        Sensor name = joint name without "joint_", plus "_pos" / "_vel", e.g. joint_left_leg_1 → left_leg_1_pos.
        """
        raise NotImplementedError("TODO: Encoders.__init__")

    def reset(self, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def read(self, d: mujoco.MjData, rng: np.random.Generator) -> EncoderReading:
        raise NotImplementedError
