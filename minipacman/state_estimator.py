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
        raise NotImplementedError("TODO: StateEstimator.__init__")

    def reset(self, qpos0: np.ndarray) -> None:
        """Start of an episode: initialise from the true pose after reset (at power-on the robot knows it's at the origin)."""
        raise NotImplementedError

    def update(self, imu: ImuReading, enc: EncoderReading, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the estimated (qpos[:robot_nq], qvel[:robot_nv]) in MuJoCo's format, so the LQR can use them directly."""
        raise NotImplementedError

    def planar_state(self):
        """Return a cbf.RobotPlanarState (xy, yaw, forward speed, turn rate, crouch level) for the CBF and the policy."""
        raise NotImplementedError
