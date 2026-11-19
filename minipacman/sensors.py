"""IMU 和關節 encoder：讀 MuJoCo 感測器並加上雜訊。【你們實作】

TITA 的 robot.xml 本來就有這些感測器（不用改 XML）：

=================  ==========================================  ====================
名稱                內容                                         sensordata 維度
=================  ==========================================  ====================
trunk_gyro         角速度，IMU 座標 (rad/s)                       3
trunk_accel        加速度計（比力 = 加速度 - 重力），IMU 座標        3
trunk_quat         姿態四元數 (w, x, y, z)                        4
*_leg_N_pos        8 個關節角度 (rad)                             各 1
*_leg_N_vel        8 個關節角速度 (rad/s)                         各 1
=================  ==========================================  ====================

怎麼讀某個感測器
----------------
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "trunk_gyro")
    adr, dim = m.sensor_adr[sid], m.sensor_dim[sid]
    value = d.sensordata[adr:adr + dim]
（在 __init__ 先把每個感測器的 adr、dim 查好存起來，每步就只做切片。）

雜訊模型（參數在 config.ImuConfig / EncoderConfig）
----------------------------------------------------
* 陀螺儀：真實值 + 每個 episode 固定的偏差 (bias，reset 時抽) + 每步的白雜訊。
* 加速度計：同上。
* 姿態：把真實四元數乘上一個小的隨機旋轉（角度 ~ N(0, quat_noise_deg)），代表 IMU 內建融合的誤差。
* encoder 角度：量化到 2π / counts_per_rev 的整數倍。
* encoder 角速度：真實值 + 白雜訊。

注意 robot.xml 的 jointactuatorfrc 有 noise="0.01" 屬性，但 MuJoCo **不會**自動加雜訊
（已驗證：noise 只存在 m.sensor_noise 給使用者參考，sensordata 是乾淨的），所以雜訊要在這裡自己加。
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import Config


@dataclass
class ImuReading:
    gyro: np.ndarray       # (3,) rad/s，IMU 座標
    accel: np.ndarray      # (3,) m/s^2，IMU 座標（靜止時約 (0, 0, +9.81)）
    quat: np.ndarray       # (4,) w, x, y, z


@dataclass
class EncoderReading:
    pos: np.ndarray        # (8,) 順序：cfg.robot.legs + cfg.robot.wheels
    vel: np.ndarray        # (8,)


class Imu:
    def __init__(self, m: mujoco.MjModel, cfg: Config):
        """查好 trunk_gyro、trunk_accel、trunk_quat 在 sensordata 的位置。"""
        raise NotImplementedError("TODO：Imu.__init__")

    def reset(self, rng: np.random.Generator) -> None:
        """每個 episode 開始：抽陀螺儀和加速度計的偏差。"""
        raise NotImplementedError

    def read(self, d: mujoco.MjData, rng: np.random.Generator) -> ImuReading:
        raise NotImplementedError


class Encoders:
    def __init__(self, m: mujoco.MjModel, cfg: Config):
        """查好 8 個關節的 *_pos、*_vel 感測器位置（順序：legs + wheels）。

        感測器名稱 = 關節名稱去掉 "joint_" 再加 "_pos" / "_vel"，例如 joint_left_leg_1 → left_leg_1_pos。
        """
        raise NotImplementedError("TODO：Encoders.__init__")

    def reset(self, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def read(self, d: mujoco.MjData, rng: np.random.Generator) -> EncoderReading:
        raise NotImplementedError
