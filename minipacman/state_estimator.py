"""狀態估測：用 IMU + encoder 估計 LQR 需要的狀態。【你們實作，四個裡最難的】

為什麼需要
----------
LQR 的輸入是完整的 qpos / qvel，其中有些量真實機器人量不到：
    * 底座位置 (x, y, z)            → 沒有感測器，要從輪子轉了多少積分（odometry）
    * 底座線速度                    → 從輪子轉速 + 車身俯仰角速度推算
    * 底座姿態、角速度              → IMU
    * 8 個關節角度、角速度          → encoder
config.EstimatorConfig.use_true_state = True 時直接用模擬器的真實值，**估測器寫好之前整條流程都用這個**。

建議做法（簡單、夠用）
----------------------
1. 姿態：直接用 IMU 的 trunk_quat（已含雜訊，當作 IMU 內建的姿態融合輸出）。
   想多做一點：用互補濾波 (complementary_alpha) 融合陀螺儀積分與加速度計的重力方向。
2. 角速度：陀螺儀讀值（IMU 座標 ≈ 車身座標）。
3. 前進速度：兩輪平均角速度 × 輪半徑，但輪子角速度是「相對車身」的，車身前後傾時輪子也會跟著轉，
   所以要加回車身的俯仰角速度：
       v_forward ≈ R * (mean(wheel_vel) * sign + pitch_rate)
   （sign 看輪軸方向，用 env.roll 裡的 coef 符號；寫完拿真實值比對就知道對不對）
4. 位置：v_forward 沿目前朝向積分；z 用腿的運動學（或固定為站姿高度）。
5. 驗證：在 tests/test_state_estimator.py 裡讓機器人用真實狀態跑一段，比較估計值和真實值。
   然後把 use_true_state 改成 False，確認 LQR 還站得住（推力測試 30/60/100 N）。
"""
from __future__ import annotations

import mujoco
import numpy as np

from .config import Config
from .sensors import EncoderReading, ImuReading


class StateEstimator:
    def __init__(self, m: mujoco.MjModel, cfg: Config, robot_nq: int, robot_nv: int):
        raise NotImplementedError("TODO：StateEstimator.__init__")

    def reset(self, qpos0: np.ndarray) -> None:
        """episode 開始：用重置後的真實姿態初始化（開機時機器人知道自己在原點）。"""
        raise NotImplementedError

    def update(self, imu: ImuReading, enc: EncoderReading, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """回傳估計的 (qpos[:robot_nq], qvel[:robot_nv])，格式和 MuJoCo 的一樣，LQR 可以直接用。"""
        raise NotImplementedError

    def planar_state(self):
        """回傳 cbf.RobotPlanarState（xy、yaw、前進速度、轉彎速度、蹲低程度），給 CBF 和 policy 用。"""
        raise NotImplementedError
