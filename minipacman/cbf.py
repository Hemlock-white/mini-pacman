"""CBF：安全函數 h、reward 用的 CBF 項、執行時的安全過濾器。【你們實作】

一、Link-CBF（給 reward 用，照 PAC-MAN）
----------------------------------------
PAC-MAN 部署版的做法（rewards.py 的 dodge_link_cbf_reward）：對機器人的每個碰撞形狀 i 和每顆球：

    h_i    = ||p_ball - c_i|| - (r_ball + r_i + margin)        # 距離減掉兩個半徑；h >= 0 代表安全
    hdot_i = d h_i / dt = n_i · (v_ball - v_i)，n_i = (p_ball - c_i) / ||p_ball - c_i||
    term   = clip(min_i(hdot_i + alpha * h_i), -constraint_clip, 0)     # 只罰「違反」的部分，<= 0
    reward += reward_weight * term                                       # 只在 is_threat() 時

c_i、r_i 用每個碰撞 geom 的外接球（m.geom_rbound）近似；v_i 是那個 geom 的線速度
（mujoco.mj_objectVelocity(m, d, mjOBJ_GEOM, g, out6, 0)，世界座標，out6[3:6] 是線速度）。
參數在 config.CBFConfig（alpha=1.0、margin=0.05、clip=2.0、weight=0.27，都和 PAC-MAN 相同）。
這個 reward 用的是**真實的球狀態**（PAC-MAN 也是，叫 privileged）。

二、安全過濾器（執行時強制；PAC-MAN 部署版沒有，是我們的實驗變因）
--------------------------------------------------------------------
輸入 policy 想做的指令 u_nom = (v, w, sdot)，輸出「盡量接近 u_nom、但預測不會撞到球」的 u_safe。
TITA 不能側移，球從正面來時距離型的 h 很快就無解（球的速度 3–6 m/s，遠大於 TITA 的 0.8 m/s），
所以建議用「預測型」：在未來 horizon 秒內，預測球的軌跡和機器人的移動，算最小距離。

    步驟：
    1. 球：每顆 BallEstimate 用 balls.predict_ball() 預測（含彈跳）；有延遲時先往前推 (t - b.stamp)。
    2. 機器人：給定候選指令 u，用「獨輪車模型 + 速度延遲」預測車身位置和高度：
           v_actual 以時間常數 tau 追 u 的 v（naive: tau=0，lag_aware: tau=cfg.cbf.lag_tau_aware）
           yaw += w*dt；xy += v_actual*dt*(cos yaw, sin yaw)；s += sdot*dt（夾在 0..1）
       車身用一個盒子近似（車身座標）：前後 ±0.30 m（含輪子）、左右 ±0.30 m、
       高度 0 到 top(s) = 0.538 - 0.13*s（站姿車頂 0.538 m；蹲到底下降 13 cm，原型量的）。
    3. h_pred(u) = 預測期間「球心到盒子的最短距離 - 球半徑」的最小值。
    4. 候選指令：v、w、sdot 各取幾個值的網格（例如 v 7 個、w 5 個、sdot {-max, 0, +max}），
       先用 lqr.clip_to_envelope 夾在安全範圍內。
    5. 選 h_pred >= filter_margin 裡面「離 u_nom 最近」的（距離可以加權，例如 sdot 乘 0.13 換成 m/s）；
       都不安全就選 h_pred 最大的。
    6. replan="every_step"：每個 policy 步重算；replan="commit"：一偵測到威脅就決定一次，
       之後一直用同一個指令，直到威脅解除（實驗 2 的變因：「反射動作」）。

參考：我們的側面丟球解答版發現，每 10 ms 重新規劃時 naive 和 lag_aware 結果一樣；
只決定一次時，naive 被打中 65%、lag_aware 30%。正面丟球是否一樣，就是實驗 2 要回答的。
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .ball_tracker import BallEstimate
from .config import Config, CBFConfig

ROBOT_HALF_LENGTH = 0.30      # m，車身盒子前後半長（含輪子）
ROBOT_HALF_WIDTH = 0.30       # m，左右半寬（含輪子）
ROBOT_TOP_STAND = 0.538       # m，站姿車頂高度（原型量的）
ROBOT_CROUCH_DROP = 0.13      # m，蹲到底車身下降多少（原型量的）


@dataclass
class RobotPlanarState:
    """CBF 和 policy 需要的機器人狀態（真實值或估測值）。"""
    xy: np.ndarray            # 底座位置 (x, y)
    yaw: float                # 朝向 (rad)
    v: float                  # 前進速度 (m/s)
    w: float                  # 轉彎速度 (rad/s)
    s: float                  # 蹲低程度 0..1


def robot_link_spheres(m: mujoco.MjModel, d: mujoco.MjData, geoms: np.ndarray):
    """每個碰撞 geom 的外接球：回傳 (centers (n,3), radii (n,), velocities (n,3))，世界座標。"""
    raise NotImplementedError("TODO：robot_link_spheres（見檔頭一）")


def link_barriers(centers: np.ndarray, radii: np.ndarray, velocities: np.ndarray,
                  ball: BallEstimate, cfg: CBFConfig) -> tuple[np.ndarray, np.ndarray]:
    """回傳每個 link 對這顆球的 (h (n,), hdot (n,))。"""
    raise NotImplementedError("TODO：link_barriers（見檔頭一）")


def is_threat(ball: BallEstimate, cfg: CBFConfig) -> bool:
    """球離地（z > r + threat_min_height_above_radius）而且水平速度 > threat_min_speed 才算威脅。"""
    raise NotImplementedError


def cbf_reward_term(m: mujoco.MjModel, d: mujoco.MjData, geoms: np.ndarray,
                    balls: list[BallEstimate], cfg: CBFConfig) -> float:
    """task_cbf reward 的 CBF 項（<= 0，已乘 reward_weight）。多顆球時取最差（最負）的那顆。"""
    raise NotImplementedError("TODO：cbf_reward_term（見檔頭一）")


class SafetyFilter:
    """執行時的安全過濾器（見檔頭二）。"""

    def __init__(self, cfg: Config, mode: str = "lag_aware", replan: str = "every_step"):
        """mode："naive" 或 "lag_aware"；replan："every_step" 或 "commit"。"""
        raise NotImplementedError("TODO：SafetyFilter.__init__")

    def reset(self) -> None:
        raise NotImplementedError

    def filter(self, u_nom: np.ndarray, robot: RobotPlanarState, balls: list[BallEstimate],
               t: float) -> tuple[np.ndarray, dict]:
        """回傳 (u_safe, info)。info 至少要有：
            "intervention": ||u_safe - u_nom||（加權後）
            "active": 這一步有沒有改指令
            "h_pred": 選中指令的預測最小距離
        沒有球時直接回傳 u_nom。"""
        raise NotImplementedError

    def predicted_clearance(self, u: np.ndarray, robot: RobotPlanarState, balls: list[BallEstimate],
                            t: float) -> float:
        """候選指令 u 在 horizon 內的預測最小距離 h_pred（見檔頭二的 1–3）。"""
        raise NotImplementedError
