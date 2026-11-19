"""LQR 平衡控制：把高層指令 (v, w, s) 變成 2 個輪子扭矩 + 6 個腿關節目標角度。【你們實作】

這個檔案的內容大部分可以從 TITA_Feasibility_Check.ipynb 第 3 格（tita_core.py）搬過來，
新的部分只有兩個：(1) 場景裡多了球，(2) 要能轉彎。

整體流程
--------
    ref_gen = ReferenceGenerator(cfg)              # 把 policy 的指令變成平滑的參考軌跡
    lqr = LQRBalancer(env, cfg, ids.robot_nv)      # env = env.py 的 TitaEnv
    lqr.design_all()                               # 啟動時算一次增益（站姿、最低蹲姿）
    每個物理步：
        ref = ref_gen.update(u_safe, dt)           # u_safe = (v, w, sdot)，CBF 過濾後的指令
        env.set_leg_targets(lqr.leg_targets(ref.s))
        env.set_wheel_torques(lqr.wheel_torques(qpos, qvel, ref))

重要的坑（原型時踩過的）
------------------------
1. **球不能進 DARE**：場景有 5 顆球、每顆 6 個 dof，它們不受輪子控制，放進 DARE 會
   「Failed to find a finite solution」。線性化完只取機器人的 dof：0 .. robot_nv-1
   （scene.scene_ids(m).robot_nv，目前是 14）。算出的 K 再補 0 擴回完整大小，或計算誤差時只取前 robot_nv 個。
2. **線性化時 frictionloss 要暫時設 0**，算完再還原（摩擦損失不可微，會讓 A 矩陣怪掉）。
3. **線性化前輪子要壓進地面**（config.robot.z_offset = -0.5 mm），否則線性化看不到接觸，LQR 一開就倒。
4. **側擺關節 (abad, joint_*_leg_1) 的增益要歸零**：硬接觸造成的假象，原本約 ±30000。
5. **輪子轉角的誤差不能直接歸零**，要用滾動關係算（tita_core.py 的 wheel_dq）：
       dq[wheel] = coef * (dq[x] - dq[yaw] * y_offset)，coef = xaxis_y / R（env.reset() 會算 env.roll）
6. **接觸用 MuJoCo 預設**（pyramidal、impratio=1），robot.xml 的 elliptic + impratio=100 會讓 LQR 漂移。
7. **env.py 的 set_pose() 會把球也歸零**：它寫 ``d.qpos[:] = 0``，新場景裡 5 顆球會全部跑到原點、
   壓在機器人底下（已實測）。線性化時球的接觸會混進 A、B。改成只歸零機器人的部分
   ``d.qpos[:robot_nq] = 0``，或是 reset 之後呼叫 BallLauncher.park(i) 把球收起來。

轉彎（已在筆電上驗證可行，見 config.CommandConfig 的安全範圍）
----------------------------------------------------------------
LQR 是在「朝向 0 度」線性化的。要追蹤轉彎速度 w，關鍵是**把誤差轉到參考的車頭座標系**：
    a. 參考姿態 q_ref = q0（線性化姿態）但底座位置換成 ref.p、底座四元數左乘「繞 z 轉 ref.psi」。
    b. dq = mj_differentiatePos(q_ref, qpos)；dq[0:3] 是世界座標的位置差 → 轉到目前車頭座標：
           前進誤差 = cos(yaw)*dx + sin(yaw)*dy；側向誤差 = 0（兩輪不能側移，不可控，不要餵給 LQR）
       dq[3:6] 是車身座標的轉角差（roll、pitch、yaw），直接用。
    c. 速度 qvel[0:3] 也轉到車頭座標；ve[0] -= ref.v；ve[5] -= ref.w。
    d. 輪子速度的參考：ve[wheel] -= coef * (ref.v - ref.w * y_offset)。
    e. 球的 dof 誤差全部設 0。
實測：站姿 v=0.5 m/s、w=1.5 rad/s 繞圈，最大前後傾 15 度；原地轉 2.5 rad/s 正常；
蹲到底 (s=1) 時 v=0.5 會傾到 50 度，所以 ReferenceGenerator 要限制「蹲越低、速度越慢」。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, CommandConfig


@dataclass
class Reference:
    """LQR 要追蹤的參考（世界座標）。"""
    p: np.ndarray          # 參考位置 (x, y)
    psi: float             # 參考朝向 (rad)
    v: float               # 參考前進速度 (m/s)
    w: float               # 參考轉彎速度 (rad/s)
    s: float               # 蹲低程度 0（站）.. 1（最低）


def clip_to_envelope(u: np.ndarray, s: float, cmd: CommandConfig) -> np.ndarray:
    """把指令 u = (v, w, sdot) 夾在安全範圍內。

    速度上限隨蹲低程度 s 線性遞減：
        v_max(s) = v_max_stand + (v_max_crouch - v_max_stand) * s
        w_max(s) = w_max_stand + (w_max_crouch - w_max_stand) * s
    v 夾在 [v_min, v_max(s)]，w 夾在 [-w_max(s), w_max(s)]，sdot 夾在 [-sdot_max, sdot_max]，
    而且 s 已經是 0 時 sdot 不能是負的、s 是 1 時不能是正的。
    回傳新的 np.ndarray（不要改到輸入）。
    """
    raise NotImplementedError("TODO：clip_to_envelope")


class ReferenceGenerator:
    """把指令 u = (v, w, sdot) 變成平滑的參考軌跡。

    每個物理步呼叫 update()：
      1. u = clip_to_envelope(u, self.s, cfg.command)
      2. 參考速度朝 u 靠近，但變化率不超過 accel_limit（v）和 yaw_accel_limit（w）。
         原型實測：速度指令一步跳太多，LQR 會被「踢」到倒下。
      3. s += sdot * dt，夾在 [0, 1]。
      4. psi += w * dt；p += v * dt * (cos psi, sin psi)。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        raise NotImplementedError("TODO：ReferenceGenerator.__init__")

    def reset(self, p0: np.ndarray, psi0: float) -> None:
        """episode 開始：參考位置 = 機器人位置，速度 0，s = 0。"""
        raise NotImplementedError

    def update(self, u: np.ndarray, dt: float) -> Reference:
        raise NotImplementedError

    @property
    def s(self) -> float:
        raise NotImplementedError


class LQRBalancer:
    """站姿與最低蹲姿各算一組 LQR 增益，中間用 s 線性內插（gain scheduling）。"""

    def __init__(self, env, cfg: Config, robot_nv: int):
        """env：env.py 的 TitaEnv（要有 m、d、poses、legs_of、balanced_k、reset、roll）。
        robot_nv：機器人的 dof 數，球的 dof 在後面（見檔頭「坑 1」）。"""
        raise NotImplementedError("TODO：LQRBalancer.__init__")

    def design_all(self) -> None:
        """設計站姿 (cfg.robot.stand_hip) 和最低蹲姿 (cfg.robot.crouch_hip) 的增益，存在 self.gains。"""
        raise NotImplementedError

    def design(self, name: str, hip: float) -> tuple[np.ndarray, np.ndarray]:
        """在一個姿勢線性化並解 DARE，回傳 (q0 線性化姿態, K 增益 (2, 2*robot_nv))。

        步驟（tita_core.py 的 Tita.lqr）：
          1. knee = env.balanced_k(hip)；env.reset(hip, knee)（輪子微微壓進地面）
          2. frictionloss 暫時設 0；mujoco.mjd_transitionFD(m, d, eps, True, A, B, None, None)；還原
          3. 只取機器人 dof 的 A、B（列和行都是 [0..robot_nv-1] 和 [nv..nv+robot_nv-1]），B 只取兩個輪子馬達
          4. Q = 對角，預設 cfg.lqr.q_default；位置 x、pitch、yaw、各自的速度用 cfg.lqr 的值；輪子轉角 = 0
             R = cfg.lqr.r_wheel * I(2)
          5. A、B 乘 sqrt(discount) 再解 scipy.linalg.solve_discrete_are
          6. K = (R + B'PB)^-1 B'PA；abad 關節的欄位歸零
        """
        raise NotImplementedError

    def leg_targets(self, s: float) -> np.ndarray:
        """6 個腿關節的目標角度：hip、knee 在站姿與蹲姿之間依 s 內插（env.legs_of(h, k)）。"""
        raise NotImplementedError

    def wheel_torques(self, qpos: np.ndarray, qvel: np.ndarray, ref: Reference) -> np.ndarray:
        """2 個輪子的扭矩 (Nm)，夾在 ±cfg.robot.wheel_torque_limit。

        qpos、qvel：完整的狀態（真實值或 state_estimator 的估計值）。
        步驟見檔頭「轉彎」a–e；增益 K 和 q0 依 ref.s 在兩組之間內插。
        """
        raise NotImplementedError
