"""強化學習環境：一隻 TITA、開向終點、球從正面丟過來。【你們實作】

PPO 透過 vec_env.py 同時跑很多個 DodgeEnv。這個檔案把所有東西接在一起：

    每個 policy 步（50 Hz，cfg.sim.policy_every 個物理步）：
      1. 觀測 obs  ← 深度相機 (depth_camera.py) + 機器人狀態（真實或 state_estimator）
      2. policy 輸出 action ∈ [-1, 1]^3 → 換成指令 u = (v, w, sdot)（見 action_to_command）
      3. [可選] SafetyFilter (cbf.py) 用球資訊來源 (ball_sources.py) 把 u 改成 u_safe
      4. launcher.update(t)（發球、收球、更新最小距離）
      5. 跑 policy_every 個物理步：每步 ReferenceGenerator + LQR (lqr.py) → 馬達 → mj_step → launcher.check_contacts()
      6. 算 reward、判斷結束

已經實作好、直接用的元件
------------------------
    scene.build_scene / scene_ids        場景與 id
    balls.BallLauncher                   發球、命中判定、最小距離、每顆球的紀錄
    depth_camera.DepthCamera / DepthFrameStack    深度圖、policy 小影像、疊影像
    ball_sources.make_ball_source        球資訊來源（truth / noisy_truth / depth_tracker）
你們要實作的元件：env.py（TitaEnv 加上 set_leg_targets / set_wheel_torques / step 等介面）、lqr.py、
sensors.py、state_estimator.py、cbf.py，以及這個檔案。

env.py 還需要的介面（建議）
--------------------------
    TitaEnv.set_leg_targets(q6)      → d.ctrl[pact] = q6
    TitaEnv.set_wheel_torques(tau2)  → d.ctrl[wact] = tau2
    TitaEnv.planar_state()           → cbf.RobotPlanarState（真實值）
    TitaEnv.pitch()                  已經有
    注意：TitaEnv.set_pose() 目前用 d.qpos[:] = 0，會把球也移到原點（壓在機器人底下）。
    改成 d.qpos[:robot_nq] = 0，或在 env.reset() 之後再呼叫 launcher.reset()（它會把球收起來）。

三種 reward（cfg.reward、cfg.cbf；variant 由建構子決定）
--------------------------------------------------------
    共同（"task"）：
        + progress_weight × (上一步到終點的距離 - 這一步到終點的距離)
        + goal_bonus（第一次進入 goal_radius）
        + at_goal_weight（每步，若在終點範圍內）
        - action_rate_weight × ||a_t - a_{t-1}||²
        - fall_penalty（倒下，episode 結束）
    "task_collision"：再加 - collision_penalty × 這一步新被打中的球數（launcher.check_contacts 的回傳）
    "task_cbf"：再加 cbf.cbf_reward_term(...)（用真實球狀態，privileged）
注意：被球打中**不結束** episode（PAC-MAN 會結束）。結束的話 "task" 也會間接學到要閃，就不是「不管安全」的基準了。

結束條件
--------
    terminated：倒下（|pitch| > cfg.task.fall_pitch）
    truncated：時間到（cfg.sim.episode_seconds）→ rsl_rl 的 extras["time_outs"]
終點到了不結束：繼續待著，球還是會丟過來（每回合 1–5 顆，排在 20 秒內）。

觀測（已確認：照 PAC-MAN 的非對稱 actor-critic，參數在 config.ObsConfig）
----------------------------------------------------------------------
actor（部署時真的在跑的網路，只能用感測器拿得到的東西），589 維：
    depth          576   疊 4 張的 16x9「只留球」深度圖（DepthFrameStack.get()），每步：
                         frame = camera.render(d)；stack.push(camera.policy_image(frame, rng, augment_depth))
    gyro             3   車身角速度（IMU；use_true_state 時用真實值）
    gravity          3   重力方向在車身座標的單位向量 = R_body^T @ (0, 0, -1)
    forward_speed    1   前進速度（state_estimator 或真實值）
    crouch           1   蹲低程度 s（ReferenceGenerator.s）
    last_action      3   上一步的 policy 輸出（reset 時填 0）
    goal             2   終點在車頭座標的 (前, 左) 距離
critic（只在訓練時用，可以看真實值），632 維 = actor 的 589 維 +
    base_vel         3   真實底座線速度（車頭座標）
    balls           40   真實球狀態，5 個欄位 × 8（ball_slot_features；用 TruthBallSource）
備案 B（config.obs.actor_ball_input = "tracker"）：actor 的 depth 576 換成追蹤器估出的球 40 維
（ball_slot_features 用 DepthTrackerBallSource 的結果），其餘不變，actor 變 53 維。
每一段在向量裡的位置：obs_layout(cfg)（已實作），actor_obs()/critic_obs() 要照這個順序填。

為什麼 critic 多看真實值：PPO 的 critic 要估「這個狀態之後能拿多少 reward」，球在哪是關鍵；
從 4 張模糊的小深度圖猜很難，給它真實值訓練會穩很多。部署（評估）時只跑 actor，所以不算作弊。

課程學習
--------
訓練時每個環境的前 cfg.obs.single_ball_env_steps 個 policy 步，每回合只丟 1 顆球
（reset 時把 launcher 的 balls_per_episode 暫時設成 (1, 1)），之後才 1–5 顆。評估時一律 1–5 顆。
"""
from __future__ import annotations

import numpy as np

from .config import Config

ACTION_DIM = 3     # (v, w, sdot)，policy 輸出在 [-1, 1]，由 action_to_command 換成實際單位


def obs_layout(cfg: Config) -> dict[str, dict[str, slice]]:
    """actor / critic 觀測裡每一段的位置。回傳 {"actor": {名稱: slice}, "critic": {名稱: slice}}。

    例如 obs_layout(cfg)["actor"]["goal"] 是終點那 2 維在 actor 向量裡的位置（ScriptedPolicy 可以用）。
    向量總長度 = 最後一個 slice 的 stop。
    """
    oc, cc = cfg.obs, cfg.camera
    slot_dim = oc.ball_slots * oc.ball_slot_features
    if oc.actor_ball_input == "depth":
        ball_part = ("depth", len(cc.frame_offsets) * cc.policy_res[0] * cc.policy_res[1])
    elif oc.actor_ball_input == "tracker":
        ball_part = ("tracked_balls", slot_dim)
    else:
        raise ValueError(f"未知的 actor_ball_input：{oc.actor_ball_input!r}")
    actor_terms = [ball_part, *oc.proprio_terms, ("goal", oc.goal_dims)]
    critic_terms = actor_terms + [("base_vel", oc.privileged_base_vel), ("balls", slot_dim)]
    out = {}
    for group, terms in (("actor", actor_terms), ("critic", critic_terms)):
        i, slices = 0, {}
        for name, dim in terms:
            slices[name] = slice(i, i + dim)
            i += dim
        out[group] = slices
    return out


def ball_slot_features(balls, robot_xy: np.ndarray, robot_yaw: float, cfg: Config) -> np.ndarray:
    """把一串 BallEstimate 變成固定長度的向量 (ball_slots * 8,)。

    每顆球 8 維：相對位置 3（球 - 機器人，轉到車頭座標：前、左、上；上 = 球的 z）、
               相對速度 3（球的速度轉到車頭座標，不扣機器人速度也可以，但要訓練和評估一致）、
               半徑 1、是否存在 1（= 1）。
    排序：離機器人近的在前；超過 ball_slots 顆就丟掉最遠的；不足的欄位全部補 0（「是否存在」= 0，
    網路才分得出「沒有球」和「球在原點」）。相對位置夾在 ±clip_rel_pos。
    有延遲的估計（b.stamp < 現在）可以先用 balls.predict_ball 推到現在再算。
    """
    raise NotImplementedError("TODO：ball_slot_features（見 docstring）")


def action_to_command(action: np.ndarray, cfg: Config) -> np.ndarray:
    """把 policy 輸出（約在 [-1, 1]）換成 u = (v, w, sdot)。

    建議：先 clip 到 [-1, 1]，再
        v    = 線性映射到 [cfg.command.v_min, cfg.command.v_max_stand]
        w    = action[1] * cfg.command.w_max_stand
        sdot = action[2] * cfg.command.sdot_max
    蹲低時的速度上限由 lqr.clip_to_envelope 再處理。
    """
    raise NotImplementedError("TODO：action_to_command")


class DodgeEnv:
    def __init__(self, cfg: Config, reward_variant: str = "task", filter_mode: str | None = None,
                 replan: str = "every_step", ball_source: str = "truth", augment_depth: bool = True,
                 seed: int = 0):
        """建立一個環境。

        reward_variant："task" / "task_collision" / "task_cbf"
        filter_mode：None（不開過濾器）/ "naive" / "lag_aware"
        replan："every_step" / "commit"（只在 filter_mode 不是 None 時有用）
        ball_source：過濾器用的球資訊來源 "truth" / "noisy_truth" / "depth_tracker"
        augment_depth：訓練時 True（policy 影像加 PAC-MAN 的增強）、評估時 False

        建議步驟：
          1. path = scene.build_scene(cfg=cfg)；self.env = TitaEnv(str(path))；self.ids = scene_ids(m, cfg)
          2. self.lqr = LQRBalancer(...)；design_all()；self.ref = ReferenceGenerator(cfg)
          3. self.launcher = BallLauncher(m, d, cfg, ids)；self.camera = DepthCamera(m, cfg, ids)
             self.stack = DepthFrameStack(policy_res 的 (高, 寬), cfg.camera.frame_offsets)
          4. self.ball_source = make_ball_source(ball_source, m, cfg, launcher, camera)
             reward 的 CBF 項另外用 TruthBallSource（privileged）
          5. 若 filter_mode：self.filter = SafetyFilter(cfg, filter_mode, replan)
          6. sensors.Imu / Encoders、state_estimator.StateEstimator（cfg.estimator.use_true_state=False 時才用）
          7. self.rng = np.random.default_rng(seed)
        """
        raise NotImplementedError("TODO：DodgeEnv.__init__")

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """重置一個 episode，回傳觀測 {"actor": ..., "critic": ...}。

        seed 不是 None 時用它重建 self.rng（評估時每個 episode 用固定 seed，才能重現）。
        要重置：機器人姿勢（站姿）、ReferenceGenerator、launcher.reset(rng)、相機疊影像 (stack.reset())、
        球資訊來源、過濾器、感測器偏差、上一步動作 (= 0)、終點位置（起點正前方 cfg.task.goal_distance）。
        課程學習：訓練環境 (augment_depth=True) 累計的 policy 步數 < cfg.obs.single_ball_env_steps 時，
        這回合只丟 1 顆（launcher.reset 之前把 cfg.balls.balls_per_episode 設成 (1, 1)，之後還原）。
        """
        raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """往前一個 policy 步。回傳 (obs, reward, terminated, truncated, info)。

        info 至少要有：
            "u_policy"、"u_safe"、"intervention"、"filter_active"、"new_hits"（這一步新被打中幾顆）
        episode 結束時另外放 "episode": self.episode_summary()。
        """
        raise NotImplementedError

    def episode_summary(self) -> dict:
        """這個 episode 的結果，格式見 evaluate.py 的 EPISODE_SCHEMA（"balls" 欄位用 launcher.episode_records()）。"""
        raise NotImplementedError

    # ---------------- 觀測（設計見檔頭「觀測」；順序照 obs_layout） ----------------
    def actor_obs(self) -> np.ndarray:
        """actor 觀測 (589,) float32。

        建議寫法：
            lay = obs_layout(self.cfg)["actor"]
            obs = np.zeros(lay["goal"].stop, np.float32)
            obs[lay["depth"]] = self.stack.get()          # 這一步的深度圖要在呼叫前就 push 進去
            obs[lay["gyro"]] = ...
            ...
        深度圖每個 policy 步只算一次（render 一次，給 policy 影像和 depth_tracker 球資訊來源共用：
        ball_source.update(d, rng, frame=frame)）。
        """
        raise NotImplementedError("TODO：actor_obs")

    def critic_obs(self) -> np.ndarray:
        """critic 觀測 (632,) float32 = actor_obs() 接上真實底座速度 (3) 和真實球狀態 (40)。"""
        raise NotImplementedError("TODO：critic_obs")

    @property
    def actor_obs_dim(self) -> int:
        """= obs_layout(cfg)["actor"] 最後一段的 stop（預設 589）。"""
        return max(s.stop for s in obs_layout(self.cfg)["actor"].values())

    @property
    def critic_obs_dim(self) -> int:
        """= obs_layout(cfg)["critic"] 最後一段的 stop（預設 632）。"""
        return max(s.stop for s in obs_layout(self.cfg)["critic"].values())
