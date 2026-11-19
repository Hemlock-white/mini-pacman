"""Mini PAC-MAN 的所有參數，集中在這一個檔案。

使用方式
--------
    from minipacman.config import default_config
    cfg = default_config()
    cfg.balls.radius_range          # 球的半徑範圍 (m)

規則
----
* 所有長度單位是公尺 (m)，時間是秒 (s)，角度是弧度 (rad)，除非變數名稱寫了 ``_deg``。
* 每個參數旁邊註明「來源」：
    - ``[PAC-MAN]``：照抄原版 repo（github.com/lzyang2000/perceptive_cbf_rl, MIT License）的設定，
      檔名寫在後面，方便報告引用。
    - ``[原型]``：我們自己的 TITA 可行性測試量到的值（Final_Project/feasibility_check_tita）。
    - ``[自訂]``：我們自己決定的值，可以調。
* 要改參數時，改這裡，不要在其他檔案寫死數字。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
# TITA 模型位置：預設是 scripts/download_tita.py 下載的位置；也可以用環境變數 MINIPACMAN_TITA_DIR 指定。
TITA_DIR = Path(os.environ.get("MINIPACMAN_TITA_DIR",
                               REPO_ROOT / "third_party" / "tita" / "urdfs" / "tita_description"))


@dataclass
class TitaSource:
    """TITA 模型的下載來源（沒有授權，所以不放進 repo，用腳本下載固定版本）。"""
    repo_url: str = "https://github.com/DDTRobot/ddt_ros2_control.git"
    commit: str = "beb05df2d585bb100b4aeaca85bcba077d3a3ba0"    # [原型] 我們測試用的版本
    subdir: str = "urdfs/tita_description"
    target_dir: Path = REPO_ROOT / "third_party" / "tita"


@dataclass
class SimConfig:
    """模擬器設定與各層的更新頻率。"""
    timestep: float = 0.002            # [原型] 物理步長 2 ms = 500 Hz
    integrator: str = "implicitfast"   # [原型]
    default_contact: bool = True       # [原型] 用 MuJoCo 預設接觸 (pyramidal, impratio=1)；robot.xml 的設定會讓 LQR 漂移
    lqr_every: int = 1                 # LQR 每幾個物理步算一次（1 = 500 Hz）
    policy_every: int = 10             # policy、CBF、深度相機每幾個物理步一次（10 = 50 Hz）[PAC-MAN] 控制頻率 50 Hz
    episode_seconds: float = 20.0      # [PAC-MAN] amp_env_cfg.py：episode 20 s

    @property
    def policy_dt(self) -> float:
        return self.timestep * self.policy_every


@dataclass
class RobotConfig:
    """TITA 的名稱、尺寸、姿勢。"""
    base_body: str = "base_link"
    legs: tuple = ("joint_left_leg_1", "joint_left_leg_2", "joint_left_leg_3",
                   "joint_right_leg_1", "joint_right_leg_2", "joint_right_leg_3")
    wheels: tuple = ("joint_left_leg_4", "joint_right_leg_4")
    wheel_motors: tuple = ("left_leg_4", "right_leg_4")
    leg_kp: float = 600.0              # [原型] 腿部位置伺服
    leg_kv: float = 15.0               # [原型]
    leg_force_limit: float = 80.0      # [原型] robot.xml 的腿部扭矩上限 (Nm)
    wheel_torque_limit: float = 20.0   # [原型] 輪子扭矩上限 (Nm)
    wheel_radius: float = 0.0925       # [原型] robot.xml 輪子圓柱半徑
    z_offset: float = -5e-4            # [原型] 重置時讓輪子微微壓進地面，線性化才看得到接觸
    stand_hip: float = 0.5             # [原型] 站姿 hip 角度；knee 由 balanced_k() 算
    crouch_hip: float = 1.0            # [原型] 最低蹲姿；hip >= 1.1 會傾斜 30–40 度（未解決）
    # 五個碰撞形狀（機器人被球打到 = 球碰到其中任何一個）
    collision_geoms: tuple = ("base_link_collision_0", "left_leg_3_collision_0", "right_leg_3_collision_0",
                              "left_leg_4_collision_0", "right_leg_4_collision_0")
    # 感測器（robot.xml 本來就有）
    imu_site: str = "trunk_imu"
    camera_bodies: tuple = ("left_camera", "right_camara")   # TITA 真實雙目相機的位置（原檔拼字就是 camara）


@dataclass
class LQRConfig:
    """LQR 平衡控制器的權重（[原型] tita_core.py，已在 AUP 驗證）。"""
    q_default: float = 1e-3
    q_pos_x: float = 20.0
    q_pitch: float = 200.0
    q_yaw: float = 20.0
    q_vel_x: float = 2.0
    q_pitch_rate: float = 2.0
    q_yaw_rate: float = 1.0
    r_wheel: float = 0.5
    discount: float = 0.9999           # 側向位置不可控，加一點折扣 DARE 才有解
    fd_eps: float = 1e-6               # mjd_transitionFD 的差分步長
    zero_abad_gains: bool = True       # 側擺關節 (abad) 增益歸零，原本 ±30000 是硬接觸造成的假象
    zero_frictionloss_when_linearizing: bool = True


@dataclass
class CommandConfig:
    """高層指令 u = (前進速度 v, 轉彎速度 w, 蹲下速度 sdot) 的範圍。

    s 是蹲低程度：0 = 站姿、1 = 最低蹲姿。

    安全範圍（envelope）[原型，筆電 MuJoCo 4.0.1 量的，AUP 上請再驗證]：
    蹲得越低，能安全行駛的速度越低。實測：s=1 時 v=0.5 會前後傾到 50 度、v=0.8 加轉彎會倒；
    s=0.5 時 v=0.5、w=2 最大傾角約 19 度。所以速度上限隨 s 線性遞減。
    """
    v_min: float = -0.5                # 可以倒退
    v_max_stand: float = 0.8
    v_max_crouch: float = 0.3
    w_max_stand: float = 2.0           # rad/s
    w_max_crouch: float = 1.0
    sdot_max: float = 1.0 / 0.3        # 0.3 秒從站姿蹲到底 [原型 duck 0.3 s]
    accel_limit: float = 1.5           # m/s^2，速度指令的變化率上限（太猛 LQR 會倒）
    yaw_accel_limit: float = 6.0       # rad/s^2
    speed_lag_tau: float = 0.16        # [原型] 速度到 90% 要 0.36 s，換算一階延遲時間常數約 0.16 s


@dataclass
class TaskConfig:
    """任務：開到前方終點。"""
    goal_distance: float = 6.0         # [自訂] 終點在起點正前方幾公尺
    goal_radius: float = 0.3           # 進入這個半徑就算到達
    fall_pitch: float = 1.0            # |pitch| > 1 rad 視為倒下（episode 結束）


@dataclass
class BallConfig:
    """球與發球器。

    一個 episode 丟 1–5 顆，從機器人「正面」（目前朝向 ±25 度）丟過來。
    有兩種軌跡：直接飛向機器人 (direct)，或先在機器人前方落地彈起再飛過來 (bounce)。
    """
    pool_size: int = 5                         # 場景裡預先放幾顆球（= 每回合最多幾顆）
    balls_per_episode: tuple = (1, 5)          # [自訂] 每回合隨機丟幾顆（含兩端）
    first_throw_range: tuple = (1.0, 3.0)      # 第一顆球的發射時間 (s)
    throw_interval_range: tuple = (1.0, 3.5)   # [PAC-MAN] 1.0–4.0 s；縮到 3.5 讓 5 顆都能在 20 s 內丟完
    radius_range: tuple = (0.075, 0.125)       # [PAC-MAN] dodge_env_cfgs.py：dr.geom_size (0.075, 0.125)
    mass_range: tuple = (0.07, 0.20)           # [PAC-MAN 固定 0.136 kg，ball.py] 依我們的要求在它附近隨機
    restitution_range: tuple = (0.5, 0.8)      # [自訂] PAC-MAN 沒有彈跳；0.5–0.8 是常見充氣球的範圍
    hollow: bool = True                        # 慣性用薄殼球 (2/3 m r^2)，像充氣球
    friction: tuple = (0.6, 0.005, 0.0001)     # [自訂] 滑動、扭轉、滾動摩擦
    # 球的接觸用「直接指定剛度、阻尼」的 solref = (-stiffness, -damping)。
    # 實測 (dt=2 ms)：stiffness=50000 時接觸約 7 步、6 m/s 撞擊最多壓進 2.7 cm，彈性係數 0–0.99 都穩定。
    # 預設的 (時間常數, 阻尼比) 寫法在阻尼比小時會變得極硬、數值不穩，量到 e > 1，所以不用。
    contact_stiffness: float = 50000.0
    calibration_impact_speed: float = 4.0      # 建彈性對照表時的撞擊速度 (m/s)
    launch_distance_range: tuple = (3.0, 5.0)  # [PAC-MAN 2–3 m，人形機器人] TITA 比較慢，拉遠一點
    launch_angle_deg: float = 25.0             # [PAC-MAN] 發射點在機器人朝向 ±25 度內
    horizontal_speed_range: tuple = (3.0, 6.0) # [PAC-MAN 約 3.2–5.2 m/s，由距離/飛行時間推得]
    arrival_height_range: tuple = (0.05, 0.65) # [自訂] 球心抵達機器人時的高度；TITA 站姿頂部約 0.54 m
    direct_launch_height_range: tuple = (0.3, 1.5)   # direct 軌跡的發射高度
    bounce_fraction: float = 0.5               # [自訂] 先落地彈跳的比例
    bounce_point_range: tuple = (0.3, 0.8)     # 彈跳點在「發射點→目標」之間的位置比例
    bounce_launch_height_max: float = 2.5      # bounce 軌跡解出的發射高度若超過就重抽
    aim_noise: float = 0.1                     # [PAC-MAN] aim_noise_scale = 0.1 m
    lead_target: bool = True                   # [PAC-MAN] 依機器人速度預判位置
    retire_behind: float = 1.0                 # 球跑到機器人後方超過這距離就收回
    retire_lifetime: float = 4.0               # 發射後超過這時間就收回
    park_xy: tuple = (-100.0, 100.0)           # 收起來的球放這裡（遠離相機視野）


@dataclass
class DepthCameraConfig:
    """深度相機。用 MuJoCo 光線投射 (mj_multiRay) 在 CPU 上算，不需要 OpenGL。

    實測（筆電）：16x9 直接投射時，2 m 外半徑 10 cm 的球一個像素都打不到（光線間距比球大），
    所以先用 64x36 投射，再用「取最小值」縮成 16x9 給 policy。64x36 一張約 0.3 ms。
    """
    hi_res: tuple = (64, 36)           # (寬, 高) 實際投射的解析度；追蹤器用這個
    policy_res: tuple = (16, 9)        # [PAC-MAN] DEPTH_WIDTH=16, DEPTH_HEIGHT=9
    fovy_deg: float = 54.0             # [PAC-MAN] 垂直視角 54 度（16:9 時水平約 84.5 度）
    near: float = 0.1                  # [PAC-MAN] DEPTH_NEAR
    far: float = 5.0                   # [PAC-MAN] DEPTH_FAR
    tilt_up_deg: float = 5.0           # [自訂] PAC-MAN 人形機器人頭部上仰 20 度；TITA 相機較低，球多在水平附近
    geomgroups: tuple = (1, 1, 1, 0, 0, 0)   # 只看可見的幾何 (group 0–2)，不看碰撞用的 group 3
    ball_only: bool = True             # [PAC-MAN] BallOnlyDepthObs：球以外的像素設成 far（理想化的分割）
    update_every: int = 1              # 每幾個 policy 步更新一次（1 = 50 Hz）[PAC-MAN] DEPTH_UPDATE_PERIOD=1
    frame_offsets: tuple = (0, 3, 8, 18)   # [PAC-MAN] train_dodge_single_ballonly.sh：疊 4 張
    # 感測雜訊（給追蹤器用的高解析度深度）：sigma = a + b * d^2，類似雙目/結構光相機
    noise_a: float = 0.002
    noise_b: float = 0.004
    pixel_dropout: float = 0.02        # 每個像素隨機失效（回傳 far）的機率
    # policy 影像的訓練增強 [PAC-MAN] _add_depth_obs_single_ballonly
    aug_ball_noise_sigma: float = 0.03
    aug_ball_pixel_dropout: float = 0.10
    aug_ball_full_dropout: float = 0.02
    aug_ball_depth_jitter: float = 0.15


@dataclass
class TrackerConfig:
    """從深度圖找球、追蹤球 (ball_tracker.py)。"""
    mode: str = "segmented"            # "segmented"：用分割找球像素（理想化，同 PAC-MAN）；"geometric"：只用深度
    floor_threshold: float = 0.06      # geometric 模式：高於地面這麼多才算候選點
    min_pixels: int = 1
    gate: float = 0.5                  # 資料關聯的距離門檻 (m)
    accel_noise: float = 3.0           # 卡爾曼濾波的過程雜訊 (m/s^2)
    confirm_hits: int = 3              # 連續看到幾次才輸出這顆球
    max_misses: int = 10               # 連續幾次沒看到就刪除（50 Hz 時 = 0.2 s）
    init_speed_std: float = 8.0        # 新軌跡速度的初始不確定度 (m/s)


@dataclass
class NoisyTruthConfig:
    """「模擬器真實值 + 雜訊 + 延遲」的球資訊來源（實驗 3 的中間組）。"""
    pos_sigma: float = 0.03
    vel_sigma: float = 0.2
    latency: float = 0.04              # 秒；會四捨五入到 policy 步
    dropout: float = 0.05              # 每一步隨機看不到某顆球的機率


@dataclass
class ImuConfig:
    """IMU（robot.xml 的 trunk_gyro / trunk_accel / trunk_quat）。數值是一般 MEMS IMU 的量級，可依實際規格調。"""
    gyro_sensor: str = "trunk_gyro"
    accel_sensor: str = "trunk_accel"
    quat_sensor: str = "trunk_quat"
    gyro_noise: float = 0.01           # rad/s
    gyro_bias_std: float = 0.005       # rad/s（每個 episode 抽一次）
    accel_noise: float = 0.05          # m/s^2
    accel_bias_std: float = 0.05
    quat_noise_deg: float = 0.5        # 姿態輸出的雜訊（當作 IMU 內建融合的結果）


@dataclass
class EncoderConfig:
    """關節 encoder（robot.xml 的 *_pos / *_vel）。"""
    counts_per_rev: int = 4096         # 12-bit
    vel_noise: float = 0.02            # rad/s


@dataclass
class EstimatorConfig:
    """state_estimator.py：用 IMU + encoder 估計 LQR 需要的狀態。"""
    complementary_alpha: float = 0.98
    use_true_state: bool = True        # True = LQR 直接用模擬器真實狀態（估測器寫好前先用這個）


@dataclass
class CBFConfig:
    """CBF：安全函數、reward 項、過濾器。"""
    # reward 用的 link-CBF [PAC-MAN] rewards.py dodge_link_cbf_reward；dodge_env_cfgs.py 的設定值
    alpha: float = 1.0
    margin: float = 0.05
    constraint_clip: float = 2.0
    reward_weight: float = 0.27
    threat_min_height_above_radius: float = 0.05   # 球離地才算威脅
    threat_min_speed: float = 0.5                   # 球水平速度 > 0.5 m/s 才算威脅
    # 執行時過濾器（我們自己的，PAC-MAN 部署版沒有用過濾器）
    filter_alpha: float = 4.0
    filter_margin: float = 0.05
    horizon: float = 1.5               # 預測多久以內的碰撞 (s)
    lag_tau_naive: float = 0.0         # 簡單版：假設速度瞬間改變
    lag_tau_aware: float = 0.16        # 考慮延遲版：用 CommandConfig.speed_lag_tau


@dataclass
class RewardConfig:
    """三種 reward（實驗 1）。variant: "task" / "task_collision" / "task_cbf"。"""
    progress_weight: float = 1.0       # 每步往終點前進的距離 (m) 乘這個
    goal_bonus: float = 10.0           # 第一次到達終點
    at_goal_weight: float = 0.1        # 停在終點時每步加分
    collision_penalty: float = 5.0     # task_collision：每被一顆球打到扣分
    fall_penalty: float = 200.0        # [PAC-MAN] is_terminated -200
    action_rate_weight: float = 0.05   # [PAC-MAN] dodge_action_rate -0.05
    # task_cbf 的 CBF 項權重在 CBFConfig.reward_weight


@dataclass
class ObsConfig:
    """PPO 的觀測（2026-11-19 確認：照 PAC-MAN 的非對稱 actor-critic）。

    actor（部署時真的在跑的網路）：深度圖疊 4 張 (576) + 本體感測 (11) + 終點 (2) = 589 維
    critic（只在訓練時幫忙評分）：actor 的全部 + 真實底座速度 (3) + 真實球狀態 (5 顆 × 8) = 632 維
    每一段在向量裡的位置由 dodge_env.obs_layout(cfg) 算出來。
    """
    actor_ball_input: str = "depth"    # "depth"：選項 A（預設，PAC-MAN 做法）
                                       # "tracker"：備案 B，actor 改看追蹤器估出的球（5 顆 × 8 = 40 維）
    # 本體感測：(名稱, 維度)。順序就是向量裡的順序
    proprio_terms: tuple = (("gyro", 3),           # 車身角速度 (rad/s)，IMU 座標
                            ("gravity", 3),        # 重力方向在車身座標的單位向量（由 IMU 姿態算）
                            ("forward_speed", 1),  # 前進速度 (m/s)
                            ("crouch", 1),         # 蹲低程度 s（ReferenceGenerator 的值）
                            ("last_action", 3))    # 上一步的 policy 輸出
    goal_dims: int = 2                 # 終點在機器人車頭座標的 (前, 左) 距離 (m)
    ball_slots: int = 5                # 最多看幾顆球（= 球池大小），不足補 0
    ball_slot_features: int = 8        # 每顆：相對位置 3、相對速度 3（車頭座標）、半徑 1、是否存在 1
    clip_rel_pos: float = 10.0         # 相對位置夾在 ±10 m
    privileged_base_vel: int = 3       # critic 多看的真實底座線速度（車頭座標）
    # 課程學習：每個訓練環境的前 N 個 policy 步，每回合只丟 1 顆球（256 個環境 × 24 步 / iteration
    # → 24000 步約等於前 1000 個 iteration）。評估時一律 1–5 顆。
    single_ball_env_steps: int = 24_000


@dataclass
class PPOConfig:
    """rsl_rl 5.4.0 的 PPO 參數（[PAC-MAN] rl_cfg.py，除非另外註明）。"""
    rsl_rl_version: str = "5.4.0"      # [PAC-MAN] pyproject.toml 註解：mjlab 1.5.3 pin rsl-rl-lib==5.4.0
    num_envs: int = 256                # [自訂] PAC-MAN 用 8192 個 GPU 環境；我們的模擬在 CPU，用 256
    num_workers: int = 0               # 0 = 自動（CPU 核心數 - 2）
    num_steps_per_env: int = 24
    max_iterations: int = 5000         # [自訂] 256 × 24 × 5000 ≈ 3,000 萬步，先跑看看
    save_interval: int = 100
    hidden_dims: tuple = (512, 256, 128)
    activation: str = "elu"
    obs_normalization: bool = True
    init_std: float = 1.0
    learning_rate: float = 1e-3
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    value_loss_coef: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    max_grad_norm: float = 1.0


@dataclass
class ExperimentConfig:
    """實驗設定。"""
    n_eval_seeds: int = 100
    eval_seed_offset: int = 10_000     # 評估用 seed 10000–10099，不和訓練重疊
    quick_seeds: int = 5               # --quick 時用幾個 seed
    reward_variants: tuple = ("task", "task_collision", "task_cbf")
    # 實驗 1：每個 policy 在過濾器關 / 開下各測一次；過濾器用哪個版本、球資訊從哪來
    exp1_filter_mode: str = "lag_aware"
    exp1_filter_source: str = "truth"          # 用真實值 = 過濾器的上限；感測不準的影響留給實驗 3
    # 實驗 2：簡單版 vs 考慮延遲版 × 每步重新規劃 vs 只決定一次；上層用寫死的指令（不需要 PPO）
    exp2_policy: str = "scripted"
    exp2_filters: tuple = ("naive", "lag_aware")
    exp2_replan: tuple = ("every_step", "commit")
    # 實驗 3：球資訊來源
    exp3_sources: tuple = ("truth", "noisy_truth", "depth_tracker")
    exp3_reward_variant: str = "task_cbf"      # 實驗 3 用哪個 policy（對應 run 腳本裡的 POLICY_FILES）
    exp3_filter_mode: str = "lag_aware"
    height_buckets: tuple = (("low", 0.0, 0.2), ("mid", 0.2, 0.4), ("high", 0.4, 10.0))   # 依球抵達高度分組


@dataclass
class Config:
    tita: TitaSource = field(default_factory=TitaSource)
    sim: SimConfig = field(default_factory=SimConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    lqr: LQRConfig = field(default_factory=LQRConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    balls: BallConfig = field(default_factory=BallConfig)
    camera: DepthCameraConfig = field(default_factory=DepthCameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    noisy_truth: NoisyTruthConfig = field(default_factory=NoisyTruthConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    cbf: CBFConfig = field(default_factory=CBFConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    exp: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> dict:
        """轉成可以存成 JSON 的 dict（Path 轉字串），每次實驗都存一份，方便重現。"""
        def clean(x):
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [clean(v) for v in x]
            if isinstance(x, Path):
                return str(x)
            return x
        return clean(asdict(self))


def default_config() -> Config:
    return Config()
