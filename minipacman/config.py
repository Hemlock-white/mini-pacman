"""All Mini PAC-MAN parameters, in one file.

Usage
-----
    from minipacman.config import default_config
    cfg = default_config()
    cfg.balls.radius_range          # ball radius range (m)

Rules
-----
* Lengths are in metres (m), time in seconds (s), angles in radians (rad), unless the name ends in ``_deg``.
* Every parameter says where it comes from:
    - ``[PAC-MAN]``: copied from the original repo (github.com/lzyang2000/perceptive_cbf_rl, MIT License);
      the file name is given so the report can cite it.
    - ``[prototype]``: measured in our own TITA feasibility test (Final_Project/feasibility_check_tita).
    - ``[ours]``: our own choice, feel free to tune.
* To change a number, change it here. Don't hard-code numbers in other files.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
# Where the TITA model lives: by default where scripts/download_tita.py puts it;
# can be overridden with the MINIPACMAN_TITA_DIR environment variable.
TITA_DIR = Path(os.environ.get("MINIPACMAN_TITA_DIR",
                               REPO_ROOT / "third_party" / "tita" / "urdfs" / "tita_description"))


@dataclass
class TitaSource:
    """Where to download the TITA model (it has no license, so it is not in the repo; a script downloads a pinned version)."""
    repo_url: str = "https://github.com/DDTRobot/ddt_ros2_control.git"
    commit: str = "beb05df2d585bb100b4aeaca85bcba077d3a3ba0"    # [prototype] the version we tested
    subdir: str = "urdfs/tita_description"
    target_dir: Path = REPO_ROOT / "third_party" / "tita"


@dataclass
class SimConfig:
    """Simulator settings and the update rate of each layer."""
    timestep: float = 0.002            # [prototype] physics step 2 ms = 500 Hz
    integrator: str = "implicitfast"   # [prototype]
    default_contact: bool = True       # [prototype] MuJoCo default contact (pyramidal, impratio=1); robot.xml's settings make the LQR drift
    lqr_every: int = 1                 # run the LQR every N physics steps (1 = 500 Hz)
    policy_every: int = 10             # policy, CBF and depth camera every N physics steps (10 = 50 Hz) [PAC-MAN] 50 Hz control
    episode_seconds: float = 20.0      # [PAC-MAN] amp_env_cfg.py: 20 s episodes

    @property
    def policy_dt(self) -> float:
        return self.timestep * self.policy_every


@dataclass
class RobotConfig:
    """TITA names, sizes and poses."""
    base_body: str = "base_link"
    legs: tuple = ("joint_left_leg_1", "joint_left_leg_2", "joint_left_leg_3",
                   "joint_right_leg_1", "joint_right_leg_2", "joint_right_leg_3")
    wheels: tuple = ("joint_left_leg_4", "joint_right_leg_4")
    wheel_motors: tuple = ("left_leg_4", "right_leg_4")
    leg_kp: float = 600.0              # [prototype] leg position servos
    leg_kv: float = 15.0               # [prototype]
    leg_force_limit: float = 80.0      # [prototype] leg torque limit in robot.xml (Nm)
    wheel_torque_limit: float = 20.0   # [prototype] wheel torque limit (Nm)
    wheel_radius: float = 0.0925       # [prototype] wheel cylinder radius in robot.xml
    z_offset: float = -5e-4            # [prototype] on reset, press the wheels slightly into the floor so the linearization sees the contact
    stand_hip: float = 0.5             # [prototype] standing hip angle; the knee comes from balanced_k()
    crouch_hip: float = 1.0            # [prototype] lowest crouch; hip >= 1.1 settles at a 30-40 degree tilt (unsolved)
    # The five collision shapes (the robot is "hit" when a ball touches any of them)
    collision_geoms: tuple = ("base_link_collision_0", "left_leg_3_collision_0", "right_leg_3_collision_0",
                              "left_leg_4_collision_0", "right_leg_4_collision_0")
    # Sensors (already in robot.xml)
    imu_site: str = "trunk_imu"
    camera_bodies: tuple = ("left_camera", "right_camara")   # where TITA's real stereo camera is ("camara" is the original spelling)


@dataclass
class LQRConfig:
    """LQR balance controller weights ([prototype] tita_core.py, verified on AUP)."""
    q_default: float = 1e-3
    q_pos_x: float = 20.0
    q_pitch: float = 200.0
    q_yaw: float = 20.0
    q_vel_x: float = 2.0
    q_pitch_rate: float = 2.0
    q_yaw_rate: float = 1.0
    r_wheel: float = 0.5
    discount: float = 0.9999           # lateral position is uncontrollable; a small discount lets the DARE be solved
    fd_eps: float = 1e-6               # finite-difference step for mjd_transitionFD
    zero_abad_gains: bool = True       # zero the hip-roll (abad) gains; the original ±30000 came from the stiff contact
    zero_frictionloss_when_linearizing: bool = True


@dataclass
class CommandConfig:
    """Limits of the high-level command u = (forward speed v, turn rate w, crouch rate sdot).

    s is the crouch level: 0 = standing, 1 = lowest crouch.

    Safe envelope [prototype, measured on a laptop with MuJoCo 4.0.1; check again on AUP]:
    the lower the robot crouches, the slower it can drive safely. Measured: at s=1, v=0.5 pitches
    to 50 degrees and v=0.8 while turning falls over; at s=0.5, v=0.5 and w=2 peak at about 19 degrees.
    So the speed limits drop linearly with s.
    """
    v_min: float = -0.5                # it can reverse
    v_max_stand: float = 0.8
    v_max_crouch: float = 0.3
    w_max_stand: float = 2.0           # rad/s
    w_max_crouch: float = 1.0
    sdot_max: float = 1.0 / 0.3        # 0.3 s from standing to full crouch [prototype duck 0.3 s]
    accel_limit: float = 1.5           # m/s^2, max change rate of the speed command (too abrupt and the LQR falls)
    yaw_accel_limit: float = 6.0       # rad/s^2
    speed_lag_tau: float = 0.16        # [prototype] 0.36 s to reach 90% speed, roughly a first-order lag with tau = 0.16 s


@dataclass
class TaskConfig:
    """Task: drive to a goal in front of the robot."""
    goal_distance: float = 6.0         # [ours] goal is this many metres straight ahead of the start
    goal_radius: float = 0.3           # inside this radius counts as reached
    fall_pitch: float = 1.0            # |pitch| > 1 rad counts as fallen (episode ends)


@dataclass
class BallConfig:
    """Balls and the launcher.

    Each episode throws 1-5 balls from the robot's "front" (its current heading ±25 degrees).
    Two kinds of trajectory: straight at the robot (direct), or bouncing once in front of the robot first (bounce).
    """
    pool_size: int = 5                         # how many balls are pre-placed in the scene (= max balls per episode)
    balls_per_episode: tuple = (1, 5)          # [ours] random number of balls per episode (inclusive)
    first_throw_range: tuple = (1.0, 3.0)      # launch time of the first ball (s)
    throw_interval_range: tuple = (1.0, 3.5)   # [PAC-MAN] 1.0-4.0 s; cut to 3.5 so 5 balls fit in 20 s
    radius_range: tuple = (0.075, 0.125)       # [PAC-MAN] dodge_env_cfgs.py: dr.geom_size (0.075, 0.125)
    mass_range: tuple = (0.07, 0.20)           # [PAC-MAN fixed at 0.136 kg, ball.py] randomised around it, as we wanted
    restitution_range: tuple = (0.5, 0.8)      # [ours] PAC-MAN has no bounce; 0.5-0.8 is typical for inflated balls
    hollow: bool = True                        # inertia of a thin shell (2/3 m r^2), like an inflated ball
    friction: tuple = (0.6, 0.005, 0.0001)     # [ours] sliding, torsional, rolling friction
    # Ball contacts use solref = (-stiffness, -damping) ("direct" stiffness and damping).
    # Measured (dt=2 ms): with stiffness=50000 a contact lasts about 7 steps and a 6 m/s impact sinks at most 2.7 cm;
    # restitution 0-0.99 is stable. The default (timeconst, dampratio) form gets extremely stiff at small damping
    # ratios, goes numerically unstable and gives e > 1, so we don't use it.
    contact_stiffness: float = 50000.0
    calibration_impact_speed: float = 4.0      # impact speed used to build the restitution table (m/s)
    launch_distance_range: tuple = (3.0, 5.0)  # [PAC-MAN 2-3 m, humanoid] TITA is slower, so a bit further
    launch_angle_deg: float = 25.0             # [PAC-MAN] launch point within ±25 degrees of the robot's heading
    horizontal_speed_range: tuple = (3.0, 6.0) # [PAC-MAN about 3.2-5.2 m/s, from distance / flight time]
    arrival_height_range: tuple = (0.05, 0.65) # [ours] ball-centre height when it reaches the robot; TITA's top is ~0.54 m standing
    direct_launch_height_range: tuple = (0.3, 1.5)   # launch height for direct throws
    bounce_fraction: float = 0.5               # [ours] fraction of throws that bounce first
    bounce_point_range: tuple = (0.3, 0.8)     # where the bounce is, as a fraction of the way from launch to target
    bounce_launch_height_max: float = 2.5      # resample if the solved launch height of a bounce throw is above this
    aim_noise: float = 0.1                     # [PAC-MAN] aim_noise_scale = 0.1 m
    lead_target: bool = True                   # [PAC-MAN] aim where the robot will be, using its velocity
    retire_behind: float = 1.0                 # take the ball back once it is this far behind the robot
    retire_lifetime: float = 4.0               # take the ball back this long after launch
    park_xy: tuple = (-100.0, 100.0)           # where unused balls are parked (far from the camera)


@dataclass
class DepthCameraConfig:
    """Depth camera. Computed on the CPU with MuJoCo ray casting (mj_multiRay); no OpenGL needed.

    Measured (laptop): casting 16x9 rays directly misses a 10 cm ball 2 m away entirely (the gap between rays
    is bigger than the ball), so we cast 64x36 and shrink to 16x9 for the policy by taking the minimum.
    One 64x36 image takes about 0.3 ms.
    """
    hi_res: tuple = (64, 36)           # (width, height) actually cast; the tracker uses this
    policy_res: tuple = (16, 9)        # [PAC-MAN] DEPTH_WIDTH=16, DEPTH_HEIGHT=9
    fovy_deg: float = 54.0             # [PAC-MAN] vertical field of view 54 degrees (about 84.5 horizontal at 16:9)
    near: float = 0.1                  # [PAC-MAN] DEPTH_NEAR
    far: float = 5.0                   # [PAC-MAN] DEPTH_FAR
    tilt_up_deg: float = 5.0           # [ours] PAC-MAN tilts the humanoid's head camera up 20 degrees; TITA's camera is lower and balls are mostly near level
    geomgroups: tuple = (1, 1, 1, 0, 0, 0)   # only see visible geoms (groups 0-2), not the collision group 3
    ball_only: bool = True             # [PAC-MAN] BallOnlyDepthObs: every non-ball pixel is set to far (idealised segmentation)
    update_every: int = 1              # update every N policy steps (1 = 50 Hz) [PAC-MAN] DEPTH_UPDATE_PERIOD=1
    frame_offsets: tuple = (0, 3, 8, 18)   # [PAC-MAN] train_dodge_single_ballonly.sh: stack 4 frames
    # Sensor noise (on the high-res depth used by the tracker): sigma = a + b * d^2, like a stereo / structured-light camera
    noise_a: float = 0.002
    noise_b: float = 0.004
    pixel_dropout: float = 0.02        # chance that a pixel randomly fails (returns far)
    # Training augmentation of the policy image [PAC-MAN] _add_depth_obs_single_ballonly
    aug_ball_noise_sigma: float = 0.03
    aug_ball_pixel_dropout: float = 0.10
    aug_ball_full_dropout: float = 0.02
    aug_ball_depth_jitter: float = 0.15


@dataclass
class TrackerConfig:
    """Finding and tracking balls in the depth image (ball_tracker.py)."""
    mode: str = "segmented"            # "segmented": find ball pixels by segmentation (idealised, same as PAC-MAN); "geometric": depth only
    floor_threshold: float = 0.06      # geometric mode: a point counts only if it is this high above the floor
    min_pixels: int = 1
    gate: float = 0.5                  # distance gate for data association (m)
    accel_noise: float = 3.0           # Kalman filter process noise (m/s^2)
    confirm_hits: int = 3              # a ball is reported only after being seen this many times
    max_misses: int = 10               # a track is deleted after this many misses in a row (0.2 s at 50 Hz)
    init_speed_std: float = 8.0        # initial speed uncertainty of a new track (m/s)


@dataclass
class NoisyTruthConfig:
    """The "simulator truth + noise + latency" ball information source (middle group in Exp 3)."""
    pos_sigma: float = 0.03
    vel_sigma: float = 0.2
    latency: float = 0.04              # seconds; rounded to policy steps
    dropout: float = 0.05              # chance per step of not seeing a ball


@dataclass
class ImuConfig:
    """IMU (trunk_gyro / trunk_accel / trunk_quat in robot.xml). Typical MEMS IMU magnitudes; adjust to a real spec if you like."""
    gyro_sensor: str = "trunk_gyro"
    accel_sensor: str = "trunk_accel"
    quat_sensor: str = "trunk_quat"
    gyro_noise: float = 0.01           # rad/s
    gyro_bias_std: float = 0.005       # rad/s (drawn once per episode)
    accel_noise: float = 0.05          # m/s^2
    accel_bias_std: float = 0.05
    quat_noise_deg: float = 0.5        # noise on the orientation output (treated as the IMU's built-in fusion)


@dataclass
class EncoderConfig:
    """Joint encoders (the *_pos / *_vel sensors in robot.xml)."""
    counts_per_rev: int = 4096         # 12-bit
    vel_noise: float = 0.02            # rad/s


@dataclass
class EstimatorConfig:
    """state_estimator.py: estimate the state the LQR needs from IMU + encoders."""
    complementary_alpha: float = 0.98
    use_true_state: bool = True        # True = the LQR uses the simulator's true state (use this until the estimator works)


@dataclass
class CBFConfig:
    """CBF: barrier function, reward term, safety filter."""
    # Link-CBF for the reward [PAC-MAN] rewards.py dodge_link_cbf_reward; values from dodge_env_cfgs.py
    alpha: float = 1.0
    margin: float = 0.05
    constraint_clip: float = 2.0
    reward_weight: float = 0.27
    threat_min_height_above_radius: float = 0.05   # a ball is a threat only when it is off the floor
    threat_min_speed: float = 0.5                   # and its horizontal speed is > 0.5 m/s
    # Runtime filter (ours; PAC-MAN's deployed config has no filter)
    filter_alpha: float = 4.0
    filter_margin: float = 0.05
    horizon: float = 1.5               # how far ahead to predict collisions (s)
    lag_tau_naive: float = 0.0         # naive: assume the speed changes instantly
    lag_tau_aware: float = 0.16        # lag-aware: use CommandConfig.speed_lag_tau


@dataclass
class RewardConfig:
    """The three rewards (Exp 1). variant: "task" / "task_collision" / "task_cbf"."""
    progress_weight: float = 1.0       # distance moved towards the goal each step (m) times this
    goal_bonus: float = 10.0           # first time the goal is reached
    at_goal_weight: float = 0.1        # per step while staying at the goal
    collision_penalty: float = 5.0     # task_collision: penalty per ball that hits
    fall_penalty: float = 200.0        # [PAC-MAN] is_terminated -200
    action_rate_weight: float = 0.05   # [PAC-MAN] dodge_action_rate -0.05
    # The weight of the CBF term in task_cbf is CBFConfig.reward_weight


@dataclass
class ObsConfig:
    """PPO observations (agreed 2026-11-19: PAC-MAN's asymmetric actor-critic).

    actor (the network that actually runs at deployment): stacked depth images (576) + proprioception (11) + goal (2) = 589
    critic (only helps with scoring during training): everything the actor sees + true base velocity (3)
                                                     + true ball states (5 balls × 8) = 632
    Where each part sits in the vector: dodge_env.obs_layout(cfg).
    """
    actor_ball_input: str = "depth"    # "depth": option A (default, the PAC-MAN way)
                                       # "tracker": fallback B, the actor sees the tracker's ball estimates instead (5 balls × 8 = 40)
    # Proprioception: (name, size). This order is the order in the vector
    proprio_terms: tuple = (("gyro", 3),           # body angular velocity (rad/s), IMU frame
                            ("gravity", 3),        # unit gravity direction in the body frame (from the IMU orientation)
                            ("forward_speed", 1),  # forward speed (m/s)
                            ("crouch", 1),         # crouch level s (from ReferenceGenerator)
                            ("last_action", 3))    # previous policy output
    goal_dims: int = 2                 # goal position in the robot's heading frame (forward, left) in m
    ball_slots: int = 5                # see at most this many balls (= pool size), pad with zeros
    ball_slot_features: int = 8        # per ball: relative position 3, relative velocity 3 (heading frame), radius 1, present 1
    clip_rel_pos: float = 10.0         # clip relative positions to ±10 m
    privileged_base_vel: int = 3       # true base linear velocity (heading frame), critic only
    # Curriculum: for the first N policy steps of each training env, throw only 1 ball per episode
    # (256 envs × 24 steps per iteration → 24000 steps is about the first 1000 iterations). Evaluation always uses 1-5.
    single_ball_env_steps: int = 24_000


@dataclass
class PPOConfig:
    """rsl_rl 5.4.0 PPO parameters ([PAC-MAN] rl_cfg.py unless noted)."""
    rsl_rl_version: str = "5.4.0"      # [PAC-MAN] pyproject.toml comment: mjlab 1.5.3 pins rsl-rl-lib==5.4.0
    num_envs: int = 256                # [ours] PAC-MAN uses 8192 GPU envs; our sim is on the CPU, so 256
    num_workers: int = 0               # 0 = automatic (CPU cores - 2)
    num_steps_per_env: int = 24
    max_iterations: int = 5000         # [ours] 256 × 24 × 5000 ≈ 30 million steps, try this first
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
    """Experiment settings."""
    n_eval_seeds: int = 100
    eval_seed_offset: int = 10_000     # evaluation seeds 10000-10099, never used in training
    quick_seeds: int = 5               # seeds used with --quick
    reward_variants: tuple = ("task", "task_collision", "task_cbf")
    # Exp 1: each policy with the filter off / on; which filter version, and where its ball info comes from
    exp1_filter_mode: str = "lag_aware"
    exp1_filter_source: str = "truth"          # truth = best case for the filter; the effect of bad sensing is Exp 3
    # Exp 2: naive vs lag-aware filter (2 groups); scripted high-level command (no PPO needed), replanning every step.
    # (SafetyFilter also supports replan="commit"; add it here if you ever want the extra comparison.)
    exp2_policy: str = "scripted"
    exp2_filters: tuple = ("naive", "lag_aware")
    exp2_replan: tuple = ("every_step",)
    # Exp 3: ball information source
    exp3_sources: tuple = ("truth", "noisy_truth", "depth_tracker")
    exp3_reward_variant: str = "task_cbf"      # which policy Exp 3 uses (matches POLICY_FILES in the run script)
    exp3_filter_mode: str = "lag_aware"
    height_buckets: tuple = (("low", 0.0, 0.2), ("mid", 0.2, 0.4), ("high", 0.4, 10.0))   # group by ball arrival height


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
        """Convert to a JSON-safe dict (Paths become strings). Saved with every run so it can be reproduced."""
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
