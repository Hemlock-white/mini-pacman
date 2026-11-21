"""RL environment: one TITA, driving to a goal, with balls thrown from the front. [YOURS]

PPO runs many DodgeEnvs at once through vec_env.py. This file wires everything together:

    every policy step (50 Hz, cfg.sim.policy_every physics steps):
      1. observation obs ← depth camera (depth_camera.py) + robot state (true or state_estimator)
      2. the policy outputs action ∈ [-1, 1]^3 → turned into the command u = (v, w, sdot) (see action_to_command)
      3. [optional] SafetyFilter (cbf.py) uses a ball information source (ball_sources.py) to change u into u_safe
      4. launcher.update(t) (launch, take back, update closest distance)
      5. run policy_every physics steps: each step ReferenceGenerator + LQR (lqr.py) → motors → mj_step → launcher.check_contacts()
      6. compute the reward, check whether the episode is over

Parts that are done, just use them
----------------------------------
    scene.build_scene / scene_ids        scene and ids
    balls.BallLauncher                   launching, hit detection, closest distance, a record per ball
    depth_camera.DepthCamera / DepthFrameStack    depth image, small policy image, frame stack
    ball_sources.make_ball_source        ball information sources (truth / noisy_truth / depth_tracker)
Parts you implement: env.py (add set_leg_targets / set_wheel_torques / step etc. to TitaEnv), lqr.py, sensors.py,
state_estimator.py, cbf.py, and this file.

What env.py still needs (suggested)
-----------------------------------
    TitaEnv.set_leg_targets(q6)      → d.ctrl[pact] = q6
    TitaEnv.set_wheel_torques(tau2)  → d.ctrl[wact] = tau2
    TitaEnv.planar_state()           → cbf.RobotPlanarState (true values)
    TitaEnv.pitch()                  already there
    Note: TitaEnv.set_pose() currently does d.qpos[:] = 0, which also moves the balls to the origin (under the robot).
    Change it to d.qpos[:robot_nq] = 0, or call launcher.reset() after env.reset() (it parks the balls).

The three rewards (cfg.reward, cfg.cbf; the variant is set in the constructor)
------------------------------------------------------------------------------
    shared ("task"):
        + progress_weight × (distance to goal last step - distance to goal this step)
        + goal_bonus (first time inside goal_radius)
        + at_goal_weight (every step while inside the goal radius)
        - action_rate_weight × ||a_t - a_{t-1}||²
        - fall_penalty (fell over, episode ends)
    "task_collision": also - collision_penalty × number of balls that newly hit this step (returned by launcher.check_contacts)
    "task_cbf": also + cbf.cbf_reward_term(...) (uses the true ball state, privileged)
Note: getting hit does **not** end the episode (PAC-MAN does end it). If it did, "task" would indirectly learn to dodge
too, and it would no longer be a "doesn't care about safety" baseline.

When an episode ends
--------------------
    terminated: fell over (|pitch| > cfg.task.fall_pitch)
    truncated: time is up (cfg.sim.episode_seconds) → rsl_rl's extras["time_outs"]
Reaching the goal does not end the episode: the robot stays there and balls keep coming (1-5 per episode, scheduled
within 20 s).

Observations (agreed: PAC-MAN's asymmetric actor-critic, parameters in config.ObsConfig)
----------------------------------------------------------------------------------------
actor (the network that runs at deployment; only things a sensor can give), 589 values:
    depth          576   4 stacked 16x9 "ball only" depth images (DepthFrameStack.get()), every step:
                         frame = camera.render(d); stack.push(camera.policy_image(frame, rng, augment_depth))
    gyro             3   body angular velocity (IMU; true value when use_true_state)
    gravity          3   unit gravity direction in the body frame = R_body^T @ (0, 0, -1)
    forward_speed    1   forward speed (state_estimator or true value)
    crouch           1   crouch level s (ReferenceGenerator.s)
    last_action      3   previous policy output (0 after reset)
    goal             2   goal position in the heading frame (forward, left)
critic (training only, may use true values), 632 values = the actor's 589 +
    base_vel         3   true base linear velocity (heading frame)
    balls           40   true ball states, 5 slots × 8 (ball_slot_features, using TruthBallSource)
Fallback B (config.obs.actor_ball_input = "tracker"): the actor's 576 depth values are replaced by the tracker's
40-value ball slots (ball_slot_features on the DepthTrackerBallSource output), everything else the same; the actor
becomes 53 values.
Where each part sits in the vector: obs_layout(cfg) (done); fill actor_obs()/critic_obs() in that order.

Why the critic gets true values: PPO's critic estimates "how much reward will come after this state", and where the
ball is matters a lot. Guessing that from 4 blurry little depth images is hard; training with the true values is much
more stable. At deployment (evaluation) only the actor runs, so this isn't cheating.

Curriculum
----------
During training, for the first cfg.obs.single_ball_env_steps policy steps of each env, throw only 1 ball per episode
(set the launcher's balls_per_episode to (1, 1) on reset), then 1-5. Evaluation always uses 1-5.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .ball_sources import TruthBallSource, make_ball_source
from .balls import BallLauncher
from .cbf import SafetyFilter, cbf_reward_term
from .config import Config
from .depth_camera import DepthCamera, DepthFrameStack
from .env import TitaEnv
from .lqr import LQRBalancer, ReferenceGenerator
from .scene import build_scene, scene_ids
from .sensors import Encoders, Imu
from .state_estimator import StateEstimator

ACTION_DIM = 3     # (v, w, sdot); the policy outputs [-1, 1], action_to_command converts to real units


def obs_layout(cfg: Config) -> dict[str, dict[str, slice]]:
    """Where each part of the actor / critic observation sits. Returns {"actor": {name: slice}, "critic": {name: slice}}.

    E.g. obs_layout(cfg)["actor"]["goal"] is where the 2 goal values sit in the actor vector (ScriptedPolicy can use it).
    The vector length = the stop of the last slice.
    """
    oc, cc = cfg.obs, cfg.camera
    slot_dim = oc.ball_slots * oc.ball_slot_features
    if oc.actor_ball_input == "depth":
        ball_part = ("depth", len(cc.frame_offsets) * cc.policy_res[0] * cc.policy_res[1])
    elif oc.actor_ball_input == "tracker":
        ball_part = ("tracked_balls", slot_dim)
    else:
        raise ValueError(f"unknown actor_ball_input: {oc.actor_ball_input!r}")
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
    """Turn a list of BallEstimate into a fixed-length vector (ball_slots * 8,).

    8 values per ball: relative position 3 (ball - robot, rotated into the heading frame: forward, left, up; up = the
               ball's z), relative velocity 3 (the ball's velocity rotated into the heading frame; subtracting the
               robot's velocity is optional, but training and evaluation must match), radius 1, present 1 (= 1).
    Order: closest to the robot first; drop the furthest if there are more than ball_slots; pad empty slots with all
    zeros ("present" = 0, so the network can tell "no ball" from "a ball at the origin"). Clip relative positions to
    ±clip_rel_pos.
    For a delayed estimate (b.stamp < now) you can push it forward to now with balls.predict_ball first.
    """
    robot_xy = np.asarray(robot_xy, dtype=float)
    if robot_xy.shape != (2,) or not np.all(np.isfinite(robot_xy)):
        raise ValueError("robot_xy must be a finite vector with shape (2,)")
    slots = np.zeros((cfg.obs.ball_slots, cfg.obs.ball_slot_features), dtype=np.float32)
    cosine, sine = np.cos(robot_yaw), np.sin(robot_yaw)
    ordered = sorted(balls, key=lambda ball: np.linalg.norm(ball.position[:2] - robot_xy))
    for slot, ball in zip(slots, ordered[:cfg.obs.ball_slots]):
        relative = np.asarray(ball.position, dtype=float).copy()
        relative[:2] -= robot_xy
        forward = cosine * relative[0] + sine * relative[1]
        left = -sine * relative[0] + cosine * relative[1]
        velocity = np.asarray(ball.velocity, dtype=float)
        v_forward = cosine * velocity[0] + sine * velocity[1]
        v_left = -sine * velocity[0] + cosine * velocity[1]
        slot[:] = [
            np.clip(forward, -cfg.obs.clip_rel_pos, cfg.obs.clip_rel_pos),
            np.clip(left, -cfg.obs.clip_rel_pos, cfg.obs.clip_rel_pos),
            np.clip(relative[2], -cfg.obs.clip_rel_pos, cfg.obs.clip_rel_pos),
            v_forward, v_left, velocity[2], float(ball.radius), 1.0,
        ]
    return slots.reshape(-1)


def action_to_command(action: np.ndarray, cfg: Config) -> np.ndarray:
    """Turn the policy output (roughly in [-1, 1]) into u = (v, w, sdot).

    Suggested: clip to [-1, 1] first, then
        v    = linear map onto [cfg.command.v_min, cfg.command.v_max_stand]
        w    = action[1] * cfg.command.w_max_stand
        sdot = action[2] * cfg.command.sdot_max
    The lower speed limits when crouched are handled afterwards by lqr.clip_to_envelope.
    """
    values = np.asarray(action, dtype=float)
    if values.shape != (ACTION_DIM,) or not np.all(np.isfinite(values)):
        raise ValueError(f"action must be a finite vector with shape ({ACTION_DIM},)")
    values = np.clip(values, -1.0, 1.0)
    command = cfg.command
    speed = command.v_min + 0.5 * (values[0] + 1.0) * (command.v_max_stand - command.v_min)
    return np.array([
        speed,
        values[1] * command.w_max_stand,
        values[2] * command.sdot_max,
    ])


class DodgeEnv:
    def __init__(self, cfg: Config, reward_variant: str = "task", filter_mode: str | None = None,
                 replan: str = "every_step", ball_source: str = "truth", augment_depth: bool = True,
                 seed: int = 0):
        """Create one environment.

        reward_variant: "task" / "task_collision" / "task_cbf"
        filter_mode: None (no filter) / "naive" / "lag_aware"
        replan: "every_step" / "commit" (only matters when filter_mode is not None)
        ball_source: ball information source for the filter, "truth" / "noisy_truth" / "depth_tracker"
        augment_depth: True in training (PAC-MAN augmentation on the policy image), False in evaluation

        Suggested steps:
          1. path = scene.build_scene(cfg=cfg); self.env = TitaEnv(str(path)); self.ids = scene_ids(m, cfg)
          2. self.lqr = LQRBalancer(...); design_all(); self.ref = ReferenceGenerator(cfg)
          3. self.launcher = BallLauncher(m, d, cfg, ids); self.camera = DepthCamera(m, cfg, ids)
             self.stack = DepthFrameStack((height, width) of policy_res, cfg.camera.frame_offsets)
          4. self.ball_source = make_ball_source(ball_source, m, cfg, launcher, camera)
             the reward's CBF term uses a separate TruthBallSource (privileged)
          5. if filter_mode: self.filter = SafetyFilter(cfg, filter_mode, replan)
          6. sensors.Imu / Encoders, state_estimator.StateEstimator (only when cfg.estimator.use_true_state=False)
          7. self.rng = np.random.default_rng(seed)
        """
        if reward_variant not in {"task", "task_collision", "task_cbf"}:
            raise ValueError(f"unknown reward variant: {reward_variant!r}")
        self.cfg = cfg
        self.reward_variant = reward_variant
        self.filter_mode = filter_mode
        self.ball_source_kind = ball_source
        self.augment_depth = bool(augment_depth)
        self.seed = int(seed)

        self.env = TitaEnv(str(build_scene(cfg=cfg)), cfg=cfg)
        self.m, self.d = self.env.m, self.env.d
        self.ids = scene_ids(self.m, cfg)
        self.lqr = LQRBalancer(self.env, cfg, self.ids.robot_nv)
        self.lqr.design_all()
        self.ref = ReferenceGenerator(cfg)

        self.launcher = BallLauncher(self.m, self.d, cfg, self.ids)
        self.camera = DepthCamera(self.m, cfg, self.ids)
        policy_width, policy_height = cfg.camera.policy_res
        self.stack = DepthFrameStack((policy_height, policy_width), cfg.camera.frame_offsets)
        self.ball_source = make_ball_source(ball_source, self.m, cfg, self.launcher, self.camera)
        self.truth_source = TruthBallSource(self.launcher)
        if cfg.obs.actor_ball_input == "tracker" and ball_source != "depth_tracker":
            self.actor_ball_source = make_ball_source(
                "depth_tracker", self.m, cfg, self.launcher, self.camera,
            )
        else:
            self.actor_ball_source = self.ball_source
        self.safety_filter = SafetyFilter(cfg, filter_mode, replan) if filter_mode else None

        self.imu = Imu(self.m, cfg)
        self.encoders = Encoders(self.m, cfg)
        self.estimator = None if cfg.estimator.use_true_state else StateEstimator(
            self.m, cfg, self.ids.robot_nq, self.ids.robot_nv,
        )
        self.total_policy_steps = 0
        self._reset_rngs(self.seed)
        self._balls = []
        self._actor_balls = []
        self._truth_balls = []
        self._imu_reading = None

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset one episode and return the observation {"actor": ..., "critic": ...}.

        If seed is not None, rebuild self.rng from it (evaluation uses a fixed seed per episode so it can be
        reproduced). Reset: robot pose (standing), ReferenceGenerator, launcher.reset(rng), frame stack
        (stack.reset()), ball information source, filter, sensor biases, last action (= 0), goal position
        (cfg.task.goal_distance straight ahead of the start).
        Curriculum: if a training env (augment_depth=True) has run fewer than cfg.obs.single_ball_env_steps policy
        steps in total, throw only 1 ball this episode (set cfg.balls.balls_per_episode to (1, 1) before
        launcher.reset, then restore it).
        """
        if seed is not None:
            self.seed = int(seed)
            self._reset_rngs(self.seed)
        stand = self.lqr.leg_poses["stand"]
        self.env.reset(*stand)
        initial = self.env.planar_state(0.0)
        self.ref.reset(initial.xy, initial.yaw)

        original_count = self.cfg.balls.balls_per_episode
        use_curriculum = (self.augment_depth
                          and self.total_policy_steps < self.cfg.obs.single_ball_env_steps)
        try:
            if use_curriculum:
                self.cfg.balls.balls_per_episode = (1, 1)
            self.launcher.reset(self.launch_rng)
        finally:
            self.cfg.balls.balls_per_episode = original_count

        self.ball_source.reset(self.perception_rng)
        if self.actor_ball_source is not self.ball_source:
            self.actor_ball_source.reset(self.perception_rng)
        self.truth_source.reset(self.launch_rng)
        if self.safety_filter is not None:
            self.safety_filter.reset()
        self.imu.reset(self.imu_rng)
        self.encoders.reset(self.imu_rng)
        if self.estimator is not None:
            self.estimator.reset(self.d.qpos)
            self._q_estimate = self.d.qpos[:self.ids.robot_nq].copy()
            self._v_estimate = self.d.qvel[:self.ids.robot_nv].copy()
            self._imu_reading = self.imu.read(self.d, self.imu_rng)

        self.stack.reset()
        self.last_action = np.zeros(ACTION_DIM)
        self.goal = initial.xy + self.cfg.task.goal_distance * np.array([
            np.cos(initial.yaw), np.sin(initial.yaw),
        ])
        self.last_distance = float(np.linalg.norm(self.goal - initial.xy))
        self.reached_goal = False
        self.fell = False
        self.t_goal = None
        self.episode_policy_steps = 0
        self.interventions: list[float] = []
        self.filter_active_steps = 0
        self._refresh_perception()
        return {"actor": self.actor_obs(), "critic": self.critic_obs()}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Advance one policy step. Returns (obs, reward, terminated, truncated, info).

        info needs at least:
            "u_policy", "u_safe", "intervention", "filter_active", "new_hits" (how many balls newly hit this step)
        When the episode ends, also put "episode": self.episode_summary().
        """
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError(f"action must be a finite vector with shape ({ACTION_DIM},)")
        action = np.clip(action, -1.0, 1.0)
        u_policy = action_to_command(action, self.cfg)
        robot = self._robot_state()
        if self.safety_filter is None:
            u_safe = u_policy.copy()
            filter_info = {"intervention": 0.0, "active": False, "h_pred": np.inf}
        else:
            u_safe, filter_info = self.safety_filter.filter(
                u_policy, robot, self._balls, float(self.d.time),
            )

        new_hits = []
        for _ in range(self.cfg.sim.policy_every):
            reference = self.ref.update(u_safe, self.cfg.sim.timestep)
            self.env.set_leg_targets(self.lqr.leg_targets(reference.s))
            if self.estimator is None:
                qpos, qvel = self.d.qpos, self.d.qvel
            else:
                qpos, qvel = self._q_estimate, self._v_estimate
            self.env.set_wheel_torques(self.lqr.wheel_torques(qpos, qvel, reference))
            self.env.step()
            new_hits.extend(self.launcher.check_contacts())
            if self.estimator is not None:
                self._imu_reading = self.imu.read(self.d, self.imu_rng)
                encoder_reading = self.encoders.read(self.d, self.imu_rng)
                self._q_estimate, self._v_estimate = self.estimator.update(
                    self._imu_reading, encoder_reading, self.cfg.sim.timestep,
                )

        self.launcher.update(float(self.d.time))
        self._refresh_perception()
        true_state = self.env.planar_state(self.ref.s)
        distance = float(np.linalg.norm(self.goal - true_state.xy))
        reward_cfg = self.cfg.reward
        reward = reward_cfg.progress_weight * (self.last_distance - distance)
        at_goal = distance <= self.cfg.task.goal_radius
        if at_goal and not self.reached_goal:
            self.reached_goal = True
            self.t_goal = float(self.d.time)
            reward += reward_cfg.goal_bonus
        if at_goal:
            reward += reward_cfg.at_goal_weight
        reward -= reward_cfg.action_rate_weight * float(np.sum((action - self.last_action) ** 2))

        self.fell = abs(self.env.pitch()) > self.cfg.task.fall_pitch
        if self.fell:
            reward -= reward_cfg.fall_penalty
        if self.reward_variant == "task_collision":
            reward -= reward_cfg.collision_penalty * len(new_hits)
        elif self.reward_variant == "task_cbf":
            reward += cbf_reward_term(
                self.m, self.d, self.ids.robot_collision_geoms,
                self._truth_balls, self.cfg.cbf,
            )

        self.last_distance = distance
        self.last_action = action.copy()
        self.episode_policy_steps += 1
        self.total_policy_steps += 1
        if self.safety_filter is not None:
            self.interventions.append(float(filter_info["intervention"]))
            self.filter_active_steps += int(filter_info["active"])

        terminated = bool(self.fell)
        truncated = bool(self.d.time + 1e-12 >= self.cfg.sim.episode_seconds)
        info = {
            "u_policy": u_policy.copy(),
            "u_safe": np.asarray(u_safe).copy(),
            "intervention": float(filter_info["intervention"]),
            "filter_active": bool(filter_info["active"]),
            "h_pred": float(filter_info["h_pred"]),
            "new_hits": len(new_hits),
        }
        if terminated or truncated:
            info["episode"] = self.episode_summary()
        observation = {"actor": self.actor_obs(), "critic": self.critic_obs()}
        return observation, float(reward), terminated, truncated, info

    def episode_summary(self) -> dict:
        """This episode's result, in the EPISODE_SCHEMA format of evaluate.py (use launcher.episode_records() for "balls")."""
        records = self.launcher.episode_records()
        filter_on = self.safety_filter is not None
        return {
            "success": bool(self.reached_goal and not self.fell),
            "fell": bool(self.fell),
            "t_goal": self.t_goal,
            "n_thrown": sum(bool(record["launched"]) for record in records),
            "n_hit": sum(bool(record["hit"]) for record in records),
            "balls": records,
            "intervention_mean": (float(np.mean(self.interventions))
                                  if filter_on and self.interventions else (0.0 if filter_on else None)),
            "filter_active_frac": (self.filter_active_steps / max(self.episode_policy_steps, 1)
                                   if filter_on else None),
            "episode_seconds": float(self.d.time),
        }

    # ---------------- observations (design: see "Observations" above; order follows obs_layout) ----------------
    def actor_obs(self) -> np.ndarray:
        """Actor observation (589,) float32.

        Suggested:
            lay = obs_layout(self.cfg)["actor"]
            obs = np.zeros(lay["goal"].stop, np.float32)
            obs[lay["depth"]] = self.stack.get()          # this step's depth image must be pushed before calling this
            obs[lay["gyro"]] = ...
            ...
        Render the depth image only once per policy step (render once, share it between the policy image and the
        depth_tracker ball source: ball_source.update(d, rng, frame=frame)).
        """
        layout = obs_layout(self.cfg)["actor"]
        observation = np.zeros(self.actor_obs_dim, dtype=np.float32)
        if "depth" in layout:
            observation[layout["depth"]] = self.stack.get()
        else:
            robot = self._robot_state()
            observation[layout["tracked_balls"]] = ball_slot_features(
                self._actor_balls, robot.xy, robot.yaw, self.cfg,
            )

        if self.estimator is None:
            gyro = self.d.sensordata[self.imu._gyro].copy()
            quat = self.d.sensordata[self.imu._quat].copy()
            robot = self.env.planar_state(self.ref.s)
        else:
            gyro = self._imu_reading.gyro
            quat = self.estimator.quat
            robot = self.estimator.planar_state()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        gravity = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        observation[layout["gyro"]] = gyro
        observation[layout["gravity"]] = gravity
        observation[layout["forward_speed"]] = robot.v
        observation[layout["crouch"]] = self.ref.s
        observation[layout["last_action"]] = self.last_action
        relative_goal = self.goal - robot.xy
        cosine, sine = np.cos(robot.yaw), np.sin(robot.yaw)
        observation[layout["goal"]] = [
            cosine * relative_goal[0] + sine * relative_goal[1],
            -sine * relative_goal[0] + cosine * relative_goal[1],
        ]
        return observation

    def critic_obs(self) -> np.ndarray:
        """Critic observation (632,) float32 = actor_obs() followed by the true base velocity (3) and true ball states (40)."""
        layout = obs_layout(self.cfg)["critic"]
        actor = self.actor_obs()
        observation = np.zeros(self.critic_obs_dim, dtype=np.float32)
        observation[:len(actor)] = actor
        true_state = self.env.planar_state(self.ref.s)
        spatial_velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self.m, self.d, mujoco.mjtObj.mjOBJ_BODY,
            self.ids.base_body, spatial_velocity, 0,
        )
        cosine, sine = np.cos(true_state.yaw), np.sin(true_state.yaw)
        vx, vy, vz = spatial_velocity[3:6]
        observation[layout["base_vel"]] = [
            cosine * vx + sine * vy,
            -sine * vx + cosine * vy,
            vz,
        ]
        observation[layout["balls"]] = ball_slot_features(
            self._truth_balls, true_state.xy, true_state.yaw, self.cfg,
        )
        return observation

    def _reset_rngs(self, seed: int) -> None:
        streams = np.random.SeedSequence(seed).spawn(4)
        self.launch_rng = np.random.default_rng(streams[0])
        self.perception_rng = np.random.default_rng(streams[1])
        self.imu_rng = np.random.default_rng(streams[2])
        self.augment_rng = np.random.default_rng(streams[3])
        self.rng = self.launch_rng

    def _refresh_perception(self) -> None:
        frame = self.camera.render(self.d)
        self.stack.push(self.camera.policy_image(
            frame, self.augment_rng, augment=self.augment_depth,
        ))
        self._balls = self.ball_source.update(self.d, self.perception_rng, frame)
        if self.actor_ball_source is self.ball_source:
            self._actor_balls = self._balls
        else:
            self._actor_balls = self.actor_ball_source.update(
                self.d, self.perception_rng, frame,
            )
        self._truth_balls = self.truth_source.update(self.d)

    def _robot_state(self):
        if self.estimator is None:
            return self.env.planar_state(self.ref.s)
        return self.estimator.planar_state()

    @property
    def actor_obs_dim(self) -> int:
        """= the stop of the last part of obs_layout(cfg)["actor"] (589 by default)."""
        return max(s.stop for s in obs_layout(self.cfg)["actor"].values())

    @property
    def critic_obs_dim(self) -> int:
        """= the stop of the last part of obs_layout(cfg)["critic"] (632 by default)."""
        return max(s.stop for s in obs_layout(self.cfg)["critic"].values())
