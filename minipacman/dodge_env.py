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

import numpy as np

from .config import Config

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
    raise NotImplementedError("TODO: ball_slot_features (see docstring)")


def action_to_command(action: np.ndarray, cfg: Config) -> np.ndarray:
    """Turn the policy output (roughly in [-1, 1]) into u = (v, w, sdot).

    Suggested: clip to [-1, 1] first, then
        v    = linear map onto [cfg.command.v_min, cfg.command.v_max_stand]
        w    = action[1] * cfg.command.w_max_stand
        sdot = action[2] * cfg.command.sdot_max
    The lower speed limits when crouched are handled afterwards by lqr.clip_to_envelope.
    """
    raise NotImplementedError("TODO: action_to_command")


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
        raise NotImplementedError("TODO: DodgeEnv.__init__")

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
        raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Advance one policy step. Returns (obs, reward, terminated, truncated, info).

        info needs at least:
            "u_policy", "u_safe", "intervention", "filter_active", "new_hits" (how many balls newly hit this step)
        When the episode ends, also put "episode": self.episode_summary().
        """
        raise NotImplementedError

    def episode_summary(self) -> dict:
        """This episode's result, in the EPISODE_SCHEMA format of evaluate.py (use launcher.episode_records() for "balls")."""
        raise NotImplementedError

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
        raise NotImplementedError("TODO: actor_obs")

    def critic_obs(self) -> np.ndarray:
        """Critic observation (632,) float32 = actor_obs() followed by the true base velocity (3) and true ball states (40)."""
        raise NotImplementedError("TODO: critic_obs")

    @property
    def actor_obs_dim(self) -> int:
        """= the stop of the last part of obs_layout(cfg)["actor"] (589 by default)."""
        return max(s.stop for s in obs_layout(self.cfg)["actor"].values())

    @property
    def critic_obs_dim(self) -> int:
        """= the stop of the last part of obs_layout(cfg)["critic"] (632 by default)."""
        return max(s.stop for s in obs_layout(self.cfg)["critic"].values())
