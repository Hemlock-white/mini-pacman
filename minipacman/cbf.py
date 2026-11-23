"""CBF: barrier function h, the CBF reward term, and the runtime safety filter. [YOURS]

1. Link-CBF (for the reward, same as PAC-MAN)
---------------------------------------------
What PAC-MAN's deployed config does (dodge_link_cbf_reward in rewards.py): for each robot collision shape i and each ball:

    h_i    = ||p_ball - c_i|| - (r_ball + r_i + margin)        # distance minus both radii; h >= 0 means safe
    hdot_i = d h_i / dt = n_i · (v_ball - v_i), n_i = (p_ball - c_i) / ||p_ball - c_i||
    term   = clip(min_i(hdot_i + alpha * h_i), -constraint_clip, 0)     # only penalise the "violating" part, <= 0
    reward += reward_weight * term                                       # only when is_threat()

Approximate c_i and r_i with each collision geom's bounding sphere (m.geom_rbound); v_i is that geom's linear velocity
(mujoco.mj_objectVelocity(m, d, mjOBJ_GEOM, g, out6, 0), world frame, out6[3:6] is linear).
Parameters are in config.CBFConfig (alpha=1.0, margin=0.05, clip=2.0, weight=0.27, all the same as PAC-MAN).
This reward uses the **true ball state** (so does PAC-MAN; it's called privileged).

2. Safety filter (enforced at runtime; PAC-MAN's deployed config has none, it's our experiment variable)
--------------------------------------------------------------------------------------------------------
Input: the command the policy wants, u_nom = (v, w, sdot). Output: u_safe, "as close to u_nom as possible, but
predicted not to hit any ball". TITA can't move sideways, and with a ball coming from the front a distance-type h
quickly becomes infeasible (the ball moves at 3-6 m/s, far faster than TITA's 0.8 m/s), so we suggest a "predictive"
version: over the next ``horizon`` seconds, predict the ball's path and the robot's motion and compute the closest
distance.

    Steps:
    1. Balls: predict each BallEstimate with balls.predict_ball() (with bounces); with latency, first push it forward
       by (t - b.stamp).
    2. Robot: for a candidate command u, predict the body position and height with a unicycle model:
           naive:     v and w take effect at once; s += sdot*dt (clipped to 0..1)
           lag_aware: u goes through ReferenceGenerator's pipeline (acceleration limits, crouch only as deep as the
                      speeds allow), then v_actual follows the reference speed with time constant cfg.cbf.lag_tau_aware
           yaw += w*dt; xy += v_actual*dt*(cos yaw, sin yaw)
       Approximate the body with a box (body frame): ±0.30 m front-back (including the wheels), ±0.30 m left-right,
       height 0 to top(s) = 0.538 - 0.13*s (top of the body is 0.538 m standing; a full crouch drops 13 cm,
       measured in the prototype).
    3. h_pred(u) = the minimum over the horizon of "distance from ball centre to the box - ball radius".
    4. Candidate commands: a grid of a few values each for v, w, sdot (e.g. 7 for v, 5 for w, sdot in
       {-max, 0, +max}), first clipped to the safe envelope with lqr.clip_to_envelope.
    5. Among those with h_pred >= filter_margin, pick the one "closest to u_nom" (you can weight the distance, e.g.
       multiply sdot by 0.13 to turn it into m/s); if none is safe, pick the one with the largest h_pred.
    6. replan="every_step" (what all experiments use): recompute every policy step. replan="commit" is optional:
       decide once when a threat is detected and keep that command until the threat is gone (a "reflex").

Exp 2 compares mode="naive" vs mode="lag_aware". For reference: in our side-throw reference solution, when
replanning every 10 ms, naive and lag_aware gave the same result (frequent replanning corrects the lag error); only
when deciding once did naive get hit 65% of the time vs 30% for lag_aware. So "no difference" is a possible, and
honest, outcome of Exp 2 with front throws.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import mujoco
import numpy as np

from .ball_tracker import BallEstimate
from .balls import predict_ball
from .config import Config, CBFConfig
from .lqr import clip_to_envelope, envelope_limits

ROBOT_HALF_LENGTH = 0.30      # m, half length of the body box front-back (including wheels)
ROBOT_HALF_WIDTH = 0.30       # m, half width left-right (including wheels)
ROBOT_TOP_STAND = 0.538       # m, top of the body when standing (measured in the prototype)
ROBOT_CROUCH_DROP = 0.13      # m, how much the body drops in a full crouch (measured in the prototype)


@dataclass
class RobotPlanarState:
    """The robot state the CBF and the policy need (true or estimated)."""
    xy: np.ndarray            # base position (x, y)
    yaw: float                # heading (rad)
    v: float                  # forward speed (m/s)
    w: float                  # turn rate (rad/s)
    s: float                  # crouch level 0..1


def robot_link_spheres(m: mujoco.MjModel, d: mujoco.MjData, geoms: np.ndarray):
    """Bounding sphere of each collision geom: returns (centers (n,3), radii (n,), velocities (n,3)), world frame."""
    geom_ids = np.asarray(geoms, dtype=int)
    if geom_ids.ndim != 1 or np.any(geom_ids < 0) or np.any(geom_ids >= m.ngeom):
        raise ValueError("geoms must be a one-dimensional array of valid geom ids")
    centers = d.geom_xpos[geom_ids].copy()
    radii = m.geom_rbound[geom_ids].copy()
    velocities = np.empty((len(geom_ids), 3))
    spatial_velocity = np.empty(6)
    for index, geom in enumerate(geom_ids):
        mujoco.mj_objectVelocity(
            m, d, mujoco.mjtObj.mjOBJ_GEOM, int(geom), spatial_velocity, 0
        )
        velocities[index] = spatial_velocity[3:6]
    return centers, radii, velocities


def link_barriers(centers: np.ndarray, radii: np.ndarray, velocities: np.ndarray,
                  ball: BallEstimate, cfg: CBFConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (h (n,), hdot (n,)) of each link with respect to this ball."""
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if centers.ndim != 2 or centers.shape[1:] != (3,):
        raise ValueError(f"centers must have shape (n, 3), got {centers.shape}")
    if radii.shape != (len(centers),) or velocities.shape != centers.shape:
        raise ValueError("radii and velocities must match centers")

    displacement = np.asarray(ball.position, dtype=float) - centers
    distances = np.linalg.norm(displacement, axis=1)
    normals = displacement / np.maximum(distances[:, None], 1e-12)
    relative_velocity = np.asarray(ball.velocity, dtype=float) - velocities
    h = distances - (radii + float(ball.radius) + cfg.margin)
    hdot = np.sum(normals * relative_velocity, axis=1)
    return h, hdot


def is_threat(ball: BallEstimate, cfg: CBFConfig) -> bool:
    """A ball is a threat only when it is off the floor (z > r + threat_min_height_above_radius) and its horizontal
    speed is > threat_min_speed."""
    return bool(
        ball.position[2] > ball.radius + cfg.threat_min_height_above_radius
        and np.linalg.norm(ball.velocity[:2]) > cfg.threat_min_speed
    )


def cbf_reward_term(m: mujoco.MjModel, d: mujoco.MjData, geoms: np.ndarray,
                    balls: list[BallEstimate], cfg: CBFConfig) -> float:
    """The CBF term of the task_cbf reward (<= 0, already multiplied by reward_weight). With several balls, take the
    worst (most negative) one."""
    threats = [ball for ball in balls if is_threat(ball, cfg)]
    if not threats:
        return 0.0
    centers, radii, velocities = robot_link_spheres(m, d, geoms)
    terms = []
    for ball in threats:
        h, hdot = link_barriers(centers, radii, velocities, ball, cfg)
        constraint = float(np.min(hdot + cfg.alpha * h))
        terms.append(float(np.clip(constraint, -cfg.constraint_clip, 0.0)))
    return cfg.reward_weight * min(terms)


class SafetyFilter:
    """Runtime safety filter (see section 2 above).

    Robot models for predicting a candidate command (the only difference between the two modes):
      naive:     the command takes effect at once: v and w jump to the command, s moves at sdot.
      lag_aware: the command goes through the same pipeline as the real robot: ReferenceGenerator's acceleration
                 limits and crouch coupling (lqr.max_crouch_for), then a first-order lag lag_tau_aware between the
                 reference speed and the robot's actual speed (measured, see CBFConfig).
    All candidates are predicted at once (vectorised), so a policy step costs a few ms instead of tens.

    Switching penalty: among safe candidates the filter picks the one closest to u_nom plus switch_weight times the
    distance to its previous output, so it doesn't jump between extreme commands every 20 ms.
    """

    def __init__(self, cfg: Config, mode: str = "lag_aware", replan: str = "every_step"):
        """mode: "naive" or "lag_aware"; replan: "every_step" or "commit"."""
        if mode not in {"naive", "lag_aware"}:
            raise ValueError(f"unknown filter mode: {mode!r}")
        if replan not in {"every_step", "commit"}:
            raise ValueError(f"unknown replanning mode: {replan!r}")
        self.cfg = cfg
        self.mode = mode
        self.replan = replan
        self.tau = cfg.cbf.lag_tau_naive if mode == "naive" else cfg.cbf.lag_tau_aware
        self.reset()

    def reset(self) -> None:
        self._committed: np.ndarray | None = None
        self._previous: np.ndarray | None = None

    def filter(self, u_nom: np.ndarray, robot: RobotPlanarState, balls: list[BallEstimate],
               t: float) -> tuple[np.ndarray, dict]:
        """Return (u_safe, info). info needs at least:
            "intervention": ||u_safe - u_nom|| (weighted)
            "active": whether the command was changed this step
            "h_pred": predicted closest distance of the chosen command
        With no balls, just return u_nom."""
        nominal = np.asarray(u_nom, dtype=float)
        if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
            raise ValueError("u_nom must be a finite vector with shape (3,)")
        threats = [ball for ball in balls if is_threat(ball, self.cfg.cbf)]
        if not threats:
            self._committed = None
            self._previous = None
            return nominal.copy(), self._info(nominal, nominal, np.inf)

        if self.replan == "commit" and self._committed is not None:
            chosen = clip_to_envelope(self._committed, robot.s, self.cfg.command)
            clearance = self.predicted_clearance(chosen, robot, threats, t)
            return chosen, self._info(nominal, chosen, clearance)

        bounded_nominal = clip_to_envelope(nominal, robot.s, self.cfg.command)
        nominal_clearance = self.predicted_clearance(bounded_nominal, robot, threats, t)
        if nominal_clearance >= self.cfg.cbf.filter_margin:
            self._previous = None
            return bounded_nominal, self._info(nominal, bounded_nominal, nominal_clearance)

        candidates = self._candidate_commands(bounded_nominal, robot.s)
        clearances = self.predicted_clearances(candidates, robot, threats, t)
        cost = self._command_distances(candidates, nominal)
        if self._previous is not None:
            cost = cost + self.cfg.cbf.switch_weight * self._command_distances(candidates, self._previous)
        safe = np.flatnonzero(clearances >= self.cfg.cbf.filter_margin)
        if safe.size:
            index = int(safe[np.argmin(cost[safe])])
        else:
            best_clearance = np.max(clearances)
            best = np.flatnonzero(np.isclose(clearances, best_clearance, atol=1e-12, rtol=0.0))
            index = int(best[np.argmin(cost[best])])
        chosen = candidates[index].copy()
        self._previous = chosen.copy()
        if self.replan == "commit" and not np.allclose(chosen, bounded_nominal):
            self._committed = chosen.copy()
        return chosen, self._info(nominal, chosen, float(clearances[index]))

    def predicted_clearance(self, u: np.ndarray, robot: RobotPlanarState, balls: list[BallEstimate],
                            t: float) -> float:
        """Predicted closest distance h_pred of candidate command u over the horizon (steps 1-3 in section 2)."""
        if not balls:
            return float("inf")
        command = clip_to_envelope(u, robot.s, self.cfg.command)
        return float(self.predicted_clearances(command[None, :], robot, balls, t)[0])

    def predicted_clearances(self, commands: np.ndarray, robot: RobotPlanarState, balls: list[BallEstimate],
                             t: float) -> np.ndarray:
        """h_pred for each row of commands (n, 3) (already inside the envelope)."""
        commands = np.atleast_2d(np.asarray(commands, dtype=float))
        if not balls:
            return np.full(len(commands), np.inf)
        dt = self.cfg.sim.policy_dt
        times = np.arange(0.0, self.cfg.cbf.horizon + 0.5 * dt, dt)
        robot_xy, robot_yaw, _, crouch = self._predict_robot(commands, robot, len(times))
        cosine, sine = np.cos(robot_yaw), np.sin(robot_yaw)                     # (n, T)
        top = ROBOT_TOP_STAND - ROBOT_CROUCH_DROP * crouch                     # (n, T)
        minimum = np.full(len(commands), np.inf)
        for ball in balls:
            latency = max(float(t) - float(ball.stamp), 0.0)
            positions, _ = predict_ball(
                ball.position, ball.velocity, ball.radius, ball.restitution,
                latency + times,
            )
            relative = positions[None, :, :2] - robot_xy                       # (n, T, 2)
            forward = cosine * relative[..., 0] + sine * relative[..., 1]
            left = -sine * relative[..., 0] + cosine * relative[..., 1]
            offsets = np.stack((
                np.abs(forward) - ROBOT_HALF_LENGTH,
                np.abs(left) - ROBOT_HALF_WIDTH,
                np.abs(positions[None, :, 2] - top / 2.0) - top / 2.0,
            ), axis=-1)                                                        # (n, T, 3)
            outside = np.linalg.norm(np.maximum(offsets, 0.0), axis=-1)
            inside = np.minimum(np.max(offsets, axis=-1), 0.0)
            clearance = outside + inside - ball.radius
            minimum = np.minimum(minimum, clearance.min(axis=1))
        return minimum

    def _predict_robot(self, commands: np.ndarray, robot: RobotPlanarState,
                       steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Roll every candidate forward (see the class docstring). Returns xy (n, T, 2), yaw, speed, crouch (n, T)."""
        cmd = self.cfg.command
        n, dt = len(commands), self.cfg.sim.policy_dt
        xy = np.empty((n, steps, 2))
        yaw = np.empty((n, steps))
        speed = np.empty((n, steps))
        crouch = np.empty((n, steps))
        p = np.tile(np.asarray(robot.xy, dtype=float), (n, 1))
        psi = np.full(n, float(robot.yaw))
        v_ref = np.full(n, float(robot.v))
        w_ref = np.full(n, float(robot.w))
        s = np.full(n, float(np.clip(robot.s, 0.0, 1.0)))
        v_actual = v_ref.copy()
        response = 1.0 if self.tau <= 0.0 else 1.0 - np.exp(-dt / self.tau)
        xy[:, 0], yaw[:, 0], speed[:, 0], crouch[:, 0] = p, psi, v_actual, s
        for index in range(1, steps):
            if self.mode == "naive":
                v_ref = commands[:, 0].copy()
                w_ref = commands[:, 1].copy()
                s = np.clip(s + commands[:, 2] * dt, 0.0, 1.0)
            else:
                # ReferenceGenerator.update, vectorised (clip to the envelope at the current s, rate limits, coupling)
                v_low = cmd.v_min + (cmd.v_min_crouch - cmd.v_min) * s
                v_high = cmd.v_max_stand + (cmd.v_max_crouch - cmd.v_max_stand) * s
                w_high = cmd.w_max_stand + (cmd.w_max_crouch - cmd.w_max_stand) * s
                v_cmd = np.clip(commands[:, 0], v_low, v_high)
                w_cmd = np.clip(commands[:, 1], -w_high, w_high)
                sdot = np.clip(commands[:, 2], -cmd.sdot_max, cmd.sdot_max)
                sdot = np.where(s <= 0.0, np.maximum(sdot, 0.0), sdot)
                sdot = np.where(s >= 1.0, np.minimum(sdot, 0.0), sdot)
                # lqr.crouch_first: a crouch request also clips (v, w) to the level crouch_lookahead ahead
                goal = np.minimum(1.0, s + sdot * cmd.crouch_lookahead)
                crouching = sdot > 0.0
                g_low = cmd.v_min + (cmd.v_min_crouch - cmd.v_min) * goal
                g_high = cmd.v_max_stand + (cmd.v_max_crouch - cmd.v_max_stand) * goal
                g_w = cmd.w_max_stand + (cmd.w_max_crouch - cmd.w_max_stand) * goal
                v_cmd = np.where(crouching, np.clip(v_cmd, g_low, g_high), v_cmd)
                w_cmd = np.where(crouching, np.clip(w_cmd, -g_w, g_w), w_cmd)
                v_ref = v_ref + np.clip(v_cmd - v_ref, -cmd.accel_limit * dt, cmd.accel_limit * dt)
                w_ref = w_ref + np.clip(w_cmd - w_ref, -cmd.yaw_accel_limit * dt, cmd.yaw_accel_limit * dt)
                s_new = np.clip(s + sdot * dt, 0.0, 1.0)
                deepest = np.minimum.reduce([
                    _deepest(np.maximum(v_ref, 0.0), cmd.v_max_stand, cmd.v_max_crouch),
                    _deepest(np.maximum(-v_ref, 0.0), -cmd.v_min, -cmd.v_min_crouch),
                    _deepest(np.abs(w_ref), cmd.w_max_stand, cmd.w_max_crouch),
                ])
                s = np.where(s_new > deepest, np.where(s <= deepest, deepest, np.minimum(s_new, s)), s_new)
            v_actual = v_actual + response * (v_ref - v_actual)
            psi = psi + w_ref * dt
            p = p + (v_actual * dt)[:, None] * np.column_stack((np.cos(psi), np.sin(psi)))
            xy[:, index], yaw[:, index], speed[:, index], crouch[:, index] = p, psi, v_actual, s
        return xy, yaw, speed, crouch

    def _candidate_commands(self, nominal: np.ndarray, s: float) -> np.ndarray:
        command = self.cfg.command
        crouch = float(np.clip(s, 0.0, 1.0))
        v_low, v_high, w_high = envelope_limits(crouch, command)
        values = [nominal]
        for candidate in product(
            np.linspace(v_low, v_high, 7),
            np.linspace(-w_high, w_high, 5),
            (-command.sdot_max, 0.0, command.sdot_max),
        ):
            values.append(clip_to_envelope(np.asarray(candidate), crouch, command))
        return np.unique(np.asarray(values), axis=0)

    @staticmethod
    def _command_distances(candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
        difference = np.atleast_2d(candidates) - np.asarray(reference)
        scaled = difference * np.array([1.0, ROBOT_HALF_LENGTH, ROBOT_CROUCH_DROP])
        return np.linalg.norm(scaled, axis=-1)

    @classmethod
    def _command_distance(cls, left: np.ndarray, right: np.ndarray) -> float:
        return float(cls._command_distances(np.asarray(left)[None, :], right)[0])

    def _info(self, nominal: np.ndarray, chosen: np.ndarray, clearance: float) -> dict:
        intervention = self._command_distance(chosen, nominal)
        return {
            "intervention": intervention,
            "active": bool(intervention > 1e-9),
            "h_pred": float(clearance),
        }


def _deepest(value: np.ndarray, at_stand: float, at_crouch: float) -> np.ndarray:
    """Vectorised lqr.max_crouch_for for one limit: the deepest s whose limit still allows |value|."""
    span = at_stand - at_crouch
    ratio = (at_stand - value) / span if span > 0 else np.zeros_like(value)
    return np.where(value <= at_crouch + 1e-12, 1.0, np.where(value >= at_stand, 0.0, ratio))
