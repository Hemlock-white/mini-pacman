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
    2. Robot: for a candidate command u, predict the body position and height with a "unicycle + speed lag" model:
           v_actual follows u's v with time constant tau (naive: tau=0, lag_aware: tau=cfg.cbf.lag_tau_aware)
           yaw += w*dt; xy += v_actual*dt*(cos yaw, sin yaw); s += sdot*dt (clipped to 0..1)
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

import mujoco
import numpy as np

from .ball_tracker import BallEstimate
from .config import Config, CBFConfig

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
    raise NotImplementedError("TODO: robot_link_spheres (see section 1 above)")


def link_barriers(centers: np.ndarray, radii: np.ndarray, velocities: np.ndarray,
                  ball: BallEstimate, cfg: CBFConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (h (n,), hdot (n,)) of each link with respect to this ball."""
    raise NotImplementedError("TODO: link_barriers (see section 1 above)")


def is_threat(ball: BallEstimate, cfg: CBFConfig) -> bool:
    """A ball is a threat only when it is off the floor (z > r + threat_min_height_above_radius) and its horizontal
    speed is > threat_min_speed."""
    raise NotImplementedError


def cbf_reward_term(m: mujoco.MjModel, d: mujoco.MjData, geoms: np.ndarray,
                    balls: list[BallEstimate], cfg: CBFConfig) -> float:
    """The CBF term of the task_cbf reward (<= 0, already multiplied by reward_weight). With several balls, take the
    worst (most negative) one."""
    raise NotImplementedError("TODO: cbf_reward_term (see section 1 above)")


class SafetyFilter:
    """Runtime safety filter (see section 2 above)."""

    def __init__(self, cfg: Config, mode: str = "lag_aware", replan: str = "every_step"):
        """mode: "naive" or "lag_aware"; replan: "every_step" or "commit"."""
        raise NotImplementedError("TODO: SafetyFilter.__init__")

    def reset(self) -> None:
        raise NotImplementedError

    def filter(self, u_nom: np.ndarray, robot: RobotPlanarState, balls: list[BallEstimate],
               t: float) -> tuple[np.ndarray, dict]:
        """Return (u_safe, info). info needs at least:
            "intervention": ||u_safe - u_nom|| (weighted)
            "active": whether the command was changed this step
            "h_pred": predicted closest distance of the chosen command
        With no balls, just return u_nom."""
        raise NotImplementedError

    def predicted_clearance(self, u: np.ndarray, robot: RobotPlanarState, balls: list[BallEstimate],
                            t: float) -> float:
        """Predicted closest distance h_pred of candidate command u over the horizon (steps 1-3 in section 2)."""
        raise NotImplementedError
