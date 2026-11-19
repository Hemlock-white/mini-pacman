"""LQR balance control: turn the high-level command (v, w, s) into 2 wheel torques + 6 leg joint targets. [YOURS]

Most of this file can be moved over from cell 3 of TITA_Feasibility_Check.ipynb (tita_core.py). Only two things are
new: (1) the scene now has balls, (2) the robot has to be able to turn.

Overall flow
------------
    ref_gen = ReferenceGenerator(cfg)              # turns the policy's command into a smooth reference
    lqr = LQRBalancer(env, cfg, ids.robot_nv)      # env = TitaEnv from env.py
    lqr.design_all()                               # compute the gains once at start-up (standing, lowest crouch)
    every physics step:
        ref = ref_gen.update(u_safe, dt)           # u_safe = (v, w, sdot), the command after the CBF filter
        env.set_leg_targets(lqr.leg_targets(ref.s))
        env.set_wheel_torques(lqr.wheel_torques(qpos, qvel, ref))

Pitfalls (we hit these in the prototype)
----------------------------------------
1. **The balls must not go into the DARE**: the scene has 5 balls with 6 dofs each. The wheels can't control them,
   and putting them in the DARE gives "Failed to find a finite solution". After linearizing, keep only the robot's
   dofs: 0 .. robot_nv-1 (scene.scene_ids(m).robot_nv, currently 14). Pad K back to full size with zeros, or only
   use the first robot_nv entries when computing errors.
2. **Set frictionloss to 0 while linearizing**, then restore it (friction loss isn't differentiable and messes up A).
3. **Press the wheels into the floor before linearizing** (config.robot.z_offset = -0.5 mm), otherwise the
   linearization doesn't see the contact and the LQR falls over immediately.
4. **Zero the gains of the hip-roll joints (abad, joint_*_leg_1)**: they are an artefact of the stiff contact
   (about ±30000 before).
5. **Wheel angle errors can't just be zeroed**; use the rolling relation (wheel_dq in tita_core.py):
       dq[wheel] = coef * (dq[x] - dq[yaw] * y_offset), coef = xaxis_y / R (env.reset() computes env.roll)
6. **Use MuJoCo's default contact** (pyramidal, impratio=1); robot.xml's elliptic + impratio=100 makes the LQR drift.
7. **env.py's set_pose() also zeroes the balls**: it does ``d.qpos[:] = 0``, so in the new scene all 5 balls jump to
   the origin, under the robot (checked). Their contacts then leak into A and B when linearizing. Change it to only
   zero the robot's part, ``d.qpos[:robot_nq] = 0``, or call BallLauncher.park(i) after resetting.

Turning (checked on a laptop, see the safe envelope in config.CommandConfig)
---------------------------------------------------------------------------
The LQR is linearized at "heading 0". To track a turn rate w, the key is to **express the errors in the reference
heading frame**:
    a. Reference pose q_ref = q0 (the linearization pose), but with the base position set to ref.p and the base
       quaternion left-multiplied by "rotate ref.psi about z".
    b. dq = mj_differentiatePos(q_ref, qpos); dq[0:3] is the position error in the world frame → rotate into the
       current heading frame:
           forward error = cos(yaw)*dx + sin(yaw)*dy; lateral error = 0 (two wheels can't move sideways; it's
           uncontrollable, don't feed it to the LQR)
       dq[3:6] is the rotation error in the body frame (roll, pitch, yaw), use it as is.
    c. Rotate the velocity qvel[0:3] into the heading frame too; ve[0] -= ref.v; ve[5] -= ref.w.
    d. Wheel speed reference: ve[wheel] -= coef * (ref.v - ref.w * y_offset).
    e. Set all ball dof errors to 0.
Measured: standing, v=0.5 m/s and w=1.5 rad/s in circles, peak pitch 15 degrees; turning on the spot at 2.5 rad/s is
fine; fully crouched (s=1), v=0.5 pitches to 50 degrees, so ReferenceGenerator has to enforce "the lower, the slower".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, CommandConfig


@dataclass
class Reference:
    """What the LQR tracks (world frame)."""
    p: np.ndarray          # reference position (x, y)
    psi: float             # reference heading (rad)
    v: float               # reference forward speed (m/s)
    w: float               # reference turn rate (rad/s)
    s: float               # crouch level 0 (standing) .. 1 (lowest)


def clip_to_envelope(u: np.ndarray, s: float, cmd: CommandConfig) -> np.ndarray:
    """Clip the command u = (v, w, sdot) to the safe envelope.

    The speed limits drop linearly with the crouch level s:
        v_max(s) = v_max_stand + (v_max_crouch - v_max_stand) * s
        w_max(s) = w_max_stand + (w_max_crouch - w_max_stand) * s
    Clip v to [v_min, v_max(s)], w to [-w_max(s), w_max(s)], sdot to [-sdot_max, sdot_max], and sdot can't be
    negative when s is already 0, or positive when s is 1.
    Return a new np.ndarray (don't modify the input).
    """
    raise NotImplementedError("TODO: clip_to_envelope")


class ReferenceGenerator:
    """Turns the command u = (v, w, sdot) into a smooth reference.

    Call update() every physics step:
      1. u = clip_to_envelope(u, self.s, cfg.command)
      2. Move the reference speeds towards u, but no faster than accel_limit (v) and yaw_accel_limit (w).
         Prototype: if the speed command jumps too much in one step, the LQR gets "kicked" and falls.
      3. s += sdot * dt, clipped to [0, 1].
      4. psi += w * dt; p += v * dt * (cos psi, sin psi).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        raise NotImplementedError("TODO: ReferenceGenerator.__init__")

    def reset(self, p0: np.ndarray, psi0: float) -> None:
        """Start of an episode: reference position = robot position, speeds 0, s = 0."""
        raise NotImplementedError

    def update(self, u: np.ndarray, dt: float) -> Reference:
        raise NotImplementedError

    @property
    def s(self) -> float:
        raise NotImplementedError


class LQRBalancer:
    """One set of LQR gains for standing and one for the lowest crouch, blended linearly by s (gain scheduling)."""

    def __init__(self, env, cfg: Config, robot_nv: int):
        """env: TitaEnv from env.py (needs m, d, poses, legs_of, balanced_k, reset, roll).
        robot_nv: number of robot dofs; the ball dofs come after (see pitfall 1)."""
        raise NotImplementedError("TODO: LQRBalancer.__init__")

    def design_all(self) -> None:
        """Design the gains for standing (cfg.robot.stand_hip) and the lowest crouch (cfg.robot.crouch_hip), store in self.gains."""
        raise NotImplementedError

    def design(self, name: str, hip: float) -> tuple[np.ndarray, np.ndarray]:
        """Linearize at one pose and solve the DARE; return (q0 linearization pose, K gains (2, 2*robot_nv)).

        Steps (Tita.lqr in tita_core.py):
          1. knee = env.balanced_k(hip); env.reset(hip, knee) (wheels pressed slightly into the floor)
          2. temporarily set frictionloss to 0; mujoco.mjd_transitionFD(m, d, eps, True, A, B, None, None); restore
          3. keep only the robot-dof rows/columns of A and B ([0..robot_nv-1] and [nv..nv+robot_nv-1]); B only the two wheel motors
          4. Q = diagonal, default cfg.lqr.q_default; position x, pitch, yaw and their rates use the cfg.lqr values; wheel angles = 0
             R = cfg.lqr.r_wheel * I(2)
          5. multiply A and B by sqrt(discount), then scipy.linalg.solve_discrete_are
          6. K = (R + B'PB)^-1 B'PA; zero the abad columns
        """
        raise NotImplementedError

    def leg_targets(self, s: float) -> np.ndarray:
        """Targets for the 6 leg joints: hip and knee blended between standing and crouched by s (env.legs_of(h, k))."""
        raise NotImplementedError

    def wheel_torques(self, qpos: np.ndarray, qvel: np.ndarray, ref: Reference) -> np.ndarray:
        """Torques for the 2 wheels (Nm), clipped to ±cfg.robot.wheel_torque_limit.

        qpos, qvel: the full state (true values or the state_estimator's estimate).
        Steps: see "Turning" a-e above; K and q0 are blended between the two sets by ref.s.
        """
        raise NotImplementedError
