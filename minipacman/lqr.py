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

import mujoco
import numpy as np
import scipy.linalg as sla

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
    command = np.asarray(u, dtype=float)
    if command.shape != (3,):
        raise ValueError(f"u must have shape (3,), got {command.shape}")
    if not np.all(np.isfinite(command)) or not np.isfinite(s):
        raise ValueError("u and s must contain only finite values")

    crouch = float(np.clip(s, 0.0, 1.0))
    v_max = cmd.v_max_stand + (cmd.v_max_crouch - cmd.v_max_stand) * crouch
    w_max = cmd.w_max_stand + (cmd.w_max_crouch - cmd.w_max_stand) * crouch
    clipped = np.array([
        np.clip(command[0], cmd.v_min, v_max),
        np.clip(command[1], -w_max, w_max),
        np.clip(command[2], -cmd.sdot_max, cmd.sdot_max),
    ], dtype=float)
    if crouch <= 0.0:
        clipped[2] = max(0.0, clipped[2])
    if crouch >= 1.0:
        clipped[2] = min(0.0, clipped[2])
    return clipped


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
        self._reference = Reference(np.zeros(2), 0.0, 0.0, 0.0, 0.0)

    def reset(self, p0: np.ndarray, psi0: float) -> None:
        """Start of an episode: reference position = robot position, speeds 0, s = 0."""
        position = np.asarray(p0, dtype=float)
        if position.shape != (2,):
            raise ValueError(f"p0 must have shape (2,), got {position.shape}")
        if not np.all(np.isfinite(position)) or not np.isfinite(psi0):
            raise ValueError("p0 and psi0 must contain only finite values")
        self._reference = Reference(position.copy(), float(psi0), 0.0, 0.0, 0.0)

    def update(self, u: np.ndarray, dt: float) -> Reference:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        command = clip_to_envelope(u, self.s, self.cfg.command)
        previous = self._reference
        limits = self.cfg.command

        dv = np.clip(command[0] - previous.v,
                     -limits.accel_limit * dt, limits.accel_limit * dt)
        dw = np.clip(command[1] - previous.w,
                     -limits.yaw_accel_limit * dt, limits.yaw_accel_limit * dt)
        v = float(previous.v + dv)
        w = float(previous.w + dw)
        s = float(np.clip(previous.s + command[2] * dt, 0.0, 1.0))
        psi = float(previous.psi + w * dt)
        heading = np.array([np.cos(psi), np.sin(psi)])
        position = previous.p + v * dt * heading
        self._reference = Reference(position, psi, v, w, s)
        return Reference(position.copy(), psi, v, w, s)

    @property
    def s(self) -> float:
        return self._reference.s


class LQRBalancer:
    """One set of LQR gains for standing and one for the lowest crouch, blended linearly by s (gain scheduling)."""

    def __init__(self, env, cfg: Config, robot_nv: int):
        """env: TitaEnv from env.py (needs m, d, poses, legs_of, balanced_k, reset, roll).
        robot_nv: number of robot dofs; the ball dofs come after (see pitfall 1)."""
        if not 0 < robot_nv <= env.m.nv:
            raise ValueError(f"robot_nv must be in [1, {env.m.nv}], got {robot_nv}")
        self.env = env
        self.cfg = cfg
        self.m, self.d = env.m, env.d
        self.robot_nv = int(robot_nv)
        self.gains: dict[str, np.ndarray] = {}
        self.q0: dict[str, np.ndarray] = {}
        self.leg_poses: dict[str, tuple[float, float]] = {}
        self.env.find_mirror()

    def design_all(self) -> None:
        """Design the gains for standing (cfg.robot.stand_hip) and the lowest crouch (cfg.robot.crouch_hip), store in self.gains."""
        for name, hip in (("stand", self.cfg.robot.stand_hip),
                          ("crouch", self.cfg.robot.crouch_hip)):
            q0, gain = self.design(name, hip)
            self.q0[name] = q0
            self.gains[name] = gain
        self.env.gains = self.gains
        self.env.poses = self.leg_poses
        self.env.reset(*self.leg_poses["stand"])

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
        knee = self.env.balanced_k(hip)
        if knee is None:
            raise RuntimeError(f"could not find a balanced knee angle for hip={hip}")
        self.leg_poses[name] = (float(hip), float(knee))
        self.env.reset(hip, knee)
        q0 = self.d.qpos.copy()

        state_dim = 2 * self.m.nv + self.m.na
        transition = np.empty((state_dim, state_dim))
        control = np.empty((state_dim, self.m.nu))
        frictionloss = self.m.dof_frictionloss.copy()
        try:
            if self.cfg.lqr.zero_frictionloss_when_linearizing:
                self.m.dof_frictionloss[:] = 0.0
            mujoco.mj_forward(self.m, self.d)
            mujoco.mjd_transitionFD(
                self.m, self.d, self.cfg.lqr.fd_eps, True,
                transition, control, None, None,
            )
        finally:
            self.m.dof_frictionloss[:] = frictionloss
            mujoco.mj_forward(self.m, self.d)

        n = self.robot_nv
        robot_state = np.r_[np.arange(n), self.m.nv + np.arange(n)]
        a = transition[np.ix_(robot_state, robot_state)]
        b = control[np.ix_(robot_state, self.env.wact)]

        weights = np.full(2 * n, self.cfg.lqr.q_default)
        weights[0] = self.cfg.lqr.q_pos_x
        weights[4] = self.cfg.lqr.q_pitch
        weights[5] = self.cfg.lqr.q_yaw
        weights[n] = self.cfg.lqr.q_vel_x
        weights[n + 4] = self.cfg.lqr.q_pitch_rate
        weights[n + 5] = self.cfg.lqr.q_yaw_rate
        weights[np.asarray(self.env.wdofs, dtype=int)] = 0.0
        q = np.diag(weights)
        r = self.cfg.lqr.r_wheel * np.eye(len(self.env.wact))

        discount = np.sqrt(self.cfg.lqr.discount)
        a_discounted, b_discounted = discount * a, discount * b
        p = sla.solve_discrete_are(a_discounted, b_discounted, q, r)
        gain = np.linalg.solve(
            r + b_discounted.T @ p @ b_discounted,
            b_discounted.T @ p @ a_discounted,
        )

        if self.cfg.lqr.zero_abad_gains:
            for joint_name in (self.cfg.robot.legs[0], self.cfg.robot.legs[3]):
                joint = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                dof = int(self.m.jnt_dofadr[joint])
                gain[:, [dof, n + dof]] = 0.0
        if not np.all(np.isfinite(gain)):
            raise RuntimeError(f"non-finite LQR gain for {name}")
        return q0, gain

    def leg_targets(self, s: float) -> np.ndarray:
        """Targets for the 6 leg joints: hip and knee blended between standing and crouched by s (env.legs_of(h, k))."""
        self._require_designed()
        crouch = float(np.clip(s, 0.0, 1.0))
        stand = np.asarray(self.leg_poses["stand"])
        low = np.asarray(self.leg_poses["crouch"])
        hip, knee = (1.0 - crouch) * stand + crouch * low
        return np.asarray(self.env.legs_of(float(hip), float(knee)))

    def wheel_torques(self, qpos: np.ndarray, qvel: np.ndarray, ref: Reference) -> np.ndarray:
        """Torques for the 2 wheels (Nm), clipped to ±cfg.robot.wheel_torque_limit.

        qpos, qvel: the full state (true values or the state_estimator's estimate).
        Steps: see "Turning" a-e above; K and q0 are blended between the two sets by ref.s.
        """
        self._require_designed()
        position = np.asarray(qpos, dtype=float)
        velocity = np.asarray(qvel, dtype=float)
        if position.shape != (self.m.nq,) or velocity.shape != (self.m.nv,):
            raise ValueError(
                f"qpos/qvel must have shapes ({self.m.nq},)/({self.m.nv},), "
                f"got {position.shape}/{velocity.shape}"
            )

        crouch = float(np.clip(ref.s, 0.0, 1.0))
        gain = (1.0 - crouch) * self.gains["stand"] + crouch * self.gains["crouch"]
        q_reference = ((1.0 - crouch) * self.q0["stand"]
                       + crouch * self.q0["crouch"])
        root = self.env.root_qadr
        q_reference[root:root + 2] = ref.p
        yaw_quat = np.array([np.cos(ref.psi / 2), 0.0, 0.0, np.sin(ref.psi / 2)])
        base_quat = q_reference[root + 3:root + 7].copy()
        mujoco.mju_mulQuat(q_reference[root + 3:root + 7], yaw_quat, base_quat)

        dq = np.zeros(self.m.nv)
        mujoco.mj_differentiatePos(self.m, dq, 1.0, q_reference, position)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, position[root + 3:root + 7])
        rotation = rotation.reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        heading = np.array([np.cos(yaw), np.sin(yaw)])

        n = self.robot_nv
        position_error = dq[:n].copy()
        position_error[0] = heading @ position_error[:2]
        position_error[1] = 0.0
        for wheel_dof, (coefficient, y_offset) in zip(self.env.wdofs, self.env.roll):
            position_error[wheel_dof] = coefficient * (
                position_error[0] - position_error[5] * y_offset
            )

        velocity_error = velocity[:n].copy()
        velocity_error[0] = heading @ velocity_error[:2] - ref.v
        velocity_error[1] = 0.0
        velocity_error[5] -= ref.w
        for wheel_dof, (coefficient, y_offset) in zip(self.env.wdofs, self.env.roll):
            velocity_error[wheel_dof] -= coefficient * (ref.v - ref.w * y_offset)

        torque = -gain @ np.r_[position_error, velocity_error]
        limit = self.cfg.robot.wheel_torque_limit
        return np.clip(torque, -limit, limit)

    def _require_designed(self) -> None:
        if set(self.gains) != {"stand", "crouch"}:
            raise RuntimeError("call design_all() before using the LQR controller")
