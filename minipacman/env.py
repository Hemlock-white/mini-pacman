"""MuJoCo wrapper for the TITA robot used by Mini PAC-MAN.

The geometry helpers in this module come from the original TITA feasibility
prototype. :class:`TitaEnv` also provides the actuator and state interface
used by the LQR controller and the RL environment.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from .config import Config, default_config


LEGS = ["joint_left_leg_1", "joint_left_leg_2", "joint_left_leg_3",
        "joint_right_leg_1", "joint_right_leg_2", "joint_right_leg_3"]
WHEELS = ["joint_left_leg_4", "joint_right_leg_4"]
WHEEL_MOTORS = ["left_leg_4", "right_leg_4"]
KP, KV = 600.0, 15.0
R_WHEEL = 0.0925
Z_OFF = -5e-4


def write_scene(tita_dir, kp=KP, kv=KV):
    """Write the small robot-only feasibility scene next to ``robot.xml``."""
    acts = "".join(
        f'    <position name="pos_{joint}" joint="{joint}" kp="{kp}" kv="{kv}" forcerange="-80 80"/>\n'
        for joint in LEGS
    )
    xml = (
        '<mujoco model="tita feasibility">\n'
        '  <include file="robot.xml"/>\n'
        '  <visual><global offwidth="1280" offheight="800"/><quality shadowsize="4096"/></visual>\n'
        '  <asset>\n'
        '    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".82 .86 .92" width="64" height="64"/>\n'
        '    <texture name="grid" type="2d" builtin="checker" rgb1=".86 .86 .86" rgb2=".72 .72 .72" width="300" height="300"/>\n'
        '    <material name="grid" texture="grid" texrepeat="10 10"/>\n'
        '  </asset>\n'
        '  <worldbody>\n'
        '    <light pos="0 0 3" dir="0 0 -1" directional="true"/>\n'
        '    <geom name="floor" type="plane" size="30 30 0.1" material="grid"/>\n'
        '  </worldbody>\n'
        '  <actuator>\n' + acts + '  </actuator>\n'
        '</mujoco>\n'
    )
    path = os.path.join(tita_dir, "mujoco", "feas_scene.xml")
    with open(path, "w", encoding="utf-8") as file:
        file.write(xml)
    return path


class TitaEnv:
    """Thin owner of a TITA ``MjModel``/``MjData`` pair.

    ``scene`` can be either the robot-only feasibility scene or the full
    Mini PAC-MAN scene. Robot resets deliberately leave all ball qpos/qvel
    untouched in the latter.
    """

    def __init__(self, scene: str, default_contact: bool | None = None,
                 ignore_abad: bool = True, cfg: Config | None = None):
        self.cfg = cfg or default_config()
        robot_cfg, sim_cfg = self.cfg.robot, self.cfg.sim

        model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.timestep = sim_cfg.timestep
        model.opt.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{sim_cfg.integrator.upper()}")
        use_default_contact = sim_cfg.default_contact if default_contact is None else default_contact
        if use_default_contact:
            # The source XML uses a very stiff elliptic contact. The default
            # pyramidal contact is the configuration validated by the prototype.
            model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            model.opt.impratio = 1.0

        self.ignore_abad = ignore_abad
        self.m, self.d = model, mujoco.MjData(model)

        self.base = self._name_id(mujoco.mjtObj.mjOBJ_BODY, robot_cfg.base_body)
        root_joint = int(model.body_jntadr[self.base])
        if root_joint < 0 or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"{robot_cfg.base_body!r} must have a free root joint")
        self.root_qadr = int(model.jnt_qposadr[root_joint])
        self.root_dadr = int(model.jnt_dofadr[root_joint])

        # Balls are appended after the robot in scene.build_scene(). The same
        # wrapper remains usable with the robot-only feasibility scene.
        first_ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_0_free")
        self.robot_nq = int(model.jnt_qposadr[first_ball]) if first_ball >= 0 else model.nq
        self.robot_nv = int(model.jnt_dofadr[first_ball]) if first_ball >= 0 else model.nv

        self.wheel_bodies = [self._name_id(mujoco.mjtObj.mjOBJ_BODY, name)
                             for name in ("left_leg_4", "right_leg_4")]
        joint_names = tuple(robot_cfg.legs) + tuple(robot_cfg.wheels)
        joint_ids = [self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        self.qadr = {name: int(model.jnt_qposadr[joint])
                     for name, joint in zip(joint_names, joint_ids)}
        self.wdofs = [int(model.jnt_dofadr[joint])
                      for joint in joint_ids[-len(robot_cfg.wheels):]]
        self.wact = np.array([self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                              for name in robot_cfg.wheel_motors], dtype=int)
        self.pact = np.array([self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_" + name)
                              for name in robot_cfg.legs], dtype=int)

        self.sign = (1, 1)
        self.gains: dict = {}
        self.poses: dict = {}
        self.roll: list[tuple[float, float]] = []
        mujoco.mj_forward(self.m, self.d)

    def _name_id(self, obj_type, name: str) -> int:
        obj_id = int(mujoco.mj_name2id(self.m, obj_type, name))
        if obj_id < 0:
            raise ValueError(f"MuJoCo object {name!r} ({obj_type}) is missing from the scene")
        return obj_id

    @staticmethod
    def _vector(values, size: int, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector

    def legs_of(self, hip: float, knee: float) -> list[float]:
        """Return the six mirrored leg angles for a hip/knee pair."""
        sign_hip, sign_knee = self.sign
        return [0.0, hip, knee, 0.0, sign_hip * hip, sign_knee * knee]

    def set_leg_targets(self, targets) -> None:
        """Set the six position-servo targets in ``cfg.robot.legs`` order."""
        self.d.ctrl[self.pact] = self._vector(targets, len(self.pact), "leg targets")

    def set_wheel_torques(self, torques) -> None:
        """Set and safety-clip the left/right wheel motor torques."""
        values = self._vector(torques, len(self.wact), "wheel torques")
        limit = self.cfg.robot.wheel_torque_limit
        self.d.ctrl[self.wact] = np.clip(values, -limit, limit)

    def step(self, steps: int = 1) -> None:
        """Advance the physics simulation by ``steps`` fixed time steps."""
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for _ in range(int(steps)):
            mujoco.mj_step(self.m, self.d)

    def set_pose(self, hip: float, knee: float, z: float = 1.0,
                 pitch: float = 0.0):
        """Set the robot pose while preserving every non-robot free joint."""
        data = self.d
        data.qpos[:self.robot_nq] = 0.0
        qadr = self.root_qadr
        data.qpos[qadr + 2] = z
        data.qpos[qadr + 3:qadr + 7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
        for joint, value in zip(self.cfg.robot.legs, self.legs_of(hip, knee)):
            data.qpos[self.qadr[joint]] = value
        mujoco.mj_kinematics(self.m, data)
        mujoco.mj_comPos(self.m, data)
        return (data.subtree_com[self.base].copy(),
                data.xpos[self.wheel_bodies[0]].copy(),
                data.xpos[self.wheel_bodies[1]].copy())

    def find_mirror(self, hip: float = 0.8, knee: float = -1.6):
        """Find the right-leg signs that mirror the left leg geometry."""
        best = None
        for sign_hip in (1, -1):
            for sign_knee in (1, -1):
                self.sign = (sign_hip, sign_knee)
                _, left, right = self.set_pose(hip, knee)
                error = abs(left[0] - right[0]) + abs(left[2] - right[2])
                if best is None or error < best[0]:
                    best = (error, (sign_hip, sign_knee))
        self.sign = best[1]
        return best

    def offset(self, hip: float, knee: float) -> float:
        center, left, right = self.set_pose(hip, knee)
        return float(center[0] - (left[0] + right[0]) / 2)

    def balanced_k(self, hip: float) -> float | None:
        """Find a knee angle that places the COM above the wheel axle."""
        knees = np.linspace(-2.65, -0.72, 120)
        offsets = [self.offset(hip, knee) for knee in knees]
        for index in range(len(knees) - 1):
            if offsets[index] * offsets[index + 1] < 0:
                low, high = knees[index], knees[index + 1]
                for _ in range(40):
                    mid = (low + high) / 2
                    if self.offset(hip, low) * self.offset(hip, mid) > 0:
                        low = mid
                    else:
                        high = mid
                return float(low)
        return None

    def geometry(self, hip: float, knee: float) -> dict[str, float]:
        center, left, right = self.set_pose(hip, knee)
        wheel_z = (left[2] + right[2]) / 2
        base_bottom = self.d.xpos[self.base][2] - 0.095
        return {
            "com_above_axle": float(center[2] - wheel_z),
            "base_clearance": float(base_bottom - (wheel_z - self.cfg.robot.wheel_radius)),
            "wheels_level": float(abs(left[2] - right[2])),
            "wheel_x_minus_base_x": float((left[0] + right[0]) / 2 - self.d.xpos[self.base][0]),
        }

    def reset(self, hip: float, knee: float, pitch: float = 0.0) -> np.ndarray:
        """Reset the robot and controls, preserving ball positions/velocities."""
        model, data = self.m, self.d
        _, left, right = self.set_pose(hip, knee)
        z = (1.0 - ((left[2] + right[2]) / 2 - self.cfg.robot.wheel_radius)
             + self.cfg.robot.z_offset)
        self.set_pose(hip, knee, z=z, pitch=pitch)
        data.qvel[:self.robot_nv] = 0.0
        data.ctrl[:] = 0.0
        data.xfrc_applied[:] = 0.0
        data.time = 0.0
        self.set_leg_targets(self.legs_of(hip, knee))
        mujoco.mj_forward(model, data)

        wheel_joint_ids = [self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                           for name in self.cfg.robot.wheels]
        radius = self.cfg.robot.wheel_radius
        self.roll = [
            (float(data.xaxis[joint][1]) / radius,
             float(data.xpos[body][1] - data.xpos[self.base][1]))
            for joint, body in zip(wheel_joint_ids, self.wheel_bodies)
        ]
        return data.qpos.copy()

    def planar_state(self, s: float = 0.0):
        """Return the true base planar state for the policy and safety filter."""
        from .cbf import RobotPlanarState

        rotation = self.d.xmat[self.base].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 0
        )
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        forward_speed = float(velocity[3:5] @ heading)
        return RobotPlanarState(
            xy=self.d.xpos[self.base, :2].copy(),
            yaw=yaw,
            v=forward_speed,
            w=float(velocity[2]),
            s=float(np.clip(s, 0.0, 1.0)),
        )

    def pitch(self) -> float:
        """Return the base pitch angle in radians."""
        rotation = self.d.xmat[self.base].reshape(3, 3)
        return float(np.arctan2(rotation[0, 2], rotation[2, 2]))
