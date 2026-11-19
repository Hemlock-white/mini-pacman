"""TITA (DDT) feasibility core: scene, poses, LQR balance, simulation.

Shared by the local prototype and the AUP notebook generator.
"""
import os
import numpy as np
import mujoco
import scipy.linalg as sla

LEGS = ['joint_left_leg_1', 'joint_left_leg_2', 'joint_left_leg_3',
        'joint_right_leg_1', 'joint_right_leg_2', 'joint_right_leg_3']
WHEELS = ['joint_left_leg_4', 'joint_right_leg_4']
WHEEL_MOTORS = ['left_leg_4', 'right_leg_4']
KP, KV = 600.0, 15.0       # leg position servos
R_WHEEL = 0.0925           # wheel cylinder radius in robot.xml
Z_OFF = -5e-4              # start slightly in contact so the linearization sees the ground


def write_scene(tita_dir, kp=KP, kv=KV):
    """Write feas_scene.xml next to robot.xml: floor, light, and position servos on the six leg joints."""
    acts = ''.join(f'    <position name="pos_{j}" joint="{j}" kp="{kp}" kv="{kv}" forcerange="-80 80"/>\n' for j in LEGS)
    x = (
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
        '</mujoco>\n')
    p = os.path.join(tita_dir, 'mujoco', 'feas_scene.xml')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(x)
    return p

class TitaEnv :
    def __init__(self, scene, default_contact=True, ignore_abad=True):
            m = mujoco.MjModel.from_xml_path(scene)
            m.opt.timestep = 0.002
            m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            if default_contact:
                # robot.xml asks for elliptic cones with impratio=100 (very stiff friction); MuJoCo's defaults
                # (pyramidal, impratio=1) give a linearization without spurious contact couplings
                m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
                m.opt.impratio = 1.0
            self.ignore_abad = ignore_abad
            self.m, self.d = m, mujoco.MjData(m)
            nid = lambda t, n: mujoco.mj_name2id(m, t, n)
            self.base = nid(mujoco.mjtObj.mjOBJ_BODY, 'base_link')
            self.wheel_bodies = [nid(mujoco.mjtObj.mjOBJ_BODY, b) for b in ['left_leg_4', 'right_leg_4']]
            self.qadr = {j: m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in LEGS + WHEELS}
            self.wdofs = [m.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in WHEELS]
            self.wact = [nid(mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in WHEEL_MOTORS]
            self.pact = [nid(mujoco.mjtObj.mjOBJ_ACTUATOR, 'pos_' + j) for j in LEGS]
            self.sign = (1, 1)  # right-leg (hip, knee) sign relative to the left leg; set by find_mirror()
            self.gains, self.poses = {}, {}
    
    def legs_of(self, h, k):
        sh, sk = self.sign
        return [0.0, h, k, 0.0, sh * h, sk * k]

    def set_pose(self, h, k, z=1.0, pitch=0.0):
        m, d = self.m, self.d
        d.qpos[:] = 0
        d.qpos[2] = z
        d.qpos[3:7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
        for j, v in zip(LEGS, self.legs_of(h, k)):
            d.qpos[self.qadr[j]] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return d.subtree_com[self.base].copy(), d.xpos[self.wheel_bodies[0]].copy(), d.xpos[self.wheel_bodies[1]].copy()

    def find_mirror(self, h=0.8, k=-1.6):
        best = None
        for sh in (1, -1):
            for sk in (1, -1):
                self.sign = (sh, sk)
                c, l, r = self.set_pose(h, k)
                err = abs(l[0] - r[0]) + abs(l[2] - r[2])
                if best is None or err < best[0]:
                    best = (err, (sh, sk))
        self.sign = best[1]
        return best

    def offset(self, h, k):
        c, l, r = self.set_pose(h, k)
        return c[0] - (l[0] + r[0]) / 2

    def balanced_k(self, h):
        ks = np.linspace(-2.65, -0.72, 120)
        v = [self.offset(h, k) for k in ks]
        for i in range(len(ks) - 1):
            if v[i] * v[i + 1] < 0:
                lo, hi = ks[i], ks[i + 1]
                for _ in range(40):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if self.offset(h, lo) * self.offset(h, mid) > 0 else (lo, mid)
                return lo
        return None

    def geometry(self, h, k):
        c, l, r = self.set_pose(h, k)
        wz = (l[2] + r[2]) / 2
        base_bottom = self.d.xpos[self.base][2] - 0.095
        return {'com_above_axle': c[2] - wz, 'base_clearance': base_bottom - (wz - R_WHEEL),
                'wheels_level': abs(l[2] - r[2]), 'wheel_x_minus_base_x': (l[0] + r[0]) / 2 - self.d.xpos[self.base][0]}

    def reset(self, h, k, pitch=0.0):
        m, d = self.m, self.d
        c, l, r = self.set_pose(h, k)
        z = 1.0 - ((l[2] + r[2]) / 2 - R_WHEEL) + Z_OFF
        self.set_pose(h, k, z=z, pitch=pitch)
        d.qvel[:] = 0
        d.ctrl[:] = 0
        d.xfrc_applied[:] = 0
        d.time = 0
        for a, v in zip(self.pact, self.legs_of(h, k)):
            d.ctrl[a] = v
        mujoco.mj_forward(m, d)
        # rolling relation per wheel: joint-angle change = coef * (forward travel - yaw * lateral offset)
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in WHEELS]
        self.roll = [(float(d.xaxis[j][1]) / R_WHEEL, float(d.xpos[b][1] - d.xpos[self.base][1]))
                     for j, b in zip(jids, self.wheel_bodies)]
        return d.qpos.copy()

    def pitch(self):
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, self.d.qpos[3:7])
        Rm = Rm.reshape(3, 3)
        return float(np.arctan2(Rm[0, 2], Rm[2, 2]))