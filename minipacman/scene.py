"""產生 Mini PAC-MAN 的 MuJoCo 場景，並查出常用的 id。

場景 = TITA (robot.xml) + 地板 + 燈光 + 腿部位置伺服 + 一池球 (ball_0 ... ball_{N-1})。

* IMU 和 encoder：robot.xml 本來就有（trunk_gyro、trunk_accel、trunk_quat、*_pos、*_vel），不用另外加。
* 深度相機：不需要 XML 相機。depth_camera.py 用 TITA 真實雙目相機的位置（left_camera / right_camara
  兩個 body 的中點）算出相機姿態，再用光線投射算深度。
* 球：每顆球一個 freejoint。球的 dof 一定排在機器人之後，所以機器人的 dof 是 0 .. robot_nv-1。
  **LQR 線性化只能用機器人的 dof**（把球放進 DARE 會解不出來），見 SceneIds.robot_nv。

和 env.py 的關係
----------------
env.py 原本的 write_scene() 只有機器人和地板。之後請把 TitaEnv 改成載入這裡產生的場景：

    from minipacman.scene import build_scene
    env = TitaEnv(str(build_scene()))

球的參數（半徑、質量、彈性）在每次發球時由 balls.py 在執行期間修改，所以 XML 裡的值只是初始值。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .config import Config, TITA_DIR, default_config


def build_scene(tita_dir: str | Path | None = None, cfg: Config | None = None,
                filename: str = "mini_pacman_scene.xml") -> Path:
    """在 <tita_dir>/mujoco/ 寫出場景 XML，回傳檔案路徑。

    放在 robot.xml 旁邊，是因為 robot.xml 用相對路徑找 mesh。
    """
    cfg = cfg or default_config()
    tita_dir = Path(tita_dir) if tita_dir is not None else TITA_DIR
    robot_xml = tita_dir / "mujoco" / "robot.xml"
    if not robot_xml.exists():
        raise FileNotFoundError(f"找不到 {robot_xml}。請先執行 python scripts/download_tita.py，"
                                f"或設定環境變數 MINIPACMAN_TITA_DIR。")
    rc, bc = cfg.robot, cfg.balls
    acts = "".join(
        f'    <position name="pos_{j}" joint="{j}" kp="{rc.leg_kp}" kv="{rc.leg_kv}" '
        f'forcerange="-{rc.leg_force_limit} {rc.leg_force_limit}"/>\n' for j in rc.legs)
    r0 = bc.radius_range[1]
    m0 = float(np.mean(bc.mass_range))
    f = " ".join(str(x) for x in bc.friction)
    balls = []
    colors = ["0.95 0.45 0.10 1", "0.20 0.55 0.95 1", "0.95 0.80 0.10 1", "0.85 0.20 0.55 1", "0.30 0.75 0.35 1"]
    for i in range(bc.pool_size):
        px, py = park_position(cfg, i, r0)[:2]
        balls.append(
            f'    <body name="ball_{i}" pos="{px} {py} {r0}">\n'
            f'      <freejoint name="ball_{i}_free"/>\n'
            f'      <geom name="ball_{i}" type="sphere" size="{r0}" mass="{m0}" priority="1" condim="3" '
            f'friction="{f}" solref="{-bc.contact_stiffness} -50" rgba="{colors[i % len(colors)]}"/>\n'
            f'    </body>\n')
    xml = (
        '<mujoco model="mini pac-man">\n'
        '  <include file="robot.xml"/>\n'
        '  <visual><global offwidth="1280" offheight="800"/><quality shadowsize="4096"/></visual>\n'
        '  <asset>\n'
        '    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".82 .86 .92" width="64" height="64"/>\n'
        '    <texture name="grid" type="2d" builtin="checker" rgb1=".86 .86 .86" rgb2=".72 .72 .72" '
        'width="300" height="300"/>\n'
        '    <material name="grid" texture="grid" texrepeat="10 10"/>\n'
        '  </asset>\n'
        '  <worldbody>\n'
        '    <light pos="0 0 3" dir="0 0 -1" directional="true"/>\n'
        '    <geom name="floor" type="plane" size="30 30 0.1" material="grid"/>\n'
        + "".join(balls) +
        '  </worldbody>\n'
        '  <actuator>\n' + acts + '  </actuator>\n'
        '</mujoco>\n')
    path = tita_dir / "mujoco" / filename
    path.write_text(xml, encoding="utf-8")
    return path


def park_position(cfg: Config, i: int, radius: float) -> np.ndarray:
    """第 i 顆球收起來時放的位置（地板上，離相機很遠，彼此不相碰）。"""
    x, y = cfg.balls.park_xy
    return np.array([x - 1.0 * i, y, radius])


def apply_sim_options(m: mujoco.MjModel, cfg: Config | None = None) -> None:
    """套用 config 裡的模擬設定（和 env.py TitaEnv.__init__ 做的事一樣）。"""
    cfg = cfg or default_config()
    m.opt.timestep = cfg.sim.timestep
    m.opt.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{cfg.sim.integrator.upper()}")
    if cfg.sim.default_contact:
        m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        m.opt.impratio = 1.0


def load_model(tita_dir: str | Path | None = None, cfg: Config | None = None):
    """產生場景並載入，回傳 (MjModel, MjData)。測試和工具用；正式流程由 env.py 載入。"""
    cfg = cfg or default_config()
    m = mujoco.MjModel.from_xml_path(str(build_scene(tita_dir, cfg)))
    apply_sim_options(m, cfg)
    return m, mujoco.MjData(m)


@dataclass
class SceneIds:
    """場景裡常用的 id。用 scene_ids(m) 建立。"""
    base_body: int
    robot_root: int
    robot_bodies: np.ndarray        # 屬於機器人的所有 body
    robot_collision_geoms: np.ndarray   # 機器人會碰撞的 geom（contype 或 conaffinity 不為 0）
    robot_nv: int                   # 機器人的 dof 數（球的 dof 在後面）
    robot_nq: int
    floor_geom: int
    ball_bodies: np.ndarray
    ball_geoms: np.ndarray
    ball_qadr: np.ndarray           # 每顆球 freejoint 的 qpos 起點（7 個：xyz + quat）
    ball_dadr: np.ndarray           # 每顆球 freejoint 的 qvel 起點（6 個：線速度 + 角速度）
    camera_bodies: np.ndarray


def scene_ids(m: mujoco.MjModel, cfg: Config | None = None) -> SceneIds:
    cfg = cfg or default_config()
    name = lambda t, n: mujoco.mj_name2id(m, t, n)
    B, G, J = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_JOINT
    base = name(B, cfg.robot.base_body)
    root = int(m.body_rootid[base])
    robot_bodies = np.where(m.body_rootid == root)[0]
    robot_geoms = np.where(np.isin(m.geom_bodyid, robot_bodies)
                           & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))[0]
    n_balls = cfg.balls.pool_size
    ball_bodies = np.array([name(B, f"ball_{i}") for i in range(n_balls)])
    ball_geoms = np.array([name(G, f"ball_{i}") for i in range(n_balls)])
    ball_joints = [name(J, f"ball_{i}_free") for i in range(n_balls)]
    if min(ball_bodies.min(), ball_geoms.min(), min(ball_joints)) < 0:
        raise ValueError("場景裡找不到球，請用 scene.build_scene() 產生的場景。")
    ball_qadr = np.array([m.jnt_qposadr[j] for j in ball_joints])
    ball_dadr = np.array([m.jnt_dofadr[j] for j in ball_joints])
    robot_nv, robot_nq = int(ball_dadr.min()), int(ball_qadr.min())
    robot_dofs = np.where(np.isin(m.dof_bodyid, robot_bodies))[0]
    assert robot_dofs.max() < robot_nv, "機器人的 dof 必須排在球之前"
    return SceneIds(
        base_body=base, robot_root=root, robot_bodies=robot_bodies, robot_collision_geoms=robot_geoms,
        robot_nv=robot_nv, robot_nq=robot_nq, floor_geom=name(G, "floor"),
        ball_bodies=ball_bodies, ball_geoms=ball_geoms, ball_qadr=ball_qadr, ball_dadr=ball_dadr,
        camera_bodies=np.array([name(B, b) for b in cfg.robot.camera_bodies]))
