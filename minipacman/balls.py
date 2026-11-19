"""球：發球器、彈跳物理、軌跡預測、碰撞判定。

主要內容
--------
* ``BallLauncher``：每個 episode 丟 1–5 顆球，從機器人正面（目前朝向 ±25 度）丟過來，
  半徑、質量、彈性、速度、高度都隨機。軌跡分兩種：
    - ``direct``：直接飛到機器人（途中不碰地）。
    - ``bounce``：先在機器人前方落地彈起，再飛到機器人。
  另外負責：記錄每顆球有沒有打到機器人、最小距離，以及把飛過去的球收回。
* ``predict_ball``：解析解預測球的未來位置（含落地彈跳），給 CBF 過濾器和追蹤器用。
* ``set_ball_properties``：執行期間改球的半徑、質量、彈性。

彈跳怎麼做
----------
MuJoCo 的接觸是「軟」的，像彈簧加阻尼。球的接觸參數用 solref = (-剛度, -阻尼)：剛度固定
(BallConfig.contact_stiffness)，阻尼越小彈越高（阻尼 0 時 e 約 0.99）。
阻尼和彈性係數 e（反彈速度 / 撞擊速度）的對應不是簡單公式，所以 ``restitution_table()`` 會在一個
只有球和地板的小場景裡做落球測試，量出對應表，之後內插查表。實測 e 和球的質量無關，
撞擊速度 1–6 m/s 之間只差約 0.05。
（MuJoCo 預設的 solref = (時間常數, 阻尼比) 寫法：阻尼比調小時剛度會變成 1/阻尼比^2 倍，
2 ms 步長下數值不穩，會量到 e > 1，所以不用。）
球的 geom 設 priority=1，所以球撞地板、撞機器人都用球自己的接觸參數。

為什麼發球時要給球「上旋」
--------------------------
真實的球落地時，如果底部在地上滑動，摩擦力會讓水平速度變慢，彈跳點後的軌跡就很難預測。
這裡發球時給球剛好「滾動不滑」的轉速 w = (z × v_水平) / r，落地瞬間接觸點速度為 0，
水平速度不變，所以 ``predict_ball`` 用「水平等速、垂直乘 -e」就能準確預測。

時間尺度
--------
``check_contacts()`` 要每個物理步呼叫（接觸只持續幾毫秒，漏掉就少算一次命中）；
``update()`` 在 policy 步呼叫（發球、收球、更新最小距離）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import BallConfig, Config, default_config
from .scene import SceneIds, park_position, scene_ids

G = 9.81
ROLL_SPEED = 0.3          # 反彈的垂直速度小於這個就當作在地上滾


# --------------------------------------------------------------------------------------------
# 彈性：阻尼比 <-> 彈性係數
# --------------------------------------------------------------------------------------------
_RESTITUTION_CACHE: dict = {}


def restitution_table(timestep: float, stiffness: float, integrator: int,
                      impact_speed: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """在「球 + 地板」的小場景做落球測試，回傳 (阻尼陣列, 量到的彈性係數陣列)。

    結果會快取，同一組參數只算一次（約 0.1 秒）。
    """
    key = (round(timestep, 6), float(stiffness), int(integrator), float(impact_speed))
    if key in _RESTITUTION_CACHE:
        return _RESTITUTION_CACHE[key]
    w0 = np.sqrt(stiffness)
    dampings = 2.0 * w0 * np.concatenate([[0.0], np.geomspace(0.005, 1.0, 29)])
    es = np.array([_drop_test(b, timestep, stiffness, integrator, impact_speed) for b in dampings])
    _RESTITUTION_CACHE[key] = (dampings, es)
    return dampings, es


def _drop_test(damping: float, timestep: float, stiffness: float, integrator: int, impact_speed: float) -> float:
    r = 0.1
    z0 = r + impact_speed ** 2 / (2 * G)
    xml = (f'<mujoco><option timestep="{timestep}" cone="pyramidal" impratio="1"/>'
           f'<worldbody><geom type="plane" size="1 1 0.1"/>'
           f'<body pos="0 0 {z0}"><freejoint/><geom type="sphere" size="{r}" mass="0.15" priority="1" '
           f'condim="3" solref="{-stiffness} {-damping}"/></body></worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.integrator = integrator
    d = mujoco.MjData(m)
    v_in, prev = None, 0.0
    for _ in range(int(3.0 / timestep)):
        mujoco.mj_step(m, d)
        if v_in is None and d.ncon > 0:
            v_in = -prev
        elif v_in is not None and d.ncon == 0:
            return float(max(d.qvel[2], 0.0) / v_in)
        prev = d.qvel[2]
    return 0.0


def damping_for_restitution(e: float, timestep: float, stiffness: float, integrator: int,
                            impact_speed: float = 4.0) -> float:
    """查表：要得到彈性係數 e，solref 的阻尼要設多少。e 超出可達範圍時取最接近的值。"""
    dampings, es = restitution_table(timestep, stiffness, integrator, impact_speed)
    order = np.argsort(es)                     # np.interp 需要遞增的 x
    return float(np.interp(e, es[order], dampings[order]))


def set_ball_properties(m: mujoco.MjModel, ids: SceneIds, i: int, radius: float, mass: float,
                        restitution: float, cfg: BallConfig) -> None:
    """執行期間修改第 i 顆球的半徑、質量、慣性、彈性。

    除了 geom_size，也要更新碰撞偵測用的外接範圍 (geom_rbound、geom_aabb)，以及質量相關的常數
    (body_subtreemass、invweight0)，否則接觸會用舊的尺寸或剛性。
    """
    g, b = int(ids.ball_geoms[i]), int(ids.ball_bodies[i])
    m.geom_size[g, 0] = radius
    m.geom_rbound[g] = radius
    if hasattr(m, "geom_aabb"):
        m.geom_aabb[g] = [0.0, 0.0, 0.0, radius, radius, radius]
    k = 2.0 / 3.0 if cfg.hollow else 2.0 / 5.0
    inertia = k * mass * radius ** 2
    dm = mass - float(m.body_mass[b])
    m.body_mass[b] = mass
    m.body_inertia[b] = inertia
    m.body_subtreemass[b] = mass
    m.body_subtreemass[0] += dm
    m.body_invweight0[b] = [1.0 / mass, 1.0 / inertia]
    da = int(ids.ball_dadr[i])
    m.dof_invweight0[da:da + 3] = 1.0 / mass
    m.dof_invweight0[da + 3:da + 6] = 1.0 / inertia
    m.dof_M0[da:da + 3] = mass
    m.dof_M0[da + 3:da + 6] = inertia
    b = damping_for_restitution(restitution, m.opt.timestep, cfg.contact_stiffness, int(m.opt.integrator),
                                cfg.calibration_impact_speed)
    m.geom_solref[g] = [-cfg.contact_stiffness, -b]


# --------------------------------------------------------------------------------------------
# 軌跡預測（含彈跳）
# --------------------------------------------------------------------------------------------
def bounce_contact_time(stiffness: float | None = None) -> float:
    """每次落地，軟接觸大約持續半個振盪週期 π/sqrt(剛度)（剛度 50000 時約 14 ms = 7 個物理步）。"""
    if stiffness is None:
        stiffness = default_config().balls.contact_stiffness
    return float(np.pi / np.sqrt(stiffness))


def predict_ball(p: np.ndarray, v: np.ndarray, radius: float, restitution: float,
                 times: np.ndarray, max_bounces: int = 6,
                 contact_time: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """預測球在 times（相對現在的秒數，可以是陣列）時的位置和速度。

    假設：只有重力、落地時水平速度不變、垂直速度乘 -e（發球時有給滾動轉速，見檔頭說明）；
    每次落地在地面停留 contact_time 秒（預設由接觸剛度算，見 bounce_contact_time）；
    反彈垂直速度 < 0.3 m/s 之後就在地上滾。不考慮撞到機器人。
    實測（e=0.5–0.8，從 1.2 m 落下）：前 1 秒誤差 < 8 cm，2 秒內（彈 3 次以上）< 15 cm。
    誤差主要來自低速撞擊時實際的 e 和名目值略有不同，彈越多次累積越多。
    回傳 (positions (n,3), velocities (n,3))。
    """
    if contact_time is None:
        contact_time = bounce_contact_time()
    times = np.atleast_1d(np.asarray(times, dtype=float))
    pos = np.zeros((len(times), 3))
    vel = np.zeros((len(times), 3))
    p = np.array(p, dtype=float)
    v = np.array(v, dtype=float)
    t0 = 0.0
    rolling = p[2] <= radius + 1e-6 and abs(v[2]) < ROLL_SPEED
    remaining = np.ones(len(times), bool)
    for _ in range(max_bounces + 1):
        if rolling:
            t_end = np.inf
        else:
            dz = max(p[2] - radius, 0.0)
            t_end = (v[2] + np.sqrt(v[2] ** 2 + 2 * G * dz)) / G      # 下一次落地的時間
        sel = remaining & (times < t0 + t_end)
        tau = times[sel] - t0
        pos[sel, 0:2] = p[0:2] + np.outer(tau, v[0:2])
        vel[sel, 0:2] = v[0:2]
        if rolling:
            pos[sel, 2] = radius
            vel[sel, 2] = 0.0
        else:
            pos[sel, 2] = p[2] + v[2] * tau - 0.5 * G * tau ** 2
            vel[sel, 2] = v[2] - G * tau
        remaining &= ~sel
        if not remaining.any():
            break
        # 落地、在地面停留 contact_time、反彈
        p = np.array([p[0] + v[0] * t_end, p[1] + v[1] * t_end, radius])
        vz_after = restitution * abs(v[2] - G * t_end)
        rolling = vz_after < ROLL_SPEED
        t0 += t_end
        if not rolling and contact_time > 0:
            sel = remaining & (times < t0 + contact_time)
            tau = times[sel] - t0
            pos[sel, 0:2] = p[0:2] + np.outer(tau, v[0:2])
            pos[sel, 2] = radius
            vel[sel, 0:2] = v[0:2]
            vel[sel, 2] = 0.0
            remaining &= ~sel
            p = np.array([p[0] + v[0] * contact_time, p[1] + v[1] * contact_time, radius])
            t0 += contact_time
        v = np.array([v[0], v[1], 0.0 if rolling else vz_after])
    if remaining.any():                       # 超過最大彈跳次數：當作在地上滾
        tau = times[remaining] - t0
        pos[remaining, 0:2] = p[0:2] + np.outer(tau, v[0:2])
        pos[remaining, 2] = radius
        vel[remaining, 0:2] = v[0:2]
    return pos, vel


# --------------------------------------------------------------------------------------------
# 發球規劃
# --------------------------------------------------------------------------------------------
@dataclass
class ThrowPlan:
    """一次發球的完整規劃（世界座標）。"""
    p0: np.ndarray            # 發射位置
    v0: np.ndarray            # 發射速度
    w0: np.ndarray            # 發射角速度（滾動不滑的上旋）
    radius: float
    mass: float
    restitution: float
    mode: str                 # "direct" 或 "bounce"
    flight_time: float        # 預計多久後抵達目標
    target: np.ndarray        # 預計抵達的位置（球心）
    bounce_point: np.ndarray | None = None

    @property
    def horizontal_speed(self) -> float:
        return float(np.linalg.norm(self.v0[0:2]))


def plan_throw(rng: np.random.Generator, cfg: BallConfig, robot_xy: np.ndarray, robot_yaw: float,
               robot_vxy: np.ndarray, radius: float, mass: float, restitution: float,
               mode: str | None = None, max_tries: int = 50) -> ThrowPlan:
    """規劃一顆從機器人正面丟過來的球。

    1. 發射點：機器人前方 D 公尺、朝向 ±launch_angle_deg 度內。
    2. 目標：機器人現在的位置 + 速度 × 飛行時間（預判）+ 瞄準雜訊；抵達高度隨機。
    3. 飛行時間 = 水平距離 / 水平速度（水平速度隨機）。
    4. direct：給定發射高度，解出垂直初速。拋物線開口向下，所以兩端都高於地面就保證途中不碰地。
       bounce：給定彈跳點（發射點到目標之間的比例位置），倒推：
         第二段（彈跳點 → 目標）要多少起跳速度 → 除以 e 得到撞地速度 → 第一段的初速和發射高度。
         發射高度超出範圍就重抽。
    """
    mode_given = mode
    for _ in range(max_tries):
        D = rng.uniform(*cfg.launch_distance_range)
        phi = np.deg2rad(rng.uniform(-cfg.launch_angle_deg, cfg.launch_angle_deg))
        p0_xy = np.asarray(robot_xy) + D * np.array([np.cos(robot_yaw + phi), np.sin(robot_yaw + phi)])
        vh = rng.uniform(*cfg.horizontal_speed_range)
        target_xy = np.array(robot_xy, dtype=float)
        T = np.linalg.norm(target_xy - p0_xy) / vh
        if cfg.lead_target:
            for _ in range(3):                      # 預判位置會改變飛行時間，迭代幾次
                target_xy = np.asarray(robot_xy) + np.asarray(robot_vxy) * T
                T = np.linalg.norm(target_xy - p0_xy) / vh
        target_xy = target_xy + rng.normal(0.0, cfg.aim_noise, 2)
        T = np.linalg.norm(target_xy - p0_xy) / vh
        z_t = max(rng.uniform(*cfg.arrival_height_range), radius + 0.01)
        m = mode_given or ("bounce" if rng.random() < cfg.bounce_fraction else "direct")
        v_h = (target_xy - p0_xy) / T
        bounce_point = None
        if m == "direct":
            z0 = max(rng.uniform(*cfg.direct_launch_height_range), radius + 0.01)
            vz0 = (z_t - z0) / T + 0.5 * G * T
        else:
            lam = rng.uniform(*cfg.bounce_point_range)
            T1, T2 = lam * T, (1.0 - lam) * T
            vz_after = (z_t - radius + 0.5 * G * T2 ** 2) / T2
            vz_impact = -vz_after / restitution
            vz0 = vz_impact + G * T1
            z0 = radius - vz0 * T1 + 0.5 * G * T1 ** 2
            if not (radius + 0.05 <= z0 <= cfg.bounce_launch_height_max):
                continue
            bounce_point = np.array([*(p0_xy + v_h * T1), radius])
        v0 = np.array([v_h[0], v_h[1], vz0])
        w0 = np.array([-v_h[1], v_h[0], 0.0]) / radius
        return ThrowPlan(p0=np.array([*p0_xy, z0]), v0=v0, w0=w0, radius=radius, mass=mass,
                         restitution=restitution, mode=m, flight_time=float(T),
                         target=np.array([*target_xy, z_t]), bounce_point=bounce_point)
    if mode_given is None:
        return plan_throw(rng, cfg, robot_xy, robot_yaw, robot_vxy, radius, mass, restitution, "direct")
    raise RuntimeError("plan_throw：找不到可行的發球，請檢查 BallConfig 的範圍")


# --------------------------------------------------------------------------------------------
# 發球器
# --------------------------------------------------------------------------------------------
@dataclass
class BallRecord:
    """這個 episode 丟出的一顆球。"""
    index: int                         # 用第幾顆球 (ball_i)
    throw_time: float                  # 預定發射時間
    radius: float
    mass: float
    restitution: float
    plan: ThrowPlan | None = None      # 發射時才規劃
    launched: bool = False
    active: bool = False               # 在場上（發射後、收回前）
    hit: bool = False
    hit_time: float | None = None
    hit_geom: str | None = None
    min_clearance: float = np.inf
    retired_time: float | None = None

    def to_dict(self) -> dict:
        """存進結果檔的欄位（report.py 用這些欄位做分組統計）。"""
        p = self.plan
        return {
            "index": self.index, "throw_time": self.throw_time, "launched": self.launched,
            "mode": p.mode if p else None,
            "arrival_height": float(p.target[2]) if p else None,
            "horizontal_speed": p.horizontal_speed if p else None,
            "flight_time": p.flight_time if p else None,
            "radius": self.radius, "mass": self.mass, "restitution": self.restitution,
            "hit": self.hit, "hit_time": self.hit_time, "hit_geom": self.hit_geom,
            "min_clearance": None if not np.isfinite(self.min_clearance) else float(self.min_clearance),
        }


class BallLauncher:
    """管理場景裡的球池：排程、發射、命中判定、最小距離、收回。

    典型用法（在 dodge_env.py 裡）::

        launcher = BallLauncher(m, d, cfg)
        launcher.reset(rng)                     # 每個 episode 開始
        for 每個 policy 步:
            launcher.update(d.time)             # 發球、收球、更新最小距離
            for 每個物理步:
                mujoco.mj_step(m, d)
                launcher.check_contacts()       # 命中判定
        records = launcher.episode_records()    # 存進結果
    """

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, cfg: Config | None = None,
                 ids: SceneIds | None = None):
        self.m, self.d = m, d
        self.cfg = cfg or default_config()
        self.bc = self.cfg.balls
        self.ids = ids or scene_ids(m, self.cfg)
        self.records: list[BallRecord] = []
        self._ball_to_record: dict[int, BallRecord] = {}
        self._geom_to_ball = {int(g): i for i, g in enumerate(self.ids.ball_geoms)}
        self._robot_geoms = set(int(g) for g in self.ids.robot_collision_geoms)
        self._fromto = np.zeros(6)
        # 先建好彈性對照表，避免第一次發球時才花時間
        restitution_table(m.opt.timestep, self.bc.contact_stiffness, int(m.opt.integrator),
                          self.bc.calibration_impact_speed)

    # ---------------- episode ----------------
    def reset(self, rng: np.random.Generator) -> None:
        """收回所有球，排好這個 episode 要丟的球（數量、時間、大小、重量、彈性）。"""
        self.rng = rng
        for i in range(self.bc.pool_size):
            self.park(i)
        lo, hi = self.bc.balls_per_episode
        n = int(rng.integers(lo, hi + 1))
        n = min(n, self.bc.pool_size)
        t = rng.uniform(*self.bc.first_throw_range)
        self.records = []
        for i in range(n):
            self.records.append(BallRecord(
                index=i, throw_time=float(t), radius=float(rng.uniform(*self.bc.radius_range)),
                mass=float(rng.uniform(*self.bc.mass_range)),
                restitution=float(rng.uniform(*self.bc.restitution_range))))
            t += rng.uniform(*self.bc.throw_interval_range)
        self._ball_to_record = {}

    def update(self, t: float) -> list[BallRecord]:
        """policy 步呼叫：發射到時間的球、收回飛走的球、更新最小距離。回傳這一步新發射的球。"""
        launched = []
        for rec in self.records:
            if not rec.launched and t >= rec.throw_time:
                self._launch(rec)
                launched.append(rec)
        self.update_clearance()
        for rec in self.active_records():
            if self._should_retire(rec, t):
                self.park(rec.index)
                rec.active = False
                rec.retired_time = t
        return launched

    def check_contacts(self) -> list[BallRecord]:
        """物理步呼叫：檢查球和機器人的接觸，第一次碰到就記為命中。回傳這一步新命中的球。"""
        new_hits = []
        d = self.d
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self._geom_to_ball and g2 in self._robot_geoms:
                ball, robot_geom = self._geom_to_ball[g1], g2
            elif g2 in self._geom_to_ball and g1 in self._robot_geoms:
                ball, robot_geom = self._geom_to_ball[g2], g1
            else:
                continue
            rec = self._ball_to_record.get(ball)
            if rec is not None and rec.active and not rec.hit:
                rec.hit, rec.hit_time = True, float(d.time)
                rec.hit_geom = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
                rec.min_clearance = min(rec.min_clearance, 0.0)
                new_hits.append(rec)
        return new_hits

    def update_clearance(self, distmax: float = 3.0) -> None:
        """更新每顆在場上的球和機器人碰撞形狀的最小距離（負數 = 穿透）。"""
        for rec in self.active_records():
            dist = self.clearance(rec.index, distmax)
            rec.min_clearance = min(rec.min_clearance, dist)

    def clearance(self, i: int, distmax: float = 3.0) -> float:
        """第 i 顆球到機器人任一碰撞形狀的最短距離（超過 distmax 回傳 distmax）。"""
        g_ball = int(self.ids.ball_geoms[i])
        return min(mujoco.mj_geomDistance(self.m, self.d, int(g), g_ball, distmax, self._fromto)
                   for g in self.ids.robot_collision_geoms)

    # ---------------- 查詢 ----------------
    def active_records(self) -> list[BallRecord]:
        return [r for r in self.records if r.active]

    def ball_state(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """第 i 顆球的 (位置, 線速度)，世界座標。"""
        qa, da = int(self.ids.ball_qadr[i]), int(self.ids.ball_dadr[i])
        return self.d.qpos[qa:qa + 3].copy(), self.d.qvel[da:da + 3].copy()

    def episode_records(self) -> list[dict]:
        return [r.to_dict() for r in self.records]

    def robot_planar_state(self) -> tuple[np.ndarray, float, np.ndarray]:
        """機器人底座的 (xy 位置, 朝向 yaw, xy 速度)，世界座標（發球器用真實值，它是「環境」的一部分）。"""
        d, b = self.d, self.ids.base_body
        R = d.xmat[b].reshape(3, 3)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        return d.xpos[b][0:2].copy(), float(np.arctan2(R[1, 0], R[0, 0])), vel[3:5].copy()

    # ---------------- 內部 ----------------
    def _launch(self, rec: BallRecord) -> None:
        xy, yaw, vxy = self.robot_planar_state()
        plan = plan_throw(self.rng, self.bc, xy, yaw, vxy, rec.radius, rec.mass, rec.restitution)
        self.launch(rec.index, plan)
        rec.plan, rec.launched, rec.active = plan, True, True
        self._ball_to_record[rec.index] = rec

    def throw_now(self, plan: ThrowPlan, index: int = 0) -> BallRecord:
        """立刻用第 index 顆球執行 plan（測試、示範影片用；正式流程由 reset/update 排程）。"""
        rec = BallRecord(index=index, throw_time=float(self.d.time), radius=plan.radius, mass=plan.mass,
                         restitution=plan.restitution, plan=plan, launched=True, active=True)
        self.launch(index, plan)
        self.records = [r for r in self.records if r.index != index] + [rec]
        self._ball_to_record[index] = rec
        return rec

    def launch(self, i: int, plan: ThrowPlan) -> None:
        """把第 i 顆球設成 plan 的初始狀態（也可以在測試裡直接呼叫）。"""
        set_ball_properties(self.m, self.ids, i, plan.radius, plan.mass, plan.restitution, self.bc)
        qa, da = int(self.ids.ball_qadr[i]), int(self.ids.ball_dadr[i])
        self.d.qpos[qa:qa + 7] = [*plan.p0, 1.0, 0.0, 0.0, 0.0]
        # freejoint 的角速度是在球自己的座標系；四元數設成單位，所以等於世界座標
        self.d.qvel[da:da + 6] = [*plan.v0, *plan.w0]

    def park(self, i: int) -> None:
        r = float(self.m.geom_size[int(self.ids.ball_geoms[i]), 0])
        qa, da = int(self.ids.ball_qadr[i]), int(self.ids.ball_dadr[i])
        self.d.qpos[qa:qa + 7] = [*park_position(self.cfg, i, r), 1.0, 0.0, 0.0, 0.0]
        self.d.qvel[da:da + 6] = 0.0

    def _should_retire(self, rec: BallRecord, t: float) -> bool:
        if t - rec.throw_time > self.bc.retire_lifetime:
            return True
        p, v = self.ball_state(rec.index)
        xy, yaw, _ = self.robot_planar_state()
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        rel = p[0:2] - xy
        behind = rel @ heading < -self.bc.retire_behind and v[0:2] @ heading < 0
        return bool(behind or np.linalg.norm(rel) > 10.0 or p[2] < -1.0)
