"""Balls: launcher, bounce physics, trajectory prediction, hit detection.

What's in here
--------------
* ``BallLauncher``: throws 1-5 balls per episode from the robot's front (current heading ±25 degrees), with random
  radius, mass, restitution, speed and height. Two kinds of trajectory:
    - ``direct``: flies straight at the robot (never touches the floor on the way).
    - ``bounce``: bounces once on the floor in front of the robot, then flies at the robot.
  It also records whether each ball hit the robot and how close it came, and takes balls back after they pass.
* ``predict_ball``: closed-form prediction of where a ball will be (including floor bounces), used by the CBF filter
  and the tracker.
* ``set_ball_properties``: change a ball's radius, mass and restitution at runtime.

How the bounce works
--------------------
MuJoCo contacts are "soft", like a spring with a damper. Ball contacts use solref = (-stiffness, -damping): the
stiffness is fixed (BallConfig.contact_stiffness) and less damping means a higher bounce (damping 0 gives e ≈ 0.99).
The mapping from damping to restitution e (rebound speed / impact speed) has no simple formula, so
``restitution_table()`` drops a ball in a tiny "ball + floor" scene, measures the table once, and then we interpolate.
Measured: e does not depend on the ball's mass, and changes by only about 0.05 between 1 and 6 m/s impacts.
(MuJoCo's default solref = (timeconst, dampratio) form makes the stiffness grow like 1/dampratio^2 as the damping
ratio gets small; at a 2 ms step that goes numerically unstable and gives e > 1, so we don't use it.)
Ball geoms have priority=1, so contacts with the floor and with the robot both use the ball's own parameters.

Why balls are launched with "topspin"
-------------------------------------
If a real ball's bottom slides along the floor when it lands, friction slows it down horizontally and the path after
the bounce is hard to predict. So each ball is launched spinning at exactly the rolling-without-slipping rate
w = (z × v_horizontal) / r. At touchdown the contact point is not moving, the horizontal speed is kept, and
``predict_ball`` can predict it with "constant horizontal speed, vertical speed times -e".

Timing
------
Call ``check_contacts()`` every physics step (a contact only lasts a few ms; miss it and you miss a hit).
Call ``update()`` every policy step (launch, take back, update the closest distance).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import BallConfig, Config, default_config
from .scene import SceneIds, park_position, scene_ids

G = 9.81
ROLL_SPEED = 0.3          # if the vertical rebound speed is below this, treat the ball as rolling on the floor


# --------------------------------------------------------------------------------------------
# Restitution: damping <-> coefficient of restitution
# --------------------------------------------------------------------------------------------
_RESTITUTION_CACHE: dict = {}


def restitution_table(timestep: float, stiffness: float, integrator: int,
                      impact_speed: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Drop a ball in a tiny "ball + floor" scene and return (damping values, measured restitution).

    Cached, so each parameter set is only measured once (about 0.1 s).
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
    """Look up the solref damping that gives restitution e. Values outside the reachable range snap to the closest one."""
    dampings, es = restitution_table(timestep, stiffness, integrator, impact_speed)
    order = np.argsort(es)                     # np.interp needs increasing x
    return float(np.interp(e, es[order], dampings[order]))


def set_ball_properties(m: mujoco.MjModel, ids: SceneIds, i: int, radius: float, mass: float,
                        restitution: float, cfg: BallConfig) -> None:
    """Change ball i's radius, mass, inertia and restitution at runtime.

    Besides geom_size we also update the bounding sizes used by collision detection (geom_rbound, geom_aabb) and the
    mass-related constants (body_subtreemass, invweight0); otherwise contacts would use the old size or stiffness.
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
# Trajectory prediction (with bounces)
# --------------------------------------------------------------------------------------------
def bounce_contact_time(stiffness: float | None = None) -> float:
    """Each touchdown, the soft contact lasts about half an oscillation, π/sqrt(stiffness) (about 14 ms = 7 physics steps at 50000)."""
    if stiffness is None:
        stiffness = default_config().balls.contact_stiffness
    return float(np.pi / np.sqrt(stiffness))


def predict_ball(p: np.ndarray, v: np.ndarray, radius: float, restitution: float,
                 times: np.ndarray, max_bounces: int = 6,
                 contact_time: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Predict the ball's position and velocity at ``times`` (seconds from now; can be an array).

    Assumes: gravity only; at touchdown the horizontal speed is kept and the vertical speed is multiplied by -e
    (balls are launched with rolling spin, see the module docstring); each touchdown stays on the floor for
    contact_time seconds (by default from the contact stiffness, see bounce_contact_time); once the vertical rebound
    is below 0.3 m/s the ball rolls. Hitting the robot is not modelled.
    Measured (e = 0.5-0.8, dropped from 1.2 m): error < 8 cm over the first second, < 15 cm within 2 s (3+ bounces).
    Most of the error comes from the real e at low impact speeds being slightly different, and it adds up with
    every bounce.
    Returns (positions (n,3), velocities (n,3)).
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
            t_end = (v[2] + np.sqrt(v[2] ** 2 + 2 * G * dz)) / G      # time until the next touchdown
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
        # touchdown, stay on the floor for contact_time, bounce
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
    if remaining.any():                       # past max_bounces: treat it as rolling
        tau = times[remaining] - t0
        pos[remaining, 0:2] = p[0:2] + np.outer(tau, v[0:2])
        pos[remaining, 2] = radius
        vel[remaining, 0:2] = v[0:2]
    return pos, vel


# --------------------------------------------------------------------------------------------
# Throw planning
# --------------------------------------------------------------------------------------------
@dataclass
class ThrowPlan:
    """Everything about one throw (world frame)."""
    p0: np.ndarray            # launch position
    v0: np.ndarray            # launch velocity
    w0: np.ndarray            # launch angular velocity (rolling-without-slipping topspin)
    radius: float
    mass: float
    restitution: float
    mode: str                 # "direct" or "bounce"
    flight_time: float        # expected time until it reaches the target
    target: np.ndarray        # expected arrival position (ball centre)
    bounce_point: np.ndarray | None = None

    @property
    def horizontal_speed(self) -> float:
        return float(np.linalg.norm(self.v0[0:2]))


def plan_throw(rng: np.random.Generator, cfg: BallConfig, robot_xy: np.ndarray, robot_yaw: float,
               robot_vxy: np.ndarray, radius: float, mass: float, restitution: float,
               mode: str | None = None, max_tries: int = 50) -> ThrowPlan:
    """Plan one ball thrown at the robot from its front.

    1. Launch point: D metres in front of the robot, within ±launch_angle_deg of its heading.
    2. Target: the robot's position now + velocity × flight time (leading the target) + aim noise; random arrival height.
    3. Flight time = horizontal distance / horizontal speed (random horizontal speed).
    4. direct: given a launch height, solve for the vertical launch speed. The parabola opens downwards, so if both
       ends are above the floor it never touches the floor on the way.
       bounce: given a bounce point (a fraction of the way from launch to target), work backwards:
         how fast it must leave the floor for the second leg (bounce point → target) → divide by e to get the
         impact speed → the launch velocity and launch height of the first leg.
         If the launch height is out of range, resample.
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
            for _ in range(3):                      # leading the target changes the flight time, so iterate a few times
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
    raise RuntimeError("plan_throw: no feasible throw found, check the ranges in BallConfig")


# --------------------------------------------------------------------------------------------
# Launcher
# --------------------------------------------------------------------------------------------
@dataclass
class BallRecord:
    """One ball thrown in this episode."""
    index: int                         # which ball (ball_i)
    throw_time: float                  # scheduled launch time
    radius: float
    mass: float
    restitution: float
    plan: ThrowPlan | None = None      # planned at launch time
    launched: bool = False
    active: bool = False               # in play (after launch, before being taken back)
    hit: bool = False
    hit_time: float | None = None
    hit_geom: str | None = None
    min_clearance: float = np.inf
    retired_time: float | None = None

    def to_dict(self) -> dict:
        """Fields saved in the results file (report.py groups the statistics by these)."""
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
    """Manages the pool of balls in the scene: schedule, launch, hit detection, closest distance, take back.

    Typical use (inside dodge_env.py)::

        launcher = BallLauncher(m, d, cfg)
        launcher.reset(rng)                     # at the start of each episode
        for each policy step:
            launcher.update(d.time)             # launch, take back, update closest distance
            for each physics step:
                mujoco.mj_step(m, d)
                launcher.check_contacts()       # hit detection
        records = launcher.episode_records()    # save with the results
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
        # build the restitution table now so the first throw doesn't pay for it
        restitution_table(m.opt.timestep, self.bc.contact_stiffness, int(m.opt.integrator),
                          self.bc.calibration_impact_speed)

    # ---------------- episode ----------------
    def reset(self, rng: np.random.Generator) -> None:
        """Take all balls back and schedule this episode's balls (count, times, size, mass, restitution)."""
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
        """Call every policy step: launch balls that are due, take back balls that have passed, update closest distances.
        Returns the balls launched in this step."""
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
        """Call every physics step: check ball-robot contacts; the first touch counts as a hit.
        Returns the balls that hit in this step."""
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
        """Update each active ball's closest distance to the robot's collision shapes (negative = penetrating)."""
        for rec in self.active_records():
            dist = self.clearance(rec.index, distmax)
            rec.min_clearance = min(rec.min_clearance, dist)

    def clearance(self, i: int, distmax: float = 3.0) -> float:
        """Shortest distance from ball i to any robot collision shape (returns distmax if further)."""
        g_ball = int(self.ids.ball_geoms[i])
        return min(mujoco.mj_geomDistance(self.m, self.d, int(g), g_ball, distmax, self._fromto)
                   for g in self.ids.robot_collision_geoms)

    # ---------------- queries ----------------
    def active_records(self) -> list[BallRecord]:
        return [r for r in self.records if r.active]

    def ball_state(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(position, linear velocity) of ball i, world frame."""
        qa, da = int(self.ids.ball_qadr[i]), int(self.ids.ball_dadr[i])
        return self.d.qpos[qa:qa + 3].copy(), self.d.qvel[da:da + 3].copy()

    def episode_records(self) -> list[dict]:
        return [r.to_dict() for r in self.records]

    def robot_planar_state(self) -> tuple[np.ndarray, float, np.ndarray]:
        """(xy position, heading yaw, xy velocity) of the robot base, world frame
        (the launcher uses true values; it is part of the "world")."""
        d, b = self.d, self.ids.base_body
        R = d.xmat[b].reshape(3, 3)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        return d.xpos[b][0:2].copy(), float(np.arctan2(R[1, 0], R[0, 0])), vel[3:5].copy()

    # ---------------- internals ----------------
    def _launch(self, rec: BallRecord) -> None:
        xy, yaw, vxy = self.robot_planar_state()
        plan = plan_throw(self.rng, self.bc, xy, yaw, vxy, rec.radius, rec.mass, rec.restitution)
        self.launch(rec.index, plan)
        rec.plan, rec.launched, rec.active = plan, True, True
        self._ball_to_record[rec.index] = rec

    def throw_now(self, plan: ThrowPlan, index: int = 0) -> BallRecord:
        """Throw ball ``index`` with ``plan`` right now (for tests and demo videos; the real pipeline schedules via reset/update)."""
        rec = BallRecord(index=index, throw_time=float(self.d.time), radius=plan.radius, mass=plan.mass,
                         restitution=plan.restitution, plan=plan, launched=True, active=True)
        self.launch(index, plan)
        self.records = [r for r in self.records if r.index != index] + [rec]
        self._ball_to_record[index] = rec
        return rec

    def launch(self, i: int, plan: ThrowPlan) -> None:
        """Put ball i in the initial state of ``plan`` (can also be called directly in tests)."""
        set_ball_properties(self.m, self.ids, i, plan.radius, plan.mass, plan.restitution, self.bc)
        qa, da = int(self.ids.ball_qadr[i]), int(self.ids.ball_dadr[i])
        self.d.qpos[qa:qa + 7] = [*plan.p0, 1.0, 0.0, 0.0, 0.0]
        # a freejoint's angular velocity is in the ball's own frame; the quaternion is identity, so that equals world
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
