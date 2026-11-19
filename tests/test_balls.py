"""Balls: bounce physics, resizing, throw trajectories, hit detection."""
import mujoco
import numpy as np
import pytest

from minipacman import balls, scene
from minipacman.balls import BallLauncher, ThrowPlan, plan_throw, predict_ball, set_ball_properties

from conftest import RobotFreezer, step

G = 9.81


def test_restitution_table_is_physical(cfg):
    b, e = balls.restitution_table(0.002, cfg.balls.contact_stiffness,
                                   int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST), cfg.balls.calibration_impact_speed)
    assert e.min() >= 0.0 and e.max() <= 1.0
    assert np.all(np.diff(e) <= 1e-3), "more damping should mean less bounce"
    assert e.max() > 0.9 and e.min() < 0.3, "the table must cover restitution_range"


def _drop_and_measure(m, d, ids, i, height):
    qa, da = int(ids.ball_qadr[i]), int(ids.ball_dadr[i])
    d.qpos[qa:qa + 7] = [5.0, 5.0, height, 1, 0, 0, 0]
    d.qvel[da:da + 6] = 0
    g = int(ids.ball_geoms[i])
    v_in, prev = None, 0.0
    for _ in range(3000):
        mujoco.mj_step(m, d)
        touching = any(g in (c.geom1, c.geom2) for c in d.contact[:d.ncon])
        if v_in is None and touching:
            v_in = -prev
        elif v_in is not None and not touching:
            return d.qvel[da + 2] / v_in
        prev = d.qvel[da + 2]
    raise AssertionError("the ball never bounced")


@pytest.mark.parametrize("e_target", [0.5, 0.65, 0.8])
@pytest.mark.parametrize("mass", [0.07, 0.2])
def test_bounce_restitution_in_scene(sim, cfg, e_target, mass):
    m, d = sim
    ids = scene.scene_ids(m, cfg)
    set_ball_properties(m, ids, 0, 0.1, mass, e_target, cfg.balls)
    e = _drop_and_measure(m, d, ids, 0, 0.1 + 4.0 ** 2 / (2 * G))
    assert abs(e - e_target) < 0.06, f"target e={e_target}, measured {e:.3f}"


@pytest.mark.parametrize("radius", [0.075, 0.125])
def test_resize_changes_contact(sim, cfg, radius):
    m, d = sim
    ids = scene.scene_ids(m, cfg)
    set_ball_properties(m, ids, 1, radius, 0.136, 0.6, cfg.balls)
    qa, da = int(ids.ball_qadr[1]), int(ids.ball_dadr[1])
    d.qpos[qa:qa + 7] = [6.0, 6.0, radius + 0.05, 1, 0, 0, 0]
    d.qvel[da:da + 6] = 0
    step(m, d, 1000)
    assert abs(d.qpos[qa + 2] - radius) < 0.003, "at rest the ball centre should be one radius above the floor"


def _throw_and_compare(m, d, cfg, mode, seed):
    ids = scene.scene_ids(m, cfg)
    launcher = BallLauncher(m, d, cfg, ids)
    rng = np.random.default_rng(seed)
    # put an imaginary robot at (20, 20) so the ball path stays well away from the real robot
    plan = plan_throw(rng, cfg.balls, np.array([20.0, 20.0]), 0.3, np.zeros(2), 0.1, 0.136, 0.7, mode=mode)
    launcher.throw_now(plan, 0)
    checks = np.linspace(0.1, plan.flight_time, 6)
    pred, _ = predict_ball(plan.p0, plan.v0, plan.radius, plan.restitution, checks)
    errs, touched = [], False
    t0 = d.time
    for k, tc in enumerate(checks):
        while d.time - t0 < tc - 1e-9:
            mujoco.mj_step(m, d)
            touched |= any(int(ids.ball_geoms[0]) in (c.geom1, c.geom2) for c in d.contact[:d.ncon])
        p, _ = launcher.ball_state(0)
        errs.append(np.linalg.norm(p - pred[k]))
    p_end, _ = launcher.ball_state(0)
    return plan, np.array(errs), np.linalg.norm(p_end - plan.target), touched


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_direct_throw_follows_plan(sim, cfg, seed):
    plan, errs, end_err, touched = _throw_and_compare(*sim, cfg, "direct", seed)
    assert not touched, "a direct throw should not touch the floor on the way"
    assert errs.max() < 0.03 and end_err < 0.03


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bounce_throw_follows_plan(sim, cfg, seed):
    plan, errs, end_err, touched = _throw_and_compare(*sim, cfg, "bounce", seed)
    assert plan.mode == "bounce" and touched, "a bounce throw should hit the floor first"
    assert end_err < 0.12, f"arrival position error {end_err:.3f} m"
    assert errs.max() < 0.12


@pytest.mark.parametrize("e", [0.5, 0.7, 0.8])
def test_predict_ball_multiple_bounces(sim, cfg, e):
    m, d = sim
    ids = scene.scene_ids(m, cfg)
    set_ball_properties(m, ids, 2, 0.1, 0.136, e, cfg.balls)
    qa, da = int(ids.ball_qadr[2]), int(ids.ball_dadr[2])
    p0, v0 = np.array([7.0, 7.0, 1.2]), np.array([1.0, 0.0, 0.0])
    d.qpos[qa:qa + 7] = [*p0, 1, 0, 0, 0]
    d.qvel[da:da + 6] = [*v0, 0, v0[0] / 0.1, 0]       # rolling-without-slipping spin
    times = np.arange(1, 101) * 0.02
    pred, _ = predict_ball(p0, v0, 0.1, e, times)
    sim_pos = []
    for _ in times:
        step(m, d, 10)
        sim_pos.append(d.qpos[qa:qa + 3].copy())
    err = np.linalg.norm(np.array(sim_pos) - pred, axis=1)
    assert err[:50].max() < 0.08, f"max error over the first second {err[:50].max():.3f} m"
    # the more bounces, the more the small differences in e at low impact speed add up;
    # the CBF only predicts 1.5 s ahead and the tracker keeps correcting
    assert err.max() < 0.15, f"max error within 2 s (3+ bounces) {err.max():.3f} m"


def test_launcher_schedule(sim, cfg):
    m, d = sim
    launcher = BallLauncher(m, d, cfg)
    counts = set()
    for seed in range(30):
        launcher.reset(np.random.default_rng(seed))
        n = len(launcher.records)
        counts.add(n)
        lo, hi = cfg.balls.balls_per_episode
        assert lo <= n <= hi
        times = [r.throw_time for r in launcher.records]
        assert all(np.diff(times) > 0) and times[-1] < cfg.sim.episode_seconds - 1.0
    assert counts == set(range(lo, hi + 1)), "30 seeds should cover every count from 1 to 5"


def _aimed_plan(base_pos, offset_y, T=0.8, radius=0.1):
    target = base_pos + np.array([0.0, offset_y, 0.0])
    p0 = base_pos + np.array([3.0, offset_y, 0.3])
    v0 = (target - p0) / T
    v0[2] += 0.5 * G * T
    return ThrowPlan(p0=p0, v0=v0, w0=np.zeros(3), radius=radius, mass=0.136, restitution=0.6,
                     mode="direct", flight_time=T, target=target)


def test_hit_detection(sim, cfg):
    m, d = sim
    ids = scene.scene_ids(m, cfg)
    launcher = BallLauncher(m, d, cfg, ids)
    freezer = RobotFreezer(m, d, ids)
    rec = launcher.throw_now(_aimed_plan(d.xpos[ids.base_body].copy(), 0.0), 0)
    for _ in range(60):
        step(m, d, 10, freezer, launcher)
        launcher.update(d.time)
    assert rec.hit and rec.hit_geom == "base_link_collision_0"
    assert rec.min_clearance <= 0.0


def test_miss_keeps_clearance(sim, cfg):
    m, d = sim
    ids = scene.scene_ids(m, cfg)
    launcher = BallLauncher(m, d, cfg, ids)
    freezer = RobotFreezer(m, d, ids)
    rec = launcher.throw_now(_aimed_plan(d.xpos[ids.base_body].copy(), 1.2), 0)
    for _ in range(60):
        step(m, d, 10, freezer, launcher)
        launcher.update(d.time)
    assert not rec.hit
    assert 0.5 < rec.min_clearance < 1.2
