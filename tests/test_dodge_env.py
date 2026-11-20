"""Tests for the single Mini PAC-MAN RL environment."""

import numpy as np
import pytest

from conftest import need_tita
from minipacman.ball_tracker import BallEstimate


def _speed_action(speed, cfg):
    value = 2 * (speed - cfg.command.v_min) / (
        cfg.command.v_max_stand - cfg.command.v_min
    ) - 1
    return np.array([value, 0.0, 0.0])


def test_action_to_command_and_ball_slots(cfg):
    from minipacman.dodge_env import action_to_command, ball_slot_features

    low = action_to_command(np.array([-1.0, -1.0, -1.0]), cfg)
    high = action_to_command(np.array([1.0, 1.0, 1.0]), cfg)
    np.testing.assert_allclose(
        low, [cfg.command.v_min, -cfg.command.w_max_stand, -cfg.command.sdot_max],
    )
    np.testing.assert_allclose(
        high, [cfg.command.v_max_stand, cfg.command.w_max_stand, cfg.command.sdot_max],
    )

    ball = BallEstimate(
        0, np.array([2.0, 1.0, 0.4]), np.array([-3.0, 2.0, 0.1]),
        0.1, 0.7, 0.0, "truth",
    )
    features = ball_slot_features([ball], np.array([1.0, 1.0]), np.pi / 2, cfg)
    slots = features.reshape(cfg.obs.ball_slots, cfg.obs.ball_slot_features)
    np.testing.assert_allclose(
        slots[0], [0.0, -1.0, 0.4, 2.0, 3.0, 0.1, 0.1, 1.0], atol=1e-7,
    )
    np.testing.assert_array_equal(slots[1:], 0.0)


@need_tita
def test_reset_is_reproducible(cfg):
    from minipacman.dodge_env import DodgeEnv

    cfg.balls.balls_per_episode = (1, 1)
    cfg.balls.first_throw_range = (0.02, 0.02)
    env = DodgeEnv(cfg, augment_depth=False, seed=17)
    stand = _speed_action(0.0, cfg)

    def snapshot():
        observation = env.reset(17)
        schedule = [
            (record.throw_time, record.radius, record.mass, record.restitution)
            for record in env.launcher.records
        ]
        while not env.launcher.records[0].launched:
            observation, _, terminated, _, _ = env.step(stand)
            assert not terminated
        plan = env.launcher.records[0].plan
        trajectory = (plan.p0.copy(), plan.v0.copy(), plan.target.copy())
        return observation, schedule, trajectory

    first = snapshot()
    second = snapshot()
    np.testing.assert_array_equal(first[0]["actor"], second[0]["actor"])
    np.testing.assert_array_equal(first[0]["critic"], second[0]["critic"])
    assert first[1] == second[1]
    for left, right in zip(first[2], second[2]):
        np.testing.assert_array_equal(left, right)


@need_tita
def test_episode_runs_to_timeout(cfg):
    from minipacman.dodge_env import DodgeEnv

    env = DodgeEnv(cfg, augment_depth=False, seed=7)
    env.reset(7)
    drive = _speed_action(0.5, cfg)
    stop = _speed_action(0.0, cfg)
    while True:
        _, _, terminated, truncated, info = env.step(stop if env.reached_goal else drive)
        if terminated or truncated:
            break
    summary = info["episode"]
    assert not terminated and truncated
    assert summary["success"] and not summary["fell"]
    assert summary["t_goal"] is not None
    assert summary["episode_seconds"] == pytest.approx(cfg.sim.episode_seconds)
    assert summary["n_thrown"] == sum(ball["launched"] for ball in summary["balls"])
    assert summary["n_hit"] == sum(ball["hit"] for ball in summary["balls"])


@need_tita
def test_reward_variants_differ_only_when_hit(cfg):
    from minipacman.balls import ThrowPlan
    from minipacman.dodge_env import DodgeEnv

    variants = {
        name: DodgeEnv(cfg, reward_variant=name, augment_depth=False, seed=5)
        for name in ("task", "task_collision", "task_cbf")
    }
    plan = ThrowPlan(
        p0=np.array([1.2, 0.0, 0.4]),
        v0=np.array([-3.0, 0.0, 1.962]),
        w0=np.array([0.0, -30.0, 0.0]),
        radius=0.1, mass=0.136, restitution=0.7,
        mode="direct", flight_time=0.4, target=np.array([0.0, 0.0, 0.4]),
    )
    for env in variants.values():
        env.reset(5)
        env.launcher.throw_now(plan)
        env._refresh_perception()

    stand = _speed_action(0.0, cfg)
    saw_hit = False
    for _ in range(100):
        results = {name: env.step(stand) for name, env in variants.items()}
        task_reward = results["task"][1]
        collision_reward = results["task_collision"][1]
        cbf_reward = results["task_cbf"][1]
        new_hits = results["task"][4]["new_hits"]
        if new_hits:
            saw_hit = True
            assert task_reward - collision_reward == pytest.approx(
                cfg.reward.collision_penalty * new_hits,
            )
            break
        assert collision_reward == pytest.approx(task_reward)
        assert cbf_reward <= task_reward + 1e-12
    assert saw_hit, "the deterministic test throw did not hit the robot"


@need_tita
@pytest.mark.parametrize("use_true_state", [True, False])
def test_obs_shapes(cfg, use_true_state):
    from minipacman.dodge_env import DodgeEnv

    cfg.estimator.use_true_state = use_true_state
    env = DodgeEnv(cfg, filter_mode="lag_aware", augment_depth=False, seed=3)
    observation = env.reset(3)
    assert observation["actor"].shape == (env.actor_obs_dim,)
    assert observation["critic"].shape == (env.critic_obs_dim,)
    assert observation["actor"].dtype == np.float32
    assert observation["critic"].dtype == np.float32
    assert np.all(np.isfinite(observation["actor"]))
    assert np.all(np.isfinite(observation["critic"]))
    observation, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert np.all(np.isfinite(observation["actor"]))
    assert np.all(np.isfinite(observation["critic"]))
    assert np.isfinite(reward)
    assert not terminated and not truncated
    assert {"u_policy", "u_safe", "intervention", "filter_active", "new_hits"} <= info.keys()


@pytest.mark.skip(reason="TODO: implement MiniPacmanVecEnv in vec_env.py")
def test_vec_env_autoreset(cfg):
    """MiniPacmanVecEnv resets automatically and reports timeouts separately."""
