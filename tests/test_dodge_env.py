"""Tests for dodge_env.py / vec_env.py / evaluate.py. [Delete pytestmark after implementing them]"""
import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="TODO: delete this line after implementing dodge_env.py")


def test_reset_is_reproducible(cfg):
    """Resetting twice with the same seed gives exactly the same throw times, sizes and trajectories (evaluation must be reproducible)."""
    pytest.fail("TODO")


def test_episode_runs_to_timeout(cfg):
    """The scripted straight-ahead command runs a full episode: doesn't fall, reaches the goal, and the summary fields match EPISODE_SCHEMA."""
    pytest.fail("TODO")


def test_reward_variants_differ_only_when_hit(cfg):
    """Same seed, same actions: on steps without a hit, the three rewards differ only by the CBF term; on the step
    with a hit, task_collision loses extra points."""
    pytest.fail("TODO")


def test_obs_shapes(cfg):
    """actor / critic observation sizes match actor_obs_dim / critic_obs_dim, and there are no nans."""
    pytest.fail("TODO")


def test_vec_env_autoreset(cfg):
    """MiniPacmanVecEnv: an environment resets automatically when done, and time_outs is True only when time ran out."""
    pytest.fail("TODO")
