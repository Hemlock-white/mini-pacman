"""dodge_env.py / vec_env.py / evaluate.py 的測試。【實作後把 pytestmark 刪掉】"""
import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="TODO：實作 dodge_env.py 後刪掉這行")


def test_reset_is_reproducible(cfg):
    """同一個 seed reset 兩次，丟球的時間、大小、軌跡要完全一樣（評估要能重現）。"""
    pytest.fail("TODO")


def test_episode_runs_to_timeout(cfg):
    """寫死的直行指令跑完整個 episode：不倒、到達終點、summary 欄位符合 EPISODE_SCHEMA。"""
    pytest.fail("TODO")


def test_reward_variants_differ_only_when_hit(cfg):
    """同一個 seed、同樣的動作：沒被打中的步，三種 reward 的差別只有 CBF 項；被打中那步 task_collision 多扣分。"""
    pytest.fail("TODO")


def test_obs_shapes(cfg):
    """actor / critic 觀測維度和 actor_obs_dim / critic_obs_dim 一致，沒有 nan。"""
    pytest.fail("TODO")


def test_vec_env_autoreset(cfg):
    """MiniPacmanVecEnv：環境 done 後自動 reset，time_outs 只在時間到時為 True。"""
    pytest.fail("TODO")
