"""Tests for the rsl_rl training adapter."""

from pathlib import Path

import pytest

from minipacman.train_ppo import make_train_cfg


def test_make_train_cfg_has_asymmetric_observations(cfg):
    train_cfg = make_train_cfg(cfg, "task_cbf")
    assert train_cfg["obs_groups"] == {"actor": ["actor"], "critic": ["critic"]}
    assert train_cfg["algorithm"]["class_name"] == "rsl_rl.algorithms.PPO"
    assert train_cfg["experiment_name"] == "mini_pacman_task_cbf"
    assert train_cfg["actor"]["hidden_dims"] == list(cfg.ppo.hidden_dims)
    assert train_cfg["actor"]["distribution_cfg"]["init_std"] == pytest.approx(0.2)
    assert train_cfg["algorithm"]["learning_rate"] == pytest.approx(3e-4)
    assert train_cfg["algorithm"]["schedule"] == "fixed"
    assert train_cfg["algorithm"]["entropy_coef"] == pytest.approx(0.001)
    assert cfg.reward.progress_weight == pytest.approx(10.0)
    assert cfg.reward.goal_bonus == pytest.approx(50.0)
    assert cfg.reward.collision_penalty == pytest.approx(50.0)


def test_train_wires_runner_and_returns_latest_checkpoint(tmp_path, monkeypatch, cfg):
    import rsl_rl.runners
    import minipacman.vec_env
    from minipacman.train_ppo import train

    calls = {}

    class FakeEnv:
        def __init__(self, *args, **kwargs):
            calls["env"] = (args, kwargs)
            self.closed = False

        def close(self):
            self.closed = True

    class FakeRunner:
        def __init__(self, env, train_cfg, log_dir, device):
            calls["runner"] = (env, train_cfg, log_dir, device)
            calls["runner_instance"] = self
            self.log_dir = Path(log_dir)
            self.alg = type("FakeAlg", (), {
                "learning_rate": 9e-3,
                "optimizer": type("FakeOptimizer", (), {"param_groups": [{"lr": 9e-3}]})(),
            })()

        def learn(self, num_learning_iterations):
            calls["iterations"] = num_learning_iterations
            (self.log_dir / "model_2.pt").write_bytes(b"older")
            (self.log_dir / "model_10.pt").write_bytes(b"latest")

        def load(self, path, map_location=None):
            calls["resume"] = (path, map_location)

    monkeypatch.setattr(minipacman.vec_env, "MiniPacmanVecEnv", FakeEnv)
    monkeypatch.setattr(rsl_rl.runners, "OnPolicyRunner", FakeRunner)
    resume = tmp_path / "resume.pt"
    resume.write_bytes(b"checkpoint")
    checkpoint = train(
        cfg, "task", seed=3, log_dir=tmp_path, device="cpu",
        max_iterations=4, num_envs=6, resume=resume,
    )
    assert checkpoint == tmp_path / "model_10.pt"
    assert calls["iterations"] == 4
    assert calls["env"][0][2] == 6
    assert calls["runner"][0].closed
    assert calls["resume"] == (str(resume), "cpu")
    assert calls["runner_instance"].alg.learning_rate == pytest.approx(cfg.ppo.learning_rate)
    assert calls["runner_instance"].alg.optimizer.param_groups[0]["lr"] == pytest.approx(
        cfg.ppo.learning_rate,
    )


def test_train_rejects_invalid_arguments(tmp_path, cfg):
    from minipacman.train_ppo import train

    with pytest.raises(ValueError, match="unknown reward"):
        train(cfg, "wrong", 0, tmp_path, "cpu", max_iterations=1, num_envs=1)
    with pytest.raises(ValueError, match="max_iterations"):
        train(cfg, "task", 0, tmp_path, "cpu", max_iterations=0, num_envs=1)
    with pytest.raises(ValueError, match="num_envs"):
        train(cfg, "task", 0, tmp_path, "cpu", max_iterations=1, num_envs=0)
