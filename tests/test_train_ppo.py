"""Tests for the rsl_rl training adapter and the training half of the protocol."""

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
    assert cfg.reward.collision_penalty == pytest.approx(100.0)
    assert cfg.cbf.reward_weight == pytest.approx(1.0)


def _fake_rsl_rl(monkeypatch, calls, crash_after=None):
    """Replace the vec env and OnPolicyRunner; the fake runner names checkpoints like rsl_rl 5.4.0 does."""
    import rsl_rl.runners

    import minipacman.vec_env

    class FakeEnv:
        def __init__(self, *args, **kwargs):
            calls["env"] = (args, kwargs)

        def close(self):
            calls["closed"] = True

    class FakeRunner:
        def __init__(self, env, train_cfg, log_dir, device):
            self.log_dir = Path(log_dir)
            self.save_interval = train_cfg["save_interval"]
            self.current_learning_iteration = 0

        def load(self, path, map_location=None):
            calls["resume"] = (path, map_location)
            self.current_learning_iteration = int(Path(path).stem.split("_")[1])   # what rsl_rl does

        def learn(self, num_learning_iterations):
            start = self.current_learning_iteration
            calls.setdefault("learn", []).append((start, num_learning_iterations))
            for it in range(start, start + num_learning_iterations):
                if crash_after is not None and it > crash_after:
                    raise KeyboardInterrupt("simulated crash")
                self.current_learning_iteration = it
                if it % self.save_interval == 0:
                    (self.log_dir / f"model_{it}.pt").write_bytes(b"x")
            (self.log_dir / f"model_{self.current_learning_iteration}.pt").write_bytes(b"x")

    monkeypatch.setattr(minipacman.vec_env, "MiniPacmanVecEnv", FakeEnv)
    monkeypatch.setattr(rsl_rl.runners, "OnPolicyRunner", FakeRunner)


def test_fresh_run_writes_manifest_and_returns_final_checkpoint(tmp_path, monkeypatch, cfg):
    from minipacman import protocol
    from minipacman.train_ppo import train

    calls = {}
    _fake_rsl_rl(monkeypatch, calls)
    cfg.ppo.save_interval = 2
    final = train(cfg, "task", seed=3, log_dir=tmp_path / "run", device="cpu", max_iterations=5, num_envs=6)
    assert final == tmp_path / "run" / "model_4.pt"
    assert calls["learn"] == [(0, 5)] and calls["closed"]
    assert calls["env"][0][2] == 6 and calls["env"][1]["filter_mode"] is None
    manifest = protocol.read_manifest(tmp_path / "run")
    assert manifest["reward_variant"] == "task" and manifest["seed"] == 3 and manifest["target_updates"] == 5
    assert manifest["segments"][0]["resume_from"] is None
    assert manifest["fingerprint"] == protocol.training_fingerprint(cfg)
    assert protocol.resolve_policy(tmp_path / "run", "task", cfg).path == final


def test_resume_continues_the_same_run_without_repeating_an_update(tmp_path, monkeypatch, cfg):
    from minipacman import protocol
    from minipacman.train_ppo import train

    calls = {}
    cfg.ppo.save_interval = 2
    run = tmp_path / "run"
    _fake_rsl_rl(monkeypatch, calls, crash_after=2)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "task", seed=0, log_dir=run, device="cpu", max_iterations=6, num_envs=4)
    calls.clear()
    _fake_rsl_rl(monkeypatch, calls)
    final = train(cfg, "task", seed=0, log_dir=run, device="cpu", max_iterations=6, num_envs=4,
                  resume=run / "model_2.pt")
    assert calls["learn"] == [(3, 3)]                      # updates 3, 4, 5: none repeated, none skipped
    assert final == run / "model_5.pt"
    segments = protocol.read_manifest(run)["segments"]
    assert [seg["resume_from"] for seg in segments] == [None, "model_2.pt"]
    assert protocol.resolve_policy(run, "task", cfg).path == final


def test_resume_from_another_run_is_refused(tmp_path, monkeypatch, cfg):
    """Round 1 built task_cbf by fine-tuning the task policy; the trainer must refuse that."""
    from minipacman.protocol import ProtocolError
    from minipacman.train_ppo import train

    calls = {}
    _fake_rsl_rl(monkeypatch, calls)
    train(cfg, "task", seed=0, log_dir=tmp_path / "task", device="cpu", max_iterations=3, num_envs=4)
    with pytest.raises(ProtocolError, match="fine-tuning"):
        train(cfg, "task_cbf", seed=0, log_dir=tmp_path / "cbf", device="cpu", max_iterations=3, num_envs=4,
              resume=tmp_path / "task" / "model_0.pt")


def test_resume_with_changed_settings_is_refused(tmp_path, monkeypatch, cfg):
    from minipacman.protocol import ProtocolError
    from minipacman.train_ppo import train

    calls = {}
    run = tmp_path / "run"
    _fake_rsl_rl(monkeypatch, calls, crash_after=1)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "task_collision", seed=0, log_dir=run, device="cpu", max_iterations=4, num_envs=4)
    cfg.reward.collision_penalty = 50.0                      # round 1 changed it between runs
    with pytest.raises(ProtocolError, match="collision_penalty"):
        train(cfg, "task_collision", seed=0, log_dir=run, device="cpu", max_iterations=4, num_envs=4,
              resume=run / "model_0.pt")
    cfg.reward.collision_penalty = 100.0
    cfg.ppo.learning_rate = 1e-4                             # and lowered the learning rate halfway
    with pytest.raises(ProtocolError, match="learning_rate"):
        train(cfg, "task_collision", seed=0, log_dir=run, device="cpu", max_iterations=4, num_envs=4,
              resume=run / "model_0.pt")


def test_fresh_run_refuses_a_used_directory(tmp_path, monkeypatch, cfg):
    from minipacman.protocol import ProtocolError
    from minipacman.train_ppo import train

    calls = {}
    _fake_rsl_rl(monkeypatch, calls)
    train(cfg, "task", seed=0, log_dir=tmp_path / "run", device="cpu", max_iterations=2, num_envs=4)
    with pytest.raises(ProtocolError, match="already holds"):
        train(cfg, "task", seed=0, log_dir=tmp_path / "run", device="cpu", max_iterations=2, num_envs=4)


def test_train_rejects_invalid_arguments(tmp_path, cfg):
    from minipacman.train_ppo import train

    with pytest.raises(ValueError, match="unknown reward"):
        train(cfg, "wrong", 0, tmp_path, "cpu", max_iterations=1, num_envs=1)
    with pytest.raises(ValueError, match="max_iterations"):
        train(cfg, "task", 0, tmp_path, "cpu", max_iterations=0, num_envs=1)
    with pytest.raises(ValueError, match="num_envs"):
        train(cfg, "task", 0, tmp_path, "cpu", max_iterations=1, num_envs=0)
