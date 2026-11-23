"""report.py and the Exp 3 flow (with fake episode results; no policy or environment needed)."""
import argparse
import csv
import importlib

import numpy as np

import pytest

from conftest import make_run
from minipacman import report


def _fake_episode(rng, hit_prob, seed, condition="c"):
    balls = []
    for k in range(int(rng.integers(1, 6))):
        balls.append({"launched": True, "mode": "direct" if k % 2 else "bounce",
                      "arrival_height": float(rng.uniform(0.05, 0.65)), "hit": bool(rng.random() < hit_prob),
                      "min_clearance": float(rng.uniform(-0.01, 0.5))})
    return {"condition": condition, "seed": seed, "success": bool(rng.random() < 0.9), "fell": False,
            "t_goal": 12.0, "n_thrown": len(balls), "n_hit": sum(b["hit"] for b in balls), "balls": balls,
            "intervention_mean": 0.1, "filter_active_frac": 0.2, "episode_seconds": 20.0, "device": "cpu"}


def test_wilson_interval():
    p, lo, hi = report.wilson(5, 100)
    assert p == 0.05 and 0.02 < lo < 0.05 < hi < 0.12


def test_summarize_and_plot(tmp_path):
    rng = np.random.default_rng(0)
    eps = [_fake_episode(rng, 0.3, s) for s in range(100)]
    row = report.summarize(eps)
    n_balls = sum(len(e["balls"]) for e in eps)
    assert row["n_balls"] == n_balls and 0.2 < row["collision_rate"] < 0.4
    assert row["n_balls_low"] + row["n_balls_mid"] + row["n_balls_high"] == n_balls
    report.save_jsonl(eps, tmp_path / "a.jsonl")
    assert len(report.load_jsonl(tmp_path / "a.jsonl")) == 100
    rows = [{"condition": n, **report.summarize(eps)} for n in ("exp3_no_filter", "exp3_truth")]
    report.write_csv(rows, tmp_path / "summary.csv")
    report.plot_exp3(rows, tmp_path)
    assert (tmp_path / "exp3_collision_rate.png").exists() and (tmp_path / "summary.csv").exists()


def test_part_exp3_flow(tmp_path, monkeypatch, cfg):
    """Exp 3 flow: all 4 conditions (filter off + 3 ball information sources) run, are saved and summarised."""
    run = importlib.import_module("run_mini_pacman_experiments")
    import minipacman.evaluate as ev
    run_dir = make_run(tmp_path / "runs" / cfg.exp.exp3_reward_variant, cfg, cfg.exp.exp3_reward_variant)
    monkeypatch.setitem(run.POLICY_FILES, cfg.exp.exp3_reward_variant, str(run_dir))
    hit_prob = {"exp3_no_filter": 0.6, "exp3_truth": 0.05, "exp3_noisy_truth": 0.1, "exp3_depth_tracker": 0.2}
    seen = []

    def fake_run_condition(condition, seeds, cfg_, out_path, device, resume=True, workers=1):
        seen.append(condition)
        rng = np.random.default_rng(len(seen))
        eps = [_fake_episode(rng, hit_prob[condition.name], s, condition.name) for s in seeds]
        report.save_jsonl(eps, out_path)
        return eps

    monkeypatch.setattr(ev, "run_condition", fake_run_condition)
    args = argparse.Namespace(seeds=None, quick=True, force=False, device="cpu", workers=1)
    rows = run.part_exp3(cfg, args, tmp_path)
    assert [c.name for c in seen] == ["exp3_no_filter", "exp3_truth", "exp3_noisy_truth", "exp3_depth_tracker"]
    assert seen[0].filter_mode is None and all(c.filter_mode == cfg.exp.exp3_filter_mode for c in seen[1:])
    assert [c.ball_source for c in seen[1:]] == list(cfg.exp.exp3_sources)
    assert len(rows) == 4 and (tmp_path / "exp3" / "summary.csv").exists()
    assert (tmp_path / "exp3" / "exp3_collision_rate.png").exists()


def test_part_exp1_flow(tmp_path, monkeypatch, cfg):
    """Exp 1 builds reward x filter conditions and writes both grouped plots."""
    run = importlib.import_module("run_mini_pacman_experiments")
    import minipacman.evaluate as ev
    seen = []
    for reward_variant in cfg.exp.reward_variants:
        run_dir = make_run(tmp_path / "runs" / reward_variant, cfg, reward_variant)
        monkeypatch.setitem(run.POLICY_FILES, reward_variant, str(run_dir))

    def fake_run_condition(condition, seeds, cfg_, out_path, device, resume=True, workers=1):
        seen.append(condition)
        probability = 0.15 if condition.filter_mode else 0.5
        rng = np.random.default_rng(len(seen) + 20)
        episodes = [_fake_episode(rng, probability, seed, condition.name) for seed in seeds]
        report.save_jsonl(episodes, out_path)
        return episodes

    monkeypatch.setattr(ev, "run_condition", fake_run_condition)
    args = argparse.Namespace(seeds=None, quick=True, force=False, device="cpu", workers=1)
    rows = run.part_exp1(cfg, args, tmp_path)
    assert [condition.name for condition in seen] == [
        f"exp1_{reward}_{suffix}"
        for reward in cfg.exp.reward_variants
        for suffix in ("nofilter", "filter")
    ]
    assert len(rows) == 6
    assert (tmp_path / "exp1" / "summary.csv").exists()
    assert (tmp_path / "exp1" / "exp1_collision_rate.png").exists()
    assert (tmp_path / "exp1" / "exp1_success_intervention.png").exists()


def test_part_exp2_flow(tmp_path, monkeypatch, cfg):
    """Exp 2 builds the naive/lag-aware scripted conditions and writes both plots."""
    run = importlib.import_module("run_mini_pacman_experiments")
    import minipacman.evaluate as ev
    seen = []

    def fake_run_condition(condition, seeds, cfg_, out_path, device, resume=True, workers=1):
        seen.append(condition)
        rng = np.random.default_rng(len(seen) + 10)
        probability = 0.4 if condition.filter_mode == "naive" else 0.2
        episodes = [_fake_episode(rng, probability, seed, condition.name) for seed in seeds]
        report.save_jsonl(episodes, out_path)
        return episodes

    monkeypatch.setattr(ev, "run_condition", fake_run_condition)
    args = argparse.Namespace(seeds=None, quick=True, force=False, device="cpu", workers=1)
    rows = run.part_exp2(cfg, args, tmp_path)
    assert [condition.name for condition in seen] == ["exp2_naive", "exp2_lag_aware"]
    assert [condition.filter_mode for condition in seen] == ["naive", "lag_aware"]
    assert all(condition.policy == "scripted" for condition in seen)
    assert len(rows) == 2
    assert (tmp_path / "exp2" / "summary.csv").exists()
    assert (tmp_path / "exp2" / "exp2_collision_rate.png").exists()
    assert (tmp_path / "exp2" / "exp2_collision_by_height.png").exists()


def test_part_summary_restores_condition_metadata(tmp_path, monkeypatch, cfg):
    """Summary rebuilds metadata that is intentionally absent from the episode JSONL schema."""
    run = importlib.import_module("run_mini_pacman_experiments")
    rng = np.random.default_rng(30)
    for reward_variant in cfg.exp.reward_variants:
        monkeypatch.setitem(run.POLICY_FILES, reward_variant, f"missing/{reward_variant}.pt")
        for suffix in ("nofilter", "filter"):
            condition = f"exp1_{reward_variant}_{suffix}"
            episodes = [_fake_episode(rng, 0.3, seed, condition) for seed in range(5)]
            report.save_jsonl(episodes, tmp_path / "exp1" / f"{condition}.jsonl")

    args = argparse.Namespace(device="cpu")
    run.part_summary(cfg, args, tmp_path)

    with open(tmp_path / "exp1" / "summary.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["reward_variant"] for row in rows] == [
        reward for reward in cfg.exp.reward_variants for _ in range(2)
    ]
    assert [row["filter_mode"] for row in rows] == [
        mode for _ in cfg.exp.reward_variants for mode in ("", cfg.exp.exp1_filter_mode)
    ]
    assert (tmp_path / "exp1" / "exp1_collision_rate.png").exists()
    assert (tmp_path / "exp1" / "exp1_success_intervention.png").exists()


def _exp1_runs(tmp_path, monkeypatch, cfg, run, **per_variant):
    for reward_variant in cfg.exp.reward_variants:
        kwargs = per_variant.get(reward_variant, {})
        run_dir = make_run(tmp_path / "runs" / reward_variant, cfg, reward_variant, **kwargs)
        monkeypatch.setitem(run.POLICY_FILES, reward_variant, str(run_dir))


def test_exp1_refuses_policies_trained_differently(tmp_path, monkeypatch, cfg):
    """Round 1 compared a 1500-update policy with a 200-update one; the runner must refuse that."""
    run = importlib.import_module("run_mini_pacman_experiments")
    _exp1_runs(tmp_path, monkeypatch, cfg, run, task_collision={"updates": 5})
    with pytest.raises(SystemExit, match="number of PPO updates differs"):
        run.exp1_conditions(cfg)


def test_exp1_refuses_different_training_seeds(tmp_path, monkeypatch, cfg):
    run = importlib.import_module("run_mini_pacman_experiments")
    _exp1_runs(tmp_path, monkeypatch, cfg, run, task_cbf={"seed": 2})
    with pytest.raises(SystemExit, match="training seed differs"):
        run.exp1_conditions(cfg)


def test_exp1_refuses_a_hand_picked_checkpoint(tmp_path, monkeypatch, cfg):
    """Round 1 picked checkpoints on evaluation seeds; only the final or validation-selected one is allowed."""
    run = importlib.import_module("run_mini_pacman_experiments")
    _exp1_runs(tmp_path, monkeypatch, cfg, run)
    monkeypatch.setitem(run.POLICY_FILES, "task", str(tmp_path / "runs" / "task" / "model_1.pt"))
    with pytest.raises(SystemExit, match="hand-picked"):
        run.exp1_conditions(cfg)


def test_part_select_uses_validation_seeds_only(tmp_path, monkeypatch, cfg):
    """--part select must evaluate on the validation seeds, never on the evaluation seeds, for all three runs."""
    run = importlib.import_module("run_mini_pacman_experiments")
    import minipacman.evaluate as ev
    from minipacman import protocol

    cfg.exp.select_every = 2
    _exp1_runs(tmp_path, monkeypatch, cfg, run, task={"updates": 5}, task_collision={"updates": 5},
               task_cbf={"updates": 5})
    used_seeds = set()

    def fake_run_condition(condition, seeds, cfg_, out_path, device, resume=True, workers=1):
        used_seeds.update(seeds)
        label = int(condition.name.rsplit("_", 1)[1])
        rng = np.random.default_rng(label)
        episodes = [_fake_episode(rng, 0.3, seed, condition.name) for seed in seeds]
        for episode in episodes:
            episode["success"] = label == 2          # make model_2 the best
        return episodes

    monkeypatch.setattr(ev, "run_condition", fake_run_condition)
    args = argparse.Namespace(force=False, device="cpu", workers=1)
    run.part_select(cfg, args, tmp_path)
    assert used_seeds == set(protocol.validation_seeds(cfg))
    assert not used_seeds & set(protocol.evaluation_seeds(cfg))
    records = [protocol.resolve_policy(tmp_path / "runs" / rv, rv, cfg) for rv in cfg.exp.reward_variants]
    assert all(r.path.name == "model_2.pt" and r.selection["rule"] == "validation" for r in records)
    protocol.check_same_protocol(records)
