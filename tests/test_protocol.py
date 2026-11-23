"""Tests for the training / evaluation protocol (protocol.py): fair Exp 1, unseen evaluation seeds."""
import json

import pytest

from conftest import make_run
from minipacman import protocol
from minipacman.protocol import ProtocolError


def test_seed_sets_are_disjoint(cfg):
    protocol.check_seed_sets(cfg, train_seed=0, num_envs=256)
    train = {protocol.train_env_seed(s, i) for s in (0, 1, 2) for i in range(256)}
    val, ev = set(protocol.validation_seeds(cfg)), set(protocol.evaluation_seeds(cfg))
    assert not (train & val) and not (train & ev) and not (val & ev)


def test_overlapping_seed_sets_are_refused(cfg):
    cfg.exp.val_seed_offset = cfg.exp.eval_seed_offset + 50
    with pytest.raises(ProtocolError, match="overlap"):
        protocol.check_seed_sets(cfg)


def test_fingerprint_tracks_training_settings_only(cfg):
    base = protocol.training_fingerprint(cfg)
    cfg.exp.n_eval_seeds = 7               # evaluation bookkeeping: not a training setting
    cfg.cbf.horizon = 2.0                  # filter: evaluation-time only
    cfg.ppo.num_workers = 7                # number of processes: speed only
    assert protocol.training_fingerprint(cfg) == base
    cfg.cbf.reward_weight = 2.7            # the CBF reward is a training setting
    assert protocol.training_fingerprint(cfg) != base


def test_final_checkpoint_is_accepted(tmp_path, cfg):
    run = make_run(tmp_path / "r", cfg, "task", updates=4)
    record = protocol.resolve_policy(run, "task", cfg)
    assert record.path.name == "model_3.pt" and record.selection == {"rule": "final"}
    assert protocol.resolve_policy(run / "model_3.pt", "task", cfg).path.name == "model_3.pt"


def test_hand_picked_checkpoint_is_refused(tmp_path, cfg):
    run = make_run(tmp_path / "r", cfg, "task", updates=4)
    with pytest.raises(ProtocolError, match="hand-picked"):
        protocol.resolve_policy(run / "model_1.pt", "task", cfg)


def test_selected_checkpoint_is_accepted_and_overrides_final(tmp_path, cfg):
    run = make_run(tmp_path / "r", cfg, "task", updates=4)
    (run / protocol.SELECTION).write_text(json.dumps({"rule": "validation", "checkpoint": "model_1.pt"}))
    assert protocol.resolve_policy(run, "task", cfg).path.name == "model_1.pt"
    with pytest.raises(ProtocolError, match="hand-picked"):
        protocol.resolve_policy(run / "model_3.pt", "task", cfg)     # final, but a selection exists


def test_unfinished_wrong_variant_and_changed_config_are_refused(tmp_path, cfg):
    run = make_run(tmp_path / "r", cfg, "task", updates=4, labels=[0, 1])
    with pytest.raises(ProtocolError, match="not finished"):
        protocol.resolve_policy(run, "task", cfg)
    run = make_run(tmp_path / "s", cfg, "task", updates=2)
    with pytest.raises(ProtocolError, match="trained with reward"):
        protocol.resolve_policy(run, "task_cbf", cfg)
    cfg.reward.collision_penalty = 50.0
    with pytest.raises(ProtocolError, match="collision_penalty"):
        protocol.resolve_policy(run, "task", cfg)


def test_round_one_checkpoints_without_manifest_are_refused(tmp_path, cfg):
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "model_1499.pt").write_bytes(b"x")
    with pytest.raises(ProtocolError, match="missing"):
        protocol.resolve_policy(tmp_path / "old" / "model_1499.pt", "task", cfg)


def test_same_protocol_check(tmp_path, cfg):
    records = [protocol.resolve_policy(make_run(tmp_path / rv, cfg, rv), rv, cfg) for rv in cfg.exp.reward_variants]
    protocol.check_same_protocol(records)
    odd = make_run(tmp_path / "odd", cfg, "task_cbf", num_envs=16)
    records[2] = protocol.resolve_policy(odd, "task_cbf", cfg)
    with pytest.raises(ProtocolError, match="number of envs differs"):
        protocol.check_same_protocol(records)


def test_mixed_selection_rules_are_refused(tmp_path, cfg):
    runs = {rv: make_run(tmp_path / rv, cfg, rv, updates=4) for rv in cfg.exp.reward_variants}
    (runs["task"] / protocol.SELECTION).write_text(json.dumps({"rule": "validation", "checkpoint": "model_1.pt"}))
    records = [protocol.resolve_policy(run, rv, cfg) for rv, run in runs.items()]
    with pytest.raises(ProtocolError, match="checkpoint rule differs"):
        protocol.check_same_protocol(records)
