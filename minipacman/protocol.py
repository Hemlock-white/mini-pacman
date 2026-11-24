"""Training / evaluation protocol: keeps Exp 1 fair and the evaluation seeds unseen. [ours, 2026-11-23]

Two rules, enforced in code because the first round of results broke both:

1. The Exp 1 policies are trained the same way: from scratch, same training seed, same config, same number of PPO
   updates, and their checkpoints are picked by the same rule. Starting one reward variant from another variant's
   checkpoint (fine-tuning), changing settings between variants, or stopping one variant early is refused.
   (Round 1: task_cbf was the task policy plus 2 PPO updates, and task_collision had 200 updates with a penalty the
   config no longer had.)
2. The evaluation seeds (cfg.exp.eval_seed_offset ..) are never used to choose anything. A policy used in an
   experiment is either the run's final checkpoint, or the checkpoint that `--part select` picked on the separate
   validation seeds (cfg.exp.val_seed_offset ..). A hand-picked checkpoint is refused.
   (Round 1: checkpoints were picked on evaluation seeds 10000-10004.)

How it works
------------
Every training run directory holds run_info.json (the "manifest"): reward variant, seed, number of envs, target
number of PPO updates, a fingerprint of every config field that affects training, the device, and one "segment" per
training session (a resume adds a segment; it may only continue the same run). --part select writes selection.json
next to it. resolve_policy() checks a checkpoint against all of that before an experiment uses it, and
check_same_protocol() checks that the Exp 1 policies match each other.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config

MANIFEST = "run_info.json"
SELECTION = "selection.json"
PROTOCOL_VERSION = 1

# Config sections that change what a policy learns (or the world it is evaluated in). Not included: tita (paths),
# exp (evaluation bookkeeping), tracker / noisy_truth (Exp 3 ball sources) and the filter fields of cbf (the filter is
# off during training; it is an evaluation-time variable).
TRAINING_SECTIONS = ("sim", "robot", "lqr", "command", "task", "balls", "camera", "imu", "encoder", "estimator",
                     "reward", "obs", "ppo")
CBF_REWARD_FIELDS = ("alpha", "margin", "constraint_clip", "reward_weight", "threat_min_height_above_radius",
                     "threat_min_speed")
SELECTION_CRITERION = "max success rate, then min fall rate, then min collision rate, then the later checkpoint"


class ProtocolError(RuntimeError):
    """A policy or training run breaks the protocol (see the module docstring)."""


# ---------------- seeds ----------------
def train_env_seed(seed: int, env_index: int) -> int:
    """Seed of training environment env_index in a run started with --seed seed."""
    return int(seed) * 100_000 + int(env_index)


def validation_seeds(cfg: Config, n: int | None = None) -> list[int]:
    n = cfg.exp.n_val_seeds if n is None else int(n)
    return list(range(cfg.exp.val_seed_offset, cfg.exp.val_seed_offset + n))


def evaluation_seeds(cfg: Config, n: int | None = None) -> list[int]:
    n = cfg.exp.n_eval_seeds if n is None else int(n)
    return list(range(cfg.exp.eval_seed_offset, cfg.exp.eval_seed_offset + n))


def check_seed_sets(cfg: Config, train_seed: int | None = None, num_envs: int | None = None) -> None:
    """Raise ProtocolError if the training, validation and evaluation seed ranges overlap."""
    ev = (cfg.exp.eval_seed_offset, cfg.exp.eval_seed_offset + cfg.exp.n_eval_seeds)
    va = (cfg.exp.val_seed_offset, cfg.exp.val_seed_offset + cfg.exp.n_val_seeds)
    ranges = {"evaluation": ev, "validation": va}
    if train_seed is not None and num_envs is not None:
        ranges["training"] = (train_env_seed(train_seed, 0), train_env_seed(train_seed, 0) + int(num_envs))
    names = list(ranges)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (a0, a1), (b0, b1) = ranges[a], ranges[b]
            if a0 < b1 and b0 < a1:
                raise ProtocolError(f"{a} seeds [{a0}, {a1}) overlap {b} seeds [{b0}, {b1})")


# ---------------- config fingerprint ----------------
PERFORMANCE_ONLY = {"ppo": ("num_workers",)}     # how fast, not what is learned: may differ between runs


def training_config(cfg: Config) -> dict:
    out = {name: asdict(getattr(cfg, name)) for name in TRAINING_SECTIONS}
    for section, fields in PERFORMANCE_ONLY.items():
        for field in fields:
            out[section].pop(field, None)
    out["cbf_reward"] = {name: getattr(cfg.cbf, name) for name in CBF_REWARD_FIELDS}
    return out


def training_fingerprint(cfg: Config) -> str:
    text = json.dumps(training_config(cfg), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def evaluation_fingerprint(cfg: Config) -> str:
    """Fingerprint of everything that changes an evaluation episode (all config sections but tita paths and exp)."""
    data = {name: value for name, value in cfg.to_dict().items() if name not in ("tita", "exp")}
    text = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def config_differences(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Human-readable list of fields that differ between two training_config() dicts."""
    out = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key, "<missing>"), new.get(key, "<missing>")
        if isinstance(a, dict) and isinstance(b, dict):
            out += config_differences(a, b, f"{prefix}{key}.")
        elif json.dumps(a, default=str) != json.dumps(b, default=str):
            out.append(f"{prefix}{key}: trained with {a!r}, config now {b!r}")
    return out


# ---------------- manifest ----------------
def read_manifest(run_dir: Path) -> dict:
    path = Path(run_dir) / MANIFEST
    if not path.is_file():
        raise ProtocolError(f"{path} is missing: this directory was not created by train_mini_pacman_policy.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(f"{path} predates the training protocol (no 'protocol' field): retrain this policy")
    return manifest


def write_manifest(run_dir: Path, manifest: dict) -> None:
    path = Path(run_dir) / MANIFEST
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def new_manifest(cfg: Config, reward_variant: str, seed: int, num_envs: int, target_updates: int,
                 device_info: dict, args: dict | None = None) -> dict:
    return {
        "protocol": PROTOCOL_VERSION,
        "reward_variant": reward_variant,
        "seed": int(seed),
        "num_envs": int(num_envs),
        "target_updates": int(target_updates),
        "fingerprint": training_fingerprint(cfg),
        "training_config": training_config(cfg),
        "device": device_info,
        "args": args or {},
        "segments": [],
        "config": cfg.to_dict(),
    }


def checkpoint_label(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", Path(path).name)
    if not match:
        raise ProtocolError(f"not a checkpoint name: {Path(path).name} (expected model_<iteration>.pt)")
    return int(match.group(1))


def final_label(manifest: dict) -> int:
    """rsl_rl names checkpoints by update index (0-based), so N updates end with model_{N-1}.pt."""
    return int(manifest["target_updates"]) - 1


def start_fresh_run(log_dir: Path, manifest: dict) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if (log_dir / MANIFEST).exists() or any(log_dir.glob("model_*.pt")):
        raise ProtocolError(f"{log_dir} already holds a training run; use a new --out, or --resume to continue it")
    manifest["segments"].append({"resume_from": None, "first_update": 0, "updates": manifest["target_updates"],
                                 "started": time.strftime("%Y-%m-%d %H:%M:%S"), "device": manifest["device"]})
    write_manifest(log_dir, manifest)


def continue_run(log_dir: Path, checkpoint: Path, cfg: Config, reward_variant: str, seed: int, num_envs: int,
                 target_updates: int, device_info: dict) -> tuple[dict, int]:
    """Check that `checkpoint` may be resumed into `log_dir` and record the new segment.
    Returns (manifest, first update index of the new segment)."""
    log_dir, checkpoint = Path(log_dir), Path(checkpoint)
    if checkpoint.parent.resolve() != log_dir.resolve():
        raise ProtocolError(
            "--resume may only continue an interrupted run in its own directory "
            f"(checkpoint in {checkpoint.parent}, --out {log_dir}). Starting from another run's policy is "
            "fine-tuning, which would make the Exp 1 policies incomparable.")
    manifest = read_manifest(log_dir)
    problems = []
    if manifest["reward_variant"] != reward_variant:
        problems.append(f"reward {manifest['reward_variant']} -> {reward_variant}")
    if manifest["seed"] != int(seed):
        problems.append(f"seed {manifest['seed']} -> {seed}")
    if manifest["num_envs"] != int(num_envs):
        problems.append(f"num_envs {manifest['num_envs']} -> {num_envs}")
    if manifest["target_updates"] != int(target_updates):
        problems.append(f"iterations {manifest['target_updates']} -> {target_updates}")
    if manifest["fingerprint"] != training_fingerprint(cfg):
        problems += config_differences(manifest["training_config"], training_config(cfg))
    if problems:
        raise ProtocolError("can't resume with different settings: " + "; ".join(problems))
    first = checkpoint_label(checkpoint) + 1
    if first >= manifest["target_updates"]:
        raise ProtocolError(f"{checkpoint.name} is already the final checkpoint; nothing to resume")
    manifest["segments"].append({"resume_from": checkpoint.name, "first_update": first,
                                 "updates": manifest["target_updates"] - first,
                                 "started": time.strftime("%Y-%m-%d %H:%M:%S"), "device": device_info})
    write_manifest(log_dir, manifest)
    return manifest, first


def finish_run(log_dir: Path, final_checkpoint: Path) -> None:
    manifest = read_manifest(log_dir)
    manifest["segments"][-1]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["segments"][-1]["final_checkpoint"] = Path(final_checkpoint).name
    write_manifest(log_dir, manifest)


# ---------------- using a policy in an experiment ----------------
@dataclass
class PolicyRecord:
    path: Path
    run_dir: Path
    reward_variant: str
    seed: int
    num_envs: int
    target_updates: int
    fingerprint: str
    selection: dict          # {"rule": "final"} or the contents of selection.json
    rocm: bool

    def summary(self) -> str:
        rule = self.selection.get("rule")
        return (f"{self.reward_variant}: {self.path.name} ({rule}), seed {self.seed}, {self.target_updates} updates, "
                f"{self.num_envs} envs, config {self.fingerprint}, ROCm {self.rocm}")


def read_selection(run_dir: Path) -> dict | None:
    path = Path(run_dir) / SELECTION
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def resolve_policy(path_or_dir: str | Path, reward_variant: str, cfg: Config) -> PolicyRecord:
    """Return the checkpoint an experiment may use for this reward variant, or raise ProtocolError.

    path_or_dir: a run directory (uses selection.json if --part select was run, otherwise the final checkpoint), or a
    checkpoint file in a run directory (must be the final one or the selected one).
    """
    target = Path(path_or_dir)
    run_dir = target if target.is_dir() else target.parent
    manifest = read_manifest(run_dir)
    selection = read_selection(run_dir)
    final = run_dir / f"model_{final_label(manifest)}.pt"
    if target.is_dir():
        path = run_dir / selection["checkpoint"] if selection else final
    else:
        path = target
    if not final.is_file():
        raise ProtocolError(f"training in {run_dir} is not finished ({final.name} is missing)")
    if not path.is_file():
        raise ProtocolError(f"{path} does not exist")
    if path.name == final.name and (selection is None or selection["checkpoint"] == final.name):
        chosen = selection or {"rule": "final"}
    elif selection is not None and selection["checkpoint"] == path.name:
        chosen = selection
    else:
        raise ProtocolError(
            f"{path.name} is neither the final checkpoint ({final.name}) nor the one --part select picked on the "
            "validation seeds: hand-picked checkpoints are not allowed (they were picked on evaluation seeds before)")
    if manifest["reward_variant"] != reward_variant:
        raise ProtocolError(f"{run_dir} was trained with reward {manifest['reward_variant']!r}, "
                            f"but it is listed as {reward_variant!r}")
    if manifest["segments"] and manifest["segments"][0]["resume_from"] is not None:
        raise ProtocolError(f"{run_dir} did not start from scratch")
    if manifest["fingerprint"] != training_fingerprint(cfg):
        diff = config_differences(manifest["training_config"], training_config(cfg))
        raise ProtocolError(f"the config changed since {run_dir} was trained: " + "; ".join(diff[:8]))
    rocm = bool(manifest.get("device", {}).get("rocm"))
    if not rocm:
        warnings.warn(f"{run_dir} was not trained on AMD ROCm; the course rules don't accept it in the report",
                      stacklevel=2)
    return PolicyRecord(path=path, run_dir=run_dir, reward_variant=reward_variant, seed=manifest["seed"],
                        num_envs=manifest["num_envs"], target_updates=manifest["target_updates"],
                        fingerprint=manifest["fingerprint"], selection=chosen, rocm=rocm)


def check_same_protocol(records: list[PolicyRecord]) -> None:
    """Exp 1: raise ProtocolError unless every policy was trained and picked the same way."""
    if len(records) < 2:
        return
    first = records[0]
    keys = {
        "training seed": lambda r: r.seed,
        "number of envs": lambda r: r.num_envs,
        "number of PPO updates": lambda r: r.target_updates,
        "config": lambda r: r.fingerprint,
        "checkpoint rule": lambda r: r.selection.get("rule"),
        "validation seeds": lambda r: tuple(r.selection.get("seeds", ())),
        "selection candidates": lambda r: r.selection.get("candidate_rule"),
        "selection criterion": lambda r: r.selection.get("criterion"),
    }
    problems = []
    for name, get in keys.items():
        values = {r.reward_variant: get(r) for r in records}
        if len({json.dumps(v, default=str) for v in values.values()}) > 1:
            problems.append(f"{name} differs: {values}")
    if problems:
        raise ProtocolError("the Exp 1 policies were not trained/picked the same way: " + "; ".join(problems))
