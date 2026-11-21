"""Train PPO with rsl_rl 5.4.0. [train() is yours; make_train_cfg() already has the parameters]

rsl_rl vs the course's MT06 (for teammates who haven't used rsl_rl)
-------------------------------------------------------------------
The PPO algorithm is exactly the same as MT06; rsl_rl just wraps each piece in a class and runs many environments at once:

============================  ===========================================  =========================
MT06 (MT06_RL_PPO.ipynb)      rsl_rl 5.4.0                                 setting here
============================  ===========================================  =========================
build_actor()                 rsl_rl.models.MLPModel ("actor")             actor.hidden_dims
build_value_function()        rsl_rl.models.MLPModel ("critic")            critic.hidden_dims
utils.PPOBuffer               rsl_rl.storage.RolloutStorage                (created automatically)
compute_gae()                 PPO.compute_returns() (gamma, lam)           algorithm.gamma / lam
ppo_update()                  PPO.update()                                 algorithm.*
collect_rollout()             the collection loop inside OnPolicyRunner.learn()  num_steps_per_env
run_training()                OnPolicyRunner.learn(num_learning_iterations)  max_iterations
args.num_steps = 2048         num_steps_per_env × num_envs (24 × 256)      steps collected per update
args.update_epochs = 10       num_learning_epochs = 5
args.num_minibatches = 32     num_mini_batches = 4
args.clip_coef = 0.2          clip_param = 0.2
args.ent_coef = 0.0           entropy_coef = 0.005
args.vf_coef = 0.5            value_loss_coef = 1.0
args.max_grad_norm = 0.5      max_grad_norm = 1.0
args.anneal_lr                schedule = "adaptive" (learning rate follows the KL)  desired_kl = 0.01
one env, envs.step()          MiniPacmanVecEnv.step(): num_envs at once    vec_env.py
============================  ===========================================  =========================

Two extra ideas:
* obs_groups: observation groups. The actor and critic can see different things (asymmetric actor-critic), e.g. the
  critic also sees the true ball state, which makes training more stable; only the actor is used at deployment.
* obs_normalization: normalises observations automatically (MT06 doesn't).

Trained checkpoints are saved as log_dir/model_<iteration>.pt. To use one in the experiments, put its path in
POLICY_FILES at the top of run_mini_pacman_experiments.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import Config


def make_train_cfg(cfg: Config, reward_variant: str) -> dict:
    """train_cfg for rsl_rl's OnPolicyRunner (parameters from config.PPOConfig, mostly the same as PAC-MAN's rl_cfg.py)."""
    p = cfg.ppo
    model = {
        "hidden_dims": list(p.hidden_dims),
        "activation": p.activation,
        "obs_normalization": p.obs_normalization,
    }
    return {
        "num_steps_per_env": p.num_steps_per_env,
        "save_interval": p.save_interval,
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "actor": {"class_name": "rsl_rl.models.MLPModel", **model,
                  "distribution_cfg": {"class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                                       "init_std": p.init_std, "std_type": "scalar"}},
        "critic": {"class_name": "rsl_rl.models.MLPModel", **model},
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "num_learning_epochs": p.num_learning_epochs,
            "num_mini_batches": p.num_mini_batches,
            "clip_param": p.clip_param,
            "gamma": p.gamma,
            "lam": p.lam,
            "value_loss_coef": p.value_loss_coef,
            "entropy_coef": p.entropy_coef,
            "learning_rate": p.learning_rate,
            "max_grad_norm": p.max_grad_norm,
            "use_clipped_value_loss": True,
            "schedule": p.schedule,
            "desired_kl": p.desired_kl,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "logger": "tensorboard",
        "experiment_name": f"mini_pacman_{reward_variant}",
        "run_name": reward_variant,
    }


def train(cfg: Config, reward_variant: str, seed: int, log_dir: Path, device: str,
          max_iterations: int | None = None, num_envs: int | None = None, resume: Path | None = None) -> Path:
    """Train one policy and return the path of the last checkpoint.

    Steps:
      1. torch.manual_seed(seed), np.random.seed(seed)
      2. env = MiniPacmanVecEnv(cfg, reward_variant, num_envs or cfg.ppo.num_envs, cfg.ppo.num_workers,
                                device, seed, filter_mode=None, augment_depth=True)
         (no filter during training, so the "taught" effect is pure)
      3. runner = rsl_rl.runners.OnPolicyRunner(env, make_train_cfg(cfg, reward_variant), str(log_dir), device)
      4. if resume is not None: runner.load(str(resume))
      5. runner.learn(num_learning_iterations=max_iterations or cfg.ppo.max_iterations)
      6. env.close(); return the model_*.pt with the highest number in log_dir

    Note: the runner modifies the make_train_cfg dict (it pops class_name), so call make_train_cfg again every time.
    """
    if reward_variant not in cfg.exp.reward_variants:
        raise ValueError(f"unknown reward variant: {reward_variant!r}")
    iterations = cfg.ppo.max_iterations if max_iterations is None else int(max_iterations)
    environment_count = cfg.ppo.num_envs if num_envs is None else int(num_envs)
    if iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if environment_count < 1:
        raise ValueError("num_envs must be at least 1")

    from rsl_rl.runners import OnPolicyRunner

    from .vec_env import MiniPacmanVecEnv

    torch.manual_seed(seed)
    np.random.seed(seed)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = MiniPacmanVecEnv(
        cfg,
        reward_variant,
        environment_count,
        cfg.ppo.num_workers,
        device,
        seed,
        filter_mode=None,
        augment_depth=True,
    )
    try:
        runner = OnPolicyRunner(
            env,
            make_train_cfg(cfg, reward_variant),
            str(log_dir),
            device,
        )
        if resume is not None:
            resume = Path(resume)
            if not resume.is_file():
                raise FileNotFoundError(f"checkpoint does not exist: {resume}")
            runner.load(str(resume), map_location=device)
        runner.learn(num_learning_iterations=iterations)
    finally:
        env.close()

    checkpoints = list(log_dir.glob("model_*.pt"))
    if not checkpoints:
        raise RuntimeError(f"training finished without a checkpoint in {log_dir}")

    def iteration(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(checkpoints, key=iteration)
