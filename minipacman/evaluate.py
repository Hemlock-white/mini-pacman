"""Evaluation: run one experiment condition on fixed seeds and save each episode's result. [Condition is done; the functions are yours]

An "experiment condition" (Condition) = which policy × whether the filter is on and which version × where the ball
information comes from. run_mini_pacman_experiments.py builds many Conditions and calls run_condition() on each.

EPISODE_SCHEMA: each episode is saved as one dict (one line of the .jsonl); report.summarize() relies on these fields
-------------------------------------------------------------------------------------------------------------------
    "condition"           str     condition name
    "seed"                int
    "success"             bool    reached the goal in time without falling
    "fell"                bool
    "t_goal"              float | None   first time the goal was reached (s)
    "n_thrown"            int     how many balls were launched
    "n_hit"               int     how many hit the robot
    "balls"               list[dict]     launcher.episode_records() (per ball: mode, height, speed, size, hit or not, closest distance)
    "intervention_mean"   float | None   how much the filter changed the command on average (filter off = None)
    "filter_active_frac"  float | None   fraction of steps where the filter stepped in
    "episode_seconds"     float   how long it actually ran
    "device"              str     PyTorch device (policy inference)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import Config


@dataclass
class Condition:
    """One experiment condition."""
    name: str
    policy: str                     # policy checkpoint path, or "scripted" (hard-coded straight-ahead command, for Exp 2)
    reward_variant: str             # which reward this policy was trained with (for the record)
    filter_mode: str | None         # None / "naive" / "lag_aware"
    replan: str = "every_step"      # "every_step" / "commit"
    ball_source: str = "truth"      # ball information source for the filter: "truth" / "noisy_truth" / "depth_tracker"

    def to_dict(self) -> dict:
        return asdict(self)


class ScriptedPolicy:
    """Hard-coded high-level command: drive straight to the goal, never dodge (for Exp 2, so the result only reflects the filter).

    act(obs) returns an action ∈ [-1, 1]^3 meaning v = 0.5 m/s, w = 0, sdot = - (stay standing).
    Stop at the goal (v = 0). Distance to the goal: obs["actor"][obs_layout(cfg)["actor"]["goal"]]
    ((forward, left) in the heading frame), or get it straight from the env.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        command = cfg.command
        self.drive_action = np.array([
            2 * (0.5 - command.v_min) / (command.v_max_stand - command.v_min) - 1,
            0.0,
            -1.0,
        ])
        self.stop_action = np.array([
            2 * (0.0 - command.v_min) / (command.v_max_stand - command.v_min) - 1,
            0.0,
            -1.0,
        ])

    def act(self, obs: dict, env) -> np.ndarray:
        from .dodge_env import obs_layout

        goal_slice = obs_layout(self.cfg)["actor"]["goal"]
        goal = np.asarray(obs["actor"][goal_slice])
        return (self.stop_action if np.linalg.norm(goal) <= self.cfg.task.goal_radius
                else self.drive_action).copy()


class _LoadedPolicy:
    """Small NumPy-facing wrapper around an rsl_rl actor model."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def act(self, obs: dict, env=None) -> np.ndarray:
        import torch
        from tensordict import TensorDict

        actor = torch.as_tensor(
            obs["actor"], dtype=torch.float32, device=self.device,
        ).reshape(1, -1)
        observation = TensorDict({"actor": actor}, batch_size=[1], device=self.device)
        with torch.inference_mode():
            action = self.model(observation)
        return action.squeeze(0).to(device="cpu").numpy()


def load_policy(path: str | Path, cfg: Config, device: str):
    """Load a checkpoint trained by rsl_rl and return a function obs_actor (np.ndarray) -> action (np.ndarray).

    An rsl_rl 5.4.0 checkpoint is a dict (checked):
        {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"}
    Suggested: build the same MLPModel as in training (the "actor" settings of train_ppo.make_train_cfg; MLPModel's
    constructor needs an example observation TensorDict to know the input size),
    load_state_dict(ck["actor_state_dict"]), use the mean of the distribution in evaluation (don't sample), and run
    inside torch.no_grad().
    Or simpler: build an OnPolicyRunner (with an evaluation vec env), runner.load(path),
    policy = runner.get_inference_policy(device).
    The device must be the ROCm GPU (course rule: CPU-only results don't count).
    """
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    from .dodge_env import ACTION_DIM, obs_layout

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"policy checkpoint does not exist: {path}")
    torch_device = torch.device(device)
    actor_dim = max(part.stop for part in obs_layout(cfg)["actor"].values())
    example = TensorDict(
        {"actor": torch.zeros((1, actor_dim), device=torch_device)},
        batch_size=[1],
        device=torch_device,
    )
    ppo = cfg.ppo
    model = MLPModel(
        example,
        {"actor": ["actor"]},
        "actor",
        ACTION_DIM,
        hidden_dims=list(ppo.hidden_dims),
        activation=ppo.activation,
        obs_normalization=ppo.obs_normalization,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": ppo.init_std,
            "std_type": "scalar",
        },
    ).to(torch_device)
    checkpoint = torch.load(path, map_location=torch_device, weights_only=True)
    if "actor_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint has no actor_state_dict: {path}")
    model.load_state_dict(checkpoint["actor_state_dict"])
    model.eval()
    return _LoadedPolicy(model, torch_device)


def run_episode(condition: Condition, seed: int, cfg: Config, policy, device: str) -> dict:
    """Run one episode and return a dict in the EPISODE_SCHEMA format.

    Create DodgeEnv(cfg, condition.reward_variant, condition.filter_mode, condition.replan,
                    condition.ball_source, augment_depth=False, seed=seed),
    obs = env.reset(seed), step until terminated or truncated, then use env.episode_summary() and add
    "condition", "seed" and "device".
    (If creating the env is slow, let run_condition reuse one env and just call reset(seed) for each seed.)
    """
    from .dodge_env import DodgeEnv

    reward_variant = "task" if condition.reward_variant == "scripted" else condition.reward_variant
    env = DodgeEnv(
        cfg, reward_variant, condition.filter_mode, condition.replan,
        condition.ball_source, augment_depth=False, seed=seed,
    )
    observation = env.reset(seed)
    while True:
        if hasattr(policy, "act"):
            action = policy.act(observation, env)
        else:
            action = policy(observation["actor"])
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=float))
        if terminated or truncated:
            break
    return {
        "condition": condition.name,
        "seed": int(seed),
        **env.episode_summary(),
        "device": str(device),
    }


def run_condition(condition: Condition, seeds: list[int], cfg: Config, out_path: Path, device: str,
                  resume: bool = True) -> list[dict]:
    """Run run_episode for every seed, append each finished one to out_path (.jsonl), return all episodes.

    resume=True: skip seeds already in out_path (read them with report.load_jsonl), so an interrupted run can continue.
    resume=False: delete out_path first and rerun everything.
    Different seeds can run in parallel with multiprocessing (each seed is independent; order doesn't matter).
    """
    from . import report

    out_path = Path(out_path)
    if not resume:
        out_path.unlink(missing_ok=True)
    existing = report.load_jsonl(out_path) if resume else []
    by_seed = {int(episode["seed"]): episode for episode in existing}
    policy = (ScriptedPolicy(cfg) if condition.policy == "scripted"
              else load_policy(condition.policy, cfg, device))
    for seed in seeds:
        if int(seed) in by_seed:
            continue
        episode = run_episode(condition, int(seed), cfg, policy, device)
        report.append_jsonl(episode, out_path)
        by_seed[int(seed)] = episode
    return [by_seed[int(seed)] for seed in seeds if int(seed) in by_seed]
