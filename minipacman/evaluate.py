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
    """Hard-coded high-level command: drive to the goal, never dodge (for Exp 2, so the result only reflects the filter).

    act(obs) returns an action in [-1, 1]^3 meaning: turn towards the goal (w = turn_gain * bearing, clipped to the
    standing turn limit), drive at cruise_speed * max(cos(bearing), 0) (slow down while facing away), and stay standing
    (sdot = -max). Stop at the goal. The goal comes from obs["actor"][obs_layout(cfg)["actor"]["goal"]]
    ((forward, left) in the heading frame).

    It has to steer back: with turning allowed, the safety filter may turn the robot away from a ball, and a policy
    that only drives straight would then miss the goal (the first version did, so Exp 2's success rate measured the
    script, not the filter).
    """

    cruise_speed = 0.5      # m/s
    turn_gain = 2.0         # rad/s per rad of bearing error

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _action(self, v: float, w: float) -> np.ndarray:
        command = self.cfg.command
        return np.array([
            2 * (v - command.v_min) / (command.v_max_stand - command.v_min) - 1,
            w / command.w_max_stand,
            -1.0,
        ])

    def act(self, obs: dict, env) -> np.ndarray:
        from .dodge_env import obs_layout

        goal_slice = obs_layout(self.cfg)["actor"]["goal"]
        forward, left = np.asarray(obs["actor"][goal_slice], dtype=float)
        if np.hypot(forward, left) <= self.cfg.task.goal_radius:
            return self._action(0.0, 0.0)
        bearing = float(np.arctan2(left, forward))
        w_max = self.cfg.command.w_max_stand
        w = float(np.clip(self.turn_gain * bearing, -w_max, w_max))
        v = self.cruise_speed * max(float(np.cos(bearing)), 0.0)
        return self._action(v, w)


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


def make_env(condition: Condition, cfg: Config, seed: int = 0, scene_path=None):
    """The DodgeEnv for a condition (evaluation: no depth augmentation). One env can run many seeds via reset(seed)."""
    from .dodge_env import DodgeEnv

    reward_variant = "task" if condition.reward_variant == "scripted" else condition.reward_variant
    return DodgeEnv(
        cfg, reward_variant, condition.filter_mode, condition.replan,
        condition.ball_source, augment_depth=False, seed=seed, scene_path=scene_path,
    )


def run_episode(condition: Condition, seed: int, cfg: Config, policy, device: str, env=None) -> dict:
    """Run one episode and return a dict in the EPISODE_SCHEMA format.

    env: reuse an env made by make_env(condition, ...) (reset(seed) fully resets MuJoCo, so the result only depends
    on the seed); None creates a new one.
    """
    env = make_env(condition, cfg, seed) if env is None else env
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
        **episode_key(condition, cfg),
    }


def episode_key(condition: Condition, cfg: Config) -> dict:
    """What produced an episode: the checkpoint (its last three path parts, e.g. task/run1/model_4999.pt, so the key
    is the same on every machine) and a fingerprint of the evaluation config. run_condition only reuses saved
    episodes whose key matches, so switching checkpoints (e.g. after --part select) can't silently reuse old ones."""
    from .protocol import evaluation_fingerprint

    policy_id = condition.policy if condition.policy == "scripted" else "/".join(Path(condition.policy).parts[-3:])
    return {"policy_id": policy_id, "eval_config": evaluation_fingerprint(cfg)}


# ---------------- parallel evaluation (each worker loads the policy once and reuses one env) ----------------
_WORKER: dict = {}


def _eval_worker_init(condition: Condition, cfg: Config, device: str, scene_path: str) -> None:
    _WORKER["policy"] = (ScriptedPolicy(cfg) if condition.policy == "scripted"
                         else load_policy(condition.policy, cfg, device))
    _WORKER["env"] = make_env(condition, cfg, scene_path=scene_path)
    _WORKER.update(condition=condition, cfg=cfg, device=device)


def _eval_worker_run(seed: int) -> dict:
    w = _WORKER
    return run_episode(w["condition"], int(seed), w["cfg"], w["policy"], w["device"], env=w["env"])


def run_condition(condition: Condition, seeds: list[int], cfg: Config, out_path: Path, device: str,
                  resume: bool = True, workers: int = 1) -> list[dict]:
    """Run run_episode for every seed, append each finished one to out_path (.jsonl), return all episodes.

    resume=True: skip seeds already in out_path (read them with report.load_jsonl), so an interrupted run can continue.
                 Only episodes made by the same checkpoint and evaluation config are reused (episode_key); the rest
                 are dropped from the file and rerun.
    resume=False: delete out_path first and rerun everything.
    workers > 1: run seeds in parallel subprocesses (each seed is independent, so the results are the same).
    """
    from . import report

    out_path = Path(out_path)
    if not resume:
        out_path.unlink(missing_ok=True)
    existing = report.load_jsonl(out_path) if resume else []
    key = episode_key(condition, cfg)
    kept = [e for e in existing if all(e.get(k) == v for k, v in key.items())]
    if len(kept) < len(existing):
        print(f"  [{condition.name}] {len(existing) - len(kept)} saved episodes came from another checkpoint or "
              f"config (or an older version of this script); running them again", flush=True)
        report.save_jsonl(kept, out_path)
    by_seed = {int(episode["seed"]): episode for episode in kept}
    todo = [int(seed) for seed in seeds if int(seed) not in by_seed]
    if todo and workers > 1:
        import multiprocessing as mp

        from .scene import build_scene
        from .vec_worker import single_threaded_children

        scene_path = str(build_scene(cfg=cfg))          # once, so the workers don't write the file at the same time
        with single_threaded_children():
            pool = mp.get_context("spawn").Pool(
                min(int(workers), len(todo)), initializer=_eval_worker_init,
                initargs=(condition, cfg, device, scene_path))
        try:
            for episode in pool.imap_unordered(_eval_worker_run, todo):
                report.append_jsonl(episode, out_path)
                by_seed[int(episode["seed"])] = episode
        finally:
            pool.close()
            pool.join()
    elif todo:
        policy = (ScriptedPolicy(cfg) if condition.policy == "scripted"
                  else load_policy(condition.policy, cfg, device))
        env = make_env(condition, cfg, todo[0])
        for seed in todo:
            episode = run_episode(condition, seed, cfg, policy, device, env=env)
            report.append_jsonl(episode, out_path)
            by_seed[seed] = episode
    return [by_seed[int(seed)] for seed in seeds if int(seed) in by_seed]
