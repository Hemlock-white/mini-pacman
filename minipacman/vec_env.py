"""Run many DodgeEnvs in parallel and connect them to rsl_rl. [YOURS]

Why we write this ourselves
---------------------------
rsl_rl was made for GPU simulators (Isaac Lab, mjlab): thousands of environments computed at once on the GPU.
Our MuJoCo runs on the CPU, so we use several subprocesses (multiprocessing), each running a few DodgeEnvs, and stitch
the results into torch tensors for rsl_rl. The bottleneck is the number of CPU cores, not the GPU.

Compared with the course's MT06
-------------------------------
MT06 uses one gymnasium environment, ``envs.step(action_np)``, one step and one environment at a time. This is the
same thing, just num_envs at once, and rsl_rl requires a fixed interface (VecEnv in rsl_rl/env/vec_env.py):

    attributes: num_envs, num_actions, max_episode_length, episode_length_buf (torch, num_envs), device, cfg
    get_observations() -> TensorDict({"actor": (num_envs, 589), "critic": (num_envs, 632)}, batch_size=[num_envs])
        (sizes come from dodge_env.obs_layout(cfg); the key names must match obs_groups in train_ppo.make_train_cfg)
    step(actions (num_envs, num_actions) torch) -> (obs TensorDict, rewards (num_envs,), dones (num_envs,), extras)
        extras["time_outs"]: (num_envs,) which ones ended because time ran out (not because they fell); rsl_rl uses
                             it for bootstrapping
        extras["log"]: {"/name": value}, shows up in tensorboard, e.g. "/episode/collision_rate"
    When an environment is done, **reset it automatically** inside step; the returned obs is the first observation
    of the new episode.

Suggested structure
-------------------
    worker(conn, env_kwargs_list): a subprocess that creates len(env_kwargs_list) DodgeEnvs and loops on commands
        ("reset", None) → return the obs of all its environments
        ("step", actions_np) → step each environment, reset the ones that are done, return obs, reward, done,
                               time_out, episode summaries
        ("close", None) → exit
    MiniPacmanVecEnv: start num_workers subprocesses (multiprocessing.get_context("spawn"), works on Windows and
        Linux), split the num_envs environments evenly between them; on step, send each its slice of actions, then
        stitch the returned numpy arrays together and convert to torch.
    Seed of each environment: base_seed * 100000 + env_index; between episodes each environment keeps using its own rng.
"""
from __future__ import annotations

import numpy as np

from .config import Config

try:
    from rsl_rl.env import VecEnv
except ImportError:          # so other modules can still be imported before rsl_rl is installed
    VecEnv = object


class MiniPacmanVecEnv(VecEnv):
    def __init__(self, cfg: Config, reward_variant: str, num_envs: int, num_workers: int = 0,
                 device: str = "cpu", seed: int = 0, **env_kwargs):
        """env_kwargs are passed to every DodgeEnv (e.g. filter_mode, ball_source, augment_depth).
        num_workers=0: use max(1, os.cpu_count() - 2) automatically."""
        raise NotImplementedError("TODO: MiniPacmanVecEnv.__init__")

    def get_observations(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
