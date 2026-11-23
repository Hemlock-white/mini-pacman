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
    Seed of each environment: protocol.train_env_seed(base_seed, env_index) = base_seed * 100000 + env_index; between
    episodes each environment keeps using its own rng. The workers live in vec_worker.py (no torch import there) and
    start with one math thread each.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os

import numpy as np
import torch
from tensordict import TensorDict

from .config import Config
from .dodge_env import ACTION_DIM
from .protocol import train_env_seed
from .scene import build_scene
from .vec_worker import env_worker, single_threaded_children

try:
    from rsl_rl.env import VecEnv
except ImportError:          # so other modules can still be imported before rsl_rl is installed
    VecEnv = object


class MiniPacmanVecEnv(VecEnv):
    def __init__(self, cfg: Config, reward_variant: str, num_envs: int, num_workers: int = 0,
                 device: str = "cpu", seed: int = 0, **env_kwargs):
        """env_kwargs are passed to every DodgeEnv (e.g. filter_mode, ball_source, augment_depth).
        num_workers=0: use max(1, os.cpu_count() - 2) automatically."""
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        self.cfg = cfg
        self.num_envs = int(num_envs)
        automatic_workers = max(1, (os.cpu_count() or 1) - 2)
        self.num_workers = min(self.num_envs, num_workers or automatic_workers)
        self.num_actions = ACTION_DIM
        self.device = torch.device(device)
        policy_dt = cfg.sim.timestep * cfg.sim.policy_every
        self.max_episode_length = int(math.ceil(cfg.sim.episode_seconds / policy_dt))
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        self._closed = False
        self._processes = []
        self._connections = []
        self._slices = []

        context = mp.get_context("spawn")
        chunks = np.array_split(np.arange(self.num_envs), self.num_workers)
        scene_path = str(build_scene(cfg=cfg))
        try:
            for indices in chunks:
                start, stop = int(indices[0]), int(indices[-1]) + 1
                parent, child = context.Pipe()
                specs = []
                for env_index in indices:
                    spec = dict(env_kwargs)
                    spec["seed"] = train_env_seed(seed, env_index)
                    spec.setdefault("scene_path", scene_path)
                    specs.append(spec)
                process = context.Process(
                    target=env_worker,
                    args=(child, cfg, reward_variant, specs),
                    daemon=True,
                )
                with single_threaded_children():
                    process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
                self._slices.append(slice(start, stop))
            self._observations = self._request_observations()
        except BaseException:
            self.close()
            raise

    def get_observations(self):
        return self._to_tensordict(self._observations)

    def step(self, actions):
        if self._closed:
            raise RuntimeError("MiniPacmanVecEnv is closed")
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError(
                f"actions must have shape ({self.num_envs}, {self.num_actions}), "
                f"got {tuple(actions.shape)}",
            )
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("actions must contain only finite values")
        actions_np = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        for conn, env_slice in zip(self._connections, self._slices):
            conn.send(("step", actions_np[env_slice]))

        observations = []
        rewards = []
        dones = []
        time_outs = []
        summaries = []
        for conn in self._connections:
            batch = self._receive(conn)
            for observation, reward, done, time_out, summary in batch:
                observations.append(observation)
                rewards.append(reward)
                dones.append(done)
                time_outs.append(time_out)
                if summary is not None:
                    summaries.append(summary)

        self._observations = observations
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        timeout_tensor = torch.as_tensor(time_outs, dtype=torch.bool, device=self.device)
        self.episode_length_buf += 1
        self.episode_length_buf[done_tensor] = 0
        extras = {"time_outs": timeout_tensor}
        log = self._episode_log(summaries)
        if log:
            extras["log"] = log
        return self.get_observations(), reward_tensor, done_tensor, extras

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    conn.send(("close", None))
                except (EOFError, BrokenPipeError, OSError):
                    pass
        for conn, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    self._receive(conn)
                except (EOFError, BrokenPipeError, RuntimeError):
                    pass
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            conn.close()

    def _request_observations(self) -> list[dict[str, np.ndarray]]:
        for conn in self._connections:
            conn.send(("observations", None))
        observations = []
        for conn in self._connections:
            observations.extend(self._receive(conn))
        return observations

    def _receive(self, conn):
        status, payload = conn.recv()
        if status == "error":
            raise RuntimeError(f"MiniPacmanVecEnv worker failed:\n{payload}")
        return payload

    def _to_tensordict(self, observations: list[dict[str, np.ndarray]]) -> TensorDict:
        data = {
            key: torch.as_tensor(
                np.stack([observation[key] for observation in observations]),
                dtype=torch.float32,
                device=self.device,
            )
            for key in ("actor", "critic")
        }
        return TensorDict(data, batch_size=[self.num_envs], device=self.device)

    @staticmethod
    def _episode_log(summaries: list[dict]) -> dict[str, float]:
        if not summaries:
            return {}
        thrown = sum(summary["n_thrown"] for summary in summaries)
        hits = sum(summary["n_hit"] for summary in summaries)
        goal_times = [summary["t_goal"] for summary in summaries if summary["t_goal"] is not None]
        log = {
            "/episode/success_rate": float(np.mean([summary["success"] for summary in summaries])),
            "/episode/fall_rate": float(np.mean([summary["fell"] for summary in summaries])),
            "/episode/collision_rate": hits / max(thrown, 1),
            "/episode/seconds": float(np.mean([summary["episode_seconds"] for summary in summaries])),
        }
        if goal_times:
            log["/episode/time_to_goal"] = float(np.mean(goal_times))
        interventions = [
            summary["intervention_mean"] for summary in summaries
            if summary["intervention_mean"] is not None
        ]
        if interventions:
            log["/episode/intervention_mean"] = float(np.mean(interventions))
        return log
