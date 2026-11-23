"""Subprocess side of MiniPacmanVecEnv (and of the parallel evaluation).

Kept apart from vec_env.py on purpose: a "spawn" subprocess imports the module that holds its target function, and
vec_env.py imports torch. Workers only need numpy + MuJoCo, so they stay a few hundred MB lighter each.
"""
from __future__ import annotations

import os
import traceback
from contextlib import contextmanager

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@contextmanager
def single_threaded_children():
    """Start subprocesses with one BLAS/OpenMP thread each (set while they start; they inherit the environment).

    Without it every worker opens one math thread per core: dozens of workers then fight over the cores, and on
    Windows starting them could hang.
    """
    saved = {name: os.environ.get(name) for name in THREAD_VARS}
    os.environ.update({name: "1" for name in THREAD_VARS})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def env_worker(conn, cfg, reward_variant: str, env_specs: list[dict]) -> None:
    """Own a small group of MuJoCo environments in one subprocess."""
    from .dodge_env import DodgeEnv

    try:
        envs = [DodgeEnv(cfg, reward_variant=reward_variant, **spec) for spec in env_specs]
        observations = [env.reset() for env in envs]
        while True:
            command, payload = conn.recv()
            if command == "observations":
                conn.send(("ok", observations))
            elif command == "step":
                batch = []
                for env, action in zip(envs, payload):
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    summary = info.get("episode") if done else None
                    if done:
                        observation = env.reset()
                    batch.append((
                        observation,
                        float(reward),
                        done,
                        bool(truncated and not terminated),
                        summary,
                    ))
                observations = [item[0] for item in batch]
                conn.send(("ok", batch))
            elif command == "close":
                conn.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown worker command: {command!r}")
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        conn.close()
