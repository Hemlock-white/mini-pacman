"""把很多個 DodgeEnv 平行跑，接到 rsl_rl。【你們實作】

為什麼要自己寫
--------------
rsl_rl 原本是給 GPU 上的模擬器（Isaac Lab、mjlab）用的：幾千個環境一次在 GPU 上算完。
我們的 MuJoCo 在 CPU 上，所以用多個子行程（multiprocessing），每個子行程負責幾個 DodgeEnv，
再把結果拼成 torch 張量交給 rsl_rl。瓶頸在 CPU 核心數，不在 GPU。

和課程 MT06 的對照
------------------
MT06 用一個 gymnasium 環境 ``envs.step(action_np)``，一次一步、一個環境。這裡是同樣的事，
只是一次 num_envs 個，而且 rsl_rl 要求固定的介面（rsl_rl/env/vec_env.py 的 VecEnv）：

    屬性：num_envs、num_actions、max_episode_length、episode_length_buf (torch, num_envs)、device、cfg
    get_observations() -> TensorDict({"actor": (num_envs, 589), "critic": (num_envs, 632)}, batch_size=[num_envs])
        （維度由 dodge_env.obs_layout(cfg) 決定；key 名稱要和 train_ppo.make_train_cfg 的 obs_groups 一致）
    step(actions (num_envs, num_actions) torch) -> (obs TensorDict, rewards (num_envs,), dones (num_envs,), extras)
        extras["time_outs"]：(num_envs,) 時間到而結束的（不是倒下），rsl_rl 會用它做 bootstrapping
        extras["log"]：{"/名稱": 數值}，會出現在 tensorboard，例如 "/episode/collision_rate"
    某個環境 done 時，要在 step 裡**自動 reset**，回傳的 obs 是新 episode 的第一個觀測。

建議結構
--------
    worker(conn, env_kwargs_list)：子行程，建立 len(env_kwargs_list) 個 DodgeEnv，迴圈接收指令
        ("reset", None) → 回傳所有環境的 obs
        ("step", actions_np) → 每個環境 step、done 就 reset，回傳 obs、reward、done、time_out、episode 摘要
        ("close", None) → 結束
    MiniPacmanVecEnv：開 num_workers 個子行程（multiprocessing.get_context("spawn")，Windows/Linux 都可用），
        把 num_envs 個環境平均分給它們；step 時把 actions 切片送出，再把回傳的 numpy 拼起來轉成 torch。
    每個環境的 seed：base_seed * 100000 + env_index，episode 之間用環境自己的 rng 繼續。
"""
from __future__ import annotations

import numpy as np

from .config import Config

try:
    from rsl_rl.env import VecEnv
except ImportError:          # 還沒裝 rsl_rl 時讓其他模組照樣能 import
    VecEnv = object


class MiniPacmanVecEnv(VecEnv):
    def __init__(self, cfg: Config, reward_variant: str, num_envs: int, num_workers: int = 0,
                 device: str = "cpu", seed: int = 0, **env_kwargs):
        """env_kwargs 會傳給每個 DodgeEnv（例如 filter_mode、ball_source、augment_depth）。
        num_workers=0：自動用 max(1, os.cpu_count() - 2)。"""
        raise NotImplementedError("TODO：MiniPacmanVecEnv.__init__")

    def get_observations(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
