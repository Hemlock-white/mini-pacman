"""評估：用固定的 seed 跑一個實驗條件，存下每個 episode 的結果。【Condition 已寫好；函式你們實作】

一個「實驗條件」(Condition) = 哪個 policy × 過濾器開不開、哪一版 × 球資訊從哪來。
run_mini_pacman_experiments.py 會建立很多個 Condition，對每個呼叫 run_condition()。

EPISODE_SCHEMA：每個 episode 存成一個 dict（.jsonl 的一行），report.summarize() 依賴這些欄位
------------------------------------------------------------------------------------------------
    "condition"           str     條件名稱
    "seed"                int
    "success"             bool    有沒有在時間內到達終點而且沒倒
    "fell"                bool
    "t_goal"              float | None   第一次到達終點的時間 (s)
    "n_thrown"            int     發射了幾顆球
    "n_hit"               int     幾顆打到機器人
    "balls"               list[dict]     launcher.episode_records()（每顆球的模式、高度、速度、大小、有沒有打中、最小距離）
    "intervention_mean"   float | None   過濾器平均改了多少（沒開過濾器 = None）
    "filter_active_frac"  float | None   過濾器有介入的步數比例
    "episode_seconds"     float   實際跑了多久
    "device"              str     PyTorch 裝置（policy 推論）
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import Config


@dataclass
class Condition:
    """一個實驗條件。"""
    name: str
    policy: str                     # policy checkpoint 路徑，或 "scripted"（寫死的直行指令，實驗 2 用）
    reward_variant: str             # 這個 policy 是用哪種 reward 練的（紀錄用）
    filter_mode: str | None         # None / "naive" / "lag_aware"
    replan: str = "every_step"      # "every_step" / "commit"
    ball_source: str = "truth"      # 過濾器的球資訊來源："truth" / "noisy_truth" / "depth_tracker"

    def to_dict(self) -> dict:
        return asdict(self)


class ScriptedPolicy:
    """寫死的上層指令：朝終點直直開，不閃球（實驗 2 用，讓結果只反映過濾器）。

    act(obs) 回傳 action ∈ [-1, 1]^3，對應 v = 0.5 m/s、w = 0、sdot = -（保持站姿）。
    到了終點就停（v = 0）。到終點的距離：actor 觀測裡的 obs["actor"][obs_layout(cfg)["actor"]["goal"]]
    （車頭座標的 (前, 左)），或直接從 env 拿。
    """

    def __init__(self, cfg: Config):
        raise NotImplementedError("TODO：ScriptedPolicy")

    def act(self, obs: dict, env) -> np.ndarray:
        raise NotImplementedError


def load_policy(path: str | Path, cfg: Config, device: str):
    """載入 rsl_rl 訓練出的 checkpoint，回傳一個函式 obs_actor (np.ndarray) -> action (np.ndarray)。

    rsl_rl 5.4.0 的 checkpoint 是一個 dict（已驗證）：
        {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"}
    建議：建立和訓練時同樣的 MLPModel（train_ppo.make_train_cfg 的 "actor" 設定；MLPModel 的建構子需要
    一份範例觀測 TensorDict 來決定輸入維度），load_state_dict(ck["actor_state_dict"])，
    評估時用分布的平均值（不要取樣），在 torch.no_grad() 裡跑。
    或者更簡單：建一個 OnPolicyRunner（用評估用的 vec env），runner.load(path)，
    policy = runner.get_inference_policy(device)。
    device 要用 ROCm GPU（課程規定 CPU-only 的結果不算）。
    """
    raise NotImplementedError("TODO：load_policy")


def run_episode(condition: Condition, seed: int, cfg: Config, policy, device: str) -> dict:
    """跑一個 episode，回傳 EPISODE_SCHEMA 格式的 dict。

    建立 DodgeEnv(cfg, condition.reward_variant, condition.filter_mode, condition.replan,
                   condition.ball_source, augment_depth=False, seed=seed)，
    obs = env.reset(seed)，迴圈 step 到 terminated 或 truncated，最後用 env.episode_summary()，
    再補上 "condition"、"seed"、"device"。
    （環境建立很花時間時，可以讓 run_condition 重複使用同一個 env，只在每個 seed 呼叫 reset(seed)。）
    """
    raise NotImplementedError("TODO：run_episode")


def run_condition(condition: Condition, seeds: list[int], cfg: Config, out_path: Path, device: str,
                  resume: bool = True) -> list[dict]:
    """對每個 seed 跑 run_episode，每跑完一個就 append 到 out_path（.jsonl），回傳全部 episode。

    resume=True：out_path 已經有的 seed 就跳過（用 report.load_jsonl 讀出來），中斷後可以接著跑。
    resume=False：先刪掉 out_path 再全部重跑。
    可以用 multiprocessing 平行跑不同 seed（每個 seed 獨立，結果和順序無關）。
    """
    raise NotImplementedError("TODO：run_condition")
