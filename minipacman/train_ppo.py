"""用 rsl_rl 5.4.0 訓練 PPO。【train() 你們實作；make_train_cfg() 已經寫好參數】

rsl_rl 和課程 MT06 的對照（給沒用過 rsl_rl 的組員）
-------------------------------------------------
PPO 演算法本身和 MT06 一模一樣，只是 rsl_rl 把每一塊包成類別，而且一次跑很多個環境：

============================  ===========================================  =========================
MT06（MT06_RL_PPO.ipynb）       rsl_rl 5.4.0                                  這裡的設定
============================  ===========================================  =========================
build_actor()                 rsl_rl.models.MLPModel（"actor"）              actor.hidden_dims
build_value_function()        rsl_rl.models.MLPModel（"critic"）             critic.hidden_dims
utils.PPOBuffer               rsl_rl.storage.RolloutStorage                 （自動建立）
compute_gae()                 PPO.compute_returns()（gamma、lam）            algorithm.gamma / lam
ppo_update()                  PPO.update()                                  algorithm.*
collect_rollout()             OnPolicyRunner.learn() 裡的收集迴圈             num_steps_per_env
run_training()                OnPolicyRunner.learn(num_learning_iterations)  max_iterations
args.num_steps = 2048         num_steps_per_env × num_envs（24 × 256）        每次更新收集的步數
args.update_epochs = 10       num_learning_epochs = 5
args.num_minibatches = 32     num_mini_batches = 4
args.clip_coef = 0.2          clip_param = 0.2
args.ent_coef = 0.0           entropy_coef = 0.005
args.vf_coef = 0.5            value_loss_coef = 1.0
args.max_grad_norm = 0.5      max_grad_norm = 1.0
args.anneal_lr                schedule = "adaptive"（依 KL 自動調學習率）     desired_kl = 0.01
一個環境 envs.step()           MiniPacmanVecEnv.step()：num_envs 個一起        vec_env.py
============================  ===========================================  =========================

多出來的兩個概念：
* obs_groups：觀測分組。actor 和 critic 可以看不同的東西（非對稱 actor-critic），
  例如 critic 多看真實的球狀態，訓練比較穩；部署時只用 actor。
* obs_normalization：自動把觀測正規化（MT06 沒有）。

訓練完的 checkpoint 存在 log_dir/model_<iteration>.pt。要用它做實驗時，
把路徑填進 run_mini_pacman_experiments.py 最上面的 POLICY_FILES。
"""
from __future__ import annotations

from pathlib import Path

from .config import Config


def make_train_cfg(cfg: Config, reward_variant: str) -> dict:
    """rsl_rl OnPolicyRunner 的 train_cfg（參數來自 config.PPOConfig，大多照 PAC-MAN 的 rl_cfg.py）。"""
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
    """訓練一個 policy，回傳最後一個 checkpoint 的路徑。

    步驟：
      1. torch.manual_seed(seed)、np.random.seed(seed)
      2. env = MiniPacmanVecEnv(cfg, reward_variant, num_envs or cfg.ppo.num_envs, cfg.ppo.num_workers,
                                device, seed, filter_mode=None, augment_depth=True)
         （訓練時不開過濾器，「教會」的效果才是單純的）
      3. runner = rsl_rl.runners.OnPolicyRunner(env, make_train_cfg(cfg, reward_variant), str(log_dir), device)
      4. resume 不是 None 時 runner.load(str(resume))
      5. runner.learn(num_learning_iterations=max_iterations or cfg.ppo.max_iterations)
      6. env.close()；回傳 log_dir 裡編號最大的 model_*.pt

    注意：rsl_rl 的 make_train_cfg 字典會被 runner 修改（pop class_name），每次都重新呼叫 make_train_cfg。
    """
    raise NotImplementedError("TODO：train（見 docstring）")
