# PPO 調整紀錄

日期：2026-11-21  
分支：`feature/ppo-tuning`  
調整 commit：`0c1e02b`  

## 目的

第一版 `task` policy 雖然完成 5,000 iterations，卻沒有學會抵達目標。本文件記錄問題現象、診斷過程、嘗試過的設定，以及目前建議用於重新訓練三個 policy 的參數。

這些測試是在 NVIDIA RTX PRO 6000 Blackwell 上執行。專案原始課程規則要求報告數據必須來自 AMD ROCm，因此本文中的 NVIDIA 結果只能用於開發與設定驗證，不能直接當作符合該規則的正式報告數據。

## 第一版訓練結果

舊 checkpoint：

```text
outputs/policies/task/20261121_nvidia_seed0/model_4999.pt
```

原始 PPO 參數：

| 參數 | 原始值 |
|---|---:|
| `init_std` | `1.0` |
| `learning_rate` | `1e-3` |
| `schedule` | `adaptive` |
| `entropy_coef` | `0.005` |

原始 task reward：

| 參數 | 原始值 |
|---|---:|
| `progress_weight` | `1.0` |
| `goal_bonus` | `10.0` |
| `collision_penalty` | `5.0` |
| `fall_penalty` | `200.0` |

觀察結果：

- 訓練完成 `5000/5000`，但整段訓練的 success rate 最大值仍為 `0`。
- 固定 evaluation seeds `10000`、`10001`、`10002` 的結果為 `0/3` 成功。
- 策略不會跌倒，但只移動約 `0.14 m`，目標距離是 `6 m`。
- deterministic actor 輸出嚴重超出標準化動作範圍，平均約為 `[8.04, -4.47, 6.75]`，進入環境後長期被 clip 成接近 `[1, -1, 1]`，造成高速轉向。
- adaptive learning rate 在 iteration 2 就曾降到約 `1e-5`，之後缺乏穩定、持續的有效更新。

## 問題分析

### 初始探索幅度過大

動作包含速度、角速度與蹲伏速度，皆會在環境中限制到 `[-1, 1]`。`init_std=1.0` 使初始 Gaussian policy 經常輸出飽和動作，機器人容易高速轉向或跌倒，難以探索平穩前進。

### Adaptive learning rate 過早降低

第一版在訓練初期就觸發較大的 KL divergence，adaptive schedule 很快把 learning rate 降到下限附近。策略因此容易停留在「不跌倒但也不前進」的局部解。

### 任務 reward 與跌倒懲罰比例失衡

原始 progress reward 是每步接近目標的距離乘以 1。完成整段 `6 m` 路程，累積 progress 最多約 `+6`；跌倒一次則是 `-200`。對 PPO 而言，保守站著不動比嘗試前進更容易避免巨大負回饋。

### Entropy 太高或太低都不理想

- `entropy_coef=0.005`：action std 從 `0.2` 上升到約 `0.53`，後期策略再次不穩定。
- `entropy_coef=0`：action std 降到約 `0.09`，400 iterations 仍沒有找到目標，探索不足。
- `entropy_coef=0.001`：保留少量探索，同時避免 action std 持續膨脹，是目前採用的折衷值。

## 調整實驗摘要

以下都是開發用 smoke runs，不是正式實驗統計。

| 設定 | 結果 |
|---|---|
| `init_std=0.2`、fixed LR `3e-4`、entropy `0.005`、原 reward | 300 iterations 開始出現成功；iteration 998 的 deterministic 評估只有 `1/5` 成功、`4/5` 跌倒，後期探索過強 |
| entropy `0`、原 reward | action std 降到約 `0.09`；400 iterations success 仍為 `0`，探索不足 |
| entropy `0.001`、原 reward | 曾短暫出現成功，但訓練不穩定，reward 比例仍使策略偏向避免移動 |
| entropy `0.001`、調高 progress/goal reward | 300 iterations 的最近 50 批平均 success 約 `71.1%`、fall 約 `9.9%`；最後一批 success `80%`、fall `0%` |

## 目前採用的設定

`minipacman/config.py` 中的建議設定：

| 類別 | 參數 | 新值 | 理由 |
|---|---|---:|---|
| PPO | `init_std` | `0.2` | 降低初始飽和動作與亂轉 |
| PPO | `learning_rate` | `3e-4` | 使用較保守但仍能有效學習的更新幅度 |
| PPO | `schedule` | `fixed` | 避免初期 KL 使 learning rate 過早降到底 |
| PPO | `entropy_coef` | `0.001` | 保留少量探索，避免 std 過度增長或完全塌縮 |
| Reward | `progress_weight` | `10.0` | 強化朝目標移動的 dense signal |
| Reward | `goal_bonus` | `50.0` | 讓完成任務明顯優於站著不動 |
| Reward | `collision_penalty` | `100.0` | 50 太弱、200 使策略放棄任務後的折衷值 |

`fall_penalty=200`、`at_goal_weight=0.1` 與 `action_rate_weight=0.05` 保持不變。

## 最終 smoke validation

驗證 checkpoint：

```text
/tmp/mini-pacman-tuned-reward-300/model_299.pt
```

訓練設定：

- reward：`task`
- seed：`4`
- environments：`256`
- iterations：`300`
- device：NVIDIA `cuda:0`

固定 deterministic evaluation seeds `10000` 到 `10004`：

| Seed | Success | Fell | 到達時間 |
|---:|:---:|:---:|---:|
| 10000 | 是 | 否 | 17.16 s |
| 10001 | 是 | 否 | 19.98 s |
| 10002 | 是 | 否 | 17.92 s |
| 10003 | 是 | 否 | 17.40 s |
| 10004 | 是 | 否 | 18.80 s |

結果為 `5/5` 成功、`0/5` 跌倒。樣本數很小，只能證明調整後的 pipeline 能學會任務，不能取代正式的多 seed 評估。

程式 regression tests：

```text
76 passed
```

## 正式重新訓練

不要覆蓋舊的失敗 run，使用新的輸出目錄：

```bash
conda activate mini-pacman

python train_mini_pacman_policy.py \
  --reward task \
  --seed 0 \
  --iterations 5000 \
  --num-envs 256 \
  --device cuda:0 \
  --out outputs/policies/task/20261121_tuned_seed0

python train_mini_pacman_policy.py \
  --reward task_collision \
  --seed 1 \
  --iterations 5000 \
  --num-envs 256 \
  --device cuda:0 \
  --out outputs/policies/task_collision/20261121_tuned_seed1

python train_mini_pacman_policy.py \
  --reward task_cbf \
  --seed 2 \
  --iterations 5000 \
  --num-envs 256 \
  --device cuda:0 \
  --out outputs/policies/task_cbf/20261121_tuned_seed2
```

建議先讓新的 `task` policy 跑到 300–500 iterations，使用固定 evaluation seeds 做一次 deterministic gate。只有 success rate 與 fall rate 正常，才繼續完整 5,000 iterations 與另外兩個 reward variants。

### Seed 0 的 500-iteration gate

固定 seeds `10000–10004` 的 checkpoint 比較：

| Checkpoint | Success | Fell | 結論 |
|---|---:|---:|---|
| `model_100.pt` | 0/5 | 1/5 | 不採用 |
| `model_200.pt` | 0/5 | 4/5 | 不採用 |
| `model_300.pt` | 0/5 | 3/5 | 不採用 |
| `model_400.pt` | 0/5 | 0/5 | 穩定但沒有到達目標 |
| `model_499.pt` | 2/5 | 0/5 | 五者最佳，作為第二階段起點 |

`model_499.pt` 的成功回合平均到達時間約為 14.88 秒。它尚未達到正式 gate 標準，因此先把 learning rate 降為 `1e-4`，接續到約 iteration 1500：

```bash
python train_mini_pacman_policy.py \
  --reward task \
  --seed 0 \
  --iterations 1001 \
  --num-envs 256 \
  --learning-rate 1e-4 \
  --device cuda:0 \
  --resume outputs/policies/task/20261121_tuned_seed0/model_499.pt \
  --out outputs/policies/task/20261121_tuned_seed0
```

訓練程式會在載入 optimizer state 後重新套用 `--learning-rate`，避免 checkpoint 中原本的 `3e-4` 覆蓋第二階段設定。

### Task policy 最終 checkpoint 選擇

第二階段的 `model_1499.pt` 在固定 seeds `10000–10004` 得到 `3/5` 成功、`0/5` 跌倒，因此繼續以 `1e-4` 訓練到 iteration 4999。後段訓練並未繼續改善 deterministic policy：

| Checkpoint | Success | Fell |
|---|---:|---:|
| `model_1499.pt` | 3/5 | 0/5 |
| `model_2200.pt` | 1/5 | 4/5 |
| `model_2300.pt` | 0/5 | 4/5 |
| `model_2400.pt` | 0/5 | 2/5 |
| `model_4999.pt` | 2/5 | 3/5 |

因此 `task` 最終選用：

```text
outputs/policies/task/20261121_tuned_seed0/model_1499.pt
```

這也說明 checkpoint 選擇不能只取 iteration 最大者；正式流程必須用固定 evaluation seeds 做 deterministic model selection。

### Task-collision 第一輪結果

`collision_penalty=50` 時，第一階段與低 learning-rate 第二階段都沒有真正學會避球：

| Checkpoint | Success | Fell | Collision rate |
|---|---:|---:|---:|
| phase 1 `model_200.pt` | 2/5 | 1/5 | 69.2% |
| phase 2 `model_1100.pt` | 1/5 | 1/5 | 90.9% |
| phase 2 `model_1200.pt` | 1/5 | 1/5 | 100% |
| phase 2 `model_1300.pt` | 1/5 | 1/5 | 91.7% |
| phase 2 `model_1499.pt` | 4/5 | 1/5 | 81.8% |

第二階段提升的是抵達目標的能力，碰撞率反而沒有改善。調高後的 task reward 可超過 +100，`-50` 的單次碰撞懲罰不足以主導策略選擇，因此後續將 `collision_penalty` 提高為 `200`，與跌倒懲罰同級，再重新訓練 `task_collision`。

`collision_penalty=200` 的 500-iteration gate 又走到另一個極端：所有固定-seed checkpoints 都是 `0/5` 成功。雖然部分 checkpoint 的碰撞率下降，但主要原因包含提早跌倒或放棄前進，不能視為成功避球。

| Checkpoint | Success | Fell | Collision rate |
|---|---:|---:|---:|
| `model_100.pt` | 0/5 | 5/5 | 50.0% |
| `model_200.pt` | 0/5 | 1/5 | 84.6% |
| `model_300.pt` | 0/5 | 2/5 | 72.7% |
| `model_400.pt` | 0/5 | 3/5 | 100% |
| `model_499.pt` | 0/5 | 4/5 | 70.0% |

因此下一輪採用中間值 `collision_penalty=100`，並從 penalty-50 階段仍保有任務能力的 `model_200.pt` 開始，以 `1e-4` 做 safety fine-tuning，而不是重新從隨機 policy 開始。

penalty-100 fine-tuning 到 iteration 699 後仍沒有改善：候選 checkpoints 的 collision rate 為 80%–92.3%，且成功率沒有超過原本的 phase-1 `model_200.pt`。因此停止繼續投入 `task_collision` 訓練，最終選用：

```text
outputs/policies/task_collision/20261121_tuned_seed1/model_200.pt
```

在相同五個 seeds 上，它得到 `2/5` success、`1/5` fall、`9/13 = 69.2%` collision；選定的 `task` policy 約為 `10/13 = 76.9%` collision。改善幅度有限，但這是目前唯一同時保留部分任務能力並降低碰撞的 checkpoint。

進入 `task_cbf` 前，將 `CBFConfig.reward_weight` 從 `0.27` 調為 `2.7`。原因是 task progress reward 已放大十倍；CBF 是逐步提供的 dense safety signal，也需要維持相同的相對尺度。

從隨機 policy 使用 CBF weight 2.7 訓練 500 iterations 後，所有固定-seed checkpoints 都是 `0/5` success。`model_499.pt` 達到 `0/5` fall、`5/13 = 38.5%` collision，證明 dense CBF signal 能學到安全行為，但策略完全放棄抵達目標。

因此後續改用 curriculum fine-tuning：從已具備任務能力的 `task model_1499.pt` 開始，將 CBF weight 降至 `1.0`，並以 learning rate `1e-4` 加入安全行為。這避免要求隨機 policy 同時學平衡、走路、到達目標與避球。

## 查看訓練進度

查看最新 checkpoint：

```bash
find outputs/policies/task/20261121_tuned_seed0 \
  -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1
```

啟動 TensorBoard：

```bash
tensorboard --logdir outputs/policies --port 6006
```

瀏覽器開啟 `http://localhost:6006`，主要觀察：

- `/episode/success_rate`
- `/episode/fall_rate`
- `/episode/collision_rate`
- `Train/mean_reward`
- `Policy/mean_std`
- `Loss/learning_rate`

## 後續流程

1. 以新設定重新訓練並驗證 `task`。
2. 訓練 `task_collision` 與 `task_cbf`。
3. 將選定 checkpoint 填入 `run_mini_pacman_experiments.py` 的 `POLICY_FILES`。
4. 執行 Exp 1 與 Exp 3。
5. Exp 2 已可獨立使用 scripted policy 執行，不依賴 PPO checkpoint。
