> **Superseded (2026-11-23). Don't cite these numbers.** They come from checkpoints that break the training
> protocol: `task_cbf` was the `task` policy plus 2 PPO updates, `task_collision` had 200 updates with a collision
> penalty the config no longer uses, and all three were picked on evaluation seeds 10000-10004. Exp 2 used a scripted
> policy that never steered back to the goal, and the LQR fell under fast command changes. All of this is fixed; see
> [training_protocol.md](training_protocol.md). The text below is kept as a record.

# NVIDIA 開發實驗結果

- 日期：2026-11-21～2026-11-22
- 分支：`feature/ppo-tuning`
- 結果目錄：`outputs/results/20261121_nvidia_100seeds`
- 執行裝置：NVIDIA RTX PRO 6000 Blackwell Max-Q、PyTorch `2.7.1+cu128`

## 使用限制

這批結果是在 NVIDIA CUDA 上產生，只能作為開發驗證、模型選擇與 AMD 正式實驗前的參考。課程規則要求報告數據必須在 AMD ROCm 上產生，因此本文數值不能直接作為符合該規則的正式報告結果。

原始 episode JSONL、CSV 與圖片保存在 `outputs/results/20261121_nvidia_100seeds`。`outputs/` 已列入 `.gitignore`，不會放入 Git commit；本文則保留可追蹤的結果摘要。

## 實驗設定

- 每個 condition 使用固定 seeds `10000–10099`，共 100 episodes。
- 每回合最長 20 秒，目標距離 6 m。
- Success rate 與 fall rate 的統計單位是 episode。
- Collision rate 定義為「擊中機器人的球數 / 已發射球數」。
- Collision rate 的區間為 95% Wilson confidence interval。
- Exp 1 與 Exp 3 的 policy checkpoint：

| Reward variant | Checkpoint |
|---|---|
| `task` | `outputs/policies/task/20261121_tuned_seed0/model_1499.pt` |
| `task_collision` | `outputs/policies/task_collision/20261121_tuned_seed1/model_200.pt` |
| `task_cbf` | `outputs/policies/task_cbf/20261121_cbf1_from_task1499_seed2/model_1500.pt` |

## Exp 1：學習式安全與執行期 filter

Exp 1 比較三種 reward policy 在 lag-aware safety filter 關閉與開啟時的結果。Filter 使用 truth ball state，並在每個 control step 重新規劃。

| Policy | Filter | Success | Fall | Balls | Collision rate (95% CI) | Mean intervention |
|---|:---:|---:|---:|---:|---:|---:|
| `task` | off | 54% | 38% | 280 | 77.5% [72.3%, 82.0%] | — |
| `task` | on | 18% | 75% | 237 | 62.4% [56.1%, 68.4%] | 0.186 |
| `task_collision` | off | 19% | 22% | 278 | 79.9% [74.7%, 84.1%] | — |
| `task_collision` | on | 12% | 44% | 260 | 62.7% [56.7%, 68.3%] | 0.140 |
| `task_cbf` | off | 43% | 38% | 264 | 82.2% [77.1%, 86.3%] | — |
| `task_cbf` | on | 23% | 58% | 246 | 63.4% [57.2%, 69.2%] | 0.173 |

觀察：

- 開啟 filter 後，三種 policy 的 collision rate 分別降低 15.1、17.2、18.8 個百分點；各組開啟與關閉的 collision confidence intervals 不重疊。
- 這個安全改善伴隨明顯的任務代價。三組 success rate 都下降，fall rate 都上升。
- `task_collision` 與 `task_cbf` policy 在關閉 filter 時沒有比純 `task` policy 更低的碰撞率。這表示目前 reward-only safety training 尚未形成可靠的避球行為。
- 現階段 runtime filter 能降低球體碰撞，但 command intervention 與平衡控制之間仍需要共同調整，否則容易以跌倒和任務失敗換取較低的碰撞率。

產生的圖片：

- `exp1/exp1_collision_rate.png`
- `exp1/exp1_success_intervention.png`

## Exp 2：naive 與 lag-aware CBF

Exp 2 使用相同的 scripted straight-ahead command，只比較 safety filter 是否納入速度延遲模型；ball state 使用 truth，並在每步重新規劃。

| Filter | Success | Fall | Balls | Collision rate (95% CI) | Mean intervention | Active fraction |
|---|---:|---:|---:|---:|---:|---:|
| naive | 6% | 24% | 291 | 71.8% [66.4%, 76.7%] | 0.174 | 0.238 |
| lag-aware | 6% | 21% | 287 | 73.5% [68.1%, 78.3%] | 0.160 | 0.226 |

依球到達高度分組的 collision rate：

| Filter | Low | Mid | High |
|---|---:|---:|---:|
| naive | 56.7% | 80.0% | 73.4% |
| lag-aware | 62.1% | 80.8% | 73.8% |

觀察：

- 兩組 success rate 相同，lag-aware 的 fall rate 低 3 個百分點。
- Lag-aware collision rate 比 naive 高 1.7 個百分點，而且兩組 95% confidence intervals 大幅重疊；目前沒有證據顯示 lag-aware 較安全。
- Lag-aware 的 mean intervention 與 active fraction 略低，但並未轉化成較低的碰撞率。
- 不應只靠增加 evaluation seeds 來修正此結果。下一輪若要改善，應先檢查 lag model、CBF horizon、margin 與 robot command dynamics 是否一致。

產生的圖片：

- `exp2/exp2_collision_rate.png`
- `exp2/exp2_collision_by_height.png`

## Exp 3：球體資訊來源

Exp 3 固定使用 `task_cbf model_1500.pt` 與 lag-aware filter，比較 truth、帶延遲與雜訊的 truth，以及 depth tracker。另保留 filter-off reference。

| Ball source | Filter | Success | Fall | Balls | Collision rate (95% CI) | Mean intervention |
|---|:---:|---:|---:|---:|---:|---:|
| truth reference | off | 43% | 38% | 264 | 82.2% [77.1%, 86.3%] | — |
| truth | on | 23% | 58% | 246 | 63.4% [57.2%, 69.2%] | 0.173 |
| noisy truth | on | 24% | 56% | 249 | 71.5% [65.6%, 76.7%] | 0.158 |
| depth tracker | on | 23% | 63% | 229 | 67.2% [60.9%, 73.0%] | 0.114 |

觀察：

- Truth 的點估計 collision rate 最低。相較 truth，depth tracker 高 3.8 個百分點，noisy truth 高 8.1 個百分點。
- 三個 filter-on sensing conditions 的 confidence intervals 仍有重疊，因此不應宣稱感測來源之間已有確定的統計優劣。
- Depth tracker 的 intervention 最小、filter active fraction 也最低，但 fall rate 最高，顯示漏偵測或追蹤延遲仍可能影響 safety command。
- Filter-on 三組的 success rate 接近，但都明顯低於 filter-off reference；主要未解問題仍是 safety intervention 與任務/平衡控制的協調。

產生的圖片：

- `exp3/exp3_collision_rate.png`
- `exp3/exp3_success_rate.png`
- `exp3/exp3_min_clearance.png`

## 開發結論

1. Runtime CBF filter 在 Exp 1 與 Exp 3 中一致降低 collision rate，但目前會降低任務成功率並增加跌倒。
2. Safety reward training 尚未單獨學出可靠的避球 policy；checkpoint selection 仍主要保留的是任務能力。
3. 目前的 lag-aware filter 沒有勝過 naive filter，需先校正 dynamics/filter 參數再做新一輪實驗。
4. Depth tracker 的結果介於 truth 與 noisy truth 之間，證明完整 perception-to-control 流程可以執行，但仍有改善空間。
5. 下一個工程重點不是盲目延長 PPO 或增加 seed，而是降低 filter 對平衡控制的破壞，並重新驗證 lag-aware model。

## 在 AMD ROCm 重跑正式數據

在新的結果目錄執行，避免混入 NVIDIA JSONL：

```bash
conda activate mini-pacman

python run_mini_pacman_experiments.py \
  --part exp1 exp2 exp3 \
  --seeds 100 \
  --device cuda:0 \
  --out outputs/results/amd_rocm_100seeds

python run_mini_pacman_experiments.py \
  --part summary \
  --device cuda:0 \
  --out outputs/results/amd_rocm_100seeds
```

執行前後應確認 `run_info.jsonl` 中 `device.torch_hip` 有值、`device.rocm` 為 `true`，再將 AMD 結果引用到正式報告。
