# Training and evaluation protocol

Written 2026-11-23, after reviewing the first round of results. Read this before training or evaluating.

## This branch: `feature/ppo-kl`, the KL control group (2026-11-24)

Branched from `feature/ppo-tuning` at `6203dcc`. **The only training difference is the learning-rate schedule:**

| | `feature/ppo-tuning` | `feature/ppo-kl` |
|---|---|---|
| `PPOConfig.schedule` | `fixed` | `adaptive` |
| `PPOConfig.desired_kl` | 0.01 (unused) | 0.01 |
| `PPOConfig.learning_rate` | 3e-4 throughout | 3e-4 at the start, then set by the KL |

`init_std`, `entropy_coef`, epochs, minibatches, rewards, environment, protocol and seeds are the same.

Why: with the fixed 3e-4, `task` and `task_collision` collapsed after about 2000 and 4000 iterations (`task_cbf`
didn't). The old note "adaptive KL hit its 1e-5 floor after two iterations" came from the first full run on the old
code (unstable LQR, `init_std` 1.0, `entropy_coef` 0.005), so the adaptive schedule was never tried fairly.

How rsl_rl 5.4.0 adapts the rate (`rsl_rl/algorithms/ppo.py`, `PPO.update`): for **every minibatch** (5 epochs x 4
= 20 times per iteration) it computes KL(old || new) of the Gaussian policy, summed over the 3 actions and averaged
over the minibatch, before that minibatch's gradient step. KL > 0.02 divides the rate by 1.5, KL < 0.005 multiplies
it by 1.5, and the rate stays within [1e-5, 1e-2].

What to watch in TensorBoard:
* `Loss/lr_mean`: the average rate of the 20 gradient steps of the iteration (added in this branch by
  `minipacman/ppo_logging.py`; training is unchanged). **This is the one to read.** Sitting at 1e-5 means the
  schedule has stopped learning.
* `Loss/learning_rate`: only the rate after the last minibatch (rsl_rl logs it already). It is misleading here: the
  KL is measured against the rollout policy, so it grows within an iteration; the rate climbs (x1.5 per minibatch)
  while the KL is small and falls back (/1.5) once it passes 0.02. In the 40-iteration smoke test (task, seed 0,
  256 envs) this value was 1e-5 from iteration 2 on, while the rate inside iterations 5-34 had a median of 7.6e-5.
* `Loss/mean_kl`: the average of the 20 minibatch KLs (also added by `ppo_logging.py`). Around 0.005-0.02 is the
  band where the rate holds still.

Early check: `feature/ppo-tuning`'s `task` run reached a validation success rate of 0.75 at `model_500`. If this
branch's `task` run still shows no `/episode/success_rate` after about 500 iterations, stop and look before
training the other two rewards.

Keep these runs apart from `feature/ppo-tuning`'s: train in a separate worktree (each worktree has its own
`outputs/`). The config fingerprint includes the schedule, so the experiments refuse a mix of runs from the two
branches.

## What went wrong in round 1, and what the code does now

| Round 1 problem | What the code does now |
|---|---|
| `task_cbf` was the `task` policy plus 2 PPO updates (resumed from `model_1499.pt`, kept `model_1500.pt`); `task_collision` had 200 updates, with a collision penalty (50) the config no longer had (100). Exp 1 compared three different training budgets. | `train_mini_pacman_policy.py` writes a manifest (`run_info.json`) for every run. `--resume` can only continue an interrupted run in its own folder with identical settings. Exp 1 refuses the three policies unless their seed, number of envs, number of PPO updates, config and checkpoint rule all match. (`minipacman/protocol.py`) |
| Checkpoints were picked on evaluation seeds 10000-10004, which are also in the 100 reported seeds. | An experiment can only use a run's **final** checkpoint, or the one `--part select` picked on separate **validation seeds** (20000-20019). A hand-picked checkpoint is refused. |
| The LQR fell under fast command changes: random high-level commands fell in 26/28 episodes, mostly while crouched, and the filter's sudden commands made robots fall. | More pitch-rate and speed damping, a leash that stops the reference running away from a robot that can't keep up, and the robot crouches only as far as its speed allows (a crouch request also slows it down). Random commands: 0/42 falls; extreme switching commands: 0/42 falls. |
| Exp 2's scripted policy never steered back after the filter turned it, so its 6 % success measured the script. | The scripted policy turns towards the goal. With 36 development seeds (30000+): no filter 93 % of balls hit, naive filter 73 %, lag-aware 51 %; 36/36 reached the goal in each, 0 falls. |
| The lag-aware filter used a first-order guess of the robot's response. | Lag-aware predicts with the same rate limits and crouch rules as the real reference, plus a measured 0.24 s tracking lag. The filter also prefers to keep its previous output, so it doesn't jump between extreme commands every 20 ms. |

The state estimator (IMU + encoders) is still **off** (`use_true_state=True`). It works for gentle driving but falls
under aggressive commands (19/28 in the stress test). Its base height doesn't follow the crouch and it uses the axle
speed as the base speed. Report this as a limitation.

## 1. Train the three policies (same settings for all three)

```bash
# once per machine: check the setup
python run_mini_pacman_experiments.py --part check

# the three runs: same --seed, same --iterations, same --num-envs; only --reward changes
python train_mini_pacman_policy.py --reward task           --seed 0 --out outputs/policies/task/run1
python train_mini_pacman_policy.py --reward task_collision --seed 0 --out outputs/policies/task_collision/run1
python train_mini_pacman_policy.py --reward task_cbf       --seed 0 --out outputs/policies/task_cbf/run1
```

* Each run uses `--num-workers` simulation processes (default: CPU cores - 2). The three runs can go one after
  another, or at the same time with `--num-workers 7` each on a 24-core machine; the total time is about the same
  (the simulation is CPU-bound). The worker count only changes speed, so it may differ between runs.
* Before the real runs, time a short one: `python train_mini_pacman_policy.py --reward task --iterations 20 --out
  outputs/policies/timing_test` and read rsl_rl's "Iteration time". 5000 iterations take 5000 times that.
* If a run crashes, continue it in place with the same command plus `--resume outputs/policies/<reward>/run1/model_<k>.pt`.
* Don't change `config.py` between the three runs. The manifest stores a fingerprint of every training setting, and
  the experiments refuse runs that don't match the current config.
* Course rule: report only runs trained on AMD ROCm. The manifest records `device.rocm`; other runs print a warning.

## 2. (Optional) pick checkpoints on the validation seeds

```bash
python run_mini_pacman_experiments.py --part select --workers 16
```

For every run it compares every 500th checkpoint plus the final one on the 20 validation seeds, with the filter off.
It picks the highest success rate, then the lowest fall rate, then the lowest collision rate, and writes
`<run>/selection.json`. Run it for all three or for none: Exp 1 refuses a mix. Without it, the final checkpoints are used.

## 3. Run the experiments

`POLICY_FILES` in `run_mini_pacman_experiments.py` already points at `outputs/policies/<reward>/run1`.

```bash
python run_mini_pacman_experiments.py --part exp1 exp2 exp3 summary --workers 16
```

Each part prints which checkpoint it uses and why (`final` or `validation`). A policy that breaks a rule stops that
part with the reason.

## Seeds

| Use | Seeds |
|---|---|
| Training environments | `seed * 100000 + env_index` (seed 0: 0-255) |
| Validation (`--part select`) | 20000-20019 |
| Evaluation (reported numbers) | 10000-10099; never used to pick anything |
| Development checks (as in this note) | 30000+ |

`protocol.check_seed_sets` refuses overlapping ranges.
