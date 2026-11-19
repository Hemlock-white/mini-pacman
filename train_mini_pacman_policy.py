"""訓練一個 Mini PAC-MAN 的 PPO policy（三種 reward 各跑一次）。

用法：
    python train_mini_pacman_policy.py --reward task
    python train_mini_pacman_policy.py --reward task_collision --seed 1
    python train_mini_pacman_policy.py --reward task_cbf --iterations 3000 --num-envs 128
    python train_mini_pacman_policy.py --reward task --quick          # 幾十秒的小訓練，確認程式能跑
    python train_mini_pacman_policy.py --reward task --resume outputs/policies/task/<run>/model_1000.pt

結果存在 outputs/policies/<reward>/<日期時間>/：checkpoint (model_*.pt)、tensorboard 紀錄、
run_info.json（GPU、版本、git commit、所有參數）。
訓練完把要用的 checkpoint 路徑填進 run_mini_pacman_experiments.py 的 POLICY_FILES。

課程規定：報告用的 policy 要在 AMD ROCm GPU 上訓練；不是 ROCm 時會印警告。
"""
import argparse
import json
import time
from pathlib import Path

from minipacman.config import OUTPUT_DIR, default_config
from minipacman.runtime import device_report, pick_device, warn_if_not_rocm


def main() -> None:
    cfg = default_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward", required=True, choices=cfg.exp.reward_variants)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=None, help=f"預設 {cfg.ppo.max_iterations}")
    ap.add_argument("--num-envs", type=int, default=None, help=f"預設 {cfg.ppo.num_envs}")
    ap.add_argument("--device", default=None, help="預設：有 GPU 用 cuda:0（ROCm 也叫 cuda）")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="2 個 iteration、8 個環境，只確認能跑")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.quick:
        args.iterations, args.num_envs = 2, 8
    device = pick_device(args.device)
    info = device_report(device)
    warn_if_not_rocm(info)
    log_dir = args.out or OUTPUT_DIR / "policies" / args.reward / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run_info.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "device": info, "config": cfg.to_dict()},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"訓練 {args.reward}，seed {args.seed}，device {device}（{info.get('gpu_name')}），輸出 {log_dir}")

    from minipacman.train_ppo import train
    try:
        ckpt = train(cfg, args.reward, args.seed, log_dir, device, max_iterations=args.iterations,
                     num_envs=args.num_envs, resume=args.resume)
    except NotImplementedError as e:
        raise SystemExit(f"[TODO] 還沒實作：{e}")
    print(f"完成：{ckpt}")
    print("把這個路徑填進 run_mini_pacman_experiments.py 的 POLICY_FILES。")


if __name__ == "__main__":
    main()
