"""Train one Mini PAC-MAN PPO policy (run once for each of the three rewards).

Usage:
    python train_mini_pacman_policy.py --reward task
    python train_mini_pacman_policy.py --reward task_collision --seed 1
    python train_mini_pacman_policy.py --reward task_cbf --iterations 3000 --num-envs 128
    python train_mini_pacman_policy.py --reward task --quick          # tiny run of a few seconds, just to check it works
    python train_mini_pacman_policy.py --reward task --resume outputs/policies/task/<run>/model_1000.pt

Results go to outputs/policies/<reward>/<date_time>/: checkpoints (model_*.pt), tensorboard logs, and run_info.json
(GPU, versions, git commit, every parameter).
When training is done, put the checkpoint you want in POLICY_FILES in run_mini_pacman_experiments.py.

Course rule: the policies in the report must be trained on an AMD ROCm GPU; a warning is printed otherwise.
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
    ap.add_argument("--iterations", type=int, default=None, help=f"default {cfg.ppo.max_iterations}")
    ap.add_argument("--num-envs", type=int, default=None, help=f"default {cfg.ppo.num_envs}")
    ap.add_argument("--device", default=None, help="default: cuda:0 if there is a GPU (ROCm is also called cuda)")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="2 iterations, 8 environments, just to check it runs")
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
    print(f"Training {args.reward}, seed {args.seed}, device {device} ({info.get('gpu_name')}), output {log_dir}")

    from minipacman.train_ppo import train
    try:
        ckpt = train(cfg, args.reward, args.seed, log_dir, device, max_iterations=args.iterations,
                     num_envs=args.num_envs, resume=args.resume)
    except NotImplementedError as e:
        raise SystemExit(f"[TODO] not implemented yet: {e}")
    print(f"Done: {ckpt}")
    print("Put this path in POLICY_FILES in run_mini_pacman_experiments.py.")


if __name__ == "__main__":
    main()
