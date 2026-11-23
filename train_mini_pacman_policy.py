"""Train one Mini PAC-MAN PPO policy (run once for each of the three rewards).

Usage (Exp 1 needs all three trained the same way: same seed, same --iterations, same --num-envs, same config):
    python train_mini_pacman_policy.py --reward task           --seed 0 --out outputs/policies/task/run1
    python train_mini_pacman_policy.py --reward task_collision --seed 0 --out outputs/policies/task_collision/run1
    python train_mini_pacman_policy.py --reward task_cbf       --seed 0 --out outputs/policies/task_cbf/run1
    python train_mini_pacman_policy.py --reward task --quick          # tiny run of a few seconds, just to check it works
    python train_mini_pacman_policy.py --reward task --resume outputs/policies/task/run1/model_1000.pt   # after a crash

Results go to --out (default outputs/policies/<reward>/<date_time>/): checkpoints (model_*.pt), tensorboard logs, and
run_info.json (the protocol manifest: reward, seed, envs, PPO updates, config fingerprint, GPU, versions).
--resume only continues an interrupted run in its own directory with the same settings; starting from another run's
checkpoint (fine-tuning) is refused, because it would make the Exp 1 comparison unfair (see minipacman/protocol.py).
When training is done, put the run directory in POLICY_FILES in run_mini_pacman_experiments.py.

Course rule: the policies in the report must be trained on an AMD ROCm GPU; a warning is printed otherwise.
"""
import argparse
import time
from pathlib import Path

from minipacman.config import OUTPUT_DIR, default_config
from minipacman.runtime import device_report, pick_device, warn_if_not_rocm


def main() -> None:
    cfg = default_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward", required=True, choices=cfg.exp.reward_variants)
    ap.add_argument("--seed", type=int, default=None, help="default 0 (on --resume: the run's seed)")
    ap.add_argument("--iterations", type=int, default=None,
                    help=f"number of PPO updates, default {cfg.ppo.max_iterations} (on --resume: the run's)")
    ap.add_argument("--num-envs", type=int, default=None,
                    help=f"default {cfg.ppo.num_envs} (on --resume: the run's)")
    ap.add_argument("--learning-rate", type=float, default=None,
                    help=f"override PPO learning rate (default {cfg.ppo.learning_rate}); use the same for all three")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="simulation processes (default: CPU cores - 2); e.g. 7 each when running three trainings at once")
    ap.add_argument("--device", default=None, help="default: cuda:0 if there is a GPU (ROCm is also called cuda)")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="2 iterations, 8 environments, just to check it runs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            ap.error("--learning-rate must be positive")
        cfg.ppo.learning_rate = args.learning_rate
    if args.num_workers is not None:
        cfg.ppo.num_workers = args.num_workers
    if args.quick:
        args.iterations, args.num_envs = 2, 8
    if args.resume is not None:
        from minipacman.protocol import read_manifest
        run = read_manifest(args.resume.parent)
        args.seed = run["seed"] if args.seed is None else args.seed
        args.iterations = run["target_updates"] if args.iterations is None else args.iterations
        args.num_envs = run["num_envs"] if args.num_envs is None else args.num_envs
        args.out = args.resume.parent if args.out is None else args.out
    args.seed = 0 if args.seed is None else args.seed
    args.iterations = cfg.ppo.max_iterations if args.iterations is None else args.iterations
    args.num_envs = cfg.ppo.num_envs if args.num_envs is None else args.num_envs
    device = pick_device(args.device)
    info = device_report(device)
    warn_if_not_rocm(info)
    log_dir = args.out or OUTPUT_DIR / "policies" / args.reward / time.strftime("%Y%m%d_%H%M%S")
    print(f"Training {args.reward}, seed {args.seed}, {args.iterations} PPO updates x {args.num_envs} envs, "
          f"device {device} ({info.get('gpu_name')}), output {log_dir}")

    from minipacman.protocol import ProtocolError
    from minipacman.train_ppo import train
    try:
        ckpt = train(cfg, args.reward, args.seed, log_dir, device, max_iterations=args.iterations,
                     num_envs=args.num_envs, resume=args.resume, device_info=info,
                     args={k: str(v) for k, v in vars(args).items()})
    except NotImplementedError as e:
        raise SystemExit(f"[TODO] not implemented yet: {e}")
    except ProtocolError as e:
        raise SystemExit(f"[protocol] {e}")
    print(f"Done: {ckpt}")
    print(f"Put the run directory {log_dir} in POLICY_FILES in run_mini_pacman_experiments.py.")


if __name__ == "__main__":
    main()
