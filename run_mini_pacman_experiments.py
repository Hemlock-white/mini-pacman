"""Run all Mini PAC-MAN experiments (training is separate: train_mini_pacman_policy.py).

Usage:
    python run_mini_pacman_experiments.py --part check                  # check the setup and how far each module is
    python run_mini_pacman_experiments.py --part exp1 exp2 exp3 summary
    python run_mini_pacman_experiments.py --part all                    # everything
    python run_mini_pacman_experiments.py --part all --quick            # only a few seeds per group, to check the whole flow
    python run_mini_pacman_experiments.py --part exp3 --force           # rerun even if results exist
    python run_mini_pacman_experiments.py --part exp2 --seeds 20        # custom number of seeds

**Which policy files to use: edit POLICY_FILES below** (paths relative to the repo root).

Experiments
-----------
    exp1  taught vs enforced: policies from 3 rewards × filter off / on = 6 groups              [YOURS, see part_exp3]
    exp2  naive vs lag-aware CBF: scripted command × (naive / lag_aware) = 2 groups              [YOURS]
    exp3  ball information source: same policy + filter, ball info from truth / noisy / depth camera estimate,
          plus a filter-off reference group                                                     [DONE]
    summary  read all results, print tables, make plots

Results go to outputs/results/<exp>/: one .jsonl per condition (one line per episode), summary.csv, plots.
The GPU, versions and parameters of every run are logged in outputs/results/run_info.jsonl.
Course rule: results in the report must be produced on AMD ROCm; otherwise a warning is printed and the log says rocm=false.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

from minipacman import report
from minipacman.config import OUTPUT_DIR, REPO_ROOT, Config, default_config
from minipacman.evaluate import Condition
from minipacman.runtime import device_report, pick_device, warn_if_not_rocm

# ============================================================================================
# Policies to evaluate (put the paths here after training)
# ============================================================================================
POLICY_FILES = {
    "task": "outputs/policies/task/20261121_tuned_seed0/model_1499.pt",
    "task_collision": "outputs/policies/task_collision/20261121_tuned_seed1/model_200.pt",
    "task_cbf": "outputs/policies/task_cbf/20261121_cbf1_from_task1499_seed2/model_1500.pt",
}

PARTS = ("check", "exp1", "exp2", "exp3", "summary")


# ============================================================================================
# Shared helpers
# ============================================================================================
def eval_seeds(cfg: Config, args) -> list[int]:
    n = args.seeds or (cfg.exp.quick_seeds if args.quick else cfg.exp.n_eval_seeds)
    return list(range(cfg.exp.eval_seed_offset, cfg.exp.eval_seed_offset + n))


def policy_path(reward_variant: str, *, require_exists: bool = True) -> str:
    path = REPO_ROOT / POLICY_FILES[reward_variant]
    if require_exists and not path.exists():
        raise SystemExit(f"Cannot find the {reward_variant} policy: {path}\n"
                         f"Train it first (python train_mini_pacman_policy.py --reward {reward_variant}), "
                         f"then put the path in POLICY_FILES in run_mini_pacman_experiments.py.")
    return str(path)


def exp1_conditions(cfg: Config, *, require_policies: bool = True) -> list[Condition]:
    """Build Exp 1 conditions in their canonical reporting order."""
    return [
        Condition(
            name=f"exp1_{reward_variant}_{'filter' if filter_mode else 'nofilter'}",
            policy=policy_path(reward_variant, require_exists=require_policies),
            reward_variant=reward_variant,
            filter_mode=filter_mode,
            replan="every_step",
            ball_source=cfg.exp.exp1_filter_source,
        )
        for reward_variant in cfg.exp.reward_variants
        for filter_mode in (None, cfg.exp.exp1_filter_mode)
    ]


def exp2_conditions(cfg: Config) -> list[Condition]:
    """Build Exp 2 conditions in their canonical reporting order."""
    include_replan = len(cfg.exp.exp2_replan) > 1
    return [
        Condition(
            name=f"exp2_{filter_mode}{f'_{replan}' if include_replan else ''}",
            policy=cfg.exp.exp2_policy,
            reward_variant="scripted",
            filter_mode=filter_mode,
            replan=replan,
            ball_source="truth",
        )
        for filter_mode in cfg.exp.exp2_filters
        for replan in cfg.exp.exp2_replan
    ]


def exp3_conditions(cfg: Config, *, require_policies: bool = True) -> list[Condition]:
    """Build Exp 3 conditions in their canonical reporting order."""
    reward_variant = cfg.exp.exp3_reward_variant
    policy = policy_path(reward_variant, require_exists=require_policies)
    return [
        Condition(
            name="exp3_no_filter",
            policy=policy,
            reward_variant=reward_variant,
            filter_mode=None,
            ball_source="truth",
        ),
        *[
            Condition(
                name=f"exp3_{source}",
                policy=policy,
                reward_variant=reward_variant,
                filter_mode=cfg.exp.exp3_filter_mode,
                ball_source=source,
            )
            for source in cfg.exp.exp3_sources
        ],
    ]


def run_conditions(conditions: list[Condition], exp_dir: Path, cfg: Config, args) -> list[dict]:
    """Run a list of conditions; each condition's episodes go to <exp_dir>/<name>.jsonl. Returns one stats row per condition."""
    from minipacman.evaluate import run_condition
    seeds = eval_seeds(cfg, args)
    rows = []
    for c in conditions:
        t0 = time.time()
        print(f"  [{c.name}] {len(seeds)} seeds ...", flush=True)
        episodes = run_condition(c, seeds, cfg, exp_dir / f"{c.name}.jsonl", args.device, resume=not args.force)
        row = {"condition": c.name, **c.to_dict(), **report.summarize(episodes, cfg.exp.height_buckets)}
        rows.append(row)
        print(f"  [{c.name}] done in {time.time() - t0:.0f} s; collision rate {row['collision_rate']:.3f} "
              f"[{row['collision_rate_lo']:.3f}, {row['collision_rate_hi']:.3f}], success rate {row['success_rate']:.3f}")
    report.write_csv(rows, exp_dir / "summary.csv")
    return rows


# ============================================================================================
# Parts
# ============================================================================================
def part_check(cfg: Config, args, out_root: Path) -> None:
    """Check: TITA model, scene, launcher, depth camera (done), and how far your modules are."""
    import mujoco
    import numpy as np
    from minipacman import scene
    from minipacman.balls import BallLauncher
    from minipacman.depth_camera import DepthCamera

    print("== Scene and the finished parts")
    m, d = scene.load_model(None, cfg)
    ids = scene.scene_ids(m, cfg)
    print(f"  scene OK: nq={m.nq} nv={m.nv}, robot dofs={ids.robot_nv}, {len(ids.ball_geoms)} balls")
    launcher = BallLauncher(m, d, cfg, ids)
    launcher.reset(np.random.default_rng(0))
    print(f"  launcher OK: seed 0 throws {len(launcher.records)} balls this episode, at "
          f"{[round(r.throw_time, 2) for r in launcher.records]} s")
    mujoco.mj_forward(m, d)
    cam = DepthCamera(m, cfg, ids)
    t0 = time.perf_counter()
    for _ in range(50):
        frame = cam.render(d)
    print(f"  depth camera OK: {cam.W}x{cam.H}, {1000 * (time.perf_counter() - t0) / 50:.2f} ms per image")

    print("== Your modules")
    checks = [
        ("env.TitaEnv interface", lambda: _has_attrs("minipacman.env", "TitaEnv",
                                                      ["set_leg_targets", "set_wheel_torques", "planar_state"])),
        ("lqr.clip_to_envelope", lambda: __import__("minipacman.lqr", fromlist=["x"]).clip_to_envelope(
            np.zeros(3), 0.0, cfg.command)),
        ("lqr.ReferenceGenerator", lambda: __import__("minipacman.lqr", fromlist=["x"]).ReferenceGenerator(cfg)),
        ("dodge_env.DodgeEnv", lambda: __import__("minipacman.dodge_env", fromlist=["x"]).DodgeEnv(cfg)),
        ("evaluate.ScriptedPolicy", lambda: __import__("minipacman.evaluate", fromlist=["x"]).ScriptedPolicy(cfg)),
    ]
    for name, fn in checks:
        try:
            fn()
            print(f"  [OK]    {name}")
        except NotImplementedError:
            print(f"  [TODO]  {name}")
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")

    print("== Policy files (POLICY_FILES)")
    for rv, p in POLICY_FILES.items():
        print(f"  {'[OK]     ' if (REPO_ROOT / p).exists() else '[MISSING]'} {rv}: {p}")


def _has_attrs(module: str, cls: str, attrs: list[str]) -> None:
    c = getattr(__import__(module, fromlist=[cls]), cls)
    missing = [a for a in attrs if not hasattr(c, a)]
    if missing:
        raise NotImplementedError(missing)


def part_exp1(cfg: Config, args, out_root: Path) -> list[dict]:
    """Exp 1: taught vs enforced. [YOURS]

    Suggested steps (same style as part_exp3):
      1. conditions = []
         for rv in cfg.exp.reward_variants:                       # "task", "task_collision", "task_cbf"
             for filt in (None, cfg.exp.exp1_filter_mode):        # filter off / on
                 conditions.append(Condition(
                     name=f"exp1_{rv}_{'filter' if filt else 'nofilter'}",
                     policy=policy_path(rv), reward_variant=rv, filter_mode=filt,
                     replan="every_step", ball_source=cfg.exp.exp1_filter_source))
      2. rows = run_conditions(conditions, out_root / "exp1", cfg, args)
      3. report.print_table(rows); try: report.plot_exp1(rows, out_root / "exp1")
      4. return rows
    """
    conditions = exp1_conditions(cfg)
    exp_dir = out_root / "exp1"
    rows = run_conditions(conditions, exp_dir, cfg, args)
    report.print_table(rows)
    report.plot_exp1(rows, exp_dir)
    print(f"  plots: {exp_dir}/exp1_*.png")
    return rows


def part_exp2(cfg: Config, args, out_root: Path) -> list[dict]:
    """Exp 2: naive vs lag-aware filter. [YOURS]

    The high-level command is scripted (Condition.policy = "scripted", reward_variant = "scripted"), so the result
    only reflects the filter, and it can run before PPO is trained.
    Conditions: for fm in cfg.exp.exp2_filters: for rp in cfg.exp.exp2_replan → 2 groups
    (exp2_replan is just ("every_step",)), e.g. name=f"exp2_{fm}". Ball info from "truth".
    Otherwise the same as part_exp1; plot with report.plot_exp2.
    """
    conditions = exp2_conditions(cfg)
    exp_dir = out_root / "exp2"
    rows = run_conditions(conditions, exp_dir, cfg, args)
    report.print_table(rows)
    report.plot_exp2(rows, exp_dir)
    print(f"  plots: {exp_dir}/exp2_*.png")
    return rows


def part_exp3(cfg: Config, args, out_root: Path) -> list[dict]:
    """Exp 3: ball information source (truth / noise and latency / depth camera estimate), plus a filter-off reference."""
    conditions = exp3_conditions(cfg)
    exp_dir = out_root / "exp3"
    rows = run_conditions(conditions, exp_dir, cfg, args)
    report.print_table(rows)
    report.plot_exp3(rows, exp_dir)
    print(f"  plots: {exp_dir}/exp3_*.png")
    return rows


def part_summary(cfg: Config, args, out_root: Path) -> None:
    """Read the .jsonl files in each experiment folder, recompute the statistics, print tables, make plots."""
    plots = {"exp1": report.plot_exp1, "exp2": report.plot_exp2, "exp3": report.plot_exp3}
    conditions = {
        "exp1": exp1_conditions(cfg, require_policies=False),
        "exp2": exp2_conditions(cfg),
        "exp3": exp3_conditions(cfg, require_policies=False),
    }
    for exp, plot in plots.items():
        exp_dir = out_root / exp
        files = {path.stem: path for path in exp_dir.glob("*.jsonl")}
        if not files:
            print(f"== {exp}: no results yet")
            continue
        expected = {condition.name: condition for condition in conditions[exp]}
        unknown = sorted(set(files) - set(expected))
        if unknown:
            raise ValueError(f"unknown {exp} condition files (metadata unavailable): {unknown}")
        rows = [
            {
                "condition": condition.name,
                **condition.to_dict(),
                **report.summarize(report.load_jsonl(files[condition.name]), cfg.exp.height_buckets),
            }
            for condition in conditions[exp]
            if condition.name in files
        ]
        print(f"== {exp}")
        report.print_table(rows)
        report.write_csv(rows, exp_dir / "summary.csv")
        try:
            plot(rows, exp_dir)
        except NotImplementedError as e:
            print(f"  (plot not implemented yet: {e})")


# ============================================================================================
def main() -> None:
    cfg = default_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="+", required=True, choices=PARTS + ("all",))
    ap.add_argument("--quick", action="store_true", help=f"only {cfg.exp.quick_seeds} seeds per group")
    ap.add_argument("--seeds", type=int, default=None, help=f"seeds per group (default {cfg.exp.n_eval_seeds})")
    ap.add_argument("--force", action="store_true", help="rerun even if results exist")
    ap.add_argument("--device", default=None, help="device for policy inference (default cuda:0; ROCm is also called cuda)")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "results")
    args = ap.parse_args()

    parts = PARTS if "all" in args.part else tuple(p for p in PARTS if p in args.part)
    args.device = pick_device(args.device)
    info = device_report(args.device)
    warn_if_not_rocm(info)
    args.out.mkdir(parents=True, exist_ok=True)
    report.append_jsonl({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "parts": parts,
                         "args": {k: str(v) for k, v in vars(args).items()},
                         "policy_files": POLICY_FILES, "device": info, "config": cfg.to_dict()},
                        args.out / "run_info.jsonl")
    funcs = {"check": part_check, "exp1": part_exp1, "exp2": part_exp2, "exp3": part_exp3, "summary": part_summary}
    for p in parts:
        print(f"\n######## {p} ########")
        try:
            funcs[p](cfg, args, args.out)
        except NotImplementedError as e:
            print(f"[TODO] {p} not implemented yet: {e}")
        except SystemExit as e:              # e.g. a policy file is missing: print why and go on to the next part
            print(f"[SKIPPED] {p}: {e}")
        except Exception:
            traceback.print_exc()
            print(f"[ERROR] {p} failed, going on to the next part")


if __name__ == "__main__":
    main()
