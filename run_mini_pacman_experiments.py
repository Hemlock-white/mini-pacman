"""Run all Mini PAC-MAN experiments (training is separate: train_mini_pacman_policy.py).

Usage:
    python run_mini_pacman_experiments.py --part check                  # check the setup and how far each module is
    python run_mini_pacman_experiments.py --part exp1 exp2 exp3 summary
    python run_mini_pacman_experiments.py --part all                    # everything
    python run_mini_pacman_experiments.py --part all --quick            # only a few seeds per group, to check the whole flow
    python run_mini_pacman_experiments.py --part exp3 --force           # rerun even if results exist
    python run_mini_pacman_experiments.py --part exp2 --seeds 20        # custom number of seeds
    python run_mini_pacman_experiments.py --part exp3 --quick --record-video --video-seeds 10090
    python run_mini_pacman_experiments.py --part select                 # optional: pick checkpoints on validation seeds
    python run_mini_pacman_experiments.py --part exp1 --workers 16      # run seeds in parallel

**Which policies to use: edit POLICY_FILES below** (run directories relative to the repo root).
Every policy is checked against the training protocol (minipacman/protocol.py) before it is used:
  * Exp 1: the three runs must have the same seed, envs, number of PPO updates and config, all from scratch
  * the checkpoint is the run's final one, or the one --part select picked on the validation seeds
    (seeds 20000-...; the evaluation seeds 10000-10099 are never used to pick anything)
A policy that breaks a rule stops the part with an explanation.

Experiments
-----------
    exp1  taught vs enforced: policies from 3 rewards × filter off / on = 6 groups              [YOURS, see part_exp3]
    exp2  naive vs lag-aware CBF: scripted command × (naive / lag_aware) = 2 groups              [YOURS]
    exp3  ball information source: same policy + filter, ball info from truth / noisy / depth camera estimate,
          plus a filter-off reference group                                                     [DONE]
    summary  read all results, print tables, make plots
    select   (optional, not part of "all") for each run in POLICY_FILES, compare every cfg.exp.select_every-th
             checkpoint + the final one on the validation seeds (filter off) and record the pick in
             <run>/selection.json; the same rule is applied to all three, so Exp 1 stays fair

Results go to outputs/results/<exp>/: one .jsonl per condition (one line per episode), summary.csv, plots.
With --record-video, selected seeds are rerun and comparison MP4s go to <out>/<exp>/videos/.
The GPU, versions and parameters of every run are logged in outputs/results/run_info.jsonl.
Course rule: results in the report must be produced on AMD ROCm; otherwise a warning is printed and the log says rocm=false.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

from minipacman import report
from minipacman.config import OUTPUT_DIR, REPO_ROOT, Config, default_config
from minipacman import protocol
from minipacman.evaluate import Condition
from minipacman.runtime import device_report, pick_device, warn_if_not_rocm

# ============================================================================================
# Policies to evaluate: the training run directories (train_mini_pacman_policy.py --out ...)
# A directory means: the checkpoint --part select picked, or else the final one. A model_*.pt file is also accepted
# if it is one of those two. (The round-1 NVIDIA checkpoints don't follow the protocol and are refused.)
# ============================================================================================
POLICY_FILES = {
    "task": "outputs/policies/task/run1",
    "task_collision": "outputs/policies/task_collision/run1",
    "task_cbf": "outputs/policies/task_cbf/run1",
}

PARTS = ("check", "select", "exp1", "exp2", "exp3", "summary")
ALL_PARTS = ("check", "exp1", "exp2", "exp3", "summary")      # --part all (select is opt-in)


# ============================================================================================
# Shared helpers
# ============================================================================================
def eval_seeds(cfg: Config, args) -> list[int]:
    n = args.seeds or (cfg.exp.quick_seeds if args.quick else cfg.exp.n_eval_seeds)
    if n > cfg.exp.n_eval_seeds:
        raise SystemExit(f"--seeds {n} is more than the {cfg.exp.n_eval_seeds} evaluation seeds reserved in the config")
    return protocol.evaluation_seeds(cfg, n)


def resolve(reward_variant: str, cfg: Config) -> protocol.PolicyRecord:
    """The checkpoint an experiment may use for this reward variant (checked against protocol.py)."""
    path = REPO_ROOT / POLICY_FILES[reward_variant]
    if not path.exists():
        raise SystemExit(f"Cannot find the {reward_variant} policy: {path}\n"
                         f"Train it first (python train_mini_pacman_policy.py --reward {reward_variant} --out ...), "
                         f"then put the run directory in POLICY_FILES in run_mini_pacman_experiments.py.")
    try:
        record = protocol.resolve_policy(path, reward_variant, cfg)
    except protocol.ProtocolError as e:
        raise SystemExit(f"[protocol] {reward_variant}: {e}")
    print(f"  policy {record.summary()}")
    return record


def policy_path(reward_variant: str, cfg: Config | None = None, *, require_exists: bool = True) -> str:
    if not require_exists:
        return str(REPO_ROOT / POLICY_FILES[reward_variant])
    return str(resolve(reward_variant, cfg or default_config()).path)


def exp1_conditions(cfg: Config, *, require_policies: bool = True) -> list[Condition]:
    """Build Exp 1 conditions in their canonical reporting order (checks that the three policies are comparable)."""
    if require_policies:
        records = {rv: resolve(rv, cfg) for rv in cfg.exp.reward_variants}
        try:
            protocol.check_same_protocol(list(records.values()))
        except protocol.ProtocolError as e:
            raise SystemExit(f"[protocol] {e}")
        paths = {rv: str(record.path) for rv, record in records.items()}
    else:
        paths = {rv: policy_path(rv, require_exists=False) for rv in cfg.exp.reward_variants}
    return [
        Condition(
            name=f"exp1_{reward_variant}_{'filter' if filter_mode else 'nofilter'}",
            policy=paths[reward_variant],
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
    policy = policy_path(reward_variant, cfg, require_exists=require_policies)
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
        episodes = run_condition(c, seeds, cfg, exp_dir / f"{c.name}.jsonl", args.device, resume=not args.force,
                                 workers=args.workers)
        row = {"condition": c.name, **c.to_dict(), **report.summarize(episodes, cfg.exp.height_buckets)}
        rows.append(row)
        print(f"  [{c.name}] done in {time.time() - t0:.0f} s; collision rate {row['collision_rate']:.3f} "
              f"[{row['collision_rate_lo']:.3f}, {row['collision_rate_hi']:.3f}], success rate {row['success_rate']:.3f}")
    report.write_csv(rows, exp_dir / "summary.csv")
    return rows


def record_videos_if_requested(
    experiment: str, conditions: list[Condition], exp_dir: Path, cfg: Config, args,
) -> list[Path]:
    """Rerun only the explicitly selected seeds and record comparison videos."""
    if not getattr(args, "record_video", False):
        return []
    from minipacman.video import record_experiment_videos

    paths = record_experiment_videos(
        experiment,
        conditions,
        list(args.video_seeds),
        cfg,
        exp_dir / "videos",
        device=args.device,
        fps=args.video_fps,
    )
    print(f"  videos: {exp_dir}/videos/*.mp4")
    return paths


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

    print("== Policies (POLICY_FILES), checked against the training protocol")
    for rv, p in POLICY_FILES.items():
        if not (REPO_ROOT / p).exists():
            print(f"  [MISSING] {rv}: {p}")
            continue
        try:
            print(f"  [OK]      {protocol.resolve_policy(REPO_ROOT / p, rv, cfg).summary()}")
        except protocol.ProtocolError as e:
            print(f"  [REFUSED] {rv}: {e}")


def _has_attrs(module: str, cls: str, attrs: list[str]) -> None:
    c = getattr(__import__(module, fromlist=[cls]), cls)
    missing = [a for a in attrs if not hasattr(c, a)]
    if missing:
        raise NotImplementedError(missing)


def part_select(cfg: Config, args, out_root: Path) -> None:
    """Pick each run's checkpoint on the validation seeds (never the evaluation seeds), the same way for all three.

    Candidates: every cfg.exp.select_every-th checkpoint plus the final one. Each is run on the validation seeds with
    the filter off and truth ball information. Pick: protocol.SELECTION_CRITERION. Written to <run>/selection.json;
    from then on Exp 1 / Exp 3 use that checkpoint (delete selection.json to go back to the final one).
    """
    from minipacman.evaluate import run_condition

    seeds = protocol.validation_seeds(cfg)
    protocol.check_seed_sets(cfg)
    candidate_rule = f"every {cfg.exp.select_every} updates + final"
    for rv in cfg.exp.reward_variants:
        target = REPO_ROOT / POLICY_FILES[rv]
        run_dir = target if target.is_dir() else target.parent
        try:
            manifest = protocol.read_manifest(run_dir)
        except protocol.ProtocolError as e:
            raise SystemExit(f"[protocol] {rv}: {e}")
        final = protocol.final_label(manifest)
        if not (run_dir / f"model_{final}.pt").is_file():
            raise SystemExit(f"[protocol] {rv}: training in {run_dir} is not finished")
        existing = protocol.read_selection(run_dir)
        if existing and not args.force:
            print(f"  {rv}: already selected {existing['checkpoint']} (use --force to redo)")
            continue
        labels = [k for k in range(cfg.exp.select_every, final, cfg.exp.select_every)
                  if (run_dir / f"model_{k}.pt").is_file()] + [final]
        rows = []
        for label in labels:
            c = Condition(name=f"select_{rv}_{label}", policy=str(run_dir / f"model_{label}.pt"), reward_variant=rv,
                          filter_mode=None, ball_source="truth")
            episodes = run_condition(c, seeds, cfg, out_root / "select" / rv / f"{c.name}.jsonl", args.device,
                                     resume=not args.force, workers=args.workers)
            stats = report.summarize(episodes, cfg.exp.height_buckets)
            rows.append({"label": label, "checkpoint": f"model_{label}.pt", "success_rate": stats["success_rate"],
                         "fall_rate": stats["fall_rate"], "collision_rate": stats["collision_rate"]})
            print(f"  {rv} model_{label}: success {stats['success_rate']:.2f}, fall {stats['fall_rate']:.2f}, "
                  f"collision {stats['collision_rate']:.2f}")
        best = max(rows, key=lambda r: (r["success_rate"], -r["fall_rate"], -r["collision_rate"], r["label"]))
        selection = {"rule": "validation", "checkpoint": best["checkpoint"], "label": best["label"],
                     "seeds": [seeds[0], len(seeds)], "criterion": protocol.SELECTION_CRITERION,
                     "candidate_rule": candidate_rule, "candidates": rows,
                     "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        (run_dir / protocol.SELECTION).write_text(json.dumps(selection, indent=2), encoding="utf-8")
        print(f"  {rv}: picked {best['checkpoint']} -> {run_dir / protocol.SELECTION}")


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
    record_videos_if_requested("exp1", conditions, exp_dir, cfg, args)
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
    record_videos_if_requested("exp2", conditions, exp_dir, cfg, args)
    return rows


def part_exp3(cfg: Config, args, out_root: Path) -> list[dict]:
    """Exp 3: ball information source (truth / noise and latency / depth camera estimate), plus a filter-off reference."""
    conditions = exp3_conditions(cfg)
    exp_dir = out_root / "exp3"
    rows = run_conditions(conditions, exp_dir, cfg, args)
    report.print_table(rows)
    report.plot_exp3(rows, exp_dir)
    print(f"  plots: {exp_dir}/exp3_*.png")
    record_videos_if_requested("exp3", conditions, exp_dir, cfg, args)
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
    ap.add_argument("--record-video", action="store_true",
                    help="rerun selected seeds after evaluation and save annotated comparison MP4s")
    ap.add_argument("--video-seeds", nargs="+", type=int, default=None,
                    help="exact seeds to record (default: the first evaluation seed only)")
    ap.add_argument("--video-fps", type=int, default=25, help="recording frame rate (default: 25)")
    ap.add_argument("--workers", type=int, default=1, help="evaluate seeds in this many parallel processes")
    args = ap.parse_args()

    parts = ALL_PARTS if "all" in args.part else tuple(p for p in PARTS if p in args.part)
    experiment_parts = tuple(part for part in parts if part in {"exp1", "exp2", "exp3"})
    if args.video_seeds is not None and not args.record_video:
        ap.error("--video-seeds requires --record-video")
    if args.record_video and not experiment_parts:
        ap.error("--record-video requires exp1, exp2, exp3, or all")
    if args.video_fps <= 0:
        ap.error("--video-fps must be positive")
    policy_rate = 1.0 / (cfg.sim.timestep * cfg.sim.policy_every)
    if args.video_fps > policy_rate:
        ap.error(f"--video-fps cannot exceed the policy rate ({policy_rate:g} Hz)")
    if args.video_seeds is not None and any(seed < 0 for seed in args.video_seeds):
        ap.error("--video-seeds must be non-negative")
    if args.record_video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        args.video_seeds = args.video_seeds or [eval_seeds(cfg, args)[0]]
    args.device = pick_device(args.device)
    info = device_report(args.device)
    warn_if_not_rocm(info)
    args.out.mkdir(parents=True, exist_ok=True)
    report.append_jsonl({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "parts": parts,
                         "args": {k: str(v) for k, v in vars(args).items()},
                         "policy_files": POLICY_FILES, "device": info, "config": cfg.to_dict()},
                        args.out / "run_info.jsonl")
    funcs = {"check": part_check, "select": part_select, "exp1": part_exp1, "exp2": part_exp2, "exp3": part_exp3,
             "summary": part_summary}
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
