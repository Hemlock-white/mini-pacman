"""跑 Mini PAC-MAN 的所有實驗（訓練另外用 train_mini_pacman_policy.py）。

用法：
    python run_mini_pacman_experiments.py --part check                  # 檢查環境與各模組進度
    python run_mini_pacman_experiments.py --part exp1 exp2 exp3 summary
    python run_mini_pacman_experiments.py --part all                    # 全部
    python run_mini_pacman_experiments.py --part all --quick            # 每組只跑幾個 seed，確認整條流程
    python run_mini_pacman_experiments.py --part exp3 --force           # 已經有結果也重跑
    python run_mini_pacman_experiments.py --part exp2 --seeds 20        # 自訂 seed 數

**要用哪個 policy 檔：改下面的 POLICY_FILES**（相對於 repo 根目錄的路徑）。

實驗
----
    exp1  教會 vs 強制：3 種 reward 的 policy × 過濾器 關 / 開 = 6 組          【你們實作，參考 part_exp3】
    exp2  簡單 CBF vs 考慮延遲：寫死的指令 × (naive / lag_aware) × (每步 / 只決定一次) = 4 組   【你們實作】
    exp3  球資訊來源：同一個 policy + 過濾器，球資訊用 真實值 / 加雜訊 / 深度相機估計，另加不開過濾器的參考組  【已實作】
    summary  讀所有結果，印表格、畫圖

結果放在 outputs/results/<exp>/：每個條件一個 .jsonl（一行一個 episode）、summary.csv、圖。
每次執行的 GPU、版本、參數記在 outputs/results/run_info.jsonl。
課程規定：報告用的結果必須在 AMD ROCm 上產生；不是 ROCm 時會印警告，紀錄裡 rocm=false。
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
# 要評估的 policy（訓練完把路徑填在這裡）
# ============================================================================================
POLICY_FILES = {
    "task": "outputs/policies/task/RUN_NAME/model_5000.pt",
    "task_collision": "outputs/policies/task_collision/RUN_NAME/model_5000.pt",
    "task_cbf": "outputs/policies/task_cbf/RUN_NAME/model_5000.pt",
}

PARTS = ("check", "exp1", "exp2", "exp3", "summary")


# ============================================================================================
# 共用
# ============================================================================================
def eval_seeds(cfg: Config, args) -> list[int]:
    n = args.seeds or (cfg.exp.quick_seeds if args.quick else cfg.exp.n_eval_seeds)
    return list(range(cfg.exp.eval_seed_offset, cfg.exp.eval_seed_offset + n))


def policy_path(reward_variant: str) -> str:
    path = REPO_ROOT / POLICY_FILES[reward_variant]
    if not path.exists():
        raise SystemExit(f"找不到 {reward_variant} 的 policy：{path}\n"
                         f"請先訓練（python train_mini_pacman_policy.py --reward {reward_variant}），"
                         f"再把路徑填進 run_mini_pacman_experiments.py 的 POLICY_FILES。")
    return str(path)


def run_conditions(conditions: list[Condition], exp_dir: Path, cfg: Config, args) -> list[dict]:
    """跑一串條件，每個條件的 episode 存成 <exp_dir>/<name>.jsonl，回傳每個條件一列的統計。"""
    from minipacman.evaluate import run_condition
    seeds = eval_seeds(cfg, args)
    rows = []
    for c in conditions:
        t0 = time.time()
        print(f"  [{c.name}] {len(seeds)} 個 seed ...", flush=True)
        episodes = run_condition(c, seeds, cfg, exp_dir / f"{c.name}.jsonl", args.device, resume=not args.force)
        row = {"condition": c.name, **c.to_dict(), **report.summarize(episodes, cfg.exp.height_buckets)}
        rows.append(row)
        print(f"  [{c.name}] 完成，{time.time() - t0:.0f} s；碰撞率 {row['collision_rate']:.3f} "
              f"[{row['collision_rate_lo']:.3f}, {row['collision_rate_hi']:.3f}]，成功率 {row['success_rate']:.3f}")
    report.write_csv(rows, exp_dir / "summary.csv")
    return rows


# ============================================================================================
# 各階段
# ============================================================================================
def part_check(cfg: Config, args, out_root: Path) -> None:
    """檢查：TITA 模型、場景、發球器、深度相機（已實作），以及你們的模組寫到哪了。"""
    import mujoco
    import numpy as np
    from minipacman import scene
    from minipacman.balls import BallLauncher
    from minipacman.depth_camera import DepthCamera

    print("== 場景與已實作的元件")
    m, d = scene.load_model(None, cfg)
    ids = scene.scene_ids(m, cfg)
    print(f"  場景 OK：nq={m.nq} nv={m.nv}，機器人 dof={ids.robot_nv}，球 {len(ids.ball_geoms)} 顆")
    launcher = BallLauncher(m, d, cfg, ids)
    launcher.reset(np.random.default_rng(0))
    print(f"  發球器 OK：seed 0 這回合丟 {len(launcher.records)} 顆，時間 "
          f"{[round(r.throw_time, 2) for r in launcher.records]} s")
    mujoco.mj_forward(m, d)
    cam = DepthCamera(m, cfg, ids)
    t0 = time.perf_counter()
    for _ in range(50):
        frame = cam.render(d)
    print(f"  深度相機 OK：{cam.W}x{cam.H}，每張 {1000 * (time.perf_counter() - t0) / 50:.2f} ms")

    print("== 你們的模組")
    checks = [
        ("env.TitaEnv 介面", lambda: _has_attrs("minipacman.env", "TitaEnv",
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
            print(f"  [OK]   {name}")
        except NotImplementedError:
            print(f"  [TODO] {name}")
        except Exception as e:
            print(f"  [錯誤] {name}: {type(e).__name__}: {e}")

    print("== policy 檔（POLICY_FILES）")
    for rv, p in POLICY_FILES.items():
        print(f"  {'[OK]  ' if (REPO_ROOT / p).exists() else '[缺少]'} {rv}: {p}")


def _has_attrs(module: str, cls: str, attrs: list[str]) -> None:
    c = getattr(__import__(module, fromlist=[cls]), cls)
    missing = [a for a in attrs if not hasattr(c, a)]
    if missing:
        raise NotImplementedError(missing)


def part_exp1(cfg: Config, args, out_root: Path) -> list[dict]:
    """實驗 1：教會 vs 強制。【你們實作】

    建議步驟（照 part_exp3 的寫法）：
      1. conditions = []
         for rv in cfg.exp.reward_variants:                       # "task", "task_collision", "task_cbf"
             for filt in (None, cfg.exp.exp1_filter_mode):        # 過濾器 關 / 開
                 conditions.append(Condition(
                     name=f"exp1_{rv}_{'filter' if filt else 'nofilter'}",
                     policy=policy_path(rv), reward_variant=rv, filter_mode=filt,
                     replan="every_step", ball_source=cfg.exp.exp1_filter_source))
      2. rows = run_conditions(conditions, out_root / "exp1", cfg, args)
      3. report.print_table(rows)；try: report.plot_exp1(rows, out_root / "exp1")
      4. return rows
    """
    raise NotImplementedError("TODO：part_exp1（見 docstring）")


def part_exp2(cfg: Config, args, out_root: Path) -> list[dict]:
    """實驗 2：簡單版 vs 考慮延遲版過濾器 × 每步重新規劃 vs 只決定一次。【你們實作】

    上層用寫死的指令（Condition.policy = "scripted"，reward_variant 填 "scripted"），
    這樣結果只反映過濾器，也不用等 PPO 練好就能跑。
    條件：for fm in cfg.exp.exp2_filters: for rp in cfg.exp.exp2_replan → 4 組，
    另外加一組不開過濾器的參考（filter_mode=None），共 5 組。球資訊用 "truth"。
    其餘同 part_exp1；畫圖用 report.plot_exp2。
    """
    raise NotImplementedError("TODO：part_exp2（見 docstring）")


def part_exp3(cfg: Config, args, out_root: Path) -> list[dict]:
    """實驗 3：球資訊來源（真實值 / 加雜訊與延遲 / 深度相機估計），加一組不開過濾器的參考。"""
    rv = cfg.exp.exp3_reward_variant
    pol = policy_path(rv)
    conditions = [Condition(name="exp3_no_filter", policy=pol, reward_variant=rv, filter_mode=None,
                            ball_source="truth")]
    for src in cfg.exp.exp3_sources:
        conditions.append(Condition(name=f"exp3_{src}", policy=pol, reward_variant=rv,
                                    filter_mode=cfg.exp.exp3_filter_mode, ball_source=src))
    exp_dir = out_root / "exp3"
    rows = run_conditions(conditions, exp_dir, cfg, args)
    report.print_table(rows)
    report.plot_exp3(rows, exp_dir)
    print(f"  圖：{exp_dir}/exp3_*.png")
    return rows


def part_summary(cfg: Config, args, out_root: Path) -> None:
    """讀每個實驗資料夾裡的 .jsonl，重新統計、印表格、畫圖。"""
    plots = {"exp1": report.plot_exp1, "exp2": report.plot_exp2, "exp3": report.plot_exp3}
    for exp, plot in plots.items():
        exp_dir = out_root / exp
        files = sorted(exp_dir.glob("*.jsonl"))
        if not files:
            print(f"== {exp}：還沒有結果")
            continue
        rows = [{"condition": f.stem, **report.summarize(report.load_jsonl(f), cfg.exp.height_buckets)}
                for f in files]
        print(f"== {exp}")
        report.print_table(rows)
        report.write_csv(rows, exp_dir / "summary.csv")
        try:
            plot(rows, exp_dir)
        except NotImplementedError as e:
            print(f"  （圖還沒實作：{e}）")


# ============================================================================================
def main() -> None:
    cfg = default_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="+", required=True, choices=PARTS + ("all",))
    ap.add_argument("--quick", action="store_true", help=f"每組只跑 {cfg.exp.quick_seeds} 個 seed")
    ap.add_argument("--seeds", type=int, default=None, help=f"每組幾個 seed（預設 {cfg.exp.n_eval_seeds}）")
    ap.add_argument("--force", action="store_true", help="已經有的結果也重跑")
    ap.add_argument("--device", default=None, help="policy 推論用的裝置（預設 cuda:0，ROCm 也叫 cuda）")
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
            print(f"[TODO] {p} 還沒實作：{e}")
        except SystemExit as e:              # 例如找不到 policy 檔：印出原因，繼續下一個階段
            print(f"[略過] {p}：{e}")
        except Exception:
            traceback.print_exc()
            print(f"[錯誤] {p} 失敗，繼續下一個階段")


if __name__ == "__main__":
    main()
