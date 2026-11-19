"""結果的存檔、統計、畫圖（matplotlib）。

結果檔格式
----------
每個實驗條件一個 ``.jsonl`` 檔，一行一個 episode（格式見 evaluate.py 的 EPISODE_SCHEMA）。
``summarize()`` 把一個條件的所有 episode 算成一列統計，``write_csv()`` 存成 CSV。
CSV 可以直接用 Excel 或 MATLAB 開（MATLAB：``readtable("summary.csv")``）。

統計方式
--------
比例（成功率、碰撞率）附 95% Wilson 信賴區間：seed 只有 100 個時，3% 和 6% 可能其實沒有差別，
報告時要看區間有沒有重疊，不要只看平均。
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


# ---------------- 存讀檔 ----------------
def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def append_jsonl(record: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


# ---------------- 統計 ----------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """比例 k/n 的 (估計值, 95% 下界, 95% 上界)。n=0 時回傳 nan。"""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def summarize(episodes: list[dict], height_buckets=(("low", 0.0, 0.2), ("mid", 0.2, 0.4), ("high", 0.4, 10.0))) -> dict:
    """把一個條件的所有 episode 算成一列統計。

    欄位：
      n_episodes、success_rate(+_lo/_hi)、fall_rate、t_goal_mean
      n_balls（發射的球數）、collision_rate(+_lo/_hi) = 被打到的球 / 發射的球
      episode_hit_rate = 至少被打到一次的 episode 比例
      min_clearance_median、intervention_mean、filter_active_frac_mean
      collision_rate_<low|mid|high>、n_balls_<...>：依球抵達高度分組
      collision_rate_<direct|bounce>：依軌跡類型分組
    """
    n = len(episodes)
    row: dict = {"n_episodes": n}
    succ = sum(bool(e.get("success")) for e in episodes)
    row["success_rate"], row["success_rate_lo"], row["success_rate_hi"] = wilson(succ, n)
    row["fall_rate"] = wilson(sum(bool(e.get("fell")) for e in episodes), n)[0]
    tg = [e["t_goal"] for e in episodes if e.get("t_goal") is not None]
    row["t_goal_mean"] = float(np.mean(tg)) if tg else float("nan")
    balls = [b for e in episodes for b in e.get("balls", []) if b.get("launched")]
    hits = sum(bool(b.get("hit")) for b in balls)
    row["n_balls"] = len(balls)
    row["collision_rate"], row["collision_rate_lo"], row["collision_rate_hi"] = wilson(hits, len(balls))
    row["episode_hit_rate"] = wilson(sum(any(b.get("hit") for b in e.get("balls", [])) for e in episodes), n)[0]
    clear = [b["min_clearance"] for b in balls if b.get("min_clearance") is not None]
    row["min_clearance_median"] = float(np.median(clear)) if clear else float("nan")
    row["intervention_mean"] = _mean(episodes, "intervention_mean")
    row["filter_active_frac_mean"] = _mean(episodes, "filter_active_frac")
    for name, lo, hi in height_buckets:
        sel = [b for b in balls if b.get("arrival_height") is not None and lo <= b["arrival_height"] < hi]
        row[f"n_balls_{name}"] = len(sel)
        row[f"collision_rate_{name}"] = wilson(sum(bool(b["hit"]) for b in sel), len(sel))[0]
    for mode in ("direct", "bounce"):
        sel = [b for b in balls if b.get("mode") == mode]
        row[f"collision_rate_{mode}"] = wilson(sum(bool(b["hit"]) for b in sel), len(sel))[0]
    return row


def _mean(episodes, key):
    vals = [e[key] for e in episodes if e.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def print_table(rows: list[dict], cols=("condition", "success_rate", "collision_rate", "collision_rate_lo",
                                         "collision_rate_hi", "min_clearance_median", "intervention_mean")) -> None:
    widths = [max(len(c), 12) for c in cols]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for r in rows:
        cells = []
        for c, w in zip(cols, widths):
            v = r.get(c, "")
            cells.append((f"{v:.3f}" if isinstance(v, float) else str(v)).rjust(w))
        print("  ".join(cells))


# ---------------- 畫圖 ----------------
def bar_chart(rows: list[dict], metric: str, path: str | Path, title: str, ylabel: str,
              label_key: str = "condition", ci: bool = True) -> None:
    """長條圖；metric 有 _lo/_hi 欄位時畫 95% 信賴區間。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [str(r.get(label_key, i)) for i, r in enumerate(rows)]
    vals = np.array([r.get(metric, np.nan) for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(max(4, 1.3 * len(rows)), 3.6))
    yerr = None
    if ci and all(f"{metric}_lo" in r for r in rows):
        lo = np.array([r[f"{metric}_lo"] for r in rows], dtype=float)
        hi = np.array([r[f"{metric}_hi"] for r in rows], dtype=float)
        yerr = np.vstack([vals - lo, hi - vals])
    ax.bar(range(len(rows)), vals, yerr=yerr, capsize=4, color="#4C78A8")
    ax.set_xticks(range(len(rows)), labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_exp3(rows: list[dict], out_dir: str | Path) -> None:
    """實驗 3：不同球資訊來源下的碰撞率、成功率、最小距離。"""
    out_dir = Path(out_dir)
    bar_chart(rows, "collision_rate", out_dir / "exp3_collision_rate.png",
              "Exp 3: ball information source", "collision rate (hit balls / thrown)")
    bar_chart(rows, "success_rate", out_dir / "exp3_success_rate.png",
              "Exp 3: ball information source", "task success rate")
    bar_chart(rows, "min_clearance_median", out_dir / "exp3_min_clearance.png",
              "Exp 3: ball information source", "median min clearance (m)", ci=False)


def plot_exp1(rows: list[dict], out_dir: str | Path) -> None:
    """實驗 1 的圖。

    TODO（你們實作）：建議兩張圖
      1. 碰撞率：x 軸三種 reward，每組兩根長條（過濾器關 / 開），附 95% 信賴區間。
      2. 成功率與 intervention_mean（過濾器開的那三組）：看「學會閃的 policy 是不是比較少需要過濾器出手」。
    可以參考 bar_chart() 和 plot_exp3() 的寫法；分組長條圖用 ax.bar(x ± 寬度/2, ...)。
    """
    raise NotImplementedError("TODO：實作 plot_exp1（見 docstring）")


def plot_exp2(rows: list[dict], out_dir: str | Path) -> None:
    """實驗 2 的圖。

    TODO（你們實作）：2×2 的碰撞率（簡單版 / 考慮延遲版 × 每步重新規劃 / 只決定一次），
    以及依球抵達高度分組的碰撞率（collision_rate_low / mid / high）。
    """
    raise NotImplementedError("TODO：實作 plot_exp2（見 docstring）")
