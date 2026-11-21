"""Saving results, statistics and plots (matplotlib).

Results format
--------------
One ``.jsonl`` file per experiment condition, one line per episode (format: EPISODE_SCHEMA in evaluate.py).
``summarize()`` turns all episodes of a condition into one row of statistics, ``write_csv()`` saves rows as CSV.
The CSV opens directly in Excel or MATLAB (MATLAB: ``readtable("summary.csv")``).

Statistics
----------
Rates (success, collision) come with a 95% Wilson confidence interval: with only 100 seeds, 3% vs 6% may be no real
difference, so check whether the intervals overlap in the report instead of only comparing means.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


# ---------------- save / load ----------------
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


# ---------------- statistics ----------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(estimate, 95% lower bound, 95% upper bound) of the rate k/n. Returns nan when n=0."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def summarize(episodes: list[dict], height_buckets=(("low", 0.0, 0.2), ("mid", 0.2, 0.4), ("high", 0.4, 10.0))) -> dict:
    """Turn all episodes of one condition into one row of statistics.

    Columns:
      n_episodes, success_rate(+_lo/_hi), fall_rate, t_goal_mean
      n_balls (balls launched), collision_rate(+_lo/_hi) = balls that hit / balls launched
      episode_hit_rate = fraction of episodes hit at least once
      min_clearance_median, intervention_mean, filter_active_frac_mean
      collision_rate_<low|mid|high>, n_balls_<...>: grouped by ball arrival height
      collision_rate_<direct|bounce>: grouped by trajectory type
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


# ---------------- plots ----------------
def bar_chart(rows: list[dict], metric: str, path: str | Path, title: str, ylabel: str,
              label_key: str = "condition", ci: bool = True) -> None:
    """Bar chart; draws the 95% confidence interval when the metric has _lo/_hi columns."""
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
    """Exp 3: collision rate, success rate and closest distance for each ball information source."""
    out_dir = Path(out_dir)
    bar_chart(rows, "collision_rate", out_dir / "exp3_collision_rate.png",
              "Exp 3: ball information source", "collision rate (hit balls / thrown)")
    bar_chart(rows, "success_rate", out_dir / "exp3_success_rate.png",
              "Exp 3: ball information source", "task success rate")
    bar_chart(rows, "min_clearance_median", out_dir / "exp3_min_clearance.png",
              "Exp 3: ball information source", "median min clearance (m)", ci=False)


def plot_exp1(rows: list[dict], out_dir: str | Path) -> None:
    """Plots for Exp 1.

    TODO (yours): two plots suggested
      1. Collision rate: three rewards on the x axis, two bars each (filter off / on), with 95% confidence intervals.
      2. Success rate and intervention_mean (the three filter-on groups): does a policy that learned to dodge need
         the filter to step in less?
    See bar_chart() and plot_exp3() for how it's done; grouped bars are ax.bar(x ± width/2, ...).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rewards = list(dict.fromkeys(str(row["reward_variant"]) for row in rows))
    modes = ((None, "filter off"), ("filter", "filter on"))
    x = np.arange(len(rewards), dtype=float)
    width = 0.36

    def select(reward, mode):
        for row in rows:
            filter_on = row.get("filter_mode") is not None
            if row.get("reward_variant") == reward and filter_on == (mode == "filter"):
                return row
        raise ValueError(f"missing Exp 1 row for {reward}, {mode}")

    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    for index, (mode, label) in enumerate(modes):
        selected = [select(reward, mode) for reward in rewards]
        values = np.array([row["collision_rate"] for row in selected], dtype=float)
        lower = np.array([row["collision_rate_lo"] for row in selected], dtype=float)
        upper = np.array([row["collision_rate_hi"] for row in selected], dtype=float)
        offset = (index - 0.5) * width
        axis.bar(
            x + offset, values, width=width, label=label,
            yerr=np.vstack([values - lower, upper - values]), capsize=4,
        )
    axis.set_xticks(x, rewards)
    axis.set_ylabel("collision rate (hit balls / thrown)")
    axis.set_title("Exp 1: learned reward vs runtime enforcement")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_collision_rate.png", dpi=150)
    plt.close(fig)

    filter_rows = [select(reward, "filter") for reward in rewards]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    success = np.array([row["success_rate"] for row in filter_rows], dtype=float)
    success_lo = np.array([row["success_rate_lo"] for row in filter_rows], dtype=float)
    success_hi = np.array([row["success_rate_hi"] for row in filter_rows], dtype=float)
    axes[0].bar(
        x, success,
        yerr=np.vstack([success - success_lo, success_hi - success]), capsize=4,
        color="#4C78A8",
    )
    axes[0].set_title("Task success with filter on")
    axes[0].set_ylabel("success rate")
    interventions = [
        np.nan if row.get("intervention_mean") is None else row["intervention_mean"]
        for row in filter_rows
    ]
    axes[1].bar(x, interventions, color="#F58518")
    axes[1].set_title("Filter intervention by policy")
    axes[1].set_ylabel("mean command intervention")
    for axis in axes:
        axis.set_xticks(x, rewards, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_success_intervention.png", dpi=150)
    plt.close(fig)


def plot_exp2(rows: list[dict], out_dir: str | Path) -> None:
    """Plots for Exp 2.

    TODO (yours): collision rate of the 2 groups (naive / lag-aware) with 95% confidence intervals, and the collision
    rate grouped by ball arrival height (collision_rate_low / mid / high) for both groups side by side.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    bar_chart(
        rows, "collision_rate", out_dir / "exp2_collision_rate.png",
        "Exp 2: speed-lag model in the safety filter",
        "collision rate (hit balls / thrown)",
    )

    buckets = ("low", "mid", "high")
    x = np.arange(len(buckets), dtype=float)
    width = 0.8 / max(len(rows), 1)
    fig, axis = plt.subplots(figsize=(6.0, 3.8))
    for index, row in enumerate(rows):
        values = [row.get(f"collision_rate_{bucket}", np.nan) for bucket in buckets]
        offset = (index - (len(rows) - 1) / 2) * width
        label = str(row.get("filter_mode") or row.get("condition", index))
        axis.bar(x + offset, values, width=width, label=label)
    axis.set_xticks(x, buckets)
    axis.set_ylabel("collision rate")
    axis.set_xlabel("ball arrival height")
    axis.set_title("Exp 2: collisions by arrival height")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exp2_collision_by_height.png", dpi=150)
    plt.close(fig)
