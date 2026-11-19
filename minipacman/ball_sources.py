"""球的資訊來源：CBF 過濾器（以及想用的話，policy）從這裡拿「球在哪、速度多少」。

實驗 3 就是比較這三種來源：

=================  ====================================================================
``truth``          模擬器的真實值（PAC-MAN 說的「只能在模擬裡用的安全上限」）
``noisy_truth``    真實值 + 位置/速度雜訊 + 延遲 + 偶爾看不到（NoisyTruthConfig）
``depth_tracker``  深度相機 → BallDetector → BallTracker（真的從感測器估出來）
=================  ====================================================================

三種來源介面一樣：

    source = make_ball_source("depth_tracker", m, cfg, launcher, camera)
    source.reset(rng)                         # 每個 episode 開始
    balls = source.update(d, rng, frame)      # 每個 policy 步；frame 可傳入已經算好的深度圖，避免重算
    for b in balls: b.position, b.velocity, b.radius, b.stamp ...

``BallEstimate.stamp`` 是這個估計對應的時間。有延遲時 stamp 會比現在早，
使用者（CBF）應該用 ``balls.predict_ball`` 把它往前推到現在：

    p_now, v_now = predict_ball(b.position, b.velocity, b.radius, b.restitution, d.time - b.stamp)
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from .ball_tracker import BallDetector, BallEstimate, BallTracker
from .balls import BallLauncher
from .config import Config, default_config
from .depth_camera import DepthCamera, DepthFrame

SOURCES = ("truth", "noisy_truth", "depth_tracker")


class TruthBallSource:
    kind = "truth"

    def __init__(self, launcher: BallLauncher):
        self.launcher = launcher

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def update(self, d: mujoco.MjData, rng: np.random.Generator | None = None,
               frame: DepthFrame | None = None) -> list[BallEstimate]:
        out = []
        for rec in self.launcher.active_records():
            p, v = self.launcher.ball_state(rec.index)
            out.append(BallEstimate(id=rec.index, position=p, velocity=v, radius=rec.radius,
                                    restitution=rec.restitution, stamp=float(d.time), source=self.kind))
        return out


class NoisyTruthBallSource:
    kind = "noisy_truth"

    def __init__(self, launcher: BallLauncher, cfg: Config | None = None):
        self.cfg = cfg or default_config()
        self.nc = self.cfg.noisy_truth
        self.truth = TruthBallSource(launcher)
        self.delay = int(round(self.nc.latency / self.cfg.sim.policy_dt))
        self.buf: deque = deque(maxlen=self.delay + 1)
        self.e_prior = float(np.mean(self.cfg.balls.restitution_range))

    def reset(self, rng: np.random.Generator) -> None:
        self.buf.clear()

    def update(self, d: mujoco.MjData, rng: np.random.Generator | None = None,
               frame: DepthFrame | None = None) -> list[BallEstimate]:
        if rng is None:
            raise ValueError("noisy_truth 需要 rng")
        self.buf.append(self.truth.update(d))
        if len(self.buf) <= self.delay:
            return []
        out = []
        for b in self.buf[0]:                       # 最舊的一筆 = delay 步之前
            if rng.random() < self.nc.dropout:
                continue
            out.append(BallEstimate(
                id=b.id, position=b.position + rng.normal(0, self.nc.pos_sigma, 3),
                velocity=b.velocity + rng.normal(0, self.nc.vel_sigma, 3),
                radius=b.radius, restitution=self.e_prior, stamp=b.stamp, source=self.kind,
                pos_std=self.nc.pos_sigma, vel_std=self.nc.vel_sigma))
        return out


class DepthTrackerBallSource:
    kind = "depth_tracker"

    def __init__(self, m: mujoco.MjModel, camera: DepthCamera, cfg: Config | None = None):
        self.cfg = cfg or default_config()
        self.camera = camera
        self.detector = BallDetector(camera, self.cfg)
        self.tracker = BallTracker(self.cfg)
        self.last_frame: DepthFrame | None = None
        self.last_detections = []

    def reset(self, rng: np.random.Generator) -> None:
        self.tracker.reset()
        self.last_frame = None

    def update(self, d: mujoco.MjData, rng: np.random.Generator | None = None,
               frame: DepthFrame | None = None) -> list[BallEstimate]:
        """frame：這一步已經算好的「乾淨」深度圖（例如 policy 觀測用的）；沒給就自己算。雜訊在這裡加。"""
        if rng is None:
            raise ValueError("depth_tracker 需要 rng（深度雜訊）")
        frame = frame if frame is not None else self.camera.render(d)
        noisy = self.camera.add_noise(frame, rng)
        self.last_frame = noisy
        self.last_detections = self.detector.detect(noisy, d)
        return self.tracker.update(self.last_detections, float(d.time))


def make_ball_source(kind: str, m: mujoco.MjModel, cfg: Config | None, launcher: BallLauncher,
                     camera: DepthCamera | None = None):
    """依名稱建立球資訊來源。kind ∈ SOURCES。"""
    cfg = cfg or default_config()
    if kind == "truth":
        return TruthBallSource(launcher)
    if kind == "noisy_truth":
        return NoisyTruthBallSource(launcher, cfg)
    if kind == "depth_tracker":
        return DepthTrackerBallSource(m, camera or DepthCamera(m, cfg, launcher.ids), cfg)
    raise ValueError(f"未知的球資訊來源 {kind!r}，可用：{SOURCES}")
