"""Ball information sources: where the CBF filter (and the policy, if you want) gets "where is the ball, how fast".

Exp 3 compares these three sources:

=================  ====================================================================
``truth``          simulator truth (what PAC-MAN calls the "simulation-only safety ceiling")
``noisy_truth``    truth + position/velocity noise + latency + occasionally missing (NoisyTruthConfig)
``depth_tracker``  depth camera → BallDetector → BallTracker (actually estimated from the sensor)
=================  ====================================================================

All three have the same interface:

    source = make_ball_source("depth_tracker", m, cfg, launcher, camera)
    source.reset(rng)                         # at the start of each episode
    balls = source.update(d, rng, frame)      # every policy step; pass an already-rendered frame to avoid recomputing
    for b in balls: b.position, b.velocity, b.radius, b.stamp ...

``BallEstimate.stamp`` is the time the estimate refers to. With latency, stamp is earlier than now, and the user
(the CBF) should push it forward to now with ``balls.predict_ball``:

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
            raise ValueError("noisy_truth needs an rng")
        self.buf.append(self.truth.update(d))
        if len(self.buf) <= self.delay:
            return []
        out = []
        for b in self.buf[0]:                       # the oldest entry = delay steps ago
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
        """frame: a clean depth image already rendered this step (e.g. the one for the policy); rendered here if not
        given. Noise is added here."""
        if rng is None:
            raise ValueError("depth_tracker needs an rng (depth noise)")
        frame = frame if frame is not None else self.camera.render(d)
        noisy = self.camera.add_noise(frame, rng)
        self.last_frame = noisy
        self.last_detections = self.detector.detect(noisy, d)
        return self.tracker.update(self.last_detections, float(d.time))


def make_ball_source(kind: str, m: mujoco.MjModel, cfg: Config | None, launcher: BallLauncher,
                     camera: DepthCamera | None = None):
    """Create a ball information source by name. kind ∈ SOURCES."""
    cfg = cfg or default_config()
    if kind == "truth":
        return TruthBallSource(launcher)
    if kind == "noisy_truth":
        return NoisyTruthBallSource(launcher, cfg)
    if kind == "depth_tracker":
        return DepthTrackerBallSource(m, camera or DepthCamera(m, cfg, launcher.ids), cfg)
    raise ValueError(f"unknown ball information source {kind!r}, choose from {SOURCES}")
