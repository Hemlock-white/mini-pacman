"""Find balls in the depth image and track their position and velocity.

The CBF filter needs numbers ("where is the ball, how fast"), but the depth camera only gives an image, so this
layer sits in between:

    depth image ──BallDetector──► detections per image (ball centre, radius)
                ──BallTracker───► an estimate per ball (position, velocity, radius), including floor bounces

BallDetector has two modes (TrackerConfig.mode)
-----------------------------------------------
* ``"segmented"``: finds ball pixels from the geom each pixel hit. This is idealised "perfect segmentation", the same
  as PAC-MAN's BallOnlyDepthObs. A real robot would need a separate detector (e.g. colour or a learned model).
* ``"geometric"``: depth only. Turn each pixel into a 3D point, drop points "near the floor" and points "on the
  robot", and what is left are candidates. More realistic, but less accurate with floor noise or a rolling ball.

How the ball centre is estimated
--------------------------------
The camera only sees the side of the ball facing it. Average the visible surface points, then push back along the
line of sight by (2/3)·r to get the centre (projected onto a disc, the average surface depth of a sphere is r·2/3).
The radius r comes from the solid angle the ball covers in the image: s = sqrt(Ω/π) ≈ r / distance to the centre.
With very few pixels this is poor, so it is clipped to BallConfig.radius_range and the tracker averages it per ball.

BallTracker
-----------
One Kalman filter per ball, state [position(3), velocity(3)], projectile motion model (known gravity).
* If a prediction puts the centre below the radius (touchdown), the vertical velocity is multiplied by -e (e = mean of
  restitution_range) and the vertical-velocity uncertainty is increased, since the true e is unknown.
* Data association: a prediction and a detection are paired if they are closer than the gate (greedy, nearest first).
* A new detection starts a "tentative" track; it is reported after confirm_hits sightings in a row and deleted after
  max_misses misses in a row.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy import ndimage

from .config import Config, default_config
from .depth_camera import DepthCamera, DepthFrame

G = 9.81


@dataclass
class BallEstimate:
    """State estimate of one ball (the format shared by every ball information source)."""
    id: int
    position: np.ndarray       # world frame (m)
    velocity: np.ndarray       # world frame (m/s)
    radius: float
    restitution: float         # true value or prior mean
    stamp: float               # the time this estimate refers to (s); earlier than now when there is latency
    source: str                # "truth" / "noisy_truth" / "depth_tracker"
    pos_std: float = 0.0       # position uncertainty (m)
    vel_std: float = 0.0       # velocity uncertainty (m/s)


@dataclass
class Detection:
    position: np.ndarray       # estimated ball centre (world frame)
    radius: float
    n_pixels: int
    range: float               # mean distance from the camera to the visible surface
    sigma: float               # position uncertainty of this detection (m)


class BallDetector:
    def __init__(self, camera: DepthCamera, cfg: Config | None = None):
        self.cam = camera
        self.cfg = cfg or default_config()
        self.tc = self.cfg.tracker
        self.r_lo, self.r_hi = self.cfg.balls.radius_range

    def detect(self, frame: DepthFrame, d: mujoco.MjData | None = None) -> list[Detection]:
        """Return the balls in this depth image. Geometric mode needs d (to remove points on the robot)."""
        finite = np.isfinite(frame.depth)
        if self.tc.mode == "segmented":
            mask = self.cam.ball_mask(frame)
            pts = frame.points().reshape(self.cam.H, self.cam.W, 3)
        elif self.tc.mode == "geometric":
            if d is None:
                raise ValueError("geometric mode needs MjData")
            pts = frame.points().reshape(self.cam.H, self.cam.W, 3)
            mask = finite & (np.nan_to_num(pts[..., 2], nan=-1.0) > self.tc.floor_threshold)
            mask &= ~self._on_robot(pts, d)
        else:
            raise ValueError(f"unknown tracker mode: {self.tc.mode}")
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        out = []
        for k in range(1, n + 1):
            sel = labels == k
            npx = int(sel.sum())
            if npx < self.tc.min_pixels:
                continue
            P = pts[sel]
            dirs = frame.dirs_world.reshape(self.cam.H, self.cam.W, 3)[sel]
            u = dirs.mean(axis=0)
            u /= np.linalg.norm(u)
            rng_mean = float(frame.depth[sel].mean())
            omega = float(self.cam.pixel_solid_angle.reshape(self.cam.H, self.cam.W)[sel].sum())
            s = np.sqrt(omega / np.pi)
            r = rng_mean * s / max(1.0 - (2.0 / 3.0) * s, 1e-3)
            r = float(np.clip(r, self.r_lo, self.r_hi))
            center = P.mean(axis=0) + (2.0 / 3.0) * r * u
            cc = self.cam.cc
            sigma_range = cc.noise_a + cc.noise_b * rng_mean ** 2
            sigma_lat = rng_mean * self.cam.pixel_angle / 2
            sigma = float(np.sqrt(sigma_range ** 2 / max(npx, 1) + sigma_lat ** 2 + (0.3 * r) ** 2 / max(npx, 1)))
            out.append(Detection(position=center, radius=r, n_pixels=npx, range=rng_mean, sigma=sigma))
        return out

    def _on_robot(self, pts: np.ndarray, d: mujoco.MjData) -> np.ndarray:
        """Points near the robot (inside a box in the body frame)."""
        b = self.cam.ids.base_body
        R = d.xmat[b].reshape(3, 3)
        local = np.nan_to_num(pts - d.xpos[b], nan=1e3) @ R
        return ((np.abs(local[..., 0]) < 0.40) & (np.abs(local[..., 1]) < 0.40)
                & (local[..., 2] > -0.70) & (local[..., 2] < 0.25))


@dataclass
class _Track:
    id: int
    x: np.ndarray              # [p, v]
    P: np.ndarray
    radius: float
    hits: int = 1
    misses: int = 0
    t: float = 0.0
    r_samples: list = field(default_factory=list)


class BallTracker:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or default_config()
        self.tc = self.cfg.tracker
        self.e_prior = float(np.mean(self.cfg.balls.restitution_range))
        lo, hi = self.cfg.balls.restitution_range
        self.e_std = (hi - lo) / np.sqrt(12)
        self.reset()

    def reset(self) -> None:
        self.tracks: list[_Track] = []
        self._next_id = 0
        self.t = None

    # ---------------- Kalman filter ----------------
    def _predict(self, tr: _Track, t: float) -> None:
        dt = t - tr.t
        if dt <= 0:
            return
        F = np.eye(6)
        F[0:3, 3:6] = dt * np.eye(3)
        x = F @ tr.x
        x[2] -= 0.5 * G * dt ** 2
        x[5] -= G * dt
        q = self.tc.accel_noise ** 2
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = dt ** 4 / 4 * q * np.eye(3)
        Q[0:3, 3:6] = Q[3:6, 0:3] = dt ** 3 / 2 * q * np.eye(3)
        Q[3:6, 3:6] = dt ** 2 * q * np.eye(3)
        P = F @ tr.P @ F.T + Q
        if x[2] < tr.radius and x[5] < 0:              # touchdown: bounce
            x[2] = tr.radius + (tr.radius - x[2])
            x[5] = -self.e_prior * x[5]
            P[5, 5] += (self.e_std * x[5] / max(self.e_prior, 1e-3)) ** 2 + 0.1
        tr.x, tr.P, tr.t = x, P, t

    def _update(self, tr: _Track, det: Detection) -> None:
        H = np.zeros((3, 6))
        H[:, 0:3] = np.eye(3)
        R = det.sigma ** 2 * np.eye(3)
        S = H @ tr.P @ H.T + R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (det.position - H @ tr.x)
        tr.P = (np.eye(6) - K @ H) @ tr.P
        tr.r_samples.append(det.radius)
        tr.radius = float(np.mean(tr.r_samples[-20:]))
        tr.hits += 1
        tr.misses = 0

    # ---------------- main loop ----------------
    def update(self, detections: list[Detection], t: float) -> list[BallEstimate]:
        """Update all tracks with this image's detections and return the confirmed balls."""
        for tr in self.tracks:
            self._predict(tr, t)
        pairs = []
        for i, tr in enumerate(self.tracks):
            for j, det in enumerate(detections):
                dist = np.linalg.norm(tr.x[0:3] - det.position)
                gate = self.tc.gate + 3 * np.sqrt(np.trace(tr.P[0:3, 0:3]) / 3)
                if dist < gate:
                    pairs.append((dist, i, j))
        used_t, used_d = set(), set()
        for _, i, j in sorted(pairs):
            if i in used_t or j in used_d:
                continue
            self._update(self.tracks[i], detections[j])
            used_t.add(i)
            used_d.add(j)
        for i, tr in enumerate(self.tracks):
            if i not in used_t:
                tr.misses += 1
        for j, det in enumerate(detections):
            if j not in used_d:
                P = np.diag([det.sigma ** 2] * 3 + [self.tc.init_speed_std ** 2] * 3)
                self.tracks.append(_Track(id=self._next_id, x=np.concatenate([det.position, np.zeros(3)]),
                                          P=P, radius=det.radius, t=t, r_samples=[det.radius]))
                self._next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.tc.max_misses and tr.x[2] > -0.5]
        self.t = t
        return self.estimates()

    def estimates(self) -> list[BallEstimate]:
        out = []
        for tr in self.tracks:
            if tr.hits < self.tc.confirm_hits:
                continue
            out.append(BallEstimate(
                id=tr.id, position=tr.x[0:3].copy(), velocity=tr.x[3:6].copy(), radius=tr.radius,
                restitution=self.e_prior, stamp=tr.t, source="depth_tracker",
                pos_std=float(np.sqrt(np.trace(tr.P[0:3, 0:3]) / 3)),
                vel_std=float(np.sqrt(np.trace(tr.P[3:6, 3:6]) / 3))))
        return out
