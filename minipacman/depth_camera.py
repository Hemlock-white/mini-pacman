"""Depth camera: depth images computed on the CPU with MuJoCo ray casting (mj_multiRay).

Why not OpenGL rendering
------------------------
* No display driver or EGL needed; same results on AUP, an NVIDIA box or a laptop, and it works inside many parallel
  environments (subprocesses).
* The resolution is tiny (64x36 = 2304 rays); casting one image on the CPU takes about 0.3 ms, faster than
  rendering and reading back.
* Every pixel also tells us which geom it hit, so the "ball only" image (PAC-MAN's BallOnlyDepthObs) needs no
  separate segmentation.

Where the camera is
-------------------
TITA's real front stereo camera: the midpoint of the left_camera / right_camara bodies in robot.xml, facing the same
way as the body, tilted up by ``tilt_up_deg``. Camera frame: x forward, y left, z up; image rows go top to bottom,
columns left to right.

Outputs
-------
* ``render(d)`` → ``DepthFrame``: high-res (64x36) "range" (distance along each ray, in m; no hit = inf) and the geom
  id each pixel hit. These are clean true values.
* ``add_noise(frame, rng)``: adds sensor noise (sigma = a + b·d², random dead pixels), for the tracker.
* ``policy_image(frame)``: shrink to 16x9 (minimum distance per block, so small balls don't vanish), ball only,
  normalised to [0, 1]: 0 = near (very close), 1 = far or nothing. PAC-MAN's augmentation can be added in training.
* ``DepthFrameStack``: stacks recent policy images at frame_offsets (default 0, 3, 8, 18 steps ago) and flattens them
  into one vector (4 × 16 × 9 = 576), so the policy can see how the ball moves.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from .config import Config, default_config
from .scene import SceneIds, scene_ids


@dataclass
class DepthFrame:
    """One high-res depth image."""
    depth: np.ndarray        # (H, W) distance along the ray (m), no hit = inf
    geomid: np.ndarray       # (H, W) id of the geom hit, no hit = -1
    position: np.ndarray     # camera position (world frame)
    rotation: np.ndarray     # camera orientation (3x3); columns = the camera's x (forward), y (left), z (up) axes
    dirs_world: np.ndarray   # (H*W, 3) ray direction of each pixel (world frame, unit vectors)
    time: float

    def points(self) -> np.ndarray:
        """3D point hit by each pixel (H*W, 3); nan where nothing was hit."""
        dist = self.depth.reshape(-1).copy()
        dist[~np.isfinite(dist)] = np.nan
        return self.position + dist[:, None] * self.dirs_world


class DepthCamera:
    def __init__(self, m: mujoco.MjModel, cfg: Config | None = None, ids: SceneIds | None = None):
        self.m = m
        self.cfg = cfg or default_config()
        self.cc = cc = self.cfg.camera
        self.ids = ids or scene_ids(m, self.cfg)
        W, H = cc.hi_res
        self.W, self.H = W, H
        self.tan_y = np.tan(np.deg2rad(cc.fovy_deg) / 2)
        self.tan_x = self.tan_y * W / H
        u = (2 * (np.arange(W) + 0.5) / W - 1) * self.tan_x       # positive to the right
        v = (2 * (np.arange(H) + 0.5) / H - 1) * self.tan_y       # positive downwards
        U, V = np.meshgrid(u, v)
        dirs = np.stack([np.ones_like(U), -U, -V], axis=-1).reshape(-1, 3)
        norm = np.linalg.norm(dirs, axis=1, keepdims=True)
        self.dirs_cam = dirs / norm
        # solid angle of each pixel (used to estimate ball size)
        du, dv = 2 * self.tan_x / W, 2 * self.tan_y / H
        self.pixel_solid_angle = (du * dv / norm[:, 0] ** 3)
        self.pixel_angle = 2 * np.deg2rad(cc.fovy_deg / 2) / H     # roughly how many radians one pixel covers
        t = np.deg2rad(cc.tilt_up_deg)
        self.R_tilt = np.array([[np.cos(t), 0, -np.sin(t)],        # rotate by -t about y: lifts the x axis up
                                [0, 1, 0],
                                [np.sin(t), 0, np.cos(t)]])
        self.geomgroup = np.array(cc.geomgroups, dtype=np.uint8)
        self.ball_geoms = np.array(self.ids.ball_geoms)
        self._geomid = np.zeros(W * H, dtype=np.int32)
        self._dist = np.zeros(W * H, dtype=np.float64)
        pw, ph = cc.policy_res
        if W % pw or H % ph:
            raise ValueError(f"hi_res {cc.hi_res} must be an integer multiple of policy_res {cc.policy_res}")
        self.block = (H // ph, W // pw)

    # ---------------- camera pose ----------------
    def pose(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(position, orientation matrix) of the camera in the world frame."""
        pos = d.xpos[self.ids.camera_bodies].mean(axis=0)
        R_base = d.xmat[self.ids.base_body].reshape(3, 3)
        return pos.copy(), R_base @ self.R_tilt

    # ---------------- depth image ----------------
    def render(self, d: mujoco.MjData) -> DepthFrame:
        """Cast all rays and return a clean (noise-free) depth image."""
        pos, R = self.pose(d)
        dirs_world = self.dirs_cam @ R.T
        vec = np.ascontiguousarray(dirs_world.reshape(-1))
        n = self.W * self.H
        # bodyexclude = the body: the camera sits inside the shell, otherwise rays hit the inside of the shell first
        mujoco.mj_multiRay(self.m, d, pos, vec, self.geomgroup, 1, int(self.ids.base_body),
                           self._geomid, self._dist, None, n, float(self.cc.far))
        dist = self._dist.copy()
        dist[dist < 0] = np.inf
        return DepthFrame(depth=dist.reshape(self.H, self.W), geomid=self._geomid.reshape(self.H, self.W).copy(),
                          position=pos, rotation=R, dirs_world=dirs_world, time=float(d.time))

    def add_noise(self, frame: DepthFrame, rng: np.random.Generator) -> DepthFrame:
        """Add sensor noise: Gaussian with sigma = a + b·d², plus randomly dead pixels (set to inf)."""
        depth = frame.depth.copy()
        finite = np.isfinite(depth)
        sigma = self.cc.noise_a + self.cc.noise_b * depth[finite] ** 2
        depth[finite] += rng.normal(0.0, 1.0, finite.sum()) * sigma
        depth[finite] = np.maximum(depth[finite], self.cc.near)
        drop = rng.random(depth.shape) < self.cc.pixel_dropout
        depth[drop] = np.inf
        return DepthFrame(depth=depth, geomid=np.where(drop, -1, frame.geomid), position=frame.position,
                          rotation=frame.rotation, dirs_world=frame.dirs_world, time=frame.time)

    def ball_mask(self, frame: DepthFrame) -> np.ndarray:
        """(H, W) boolean array: which pixels hit a ball."""
        return np.isin(frame.geomid, self.ball_geoms) & np.isfinite(frame.depth)

    # ---------------- small image for the policy ----------------
    def policy_image(self, frame: DepthFrame, rng: np.random.Generator | None = None,
                     augment: bool = False) -> np.ndarray:
        """16x9 image normalised to [0, 1] (float32). With ball_only=True everything except balls is set to far.

        augment=True (training only) adds PAC-MAN's augmentation, on ball pixels only:
        the whole ball disappears (aug_ball_full_dropout), ball pixels randomly disappear (aug_ball_pixel_dropout),
        a per-image depth offset on the ball (±aug_ball_depth_jitter), and Gaussian noise (aug_ball_noise_sigma).
        """
        cc = self.cc
        depth = frame.depth.copy()
        ball = self.ball_mask(frame)
        if cc.ball_only:
            depth[~ball] = np.inf
        if augment:
            if rng is None:
                raise ValueError("augment=True needs an rng")
            if rng.random() < cc.aug_ball_full_dropout:
                depth[ball] = np.inf
            else:
                drop = ball & (rng.random(depth.shape) < cc.aug_ball_pixel_dropout)
                keep = ball & ~drop
                depth[drop] = np.inf
                depth[keep] += rng.uniform(-cc.aug_ball_depth_jitter, cc.aug_ball_depth_jitter)
                depth[keep] += rng.normal(0.0, cc.aug_ball_noise_sigma, keep.sum())
        bh, bw = self.block
        ph, pw = self.H // bh, self.W // bw
        pooled = depth.reshape(ph, bh, pw, bw).min(axis=(1, 3))
        pooled = np.clip(pooled, cc.near, cc.far)
        return ((pooled - cc.near) / (cc.far - cc.near)).astype(np.float32)


class DepthFrameStack:
    """Keeps recent policy images and returns them at the given offsets, flattened.

    offsets=(0, 3, 8, 18) means "this step, 3 steps ago, 8 steps ago, 18 steps ago" (in policy steps, about 0-0.36 s
    at 50 Hz). On reset it is filled with "all far (= 1)", meaning nothing seen yet.
    """

    def __init__(self, image_shape: tuple[int, int], offsets: tuple[int, ...]):
        self.shape = image_shape
        self.offsets = tuple(offsets)
        self.buf: deque = deque(maxlen=max(self.offsets) + 1)
        self.reset()

    def reset(self) -> None:
        self.buf.clear()
        for _ in range(self.buf.maxlen):
            self.buf.append(np.ones(self.shape, dtype=np.float32))

    def push(self, image: np.ndarray) -> None:
        self.buf.append(image.astype(np.float32))

    def get(self) -> np.ndarray:
        """Return a (len(offsets) * H * W,) vector, newest first."""
        return np.concatenate([self.buf[-1 - k].reshape(-1) for k in self.offsets])

    @property
    def dim(self) -> int:
        return len(self.offsets) * self.shape[0] * self.shape[1]
