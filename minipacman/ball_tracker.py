"""從深度圖找出球、追蹤球的位置和速度。

CBF 過濾器需要「球在哪、速度多少」的數字，深度相機只給影像，所以中間需要這一層：

    深度圖 ──BallDetector──► 每張圖的偵測結果（球心位置、半徑）
           ──BallTracker───► 每顆球的估計（位置、速度、半徑），含落地彈跳

BallDetector 兩種模式（TrackerConfig.mode）
------------------------------------------
* ``"segmented"``：用每個像素打到的 geom 找出球的像素。這是理想化的「完美分割」，
  和 PAC-MAN 的 BallOnlyDepthObs 一樣。真實機器人需要另外做偵測（例如顏色或學習的模型）。
* ``"geometric"``：只用深度。把每個像素換成 3D 點，去掉「接近地面」和「在機器人身上」的點，
  剩下的就是候選。比較真實，但地面雜訊或球在地上滾時會比較不準。

球心怎麼估
----------
相機只看到球朝向相機的那一面。把看到的表面點平均，再沿著視線往後推 (2/3)·r 就是球心
（球面投影到圓盤上，表面深度的平均是 r·2/3）。半徑 r 由球在影像中佔的立體角估計：
s = sqrt(Ω/π) ≈ r / 球心距離。像素很少時估得很差，所以會夾在 BallConfig.radius_range 內，
追蹤器再對每顆球的半徑做平均。

BallTracker
-----------
每顆球一個卡爾曼濾波器，狀態 [位置(3), 速度(3)]，運動模型是拋體（已知重力）。
* 預測時如果球心低於半徑（落地），就把垂直速度乘 -e（e 用 restitution_range 的平均），
  並加大垂直速度的不確定度，因為真實的 e 不知道。
* 資料關聯：預測位置和偵測位置距離小於 gate 就配對（由近到遠貪婪配對）。
* 新的偵測先開一條「暫定」軌跡，連續看到 confirm_hits 次才輸出；連續 max_misses 次沒看到就刪除。
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
    """一顆球的狀態估計（所有球資訊來源共用這個格式）。"""
    id: int
    position: np.ndarray       # 世界座標 (m)
    velocity: np.ndarray       # 世界座標 (m/s)
    radius: float
    restitution: float         # 真實值或先驗平均
    stamp: float               # 這個估計對應的時間 (s)；有延遲時 < 現在時間
    source: str                # "truth" / "noisy_truth" / "depth_tracker"
    pos_std: float = 0.0       # 位置不確定度 (m)
    vel_std: float = 0.0       # 速度不確定度 (m/s)


@dataclass
class Detection:
    position: np.ndarray       # 估計的球心（世界座標）
    radius: float
    n_pixels: int
    range: float               # 相機到看到的表面的平均距離
    sigma: float               # 這個偵測的位置不確定度 (m)


class BallDetector:
    def __init__(self, camera: DepthCamera, cfg: Config | None = None):
        self.cam = camera
        self.cfg = cfg or default_config()
        self.tc = self.cfg.tracker
        self.r_lo, self.r_hi = self.cfg.balls.radius_range

    def detect(self, frame: DepthFrame, d: mujoco.MjData | None = None) -> list[Detection]:
        """回傳這張深度圖裡的球。geometric 模式需要 d（用來去掉機器人身上的點）。"""
        finite = np.isfinite(frame.depth)
        if self.tc.mode == "segmented":
            mask = self.cam.ball_mask(frame)
            pts = frame.points().reshape(self.cam.H, self.cam.W, 3)
        elif self.tc.mode == "geometric":
            if d is None:
                raise ValueError("geometric 模式需要 MjData")
            pts = frame.points().reshape(self.cam.H, self.cam.W, 3)
            mask = finite & (np.nan_to_num(pts[..., 2], nan=-1.0) > self.tc.floor_threshold)
            mask &= ~self._on_robot(pts, d)
        else:
            raise ValueError(f"未知的 tracker mode: {self.tc.mode}")
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
        """在機器人附近（車身座標的一個盒子內）的點。"""
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

    # ---------------- 卡爾曼濾波 ----------------
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
        if x[2] < tr.radius and x[5] < 0:              # 落地：反彈
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

    # ---------------- 主流程 ----------------
    def update(self, detections: list[Detection], t: float) -> list[BallEstimate]:
        """用這一張圖的偵測結果更新所有軌跡，回傳已確認的球。"""
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
