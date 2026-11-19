"""深度相機：用 MuJoCo 光線投射 (mj_multiRay) 在 CPU 上算深度圖。

為什麼不用 OpenGL 渲染
----------------------
* 不需要顯示驅動或 EGL，AUP、NVIDIA 電腦、筆電上結果一樣，也能在多個平行環境（子行程）裡用。
* 解析度很低（64x36 = 2304 條光線），CPU 投射一張約 0.3 ms，比渲染再讀回快。
* 每個像素順便知道打到哪個 geom，「只留球」的影像（PAC-MAN 的 BallOnlyDepthObs）不用另外做分割。

相機在哪裡
----------
TITA 前方真實的雙目相機：robot.xml 的 left_camera / right_camara 兩個 body 的中點，朝向和車身相同，
再往上仰 ``tilt_up_deg`` 度。相機座標：x 朝前、y 朝左、z 朝上；影像的列 (row) 由上到下、行 (col) 由左到右。

輸出
----
* ``render(d)`` → ``DepthFrame``：高解析度 (64x36) 的「距離」（沿光線的距離，單位 m，沒打到 = inf），
  以及每個像素打到的 geom id。這是乾淨的真實值。
* ``add_noise(frame, rng)``：加上感測雜訊（sigma = a + b·d²、隨機失效像素），給追蹤器用。
* ``policy_image(frame)``：縮成 16x9（每格取最小距離，小球才不會消失）、只留球、正規化到 [0, 1]：
  0 = near（很近）、1 = far 或沒東西。訓練時可以加 PAC-MAN 的資料增強。
* ``DepthFrameStack``：把最近的 policy 影像依 frame_offsets（預設 0, 3, 8, 18 步前）疊起來，
  攤平成一個向量（4 × 16 × 9 = 576 維），讓 policy 看得出球的速度。
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
    """一張高解析度深度圖。"""
    depth: np.ndarray        # (H, W) 沿光線的距離 (m)，沒打到 = inf
    geomid: np.ndarray       # (H, W) 打到的 geom id，沒打到 = -1
    position: np.ndarray     # 相機位置（世界座標）
    rotation: np.ndarray     # 相機姿態 (3x3)，欄向量 = 相機的 x(前)、y(左)、z(上) 軸
    dirs_world: np.ndarray   # (H*W, 3) 每個像素的光線方向（世界座標，單位向量）
    time: float

    def points(self) -> np.ndarray:
        """每個像素打到的 3D 點 (H*W, 3)，沒打到的是 nan。"""
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
        u = (2 * (np.arange(W) + 0.5) / W - 1) * self.tan_x       # 往右為正
        v = (2 * (np.arange(H) + 0.5) / H - 1) * self.tan_y       # 往下為正
        U, V = np.meshgrid(u, v)
        dirs = np.stack([np.ones_like(U), -U, -V], axis=-1).reshape(-1, 3)
        norm = np.linalg.norm(dirs, axis=1, keepdims=True)
        self.dirs_cam = dirs / norm
        # 每個像素的立體角（估計球的大小用）
        du, dv = 2 * self.tan_x / W, 2 * self.tan_y / H
        self.pixel_solid_angle = (du * dv / norm[:, 0] ** 3)
        self.pixel_angle = 2 * np.deg2rad(cc.fovy_deg / 2) / H     # 一個像素約幾弧度
        t = np.deg2rad(cc.tilt_up_deg)
        self.R_tilt = np.array([[np.cos(t), 0, -np.sin(t)],        # 繞 y 軸轉 -t：x 軸往上抬
                                [0, 1, 0],
                                [np.sin(t), 0, np.cos(t)]])
        self.geomgroup = np.array(cc.geomgroups, dtype=np.uint8)
        self.ball_geoms = np.array(self.ids.ball_geoms)
        self._geomid = np.zeros(W * H, dtype=np.int32)
        self._dist = np.zeros(W * H, dtype=np.float64)
        pw, ph = cc.policy_res
        if W % pw or H % ph:
            raise ValueError(f"hi_res {cc.hi_res} 必須是 policy_res {cc.policy_res} 的整數倍")
        self.block = (H // ph, W // pw)

    # ---------------- 相機姿態 ----------------
    def pose(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """相機在世界座標的 (位置, 姿態矩陣)。"""
        pos = d.xpos[self.ids.camera_bodies].mean(axis=0)
        R_base = d.xmat[self.ids.base_body].reshape(3, 3)
        return pos.copy(), R_base @ self.R_tilt

    # ---------------- 深度圖 ----------------
    def render(self, d: mujoco.MjData) -> DepthFrame:
        """投射所有光線，回傳乾淨（沒有雜訊）的深度圖。"""
        pos, R = self.pose(d)
        dirs_world = self.dirs_cam @ R.T
        vec = np.ascontiguousarray(dirs_world.reshape(-1))
        n = self.W * self.H
        # bodyexclude = 車身：相機在車殼裡面，不排除的話光線會先打到車殼內側
        mujoco.mj_multiRay(self.m, d, pos, vec, self.geomgroup, 1, int(self.ids.base_body),
                           self._geomid, self._dist, None, n, float(self.cc.far))
        dist = self._dist.copy()
        dist[dist < 0] = np.inf
        return DepthFrame(depth=dist.reshape(self.H, self.W), geomid=self._geomid.reshape(self.H, self.W).copy(),
                          position=pos, rotation=R, dirs_world=dirs_world, time=float(d.time))

    def add_noise(self, frame: DepthFrame, rng: np.random.Generator) -> DepthFrame:
        """加上感測雜訊：高斯雜訊 sigma = a + b·d²，以及隨機失效的像素（變成 inf）。"""
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
        """(H, W) 布林陣列：哪些像素打到球。"""
        return np.isin(frame.geomid, self.ball_geoms) & np.isfinite(frame.depth)

    # ---------------- policy 用的小影像 ----------------
    def policy_image(self, frame: DepthFrame, rng: np.random.Generator | None = None,
                     augment: bool = False) -> np.ndarray:
        """16x9、正規化到 [0, 1] 的影像（float32）。ball_only=True 時球以外都設成 far。

        augment=True（只在訓練時）加 PAC-MAN 的增強，只作用在球的像素上：
        整顆球消失 (aug_ball_full_dropout)、球的像素隨機消失 (aug_ball_pixel_dropout)、
        整張的球深度偏移 (±aug_ball_depth_jitter)、高斯雜訊 (aug_ball_noise_sigma)。
        """
        cc = self.cc
        depth = frame.depth.copy()
        ball = self.ball_mask(frame)
        if cc.ball_only:
            depth[~ball] = np.inf
        if augment:
            if rng is None:
                raise ValueError("augment=True 需要 rng")
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
    """保存最近的 policy 影像，依 offsets 取出並攤平。

    offsets=(0, 3, 8, 18) 表示「這一步、3 步前、8 步前、18 步前」（以 policy 步計算，50 Hz 時約 0–0.36 s）。
    重置時用「全部 far (=1)」填滿，代表還沒看到任何東西。
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
        """回傳 (len(offsets) * H * W,) 的向量，最新的在最前面。"""
        return np.concatenate([self.buf[-1 - k].reshape(-1) for k in self.offsets])

    @property
    def dim(self) -> int:
        return len(self.offsets) * self.shape[0] * self.shape[1]
