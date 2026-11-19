"""深度相機：幾何正確、看得到球、雜訊大小、疊影像。"""
import mujoco
import numpy as np
import pytest

from minipacman import scene
from minipacman.ball_tracker import BallDetector
from minipacman.balls import set_ball_properties
from minipacman.depth_camera import DepthCamera, DepthFrameStack


def _place_ball(m, d, ids, i, pos, radius, cfg):
    set_ball_properties(m, ids, i, radius, 0.136, 0.6, cfg.balls)
    qa = int(ids.ball_qadr[i])
    d.qpos[qa:qa + 7] = [*pos, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)


def test_floor_depth_matches_geometry(sim, cfg):
    m, d = sim
    ids = scene.scene_ids(m, cfg)
    cam = DepthCamera(m, cfg, ids)
    frame = cam.render(d)
    floor = (frame.geomid == ids.floor_geom).reshape(-1)
    assert floor.sum() > 100, "相機應該看得到地板"
    t = -frame.position[2] / frame.dirs_world[floor, 2]          # 光線和 z=0 平面的交點
    assert np.allclose(frame.depth.reshape(-1)[floor], t, atol=1e-6)


@pytest.mark.parametrize("dist", [1.5, 3.0, 4.5])
def test_ball_visible_and_located(sim, cfg, dist):
    m, d = sim
    ids = scene.scene_ids(m, cfg)
    cam = DepthCamera(m, cfg, ids)
    pos, R = cam.pose(d)
    target = pos + dist * R[:, 0] + 0.1 * R[:, 1]
    _place_ball(m, d, ids, 0, target, 0.1, cfg)
    frame = cam.render(d)
    assert cam.ball_mask(frame).sum() >= 1, "高解析度圖裡應該有球的像素"
    img = cam.policy_image(frame)
    assert img.shape == (cfg.camera.policy_res[1], cfg.camera.policy_res[0])
    assert img.min() < 1.0, "縮成 16x9 後球不能消失（取最小值縮圖）"
    assert np.isclose(img.min(), (dist - 0.1 - cfg.camera.near) / (cfg.camera.far - cfg.camera.near), atol=0.03)
    dets = BallDetector(cam, cfg).detect(frame)
    assert len(dets) == 1
    tol = 0.03 if dist < 3.5 else 0.08          # 遠處像素少，半徑估計較差
    assert np.linalg.norm(dets[0].position - target) < tol


def test_geometric_detector_without_segmentation(sim, cfg):
    m, d = sim
    ids = scene.scene_ids(m, cfg)
    cfg.tracker.mode = "geometric"
    cam = DepthCamera(m, cfg, ids)
    pos, R = cam.pose(d)
    target = pos + 2.5 * R[:, 0]
    target[2] = 0.5
    _place_ball(m, d, ids, 0, target, 0.1, cfg)
    dets = BallDetector(cam, cfg).detect(cam.render(d), d)
    assert len(dets) == 1, f"應該只找到 1 顆球，找到 {len(dets)}"
    assert np.linalg.norm(dets[0].position - target) < 0.05


def test_noise_magnitude(sim, cfg):
    m, d = sim
    cam = DepthCamera(m, cfg)
    frame = cam.render(d)
    frame.depth[:] = 3.0
    cfg.camera.pixel_dropout = 0.0
    noisy = cam.add_noise(frame, np.random.default_rng(0))
    expected = cfg.camera.noise_a + cfg.camera.noise_b * 9.0
    assert abs(np.std(noisy.depth - 3.0) - expected) < 0.15 * expected


def test_frame_stack_offsets():
    stack = DepthFrameStack((9, 16), (0, 3, 8, 18))
    assert stack.dim == 4 * 9 * 16
    assert np.all(stack.get() == 1.0), "重置後全部是 far"
    for k in range(30):
        stack.push(np.full((9, 16), float(k)))
    v = stack.get().reshape(4, 9 * 16)
    assert list(v[:, 0]) == [29, 26, 21, 11]
