"""測試共用的 fixture。

需要 TITA 模型的測試，在模型還沒下載時會自動跳過（python scripts/download_tita.py）。
"""
import mujoco
import numpy as np
import pytest

from minipacman.config import TITA_DIR, default_config

HAS_TITA = (TITA_DIR / "mujoco" / "robot.xml").exists()
need_tita = pytest.mark.skipif(not HAS_TITA, reason="需要先下載 TITA 模型：python scripts/download_tita.py")


@pytest.fixture
def cfg():
    return default_config()


@pytest.fixture
def sim(cfg):
    if not HAS_TITA:
        pytest.skip("需要先下載 TITA 模型")
    from minipacman import scene
    m, d = scene.load_model(None, cfg)
    mujoco.mj_forward(m, d)
    return m, d


class RobotFreezer:
    """讓機器人固定不動（每一步把機器人的 qpos/qvel 還原）。

    測試球和相機時還沒有平衡控制器，機器人約 0.7 秒就會倒，所以先把它「釘住」。
    球撞到被釘住的機器人，就像撞到一個質量無限大的物體。
    """

    def __init__(self, m, d, ids):
        self.d, self.nq, self.nv = d, ids.robot_nq, ids.robot_nv
        self.qpos = d.qpos[:self.nq].copy()
        self.qvel = np.zeros(self.nv)

    def restore(self):
        self.d.qpos[:self.nq] = self.qpos
        self.d.qvel[:self.nv] = self.qvel


def step(m, d, n, freezer=None, launcher=None):
    """往前模擬 n 個物理步；有 launcher 時每步做命中判定。"""
    for _ in range(n):
        mujoco.mj_step(m, d)
        if freezer is not None:
            freezer.restore()
        if launcher is not None:
            launcher.check_contacts()
