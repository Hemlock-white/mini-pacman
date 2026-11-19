"""Shared test fixtures.

Tests that need the TITA model are skipped automatically until it is downloaded (python scripts/download_tita.py).
"""
import mujoco
import numpy as np
import pytest

from minipacman.config import TITA_DIR, default_config

HAS_TITA = (TITA_DIR / "mujoco" / "robot.xml").exists()
need_tita = pytest.mark.skipif(not HAS_TITA, reason="download the TITA model first: python scripts/download_tita.py")


@pytest.fixture
def cfg():
    return default_config()


@pytest.fixture
def sim(cfg):
    if not HAS_TITA:
        pytest.skip("download the TITA model first")
    from minipacman import scene
    m, d = scene.load_model(None, cfg)
    mujoco.mj_forward(m, d)
    return m, d


class RobotFreezer:
    """Keep the robot still (restore its qpos/qvel every step).

    There is no balance controller yet when testing balls and the camera, and the robot falls in about 0.7 s, so we
    "pin" it. A ball hitting the pinned robot is like hitting an object with infinite mass.
    """

    def __init__(self, m, d, ids):
        self.d, self.nq, self.nv = d, ids.robot_nq, ids.robot_nv
        self.qpos = d.qpos[:self.nq].copy()
        self.qvel = np.zeros(self.nv)

    def restore(self):
        self.d.qpos[:self.nq] = self.qpos
        self.d.qvel[:self.nv] = self.qvel


def step(m, d, n, freezer=None, launcher=None):
    """Simulate n physics steps; with a launcher, check for hits every step."""
    for _ in range(n):
        mujoco.mj_step(m, d)
        if freezer is not None:
            freezer.restore()
        if launcher is not None:
            launcher.check_contacts()
