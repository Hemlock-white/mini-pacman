"""lqr.py 的測試。【實作 lqr.py 和 env.py 的介面後，把下一行 pytestmark 刪掉】

這些數字都是原型（TITA_Feasibility_Check.ipynb，AUP 上跑過）量到的，拆檔之後應該不變。
"""
import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="TODO：實作 lqr.py 後刪掉這行")

from conftest import need_tita  # noqa: E402


def test_clip_to_envelope(cfg):
    from minipacman.lqr import clip_to_envelope
    c = cfg.command
    u = clip_to_envelope(np.array([5.0, 5.0, 5.0]), 0.0, c)
    assert np.allclose(u, [c.v_max_stand, c.w_max_stand, c.sdot_max])
    u = clip_to_envelope(np.array([5.0, -5.0, 5.0]), 1.0, c)
    assert np.allclose(u, [c.v_max_crouch, -c.w_max_crouch, 0.0]), "蹲到底：速度變慢、不能再往下蹲"
    u = clip_to_envelope(np.array([0.0, 0.0, -5.0]), 0.0, c)
    assert u[2] == 0.0, "站直時不能再往上"


def test_reference_rate_limit(cfg):
    from minipacman.lqr import ReferenceGenerator
    ref_gen = ReferenceGenerator(cfg)
    ref_gen.reset(np.zeros(2), 0.0)
    ref = ref_gen.update(np.array([0.8, 0.0, 0.0]), cfg.sim.timestep)
    assert ref.v <= cfg.command.accel_limit * cfg.sim.timestep + 1e-12, "速度指令不能一步跳上去"


@need_tita
def test_gains_are_finite_and_abad_zero(cfg):
    """DARE 要有解（只用機器人的 dof），而且 abad 關節的增益是 0。"""
    pytest.fail("TODO：建立 TitaEnv + LQRBalancer，design_all() 後檢查 K 沒有 nan、abad 欄位是 0")


@need_tita
@pytest.mark.parametrize("push", [30.0, 60.0, 100.0])
def test_survives_push(cfg, push):
    """站姿受到 0.1 s 的水平推力後不倒（原型：30/60/100 N 都站得住）。"""
    pytest.fail("TODO：模擬 3 秒，第 1 秒時對 base 施加 push N（d.xfrc_applied），檢查 |pitch| < 1")


@need_tita
def test_drive_speed(cfg):
    """直行指令 0.5 m/s：3 秒後速度約 0.5 m/s（原型 0.516），速度到 90% 約 0.36 s。"""
    pytest.fail("TODO")


@need_tita
def test_turning(cfg):
    """站姿 v=0.5、w=1.5 繞圈 2 秒：不倒，轉彎速度誤差 < 0.2 rad/s（筆電實測可行）。"""
    pytest.fail("TODO")
