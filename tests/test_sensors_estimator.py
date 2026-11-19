"""sensors.py 和 state_estimator.py 的測試。【實作後把 pytestmark 刪掉】"""
import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="TODO：實作 sensors.py / state_estimator.py 後刪掉這行")


def test_imu_at_rest(sim, cfg):
    """機器人靜止（凍結）時：陀螺儀約 0，加速度計約 (0, 0, 9.81)（誤差在雜訊與偏差範圍內）。"""
    pytest.fail("TODO")


def test_encoder_quantization(sim, cfg):
    """encoder 角度是 2π / counts_per_rev 的整數倍。"""
    pytest.fail("TODO")


def test_estimator_tracks_truth(sim, cfg):
    """用真實狀態讓 LQR 開 3 秒（0.5 m/s），估計的前進速度誤差 < 0.05 m/s、位置誤差 < 0.1 m、pitch 誤差 < 2 度。"""
    pytest.fail("TODO")


def test_lqr_with_estimator(sim, cfg):
    """cfg.estimator.use_true_state = False 時，LQR 仍然能站穩並承受 60 N 推力。"""
    pytest.fail("TODO")
