"""Tests for sensors.py and state_estimator.py. [Delete pytestmark after implementing them]"""
import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="TODO: delete this line after implementing sensors.py / state_estimator.py")


def test_imu_at_rest(sim, cfg):
    """Robot still (pinned): gyro ≈ 0, accelerometer ≈ (0, 0, 9.81) (within noise and bias)."""
    pytest.fail("TODO")


def test_encoder_quantization(sim, cfg):
    """Encoder angles are integer multiples of 2π / counts_per_rev."""
    pytest.fail("TODO")


def test_estimator_tracks_truth(sim, cfg):
    """Drive 3 s at 0.5 m/s with the LQR on the true state: forward speed error < 0.05 m/s, position error < 0.1 m,
    pitch error < 2 degrees."""
    pytest.fail("TODO")


def test_lqr_with_estimator(sim, cfg):
    """With cfg.estimator.use_true_state = False, the LQR still stands and survives a 60 N push."""
    pytest.fail("TODO")
