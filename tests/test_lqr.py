"""Tests for lqr.py. [Delete the pytestmark line below once lqr.py and the env.py interface are implemented]

These numbers were all measured in the prototype (TITA_Feasibility_Check.ipynb, run on AUP); splitting the code into
files shouldn't change them.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="TODO: delete this line after implementing lqr.py")

from conftest import need_tita  # noqa: E402


def test_clip_to_envelope(cfg):
    from minipacman.lqr import clip_to_envelope
    c = cfg.command
    u = clip_to_envelope(np.array([5.0, 5.0, 5.0]), 0.0, c)
    assert np.allclose(u, [c.v_max_stand, c.w_max_stand, c.sdot_max])
    u = clip_to_envelope(np.array([5.0, -5.0, 5.0]), 1.0, c)
    assert np.allclose(u, [c.v_max_crouch, -c.w_max_crouch, 0.0]), "fully crouched: slower, and can't go lower"
    u = clip_to_envelope(np.array([0.0, 0.0, -5.0]), 0.0, c)
    assert u[2] == 0.0, "standing straight: can't go higher"


def test_reference_rate_limit(cfg):
    from minipacman.lqr import ReferenceGenerator
    ref_gen = ReferenceGenerator(cfg)
    ref_gen.reset(np.zeros(2), 0.0)
    ref = ref_gen.update(np.array([0.8, 0.0, 0.0]), cfg.sim.timestep)
    assert ref.v <= cfg.command.accel_limit * cfg.sim.timestep + 1e-12, "the speed command can't jump up in one step"


@need_tita
def test_gains_are_finite_and_abad_zero(cfg):
    """The DARE must be solvable (robot dofs only) and the abad joint gains must be 0."""
    pytest.fail("TODO: build TitaEnv + LQRBalancer, call design_all(), check K has no nan and the abad columns are 0")


@need_tita
@pytest.mark.parametrize("push", [30.0, 60.0, 100.0])
def test_survives_push(cfg, push):
    """Standing, the robot doesn't fall after a 0.1 s horizontal push (prototype: fine at 30/60/100 N)."""
    pytest.fail("TODO: simulate 3 s, push the base with `push` N at t=1 s (d.xfrc_applied), check |pitch| < 1")


@need_tita
def test_drive_speed(cfg):
    """Driving at 0.5 m/s: about 0.5 m/s after 3 s (prototype 0.516), about 0.36 s to reach 90% speed."""
    pytest.fail("TODO")


@need_tita
def test_turning(cfg):
    """Standing, circling at v=0.5, w=1.5 for 2 s: doesn't fall, turn-rate error < 0.2 rad/s (checked on a laptop)."""
    pytest.fail("TODO")
