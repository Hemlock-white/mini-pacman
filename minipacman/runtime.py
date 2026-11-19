"""Run environment info: GPU, ROCm/CUDA, package versions, git commit.

Course grading rules (Final Project Reference.pdf page 10, mandatory; miss either and the project is not graded):
    * Every result you report must be produced on AMD ROCm. CUDA or CPU-only runs are not accepted.
    * The implementation must run on the Ryzen AI MAX+ 395 machines provided in class.

So every training run and experiment calls ``device_report()`` and saves a record, and warns loudly when it is not
ROCm. Note: the MuJoCo simulation always runs on the CPU; "ROCm" means PyTorch (policy training and inference) must
use the AMD GPU.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

from .config import REPO_ROOT


def pick_device(requested: str | None = None) -> str:
    """Pick the PyTorch device. The ROCm build of PyTorch also calls it "cuda"."""
    import torch
    if requested:
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def device_report(device: str | None = None) -> dict:
    """Return a dict with the hardware and software versions of this run."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit(),
    }
    try:
        import mujoco
        info["mujoco"] = mujoco.__version__
    except ImportError:
        info["mujoco"] = None
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_hip"] = getattr(torch.version, "hip", None)
        info["torch_cuda"] = torch.version.cuda
        dev = device or pick_device()
        info["device"] = dev
        info["gpu_name"] = torch.cuda.get_device_name(0) if dev.startswith("cuda") else None
        info["rocm"] = bool(info["torch_hip"]) and dev.startswith("cuda")
    except ImportError:
        info.update(torch=None, torch_hip=None, torch_cuda=None, device="cpu", gpu_name=None, rocm=False)
    try:
        import rsl_rl  # noqa: F401
        from importlib.metadata import version
        info["rsl_rl"] = version("rsl-rl-lib")
    except Exception:
        info["rsl_rl"] = None
    return info


def warn_if_not_rocm(info: dict) -> None:
    """Warn when not running on an AMD ROCm GPU (results can't go in the report)."""
    if not info.get("rocm"):
        msg = (f"This run is NOT on an AMD ROCm GPU (device={info.get('device')}, gpu={info.get('gpu_name')}, "
               f"hip={info.get('torch_hip')}). Under the course rules these results can't go in the report; "
               f"use them for development and debugging only.")
        warnings.warn(msg, stacklevel=2)
        print("!" * 100 + "\n" + msg + "\n" + "!" * 100)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=5)
        commit = out.stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return f"{commit}{'-dirty' if dirty and commit else ''}" if commit else None
    except Exception:
        return None
