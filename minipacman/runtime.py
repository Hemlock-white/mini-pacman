"""執行環境資訊：GPU、ROCm/CUDA、套件版本、git commit。

課程評分規定（Final Project Reference.pdf 第 10 頁，必要條件，沒達到就不評分）：
    * Every result you report must be produced on AMD ROCm. CUDA or CPU-only runs are not accepted.
    * The implementation must run on the Ryzen AI MAX+ 395 machines provided in class.

所以每次訓練和實驗都會呼叫 ``device_report()`` 存一份紀錄，不是 ROCm 時大聲警告。
注意：MuJoCo 模擬本來就在 CPU 上跑；「ROCm」指的是 PyTorch（policy 訓練與推論）要用 AMD GPU。
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
    """選 PyTorch 裝置。ROCm 版的 PyTorch 也是用 "cuda" 這個名字。"""
    import torch
    if requested:
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def device_report(device: str | None = None) -> dict:
    """回傳一個 dict，記錄這次執行的硬體與軟體版本。"""
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
    """不是在 AMD ROCm GPU 上跑時發出警告（結果不能放進報告）。"""
    if not info.get("rocm"):
        msg = (f"這次不是在 AMD ROCm GPU 上執行（device={info.get('device')}, gpu={info.get('gpu_name')}, "
               f"hip={info.get('torch_hip')}）。依課程規定，這次的結果不能放進報告，只能用來開發和除錯。")
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
