"""Download the pinned version of the TITA model into third_party/tita/. [YOURS]

Why the model isn't in the repo: tita_description in DDTRobot/ddt_ros2_control has no license (package.xml says TODO),
so we can't redistribute it. This script downloads a pinned commit from the official repo instead.
third_party/ is already in .gitignore.

Usage:
    python scripts/download_tita.py

What to do (four git commands; the parameters are in minipacman.config.TitaSource):
    git init <target_dir>
    git -C <target_dir> remote add origin <repo_url>
    git -C <target_dir> fetch --depth 1 origin <commit>
    git -C <target_dir> checkout FETCH_HEAD
Run them with subprocess.run([...], check=True). If it's already downloaded (<target_dir>/<subdir>/mujoco/robot.xml
exists), just exit. At the end print the model path and check that robot.xml exists.
Bonus: use sparse checkout to download only urdfs/tita_description (git sparse-checkout set <subdir>); it's faster.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minipacman.config import TitaSource  # noqa: E402


def main() -> None:
    src = TitaSource()
    raise NotImplementedError("TODO: download_tita.py (see the docstring at the top)")


if __name__ == "__main__":
    main()
